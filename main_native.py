import os
import fsspec
import hydra
import omegaconf
import rich.syntax
import rich.tree
import torch
import transformers
from tqdm import tqdm
import pandas as pd

import dataloader
from diffusion_native import Diffusion 
import utils
import metrics

from torch.profiler import profile, record_function, ProfilerActivity

# Registering custom resolvers for OmegaConf
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

def get_optimizer_and_scheduler(model, config):
    """Create optimizer and learning rate scheduler."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        betas=(config.optim.beta1, config.optim.beta2),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay
    )
    
    scheduler = hydra.utils.instantiate(
        config.lr_scheduler, optimizer=optimizer
    )
    return optimizer, scheduler

def _load_from_checkpoint(config, tokenizer, device):
  """Loads a model from a checkpoint."""
  if 'hf' in config.algo.backbone:
    model = Diffusion(config, tokenizer=tokenizer).to(device)
    # If it's a huggingface model, the logic to load might be different
    # For now, we assume it's a raw state_dict
    if os.path.exists(config.eval.checkpoint_path):
        state_dict = torch.load(config.eval.checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
  else:
    model = Diffusion(config, tokenizer=tokenizer).to(device)
    if os.path.exists(config.eval.checkpoint_path):
        checkpoint = torch.load(config.eval.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
  return model

def _print_config(config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library."""
  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)
  for field in config.keys():
    branch = tree.add(field, style=style, guide_style=style)
    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(config_section, resolve=resolve)
    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(f'{config.checkpointing.save_dir}/config_tree.txt', 'w') as fp:
      rich.print(tree, file=fp)

def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  """Prints a batch from the dataloader for verification."""
  for dl_type, dl in [('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)

def generate_samples(config, logger, tokenizer, device):
  """Generates samples from a trained model."""
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer, device=device)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  
  text_samples, nfes = model.sample(num_steps=config.algo.T)
  
  # Metrics calculation
  gen_metrics = metrics.Metrics(config)
  gen_metrics.record_generative_perplexity(
      text_samples,
      config.model.length,
      config.loader.eval_batch_size,
      device
  )

  print('Text samples:', text_samples)
  print('Generative perplexity:', gen_metrics.gen_ppl.compute())
  print('Entropy:', gen_metrics.gen_entropy.compute())
  
  csv_path = config.sampling.logdir
  save_dict = {
      'gen_ppl': gen_metrics.gen_ppls,
      'gen_nfes': nfes,
      'gen_entropy': gen_metrics.gen_entropies,
      'gen_lengths': gen_metrics.gen_lengths,
      'samples': [[i] for i in text_samples],
      'seed': [config.seed for _ in range(len(text_samples))]
  }
  if config.sampling.var_length:
    save_dict['samples'] = ['' for _ in range(len(text_samples))]
  utils.update_and_save_csv(save_dict, csv_path)
  return text_samples

def _ppl_eval(config, logger, tokenizer, device):
  """Performs perplexity evaluation on the validation set."""
  logger.info('Starting PPL Eval.')
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer, device=device)

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  _, valid_ds = dataloader.get_dataloaders(config, tokenizer, skip_train=True, valid_seed=config.seed)
  
  model.eval()
  val_loss = 0
  val_pbar = tqdm(valid_ds, desc="Perplexity Evaluation")
  with torch.no_grad():
      for batch in val_pbar:
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          loss_obj = model.compute_loss(input_ids, attention_mask)
          val_loss += loss_obj.loss.item()
          val_pbar.set_postfix({"loss": loss_obj.loss.item()})

  avg_val_loss = val_loss / len(valid_ds)
  perplexity = torch.exp(torch.tensor(avg_val_loss))
  print(f"Validation Perplexity: {perplexity.item():.4f}")
  return perplexity.item()


def _train(config, logger, tokenizer, device):
  """Main training loop."""
  logger.info('Starting Training.')

  # --- Setup ---
  train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  model = Diffusion(config, tokenizer).to(device)
  optimizer, scheduler = get_optimizer_and_scheduler(model, config)

  # --- Checkpointing ---
  start_epoch = 0
  iteration_step = 0
  max_steps = config.trainer.max_steps
  if config.checkpointing.resume_from_ckpt and config.checkpointing.resume_ckpt_path:
    if utils.fsspec_exists(config.checkpointing.resume_ckpt_path):
      logger.info(f'Resuming training from {config.checkpointing.resume_ckpt_path}')
      checkpoint = torch.load(config.checkpointing.resume_ckpt_path, map_location=device)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      start_epoch = checkpoint.get('epoch', 0) + 1
    else:
      logger.warning(f"Checkpoint not found at {config.checkpointing.resume_ckpt_path}. Starting from scratch.")
  elif config.training.from_pretrained:
      logger.info(f'Loading pretrained model from {config.training.from_pretrained}')
      if 'kuleshov-group/' in config.training.from_pretrained:
          hf_model = transformers.AutoModelForMaskedLM.from_pretrained(
              config.training.from_pretrained, trust_remote_code=True
          )
          model.load_state_dict(hf_model.state_dict(), strict=False)
      else:
          pretrained_ckpt = torch.load(config.training.from_pretrained, map_location=device)
          model.load_state_dict(pretrained_ckpt, strict=False)
  #with profile(
  #    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
  #    schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
  #    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir/bd3lm'),
  #    record_shapes=False,
  #    profile_memory=True,
  #    with_stack=True,
  #    with_flops=False,
  #) as prof:
  #    for _ in range(1):
  #        prof.step()
  #        with record_function("BD3LM_sample"):
    # --- Training Loop ---
  for epoch in range(start_epoch, config.trainer.max_epochs):
    # --- Training ---
    model.train()
    train_loss = 0
    train_pbar = tqdm(train_ds, desc=f"Epoch {epoch+1}/{config.trainer.max_epochs} [Training]")
    for batch in train_pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        optimizer.zero_grad()
        loss_obj = model.compute_loss(input_ids, attention_mask)
        loss = loss_obj.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if model.ema:
            model.ema.update(model.parameters())
        train_loss += loss.item()
        train_pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
        iteration_step += 1
        if iteration_step >= max_steps:
            break
          
    avg_train_loss = train_loss / iteration_step
    logger.info(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")

    # --- Validation ---
    #model.eval()
    #val_loss = 0
    #val_pbar = tqdm(valid_ds, desc=f"Epoch {epoch+1}/{config.training.num_epochs} [Validation]")
    #with torch.no_grad():
    #    for batch in val_pbar:
    #        input_ids = batch['input_ids'].to(device)
    #        attention_mask = batch['attention_mask'].to(device)
    #        loss_obj = model.compute_loss(input_ids, attention_mask)
    #        val_loss += loss_obj.loss.item()
    #        val_pbar.set_postfix({"loss": loss_obj.loss.item()})
#
    #avg_val_loss = val_loss / len(valid_ds)
    #logger.info(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f}")
#
    ## --- Save Checkpoint ---
    #if (epoch + 1) % config.checkpointing.save_interval == 0:
    #    ckpt_path = os.path.join(config.checkpointing.save_dir, f"epoch_{epoch+1}.pt")
    #    os.makedirs(config.checkpointing.save_dir, exist_ok=True)
    #    torch.save({
    #        'epoch': epoch,
    #        'model_state_dict': model.state_dict(),
    #        'optimizer_state_dict': optimizer.state_dict(),
    #        'scheduler_state_dict': scheduler.state_dict(),
    #        'loss': avg_val_loss,
    #    }, ckpt_path)
    #    logger.info(f"Checkpoint saved to {ckpt_path}")
    if iteration_step >= max_steps:
        logger.info(f"Reached max steps {max_steps}. Ending training.")
        break
  #print("Profiler run complete. Printing summary...")
  #print("-" * 50)
  #print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=50))
  #print("\n" + "-" * 50)
  #print("To view the detailed trace, run the following command in your terminal:")
  #print("tensorboard --logdir=./log")
  #print("-" * 50)

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
  """Main entry point."""
  torch.manual_seed(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    generate_samples(config, logger, tokenizer, device)
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer, device)
  else:
    _train(config, logger, tokenizer, device)

if __name__ == '__main__':
  main()