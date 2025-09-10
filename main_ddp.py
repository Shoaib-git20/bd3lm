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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import dataloader
from diffusion_native import Diffusion 
import utils
import metrics

# DDP Setup
def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def is_main_process():
    """Checks if the current process is the main one (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

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

def _print_config(config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library on the main process."""
  if not is_main_process():
      return
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
    save_dir = config.checkpointing.save_dir
    os.makedirs(save_dir, exist_ok=True)
    with fsspec.open(f'{save_dir}/config_tree.txt', 'w') as fp:
      rich.print(tree, file=fp)

def _train(config, logger, tokenizer, device):
  """Main DDP training loop."""
  rank = int(os.environ["RANK"])
  local_rank = int(os.environ["LOCAL_RANK"])
  
  if is_main_process():
      logger.info('Starting DDP Training.')

  # --- Setup DDP DataLoaders ---
  train_set, valid_set = dataloader.get_dataloaders(config, tokenizer)

  train_sampler = DistributedSampler(train_set, shuffle=True)
  valid_sampler = DistributedSampler(valid_set, shuffle=False)

  train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.loader.batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=not config.data.streaming and train_sampler is None,
    sampler=train_sampler,
    persistent_workers=True)
  
  train_loader.tokenizer = tokenizer
  print("Train loader created and train tokenizer set is ready")

  valid_loader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=config.loader.eval_batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=False,
    sampler=valid_sampler,
    generator=None)
  # Will be used in generative perplexity calculation
  valid_loader.tokenizer = tokenizer
  print("Valid loader created and valid tokenizer set is ready")

  # --- Model, Optimizer, Scheduler ---
  model = Diffusion(config, tokenizer).to(local_rank)
  model = DDP(model, device_ids=[local_rank])
  
  optimizer, scheduler = get_optimizer_and_scheduler(model, config)

  # --- Checkpointing ---
  start_epoch = 0
  current_step = 0
  if config.checkpointing.resume_from_ckpt and config.checkpointing.resume_ckpt_path:
    if utils.fsspec_exists(config.checkpointing.resume_ckpt_path):
      if is_main_process():
          logger.info(f'Resuming training from {config.checkpointing.resume_ckpt_path}')
      map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
      checkpoint = torch.load(config.checkpointing.resume_ckpt_path, map_location=map_location)
      model.module.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      start_epoch = checkpoint.get('epoch', 0) + 1
      current_step = checkpoint.get('step', 0)
    else:
      if is_main_process():
          logger.warning(f"Checkpoint not found at {config.checkpointing.resume_ckpt_path}. Starting from scratch.")
  
  # --- Training Loop ---
  max_steps = config.trainer.max_steps
  training_complete = False

  for epoch in range(start_epoch, 1000): # Loop for a large number of epochs
    train_sampler.set_epoch(epoch)
    model.train()
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", disable=not is_main_process())
    
    for batch in train_pbar:
        if current_step >= max_steps:
            training_complete = True
            break
            
        input_ids = batch['input_ids'].to(local_rank)
        attention_mask = batch['attention_mask'].to(local_rank)

        optimizer.zero_grad()
        loss_obj = model.module.compute_loss(input_ids, attention_mask)
        loss = loss_obj.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if model.module.ema:
            model.module.ema.update(model.parameters())

        current_step += 1
        if is_main_process():
            train_pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0], "step": current_step})

    # --- Validation ---
    #model.eval()
    #val_loss = 0
    #with torch.no_grad():
    #    for batch in valid_loader:
    #        input_ids = batch['input_ids'].to(local_rank)
    #        attention_mask = batch['attention_mask'].to(local_rank)
    #        loss_obj = model.compute_loss(input_ids, attention_mask)
    #        val_loss += loss_obj.loss.item()
    #
    ## Gather validation loss
    #val_loss_tensor = torch.tensor(val_loss).to(local_rank)
    #dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
    #total_val_samples = len(valid_sampler)
    #avg_val_loss = val_loss_tensor.item() / total_val_samples
#
    #if is_main_process():
    #    logger.info(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f}")
#
    ## --- Save Checkpoint ---
    #if is_main_process() and (epoch + 1) % config.checkpointing.save_interval == 0:
    #    ckpt_path = os.path.join(config.checkpointing.save_dir, f"epoch_{epoch+1}_step_{current_step}.pt")
    #    os.makedirs(config.checkpointing.save_dir, exist_ok=True)
    #    torch.save({
    #        'epoch': epoch,
    #        'step': current_step,
    #        'model_state_dict': model.module.state_dict(),
    #        'optimizer_state_dict': optimizer.state_dict(),
    #        'scheduler_state_dict': scheduler.state_dict(),
    #        'loss': avg_val_loss,
    #    }, ckpt_path)
    #    logger.info(f"Checkpoint saved to {ckpt_path}")

    if training_complete:
        if is_main_process():
            logger.info(f"Reached max_steps ({max_steps}). Stopping training.")
        break

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
  """Main entry point for DDP training."""
  # DDP is assumed to be used if this script is run
  use_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
  if use_ddp:
      setup_ddp()
  
  # Ensure each process has a different seed
  seed = config.seed + int(os.environ.get("RANK", 0))
  torch.manual_seed(seed)
  
  logger = utils.get_logger(__name__)
  device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}") if use_ddp else torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = dataloader.get_tokenizer(config)

  if is_main_process():
      _print_config(config, resolve=True, save_cfg=True)

  if config.mode == 'train':
    _train(config, logger, tokenizer, device)
  else:
      if is_main_process():
          logger.warning("sample_eval and ppl_eval modes are not configured for DDP. Please run them on a single GPU using main_native.py.")

  if use_ddp:
      cleanup_ddp()

if __name__ == '__main__':
  main()
