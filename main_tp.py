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
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
    fully_shard,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
)

from torch.distributed._symmetric_memory import enable_symm_mem_for_group
import torch._inductor
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.fsdp.wrap import wrap as fsdp_wrap

import dataloader
from diffusion_native import Diffusion
from models.native_dit import DDiTBlock, DDiTBlockCausal, TimestepEmbedder, EmbeddingLayer, DDiTFinalLayer 
import utils
import metrics
from dataclasses import dataclass
import re


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor

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
    dtensor_params = []
    local_params = []
    
    for p in model.parameters():
        if p.requires_grad:
            if isinstance(p, DTensor) or isinstance(p.data, DTensor):
                dtensor_params.append(p)
            else:
                local_params.append(p)

    param_groups = [
        {'params': dtensor_params, 'name': 'dtensor_group'},
        {'params': local_params, 'name': 'local_group'}
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.optim.lr,
        betas=(config.optim.beta1, config.optim.beta2),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay
    )
    
    scheduler = hydra.utils.instantiate(
        config.lr_scheduler, optimizer=optimizer
    )
    return optimizer, scheduler

def clip_grad_norm_hybrid(parameters, max_norm, norm_type=2.0):
    """
    Clips gradients for a model containing both DTensors (TP) and standard Tensors (Local).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    # Filter parameters that have gradients
    params = [p for p in parameters if p.grad is not None]
    
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    total_norm = 0.0
    
    for p in params:
        grad = p.grad.detach()
        if isinstance(grad, DTensor):
            param_norm = grad.norm(norm_type)
            total_norm += param_norm.to_local().item() ** norm_type
        else:
            param_norm = grad.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
            
    total_norm = total_norm ** (1.0 / norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for p in params:
            p.grad.detach().mul_(clip_coef)
            
    return total_norm

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


def debug_types(x, shift, scale, prefix="modulate_debug"):
    def tinfo(t):
        if isinstance(t, DTensor):
            return f"DTensor shape={tuple(t.shape)} placements={getattr(t, 'placements', None)}"
        elif isinstance(t, torch.Tensor):
            return f"Tensor shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"
        else:
            return f"Other type: {type(t)}"
    print(f"[{prefix}] x: {tinfo(x)}")
    print(f"[{prefix}] shift: {tinfo(shift)}")
    print(f"[{prefix}] scale: {tinfo(scale)}")

def summarize_param_types(model):
    dtensor_params = []
    tensor_params = []
    for name, p in model.named_parameters():
        if isinstance(p, DTensor):
            dtensor_params.append((name, getattr(p, "placements", None), tuple(p.shape)))
        else:
            tensor_params.append((name, p.shape))
    if is_main_process():
        print("DTensor params (sharded):")
    for n, placements, shape in dtensor_params[:200]:
        if is_main_process():
            print(f"  {n} shape={shape} placements={placements}")
    if is_main_process():
        print(f"... total DTensor params: {len(dtensor_params)}")
        print("Regular torch.Tensor params: (sample)")
    for n, shape in tensor_params[:200]:
        if is_main_process():
            print(f"  {n} shape={shape}")
    if is_main_process():
        print(f"... total tensor params: {len(tensor_params)}")
    return dtensor_params, tensor_params

def module_contains_dtensor(module) -> bool:
    """
    Returns True if *any* parameter inside `module` (recursively) is a DTensor.
    """
    for p in module.parameters(recurse=True):
        if isinstance(p, DTensor):
            return True
    return False

def _train(config, logger, tokenizer, device):
  """Main distributed training loop."""
  rank = int(os.environ["RANK"])
  local_rank = int(os.environ["LOCAL_RANK"])
  
  if is_main_process():
      logger.info(f'Starting {config.strategy.name} Training with world size {os.environ["WORLD_SIZE"]}.')

  # --- Model ---
  model = Diffusion(config, tokenizer)
  train_set, valid_set = dataloader.get_dataloaders(config, tokenizer)

  dp_size = None
  dp_rank = None
  tp_mesh = None
  dp_mesh = None
  mesh_2d = None
  device_type = torch.accelerator.current_accelerator().type
  
  if config.strategy.name == 'tp':
    world_size = dist.get_world_size()
    tp_size = 4
    assert world_size % (tp_size) == 0, f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
    dp_size = world_size // (tp_size)
    model = model.to(device_type)
    model, dp_mesh, tp_mesh, mesh_2d = apply_native_tp(model, config, tp_size, dp_size)
    device_type = torch.accelerator.current_accelerator().type
    if is_main_process():
        print(f"[Debug][3D] Initializing 2D device mesh with shape ({dp_size}, {tp_size})")
    
    mesh_2d = init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

    if is_main_process():
        print(f"[Debug][3D] Device mesh initialized: {mesh_2d}")

    tp_mesh = mesh_2d["tp"]
    dp_mesh = mesh_2d["dp"]

    tp_plan = {
        "backbone.vocab_embed.embedding": RowwiseParallel(input_layouts=Replicate(),),

        # "backbone.output_layer.norm_final": SequenceParallel(sequence_dim=1),

        # "backbone.output_layer.linear": PrepareModuleInput(
        #     input_layouts=Shard(1), 
        #     desired_input_layouts=Replicate()
        # ),
        "backbone.output_layer.linear": ColwiseParallel(
            output_layouts=Replicate()
        ),
    }

    # --- Looping over DiT Blocks ---
    n_blocks = len(model.backbone.blocks)
    for i in range(n_blocks):
        p = f"backbone.blocks.{i}"

        tp_plan[f"{p}.atten.attn_qkv"] = ColwiseParallel(use_local_output=False)
        tp_plan[f"{p}.atten.attn_out"] = RowwiseParallel()

        tp_plan[f"{p}.mlp.w1"] = ColwiseParallel()

        tp_plan[f"{p}.mlp.w2"] = RowwiseParallel()

    parallelize_module(model, tp_mesh, parallelize_plan=tp_plan)
    torch.cuda.synchronize()

    #model = fully_shard(model, mesh=dp_mesh)
    torch.cuda.synchronize()
    dp_rank = dp_mesh.get_local_rank()
    torch.manual_seed(dp_rank + 100)
    summarize_param_types(model)

    train_sampler = DistributedSampler(train_set, num_replicas=dp_size, rank=dp_rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_set, num_replicas=dp_size, rank=dp_rank, shuffle=False)
  elif config.strategy.name == 'ddp':
    model = DDP(model.to(local_rank), device_ids=[local_rank])
    train_sampler = DistributedSampler(train_set, shuffle=True)
    valid_sampler = DistributedSampler(valid_set, shuffle=False)
  else:
    raise ValueError(f"Unknown distributed strategy: {strategy_name}")

  print(f"Model type after whole setup: {type(model)}")
  # --- Setup DataLoaders ---

  train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.loader.batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    #shuffle=not config.data.streaming and train_sampler is None,
    shuffle=False,
    sampler=train_sampler,
    persistent_workers=True)
  
  train_loader.tokenizer = tokenizer
  logger.info("Train loader created and train tokenizer set is ready")

  valid_loader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=config.loader.eval_batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=False,
    sampler=valid_sampler,
    generator=None)
  valid_loader.tokenizer = tokenizer
  logger.info("Valid loader created and valid tokenizer set is ready")

  # --- Optimizer, Scheduler ---

  optimizer, scheduler = get_optimizer_and_scheduler(model, config)

  # --- Checkpointing ---
  start_epoch = 0
  current_step = 0
  sampling_eps_min = None
  sampling_eps_max = None

  if config.checkpointing.resume_from_ckpt and config.checkpointing.resume_ckpt_path:
    if utils.fsspec_exists(config.checkpointing.resume_ckpt_path):
      if is_main_process():
          logger.info(f'Resuming training from {config.checkpointing.resume_ckpt_path}')
      
      if config.strategy.name == 'fsdp':
          with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
              checkpoint = torch.load(config.checkpointing.resume_ckpt_path)
              model.load_state_dict(checkpoint['model_state_dict'])
      else: # DDP
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

  if config.strategy.name in ('ddp', 'fsdp', 'tp', '3d', 'async_tp'):
    model.sampling_eps_min = torch.tensor(config.training.sampling_eps_min)
    model.sampling_eps_max = torch.tensor(config.training.sampling_eps_max)

  for epoch in range(start_epoch, 1000): # Loop for a large number of epochs
    train_sampler.set_epoch(epoch)
    model.train()
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", disable=not is_main_process())
    
    for batch in train_pbar:
        if current_step >= max_steps:
            training_complete = True
            break
            
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)

        optimizer.zero_grad()
        if config.strategy.name in ('fsdp', 'tp', '3d', 'async_tp'):
            if sampling_eps_min is None and hasattr(model, 'sampling_eps_min'):
              sampling_eps_min = model.sampling_eps_min.item()
              sampling_eps_max = model.sampling_eps_max.item()
            elif not hasattr(model, 'sampling_eps_min'):
              sampling_eps_min = 1e-3
              sampling_eps_max = 1.0

            (input_tokens, output_tokens, attention_mask) = model._maybe_sub_sample(input_ids, attention_mask)

            if model.parameterization == 'ar':
              output = model.forward(input_tokens, None)
              loss = - output.gather(-1, output_tokens[:, :, None])[:, :, 0]
            else:
              loss = model._forward_pass_diffusion(
                        input_tokens, sampling_eps_min=sampling_eps_min, sampling_eps_max=sampling_eps_max)
            if model.ignore_bos and not model.training:
                attention_mask[:, 0] = 0

            nlls = (loss * attention_mask)
            token_nll = nlls.sum() / attention_mask.sum()
            loss_obj = Loss(loss=token_nll, nlls=nlls, token_mask=attention_mask)
        else:
            loss_obj = model.module.compute_loss(input_ids, attention_mask)
        
        loss = loss_obj.loss
        
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if config.strategy.name == '3d':
            local_grads = [p.grad for p in model.parameters()
               if p.grad is not None and not isinstance(p.grad, DTensor)]

            for g in local_grads:
                dist.all_reduce(g, op=dist.ReduceOp.AVG)

        clip_grad_norm_hybrid(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

        if config.strategy.name in ('fsdp', 'tp', '3d', 'async_tp'):
            if model.ema:
                model.ema.update(model.parameters())
        else:
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
        # --- Save Checkpoint ---
        #if is_main_process() and (epoch + 1) % config.checkpointing.save_interval == 0:
        #    ckpt_path = os.path.join(config.checkpointing.save_dir, f"epoch_{epoch+1}_step_{current_step}.pt")
        #    os.makedirs(config.checkpointing.save_dir, exist_ok=True)
        #    
        #    if config.strategy.name == 'fsdp':
        #        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        #            model_state = model.state_dict()
        #    else: # DDP
        #        model_state = model.module.state_dict()
#
        #    torch.save({
        #        'epoch': epoch,
        #        'step': current_step,
        #        'model_state_dict': model_state,
        #        'optimizer_state_dict': optimizer.state_dict(),
        #        'scheduler_state_dict': scheduler.state_dict(),
        #        'loss': avg_val_loss,
        #    }, ckpt_path)
        #    logger.info(f"Checkpoint saved to {ckpt_path}")

    if current_step >= max_steps:
        if is_main_process():
            logger.info(f"Reached max_steps ({max_steps}). Stopping training.")
        break

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
  """Main entry point for DDP training."""
  # DDP is assumed to be used if this script is run
  use_ddp = int(os.environ.get("WORLD_SIZE", 1)) >= 1
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
          logger.warning("sample_eval and ppl_eval modes are not configured yet")

  if use_ddp:
      cleanup_ddp()

if __name__ == '__main__':
  main()