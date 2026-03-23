import os
import time
import pynvml
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
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

import dataloader
from diffusion_native import Diffusion
from models.native_dit import DDiTBlock, DDiTBlockCausal, TimestepEmbedder, EmbeddingLayer, DDiTFinalLayer 
import utils
import metrics
from dataclasses import dataclass
import re

def debug_topology_and_network(mesh_2d):
    global_rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    hostname = dist.gethostname()
    
    # Extract the actual global ranks in this GPU's TP and DP groups
    tp_group = mesh_2d["tp"].get_group()
    dp_group = mesh_2d["dp"].get_group()
    
    tp_ranks = dist.get_process_group_ranks(tp_group)
    dp_ranks = dist.get_process_group_ranks(dp_group)
    
    # Grab all NCCL environment variables to check the network backend
    #nccl_env = {k: v for k, v in os.environ.items() if "NCCL" in k}
    
    # Print the diagnostics
    print(f"[Host: {hostname} | Global Rank: {global_rank} | Local Rank: {local_rank}] \n"
          f"  -> TP Ranks (must be on same host!): {tp_ranks}\n"
          f"  -> DP Ranks (cross-host is fine): {dp_ranks}\n"
          #f"  -> NCCL Env: {nccl_env}\n"
          f"-"*60)
    
    # Force all GPUs to wait so the prints don't scramble
    dist.barrier()

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor

# DDP Setup
def setup_distributed():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_distributed():
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

def clip_grad_norm_hybrid(parameters, max_norm, norm_type=2.0, tp_size=1):
    """
    Clips gradients for a model containing both DTensors (TP) and standard Tensors (Local).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    params = [p for p in parameters if p.grad is not None]

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    total_sq_norm = torch.zeros((), device=params[0].grad.device, dtype=torch.float32)

    for p in params:
        g = p.grad.detach()

        if isinstance(g, DTensor):
            is_replicated = any(isinstance(placement, Replicate) for placement in g.placements)
            local_grad = g.to_local()
        else:
            is_replicated = True
            local_grad = g
        
        param_sq_norm = torch.sum(local_grad.float() ** norm_type)
        
        if is_replicated:
            param_sq_norm /= tp_size

        total_sq_norm += param_sq_norm

    if dist.is_initialized():
        dist.all_reduce(total_sq_norm, op=dist.ReduceOp.SUM)

    total_norm = total_sq_norm.pow(1.0 / norm_type)
    if torch.isnan(total_norm) or torch.isinf(total_norm):
        print(f"[Warning] Gradient norm is {total_norm}. Skipping clipping.")
        return total_norm
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)

    if clip_coef < 1:
        for p in params:
            if isinstance(p.grad, DTensor):
                p.grad = p.grad * clip_coef
            else:
                p.grad.mul_(clip_coef)

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

def _train(config, logger, tokenizer, device, nvml_handle):
  """Main distributed training loop."""
  rank = int(os.environ["RANK"])
  local_rank = int(os.environ["LOCAL_RANK"])
  
  if is_main_process():
      logger.info(f'Starting {config.strategy.name} training with world size {os.environ["WORLD_SIZE"]}.')

  seed = config.seed
  #torch.manual_seed(seed)
  #torch.cuda.manual_seed(seed)
  #torch.cuda.manual_seed_all(seed)

  # --- Model ---
  model = Diffusion(config, tokenizer)
  model = model.to(device)

  tp_mesh = None
  dp_mesh = None
  
  if config.strategy.name == '3d':
    # -------------------------------------------------
    # APPLY TPARALLELISM TO MODEL
    # -------------------------------------------------
    world_size = dist.get_world_size()
    tp_size = config.strategy.tp_degree
    dp_size = config.strategy.dp_degree
    if is_main_process():
        logger.info(f"[TP] World size: {world_size}, TP size: {tp_size}, DP size: {dp_size}, device type {device.type}")
    mesh_2d = init_device_mesh(device.type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
    tp_mesh = mesh_2d["tp"]
    dp_mesh = mesh_2d["dp"]

    if is_main_process():
        print(f"[Debug][3d] Device mesh initialized: {mesh_2d}\n - TP mesh: {tp_mesh}\n - DP mesh: {dp_mesh}")
    
    # --- Define TP plan ---
    tp_plan = {
        "backbone.vocab_embed.embedding": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1), use_local_output=False),
        "backbone.sigma_map.mlp.0": ColwiseParallel(),
        "backbone.sigma_map.mlp.2": RowwiseParallel(),
        "backbone.output_layer.adaLN_modulation": ColwiseParallel(input_layouts=Replicate(), use_local_output=False),
        "backbone.output_layer.norm_final": SequenceParallel(),
        "backbone.output_layer.linear": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
    }

    ## --- Looping over DiT Blocks ---
    n_blocks = len(model.backbone.blocks)
    for i in range(n_blocks):
        p = f"backbone.blocks.{i}"

        tp_plan[f"{p}.adaLN_modulation"] = ColwiseParallel(input_layouts=Replicate(), use_local_output=False)

        tp_plan[f"{p}.norm1"] = SequenceParallel()

        tp_plan[f"{p}.atten"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )

        tp_plan[f"{p}.atten.attn_qkv"] = ColwiseParallel(use_local_output=False)
        tp_plan[f"{p}.atten.attn_out"] = RowwiseParallel(output_layouts=Shard(1), use_local_output=False)
        
        tp_plan[f"{p}.norm2"] = SequenceParallel()
        tp_plan[f"{p}.mlp"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        tp_plan[f"{p}.mlp.w1"] = ColwiseParallel()
        tp_plan[f"{p}.mlp.w2"] = RowwiseParallel(use_local_output=False)

    parallelize_module(model, tp_mesh, parallelize_plan=tp_plan)
    if is_main_process():
        print("[Debug] Tensor Parallelism applied")

    # -------------------------------------------------
    # APPLY FSDP SHARDING ON TOP OF TP-PARALLELIZED MODEL
    # -------------------------------------------------

    fsdp_config = {
        "mesh": dp_mesh,
        "reshard_after_forward": False,
    }

    # shard embedding
    fully_shard(
        model.backbone.vocab_embed.embedding,
        **fsdp_config,
    )

    # shard each transformer block
    for block in model.backbone.blocks:
        fully_shard(
            block,
            **fsdp_config,
        )

    # shard output layers
    fully_shard(
        [
            model.backbone.output_layer.norm_final,
            model.backbone.output_layer.linear,
        ],
        **fsdp_config,
    )

    # final root wrap
    fully_shard(model.backbone, **fsdp_config)

    torch.cuda.synchronize()

    if is_main_process():
        print("[Debug][3d] Applied TP + FSDP to model")

    #summarize_param_types(model)

    #debug_topology_and_network(mesh_2d)

    if is_main_process():
        print(f"[Debug][3d] Model parallelized according to TP plan. Uncomment the debug line at 366 in main_3d.py to see parameter types.")

  else:
    raise ValueError(f"Not/Unknown distributed strategy {strategy_name} for main_3d.py file type")
  if is_main_process():
    print(f"Model type after whole setup: {type(model)}")

  # ---- GPU utilization trackers ----
  torch.cuda.reset_peak_memory_stats()
  max_gpu_util = 0
  max_mem_util = 0

  # --- Setup DataLoaders ---

  train_set, valid_set = dataloader.get_dataloaders(config, tokenizer)
  dp_rank = dp_mesh.get_local_rank()
  dp_world_size = dp_mesh.size()

  train_sampler = DistributedSampler( train_set, num_replicas=dp_world_size, rank=dp_rank, shuffle=True )
  valid_sampler = DistributedSampler( valid_set, num_replicas=dp_world_size, rank=dp_rank, shuffle=False )

  torch.manual_seed(config.seed)

  train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      sampler=train_sampler,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      persistent_workers=True,
  )
  train_loader.tokenizer = tokenizer
  logger.info("Train loader created and train tokenizer set is ready")

  valid_loader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=config.loader.eval_batch_size,
    sampler=valid_sampler,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=False,
    )
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
  accumulation_steps = config.trainer.accumulate_grad_batches
  training_complete = False
  start_time = time.time()
  step_start_time = time.time()

  if config.strategy.name in ('ddp', 'fsdp', 'tp', '3d', 'async_tp'):
    model.sampling_eps_min = torch.tensor(config.training.sampling_eps_min)
    model.sampling_eps_max = torch.tensor(config.training.sampling_eps_max)

  for epoch in range(start_epoch, 1000): # Loop for a large number of epochs
    train_sampler.set_epoch(epoch)
    model.train()
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", disable=not is_main_process())
    with profile(
      activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
      schedule=schedule(wait=1, warmup=2, active=3, repeat=1),
      on_trace_ready=tensorboard_trace_handler('./profiler_logs/run_2node_tp4_dp2'),
      record_shapes=False,
      profile_memory=True,
      with_stack=True 
    ) as prof:
      for batch in train_pbar:
        if current_step >= max_steps:
          training_complete = True
          break
        with record_function("train_step"):
          iter_start = time.time()
          with record_function("data_transfer"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
          with record_function("forward_pass"):
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
          with record_function("backward_pass"):
            loss.backward()
            clip_grad_norm_hybrid(model.parameters(), 1.0, tp_size=tp_size)
          if (current_step + 1) % accumulation_steps == 0:
            with record_function("optimizer_step"):
              optimizer.step()
              optimizer.zero_grad()
            with record_function("lr_scheduler_step"):
              scheduler.step()
            with record_function("ema_update"):
              if config.strategy.name in ('fsdp', 'tp', '3d', 'async_tp'):
                if model.ema:
                    model.ema.update(model.parameters())
              else:
                if model.module.ema:
                    model.module.ema.update(model.parameters())
          
            with record_function("log_metrics"):
              # ---- GPU utilization sampling ----
              util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
              max_gpu_util = max(max_gpu_util, util.gpu)
              max_mem_util = max(max_mem_util, util.memory)
              iter_end = time.time()
              iter_time = iter_end - iter_start
              tokens_processed = input_ids.numel()*accumulation_steps 
              tps = tokens_processed / iter_time
              if is_main_process():
                loss_val = loss.item()*accumulation_steps
                lr_val = scheduler.get_last_lr()[0]
                train_pbar.set_postfix({
                    "loss": f"{loss_val:.4f}", 
                    "lr": f"{lr_val:.2e}", 
                    "step": current_step,
                    "ms/it": f"{iter_time*1000:.1f}"
                })
                logger.info(
                    f"Step {current_step} | Epoch {epoch+1} | "
                    f"Loss: {loss_val:.4f} | LR: {lr_val:.6f} | "
                    f"Time: {iter_time*1000:.2f}ms | Tokens/s: {tps:.0f}"
                )
                logger.info(
                    f"[Rank {dist.get_rank()}] | "
                    f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB | "
                    f"GPU util: {util.gpu}% | Mem util: {util.memory}%"
                )
        current_step += 1
        prof.step()
    if current_step >= max_steps:
      if is_main_process():
        logger.info(f"Reached max_steps ({max_steps}). Stopping training.")
      peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
      logger.info(f"[Rank {dist.get_rank()}] "f"Peak GPU memory allocated: {peak_mem_mb:.2f} MB | "f"Peak GPU util: {max_gpu_util}% | "f"Peak MEM util: {max_mem_util}%")
      break

                #logger.info(f"memory summary:\n{torch.cuda.memory_summary()}")
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

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
  """Main entry point for DDP training."""
  
  use_distributed = int(os.environ.get("WORLD_SIZE", 1)) >= 1
  if is_main_process():
      print("Distributed training:", use_distributed)
  if use_distributed:
      setup_distributed()
  
  # torch.cuda.empty_cache()
  # ---- NVML INIT (per process / per rank) ----
  pynvml.nvmlInit()
  local_rank = int(os.environ["LOCAL_RANK"])
  nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

  logger = utils.get_logger(__name__)
  device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}") if use_distributed else torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = dataloader.get_tokenizer(config)

  if is_main_process():
      _print_config(config, resolve=True, save_cfg=True)

  if config.mode == 'train':
    _train(config, logger, tokenizer, device, nvml_handle)
  else:
      if is_main_process():
          logger.warning("sample_eval and ppl_eval modes are not configured yet")

  #if use_distributed:
  #    cleanup_distributed()

  pynvml.nvmlShutdown()
  # torch.cuda.empty_cache()

if __name__ == '__main__':
  main()