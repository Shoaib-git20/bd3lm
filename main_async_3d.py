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
)

from torch.distributed._symmetric_memory import enable_symm_mem_for_group
import torch._inductor

import dataloader
from diffusion_native import Diffusion
from models.dit import DDiTBlock, DDiTBlockCausal, TimestepEmbedder, EmbeddingLayer, DDiTFinalLayer 
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

def setup_fsdp_model(model, config, local_rank, mesh=None):
   # Debug: surface fsdp setup inputs
    try:
        _ws = dist.get_world_size()
        _rk = dist.get_rank()
    except Exception:
        _ws = None
        _rk = None
    print(f"[Debug][FSDP setup] local_rank={local_rank} rank={_rk} world_size={_ws} mesh_provided={'yes' if mesh is not None else 'no'}")
    print(f"[Debug][FSDP setup] requested sharding_strategy={getattr(config.strategy, 'sharding_strategy', None)} mixed_precision={config.strategy.get('mixed_precision', 'fp32')}")
   # Configure mixed precision
    mixed_precision_policy = None
    if config.strategy.get('mixed_precision', 'fp32') == 'bf16':
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif config.strategy.get('mixed_precision', 'fp32') == 'fp16':
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    elif config.strategy.get('name', 'ddp') == '3d':
        mixed_precision_policy = None
    # Define transformer wrapping policy
    dit_auto_wrap_policy = functools.partial(
       transformer_auto_wrap_policy,
       transformer_layer_cls={
           DDiTBlock, DDiTBlockCausal, TimestepEmbedder, EmbeddingLayer, DDiTFinalLayer
       })
    if config.strategy.sharding_strategy == 'FULL_SHARD':
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif config.strategy.sharding_strategy == 'SHARD_GRAD_OP':
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif config.strategy.sharding_strategy == 'HYBRID_SHARD':
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif config.strategy.get('name', 'ddp') == '3d':
        sharding_strategy = None
    else:
        raise ValueError(f"Unknown sharding strategy: {config.strategy.sharding_strategy}")
    # Initialize FSDP wrapped model
    if config.strategy.get('name') == '3d':
        model = FSDP(
            model,
            auto_wrap_policy=dit_auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            device_mesh=mesh,
        )
        return model
    else :
        model = FSDP(
            model,
            auto_wrap_policy=dit_auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )
    return model

def setup_tp_model(model, config, local_rank, tp_size, rank, world_size, tp_mesh=None):
    # Debug: entering TP model setup
    print(f"[Debug][TP setup] entering setup_tp_model: rank={rank} world_size={world_size} local_rank={local_rank} tp_size={tp_size}")
    if tp_mesh is None:
        tp_mesh = init_device_mesh("cuda", (tp_size,))
    print(f"[Debug][TP setup] DeviceMesh initialized with shape {(tp_size,)} -> {tp_mesh}")
    # mesh_shape = (world_size,)
    # device_ids = list(range(world_size))
    # mesh = init_device_mesh(device_ids, mesh_shape)
    # Define parallelization rules for layers
    tp_layer_plan_patterns = {
       "vocab_embed.embedding": "replicated",
       "blocks.*.norm1": "sequence",
       "blocks.*.norm2": "sequence",
       "blocks.*.attn_qkv": "column",
       "blocks.*.attn_out": "row",
       "blocks.*.mlp.0": "column",
       "blocks.*.mlp.2": "row",
       "blocks.*.adaLN_modulation": "replicated",
       "output_layer.norm_final": "sequence",
       "output_layer.linear": "row",
    }
    def _pattern_to_regex(pattern: str):
        # Escape dots and other chars, convert '*' to '.*'
        # e.g. "blocks.*.attn_qkv" -> ^blocks\..*\.attn_qkv$
        esc = re.escape(pattern)
        regex = "^" + esc.replace(r"\*", ".*") + "$"
        return re.compile(regex)
    # Create list of compiled patterns
    compiled_patterns = [(repl, _pattern_to_regex(pat)) for pat, repl in tp_layer_plan_patterns.items()]
    def build_parallelize_plan(model):
        """
        Return dict mapping module FQN -> ParallelStyle() (ColwiseParallel/RowwiseParallel)
        Only include modules that match patterns and have a parallel style (i.e., skip 'replicated').
        """
        plan = {}
        for name, module in model.named_modules():
            # try matching each pattern; the first matching pattern will apply
            for parallel_style, cre in compiled_patterns:
                if cre.match(name):
                    if parallel_style == "column":
                        plan[name] = ColwiseParallel()
                    elif parallel_style == "row":
                        plan[name] = RowwiseParallel()
                    elif parallel_style == "sequence":
                        plan[name] = SequenceParallel()
                    # Skip 'replicated' style as it doesn't need explicit parallelization
                    break
        return plan
    # Build the plan using the actual model instance
    # Note: ensure 'model' is defined above this snippet (your instantiated DIT model)
    parallelize_plan = build_parallelize_plan(model)
    # Debug print the plan (for rank 0)
    if rank == 0:
        print("Tensor Parallelize plan (FQN -> ParallelStyle):")
        for k, v in parallelize_plan.items():
            print(f"  {k} -> {v}")
    # ---------- 5) Apply parallelize_module ----------
    # This will transform parameters into DTensor and shard weights according to the plan.
    # It must be called on all ranks.
    # The src_data_rank argument defaults to 0 (rank that has the data if needed); we keep default here.
    try:
        parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=parallelize_plan)
        print(f"[rank {rank}] parallelize_module done")
    except Exception as e:
        # If the API call fails (version mismatch), give a helpful message:
        raise RuntimeError(
            "parallelize_module failed. Make sure your PyTorch version supports torch.distributed.tensor.parallel "
            "and DeviceMesh (PyTorch >= 2.3 / 2.4 recommended). Original error: " + str(e)
        )
    return model
   

def setup_distributed_model(model, config, local_rank):
    """Sets up the model for distributed training with DDP or FSDP."""
    
    strategy_name = config.strategy.get('name', 'ddp') # Default to DDP
    # Debug: surface selection info
    try:
        _world_size = dist.get_world_size()
        _rank = dist.get_rank()
    except Exception:
        _world_size = None
        _rank = None
    print(f"[Debug][setup_distributed_model] strategy={strategy_name} local_rank={local_rank} rank={_rank} world_size={_world_size}")
    
    if strategy_name == 'ddp':
        # DDP: move model to the local CUDA device and wrap with DistributedDataParallel
        print(f"[Debug][DDP] rank={_rank} local_rank={local_rank} wrapping model with DDP on cuda:{local_rank}")
        model = model.to(local_rank)
        return DDP(model, device_ids=[local_rank])

    elif strategy_name == 'fsdp':
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"[rank {rank}/{world_size}] running on device {device}")
        print(f"[Debug][FSDP] wrapping model with FSDP (local_rank={local_rank})")
        model = setup_fsdp_model(model, config, local_rank)
        return model
        
    elif strategy_name == 'tp':
        # Initialize device mesh for tensor parallelism
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        #local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        tp_size = 2
        print(f"[rank {rank}/{world_size}] running on device {device} (cuda:local-rank)")
        print(f"[Debug][TP] preparing tensor-parallel setup: tp_size={tp_size} local_rank={local_rank} rank={rank} world_size={world_size}")
        if world_size < tp_size or world_size % tp_size != 0:
            pass

        model = model.to(device)

        model = setup_tp_model(model, config, local_rank, tp_size, rank, world_size)

        return model
    elif strategy_name == '3d':
        print(f"[Debug] Initializing 3D parallel strategy...")
        print(f"[Debug][3D] local_rank={local_rank} rank={dist.get_rank()} world_size={dist.get_world_size()}")
        tp_size = 2
        world_size = dist.get_world_size()
        print(f"[Debug] World size: {world_size}, TP size: {tp_size}")
        
        assert world_size % (tp_size) == 0, f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
        dp_size = world_size // (tp_size)
        print(f"[Debug] Calculated DP size: {dp_size}")
        
        print(f"[Debug] Initializing 2D device mesh with shape ({tp_size}, {dp_size})")
        mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        print(f"[Debug] Device mesh initialized: {mesh_2d}")
        
        tp_mesh = mesh_2d["tp"]  # a submesh that connects intra-host devices
        dp_mesh = mesh_2d["dp"]  # a submesh that connects inter-host devices
        print(f"[Debug] Created TP mesh: {tp_mesh}")
        print(f"[Debug] Created DP mesh: {dp_mesh}")
        
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        print(f"[rank {rank}/{world_size}] running on device {device} (cuda:local-rank)")
        print(f"[Debug] Process rank {rank} moved model to device {device}")

        model = model.to(device)
        print(f"[Debug] Setting up Tensor Parallel model...")
        model_tp = setup_tp_model(model, config, local_rank, tp_size, rank, world_size, tp_mesh=tp_mesh)
        print(f"[Debug] TP model setup complete")
        
        # apply FSDP inter-host on dp_mesh
        print(f"[Debug] Setting up FSDP with DP mesh...")
        model = setup_fsdp_model(model_tp, config, local_rank, mesh=dp_mesh)
        print(f"[Debug] FSDP setup complete")
        print(f"[Debug] 3D parallel strategy initialization complete")

    elif strategy_name == "async_tp":
        # async tensor-parallel only (1 node, 2 GPUs per node)
        print("[Debug] Initializing official async-TP strategy (1 node, 2 GPUs)...")

        tp_size = 2
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        assert world_size == tp_size, (
            f"async_tp expects single node with {tp_size} GPUs, got world_size={world_size}"
        )

        # --- Build the TP mesh ---
        mesh_1d = init_device_mesh("cuda", (tp_size,), mesh_dim_names=("tp",))
        tp_mesh = mesh_1d["tp"]
        tp_group = tp_mesh.get_group()
        print(f"[Debug] TP mesh: {tp_mesh}, group size={tp_size}")

        # --- Enable symmetric memory for this TP group ---

        print("[Debug] Enabling symmetric memory for TP group...")
        try:
            # If user code already provided a name string
            if isinstance(tp_group, str):
                group_name = tp_group
            else:
                # Common attribute added by DeviceMesh.get_group() in many PyTorch builds
                group_name = getattr(tp_group, "group_name", None)

                # alternate attribute names some builds expose
                if group_name is None:
                    group_name = getattr(tp_group, "name", None)
                if group_name is None and hasattr(tp_group, "get_group_name"):
                    try:
                        group_name = tp_group.get_group_name()
                    except Exception:
                        group_name = None

            if not group_name:
                raise RuntimeError(
                    "Could not determine the process-group name for tp_group. "
                    "enable_symm_mem_for_group requires a string group name (e.g. tp_group.group_name). "
                    "If your ProcessGroup object has no name, create or obtain a named ProcessGroup "
                    "from DeviceMesh.get_group() or upgrade PyTorch so DeviceMesh exposes group_name."
                )

            # Now call the symmetric memory helper with the string name:
            enable_symm_mem_for_group(group_name)
            print(f"[Debug] Symmetric memory enabled for group '{group_name}'")

        except Exception as e:
            # bubble up a clearer error showing what we received
            raise RuntimeError(
                f"Failed to enable symmetric memory for TP group. tp_group type={type(tp_group)}, "
                f"attrs={[a for a in dir(tp_group) if not a.startswith('__')] if not isinstance(tp_group, str) else 'string'}; "
                f"error={e}"
            ) from e
    
        print("[Debug] Symmetric memory enabled.")

        # --- Configure torch.compile for async TP ---
        torch._inductor.config._micro_pipeline_tp = True
        print("[Debug] torch.compile micro-pipeline async-TP mode enabled.")

        # --- Move model to GPU and prepare TP ---
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = setup_tp_model(model, config, local_rank, tp_size, rank, world_size)

        # --- Compile the model for async overlapped execution ---
        # You can compile the whole model, or selectively only TP-heavy parts
        try:
            model = torch.compile(model)
            print("[Debug] Model compiled with torch.compile for async TP.")
        except Exception as e:
            print(f"[Warning] torch.compile(model) failed: {e}")

        # Example: compile only specific TP region if you want finer control
        # if hasattr(model, "tp_submodule"):
        #     model.tp_submodule = torch.compile(model.tp_submodule)
        #     print("[Debug] Only TP submodule compiled for async TP.")

        print("[Debug] async-TP initialization complete (no FSDP/DP applied).")

    else:
        raise ValueError(f"Unknown distributed strategy: {strategy_name}")

def _train(config, logger, tokenizer, device):
  """Main distributed training loop."""
  rank = int(os.environ["RANK"])
  local_rank = int(os.environ["LOCAL_RANK"])
  
  if is_main_process():
      logger.info(f'Starting {config.strategy.name} Training.')

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
  model = Diffusion(config, tokenizer)
  
  setup_distributed_model(model, config, local_rank)

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
  if config.strategy.name in ('fsdp', 'tp', '3d', 'async_tp'):
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
            
        input_ids = batch['input_ids'].to(local_rank)
        attention_mask = batch['attention_mask'].to(local_rank)

        optimizer.zero_grad()
        if config.strategy.name in ('fsdp', 'tp', '3d', 'async_tp'):
            loss_obj = model.compute_loss(input_ids, attention_mask)
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
                input_tokens,
                sampling_eps_min=sampling_eps_min,
                sampling_eps_max=sampling_eps_max)

            if model.ignore_bos and not model.training:
              attention_mask[:, 0] = 0

            nlls = (loss * attention_mask)
            token_nll = nlls.sum() / attention_mask.sum()
            loss_obj = Loss(loss=token_nll, nlls=nlls, token_mask=attention_mask)
        else:
            loss_obj = model.module.compute_loss(input_ids, attention_mask)
        
        loss = loss_obj.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
