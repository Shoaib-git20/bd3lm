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
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.fsdp.wrap import wrap as fsdp_wrap

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

def fsdp_wrap_safe(root_model, fsdp_class=FSDP, fsdp_kwargs=None, max_depth=None):
    """
    Wraps submodules of `root_model` with FSDP safely:
    - If a submodule contains ANY DTensor params, it is NOT wrapped.
    - Instead, we recurse into its children and try to wrap safe grandchildren.
    - If a submodule contains NO DTensor params (i.e., fully local Tensor params),
      it is wrapped as a unit with FSDP.

    Parameters:
      - root_model: top-level model to process (modify in-place)
      - fsdp_class: FSDP class to use (default torch.distributed.fsdp.FullyShardedDataParallel)
      - fsdp_kwargs: kwargs forwarded to FSDP(...) when wrapping
      - max_depth: optional int limiting recursion depth (None => unlimited)

    Returns:
      - root_model (modified)
    """
    if fsdp_kwargs is None:
        fsdp_kwargs = {}

    def _rec_wrap(module, prefix="", depth=0):
        # stop recursion if depth limit reached
        if (max_depth is not None) and (depth > max_depth):
            return module

        # If module contains ANY DTensor parameters, do NOT wrap it.
        # Instead, attempt to wrap its children individually.
        if module_contains_dtensor(module):
            # recurse into children
            for child_name, child in list(module.named_children()):
                full_name = f"{prefix}.{child_name}" if prefix else child_name
                # process child and replace if changed
                wrapped_child = _rec_wrap(child, prefix=full_name, depth=depth+1)
                if wrapped_child is not child:
                    setattr(module, child_name, wrapped_child)
            return module
        else:
            # no DTensor inside this module -> safe to wrap
            try:
                wrapped = fsdp_class(module, **fsdp_kwargs)
                print(f"[fsdp_wrap_safe] Wrapped '{prefix or module.__class__.__name__}' with FSDP")
                return wrapped
            except Exception as e:
                # fallback: log and leave module unwrapped
                print(f"[fsdp_wrap_safe][WARN] Failed to wrap '{prefix}': {e}")
                return module

    # Start recursion at immediate named children of root_model:
    for name, child in list(root_model.named_children()):
        wrapped = _rec_wrap(child, prefix=name, depth=0)
        if wrapped is not child:
            setattr(root_model, name, wrapped)

    return root_model

# corrected snippet (only the tp_plan part and parallelize call shown)

def apply_tp_fsdp_with_sp(model, config, tp_size, dp_size):

    device_type = torch.accelerator.current_accelerator().type
    if is_main_process():
        print(f"[Debug][3D] Initializing 2D device mesh with shape ({dp_size}, {tp_size})")
    
    mesh_2d = init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

    if is_main_process():
        print(f"[Debug][3D] Device mesh initialized: {mesh_2d}")

    tp_mesh = mesh_2d["tp"]  # a submesh that connects intra-host devices
    dp_mesh = mesh_2d["dp"]  # a submesh that connects inter-host devices

    tp_plan = {

        # --- EMBEDDING ---
        # Embedding weights are typically row-sharded on vocab dimension
        # Output should be sequence-sharded to align with SP pipeline
        "backbone.vocab_embed.embedding": RowwiseParallel(
            input_layouts=Replicate(),   # input token IDs are replicated across TP ranks
            output_layouts=Shard(dim=1), # outputs sharded along sequence dim for SP
            use_local_output=False,
        ),
        "backbone.output_layer.linear": ColwiseParallel(
            input_layouts=(Shard(dim=1),),
            output_layouts=(Replicate(),),
            use_local_output=True,   # ensure local tensor for loss / FSDP consumption
        ),

        # final norm: run on sequence-sharded slices and return local output
        "backbone.output_layer.norm_final": SequenceParallel(sequence_dim=1, use_local_output=True),

        "backbone.sigma_map.mlp.0": SequenceParallel(sequence_dim=1, use_local_output=False),
        # The output (t_cond) must be replicated for AdaLN, so use RowwiseParallel last
        "backbone.sigma_map.mlp.2": SequenceParallel(sequence_dim=1, use_local_output=False),
    }

    n_blocks = len(model.backbone.blocks)
    for i in range(n_blocks):
        p = f"backbone.blocks.{i}"

        tp_plan[f"{p}.norm1"] = SequenceParallel(sequence_dim=1, use_local_output=False)
        tp_plan[f"{p}.norm2"] = SequenceParallel(sequence_dim=1, use_local_output=False)
        
        # AdaLN modulation: sequence_dim=1
        tp_plan[f"{p}.adaLN_modulation"] = SequenceParallel(sequence_dim=1, use_local_output=False)

        # attention qkv: assume sequence is sharded, project features (colwise on feature dim)
        tp_plan[f"{p}.attn_qkv"] = ColwiseParallel(input_layouts=Shard(1),output_layouts=Shard(-1),use_local_output=False)

        #tp_plan[f"{p}.attn_qkv"] = ColwiseParallel(input_layouts=(Shard(1),Replicate()),use_local_output=False)
        tp_plan[f"{p}.attn_out"] = RowwiseParallel(output_layouts=Shard(1), use_local_output=False)

        # MLP: use dot notation mlp.0 and mlp.2 (not mlp[0])
        tp_plan[f"{p}.mlp.0"] = ColwiseParallel(use_local_output=False)
        tp_plan[f"{p}.mlp.2"] = RowwiseParallel(output_layouts=Shard(1) , use_local_output=False)

    if is_main_process():
        print("Tensor Parallelize plan (FQN -> ParallelStyle):")
        for k, v in tp_plan.items():
            print(f"  {k} -> {v}")

    # call parallelize_module with explicit kwargs (safer)
    parallelize_module(model, tp_mesh, parallelize_plan=tp_plan)

    torch.cuda.synchronize()

    model = fully_shard(model, mesh=dp_mesh)

    return model, dp_mesh, tp_mesh, mesh_2d

def setup_fsdp_model(model, config, local_rank, mesh=None):
   # Debug: surface fsdp setup inputs
    #try:
    #    _ws = dist.get_world_size()
    #    _rk = dist.get_rank()
    #except Exception:
    #    _ws = None
    #    _rk = None
    #print(f"[Debug][FSDP setup] local_rank={local_rank} rank={_rk} world_size={_ws} mesh_provided={'yes' if mesh is not None else 'no'}")
    #print(f"[Debug][FSDP setup] requested sharding_strategy={getattr(config.strategy, 'sharding_strategy', None)} mixed_precision={config.strategy.get('mixed_precision', 'fp32')}")
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
        fsdp_kwargs = dict(auto_wrap_policy=dit_auto_wrap_policy, 
                            device_id=torch.cuda.current_device(), 
                            sharding_strategy=sharding_strategy, mixed_precision=mixed_precision_policy,
                            use_orig_params=True, device_mesh=mesh,)
        #model = fsdp_wrap_safe(model, FSDP, fsdp_kwargs)

        for block in model.backbone.blocks:
            fully_shard(block, mesh=mesh,mp_policy=mixed_precision_policy)

        fully_shard(model, mesh=mesh,mp_policy=mixed_precision_policy)
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
    #print(f"[Debug][TP setup] entering setup_tp_model: rank={rank} world_size={world_size} local_rank={local_rank} tp_size={tp_size}")
    if tp_mesh is None:
        tp_mesh = init_device_mesh("cuda", (tp_size,))
    #print(f"[Debug][TP setup] DeviceMesh initialized with shape {(tp_size,)} -> {tp_mesh}")
    # mesh_shape = (world_size,)
    # device_ids = list(range(world_size))
    # mesh = init_device_mesh(device_ids, mesh_shape)
    # Define parallelization rules for layers
    tp_layer_plan_patterns = {
        "backbone.vocab_embed.embedding": "replicated",
        "backbone.blocks.*.norm1": "sequence",
        "backbone.blocks.*.norm2": "sequence",
        "backbone.blocks.*.attn_qkv": "column",
        "backbone.blocks.*.attn_out": "row",
        #"backbone.blocks.*.mlp.*0": "column",
        #"backbone.blocks.*.mlp.*2": "row",
        "backbone.blocks.*.adaLN_modulation": "replicated",
        "backbone.output_layer.norm_final": "sequence",
        "backbone.output_layer.linear": "row",
        #"backbone.blocks.*": "replicated",
    }

    # Build the plan using the actual model instance
    # Note: ensure 'model' is defined above this snippet (your instantiated DIT model)

    # precompile regexes safely (escape dots, convert '*' -> '.*')
    compiled_patterns = []
    for pat, style in tp_layer_plan_patterns.items():
        esc = re.escape(pat)
        regex = "^" + esc.replace(r"\*", ".*") + "$"
        compiled_patterns.append((re.compile(regex), style, pat))

    parallelize_plan = {}
    skipped = []

    for name, module in model.named_modules():
        #if is_main_process():
        #    print(f"[Debug][TP setup] Examining module: {name} ({type(module)})")
        matched_any = False
        for cre, style, pat in compiled_patterns:
            if cre.match(name):
                matched_any = True
                # column/row require Linear or Embedding
                if style in ("column", "row"):
                    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                        parallelize_plan[name] = ColwiseParallel() if style == "column" else RowwiseParallel()
                    elif isinstance(module, torch.nn.Sequential):
                        for child_name, child in module.named_children():
                            full_name = f"{name}.{child_name}"
                            if isinstance(child, torch.nn.Linear):
                                if "mlp.0" in full_name:
                                    parallelize_plan[full_name] = ColwiseParallel()
                                elif "mlp.2" in full_name:
                                    parallelize_plan[full_name] = RowwiseParallel()
                    else:
                        skipped.append((name, style, type(module).__name__, pat))
                        #if is_main_process():
                        #    print(f"[TP planner] Skipping {name}: requested '{style}' but module is {type(module).__name__} (pattern={pat})")
                elif style == "sequence":
                    # accept if name includes 'norm' or module looks like LayerNorm (or custom norm)
                    if "norm" in name.lower() or "norm" in module.__class__.__name__.lower() or isinstance(module, torch.nn.LayerNorm):
                        parallelize_plan[name] = SequenceParallel()
                    else:
                        skipped.append((name, style, type(module).__name__, pat))
                        #if is_main_process():
                        #    print(f"[TP planner] Skipping {name}: requested 'sequence' but module is {type(module).__name__} (pattern={pat})")
                #else:
                    # 'replicated' -> intentionally no entry (kept replicated)
                    #if is_main_process():
                    #    print(f"[TP planner] {name} set to 'replicated' by pattern {pat}")
                #break  # stop checking other patterns after first match
        # optional: track names that matched no pattern
        #if not matched_any:
        #    if is_main_process():
        #        print(f"[TP planner] No pattern matched for {name}")
    # summary
    # TEMP DEBUG: check Colwise/Rowwise targets before calling parallelize_module
    #if is_main_process():
    #    module_map = dict(model.named_modules())  # compute once
    #    colrow_warnings = []  # (name, expected_style, actual_type)
    #    colrow_valid = []     # (name, style, actual_type)
#
    #    for name, style in parallelize_plan.items():
    #        # style is an instance of ParallelStyle (ColwiseParallel, RowwiseParallel, SequenceParallel, ...)
    #        if isinstance(style, (ColwiseParallel, RowwiseParallel)):
    #            module = module_map.get(name)
    #            if module is None:
    #                colrow_warnings.append((name, style.__class__.__name__, "NOT FOUND"))
    #            else:
    #                # strict check first
    #                if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
    #                    colrow_valid.append((name, style.__class__.__name__, module.__class__.__name__))
    #                else:
    #                    # broadened "linear-like" heuristic: subclass or custom wrapper
    #                    cls_name = module.__class__.__name__.lower()
    #                    if "linear" in cls_name or "embedding" in cls_name:
    #                        colrow_valid.append((name, style.__class__.__name__, module.__class__.__name__ + " (linear-like)"))
    #                    else:
    #                        colrow_warnings.append((name, style.__class__.__name__, module.__class__.__name__))
#
    #    # Print results
    #    if colrow_valid:
    #        print("[TP check] Colwise/Rowwise valid targets:")
    #        for n, s, t in colrow_valid:
    #            print(f"  {n} -> {s} (module type: {t})")
    #    else:
    #        print("[TP check] No Colwise/Rowwise valid targets found.")
#
    #    if colrow_warnings:
    #        print("[TP check][WARN] Colwise/Rowwise targets with issues:")
    #        for n, s, t in colrow_warnings:
    #            print(f"  {n} -> {s} (module type: {t})")
    #    else:
    #        print("[TP check] No warnings for Colwise/Rowwise targets.")


    # ---------- 5) Apply parallelize_module ----------
    # This will transform parameters into DTensor and shard weights according to the plan.
    # It must be called on all ranks.
    # The src_data_rank argument defaults to 0 (rank that has the data if needed); we keep default here.
    # run this BEFORE calling parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=parallelize_plan)
    if is_main_process():
        print("Tensor Parallelize plan (FQN -> ParallelStyle):")
        for k, v in parallelize_plan.items():
            print(f"  {k} -> {v}")

    #from torch.distributed.tensor.parallel.api import parallelize_module as _parallelize_module
    #import traceback
#
    #if is_main_process():
    #    print("[TP debug] Starting per-mapping check...")
#
    #bad = []
    #for fname, style in list(parallelize_plan.items()):
    #    small_plan = {fname: style}
    #    try:
    #        # Apply only this single mapping to a fresh copy of the model structure.
    #        # IMPORTANT: we apply in-place to the real model but with a single-plan so we can localize failures.
    #        if is_main_process():    
    #            print(f"[TP debug] Testing mapping {fname} -> {style.__class__.__name__}")
    #        _parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=small_plan)
    #        if is_main_process():
    #            print(f"[TP debug] OK: {fname}")
    #    except NotImplementedError as e:
    #        if is_main_process():
    #            print(f"[TP debug][ERROR] Mapping failed for {fname}: {e}")
    #        bad.append(fname)
    #    except Exception as e:
    #        if is_main_process():
    #            print(f"[TP debug][ERROR] Unexpected error for {fname}: {type(e).__name__}: {e}")
    #            traceback.print_exc()
    #if is_main_process():
    #    print(f"[TP debug] Per-mapping check complete. {len(bad)} failing mappings: {bad}")
    model = parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=parallelize_plan)
        #print(f"[rank {rank}] parallelize_module done")
    return model
   

def setup_distributed_model(model, logger, config, local_rank):
    """Sets up the model for distributed training with DDP or FSDP."""
    
    strategy_name = config.strategy.get('name', 'ddp') # Default to DDP
    # Debug: surface selection info
    #try:
    #    _world_size = dist.get_world_size()
    #    _rank = dist.get_rank()
    #except Exception:
    #    _world_size = None
    #    _rank = None
    #print(f"[Debug][setup_distributed_model] strategy={strategy_name} local_rank={local_rank} rank={_rank} world_size={_world_size}")
    
    if strategy_name == 'ddp':
        # DDP: move model to the local CUDA device and wrap with DistributedDataParallel
        #print(f"[Debug][DDP] rank={_rank} local_rank={local_rank} wrapping model with DDP on cuda:{local_rank}")
        model = model.to(local_rank)
        return DDP(model, device_ids=[local_rank])

    elif strategy_name == 'fsdp':
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        #print(f"[rank {rank}/{world_size}] running on device {device}")
        #print(f"[Debug][FSDP] wrapping model with FSDP (local_rank={local_rank})")
        model = setup_fsdp_model(model, config, local_rank)
        logger.info(f"FSDP model setup complete on world_size={world_size} rank={rank}")
        return model
        
    elif strategy_name == 'tp':
        # Initialize device mesh for tensor parallelism
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        #local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        tp_size = 2
        #print(f"[rank {rank}/{world_size}] running on device {device} (cuda:local-rank)")
        #print(f"[Debug][TP] preparing tensor-parallel setup: tp_size={tp_size} local_rank={local_rank} rank={rank} world_size={world_size}")
        if world_size < tp_size or world_size % tp_size != 0:
            pass

        model = model.to(device)

        model = setup_tp_model(model, config, local_rank, tp_size, rank, world_size)
        logger.info(f"TP model setup complete with tp_size={tp_size} on world_size={world_size} rank={rank}")

        return model
    elif strategy_name == '3d':
        tp_size = 4
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        logger.info(f"[Debug][3D] Initializing 3D parallel strategy with tp_size={tp_size} on world_size={world_size}...")
        #print(f"[Debug][3D] local_rank={local_rank} rank={rank} World size: {world_size}, TP size: {tp_size}")

        assert world_size % (tp_size) == 0, f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
        dp_size = world_size // (tp_size)

        #logger.info(f"[Debug][3D] Initializing 2D device mesh with shape ({dp_size}, {tp_size})")
        #mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        #logger.info(f"[Debug][3D] Device mesh initialized: {mesh_2d}")
#
        #tp_mesh = mesh_2d["tp"]  # a submesh that connects intra-host devices
        #dp_mesh = mesh_2d["dp"]  # a submesh that connects inter-host devices
      #  tp_group = tp_mesh.get_group()

        #logger.info(f"[Debug][3D] TP mesh: {tp_mesh}, group size={tp_size}")
        #logger.info(f"[Debug][3D] Created DP mesh: {dp_mesh}")

        # --- Enable symmetric memory for this TP group ---

        #logger.info(f"[Debug][3D] Enabling symmetric memory for TP group...")
      #  try:
      #      # If user code already provided a name string
      #      if isinstance(tp_group, str):
      #          group_name = tp_group
      #      else:
      #          # Common attribute added by DeviceMesh.get_group() in many PyTorch builds
      #          group_name = getattr(tp_group, "group_name", None)
#
      #          # alternate attribute names some builds expose
      #          if group_name is None:
      #              group_name = getattr(tp_group, "name", None)
      #          if group_name is None and hasattr(tp_group, "get_group_name"):
      #              try:
      #                  group_name = tp_group.get_group_name()
      #              except Exception:
      #                  group_name = None
#
      #      if not group_name:
      #          raise RuntimeError(
      #              "Could not determine the process-group name for tp_group. "
      #              "enable_symm_mem_for_group requires a string group name (e.g. tp_group.group_name). "
      #              "If your ProcessGroup object has no name, create or obtain a named ProcessGroup "
      #              "from DeviceMesh.get_group() or upgrade PyTorch so DeviceMesh exposes group_name."
      #          )
#
      #      # Now call the symmetric memory helper with the string name:
      #      enable_symm_mem_for_group(group_name)
      #      #logger.info(f"[Debug][3D] Symmetric memory enabled for group '{group_name}'")
#
      #  except Exception as e:
      #      # bubble up a clearer error showing what we received
      #      raise RuntimeError(
      #          f"Failed to enable symmetric memory for TP group. tp_group type={type(tp_group)}, "
      #          f"attrs={[a for a in dir(tp_group) if not a.startswith('__')] if not isinstance(tp_group, str) else 'string'}; "
      #          f"error={e}"
      #      ) from e
#
      #  # --- Configure torch.compile for async TP ---
      #  torch._inductor.config._micro_pipeline_tp = True
      #  #logger.info(f"[Debug][3D] torch.compile micro-pipeline async-TP mode enabled.")

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        #print(f"[rank {rank}/{world_size}] running on device {device} (cuda:local-rank)")
        #logger.info(f"[Debug][3D] Process rank {rank} moved model to device {device}")

        model = model.to(device)
        #logger.info(f"[Debug][3D] Setting up Tensor Parallel model...")
        model = apply_tp_fsdp_with_sp(model, config, tp_size, dp_size)
        torch.cuda.synchronize()

        # apply FSDP inter-host on dp_mesh
        #logger.info(f"[Debug][3D] Setting up FSDP with DP mesh = {dp_mesh}...")
        #model = setup_fsdp_model(model, config, local_rank, mesh=dp_mesh)
        summarize_param_types(model)
        # and list which top-level modules were wrapped:
        #if is_main_process():
        #    print("Going into FSDP-wrapped model, listing top-level children:")
        #    for name, child in model.named_children():
        #        print(name, type(child))
        #    print("FSDP naming complete.")
#
        #print(f"Model type after fsdp setup: {type(model)}")
        logger.info(f"[Debug][3D] FSDP setup complete")

        #try:
        #    model = torch.compile(model)
        #    #logger.info(f"[Debug][3D] Model compiled with torch.compile for async TP.")
        #except Exception as e:
        #    logger.warning(f"[Warning] torch.compile(model) failed: {e}")
        logger.info(f"[Debug][3D] Model is now wrapped with 3D parallelism (TP + FSDP).")

        return model

    elif strategy_name == "async_tp":
        # async tensor-parallel only (1 node, 2 GPUs per node)
        print("[Debug] Initializing official async-TP strategy (1 node, 2 GPUs)...")

        tp_size = 2
        world_size = dist.get_world_size()
        rank = dist.get_rank()

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

        print("[Debug] async-TP initialization complete (no FSDP/DP applied).")

    else:
        raise ValueError(f"Unknown distributed strategy: {strategy_name}")

def _train(config, logger, tokenizer, device):
  """Main distributed training loop."""
  rank = int(os.environ["RANK"])
  local_rank = int(os.environ["LOCAL_RANK"])
  
  if is_main_process():
      logger.info(f'Starting {config.strategy.name} Training with world size {os.environ["WORLD_SIZE"]}.')

  # --- Model ---
  model = Diffusion(config, tokenizer)

  train_set, valid_set = dataloader.get_dataloaders(config, tokenizer)

  print(f"Model type before whole setup: {type(model)}")

  dp_size = None
  dp_rank = None
  tp_mesh = None
  dp_mesh = None
  mesh_2d = None
  device_type = torch.accelerator.current_accelerator().type
  
  if config.strategy.name == '3d':
    world_size = dist.get_world_size()
    tp_size = 4
    assert world_size % (tp_size) == 0, f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
    dp_size = world_size // (tp_size)
    model = model.to(device_type)
    model, dp_mesh, tp_mesh, mesh_2d = apply_tp_fsdp_with_sp(model, config, tp_size, dp_size)
    torch.cuda.synchronize()

    dp_rank = dp_mesh.get_local_rank()
    torch.manual_seed(dp_rank + 100)
    summarize_param_types(model)

    train_sampler = DistributedSampler(train_set, num_replicas=dp_size, rank=dp_rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_set, num_replicas=dp_size, rank=dp_rank, shuffle=False)
  else:
    model = setup_distributed_model(model, logger, config, local_rank)
    train_sampler = DistributedSampler(train_set, shuffle=True)
    valid_sampler = DistributedSampler(valid_set, shuffle=False)



  print(f"Model type after whole setup: {type(model)}")
  # --- Setup DDP DataLoaders ---

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
  # Will be used in generative perplexity calculation
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
        if config.strategy.name in ('3d',):
            input_ids = DTensor.from_local(
                input_ids, 
                mesh_2d, 
                placements=[Shard(0), Replicate()]
            )
            attention_mask = DTensor.from_local(
                attention_mask, 
                mesh_2d, 
                placements=[Shard(0), Replicate()]
            )

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
