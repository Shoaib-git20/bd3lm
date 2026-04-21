import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed._composable.fsdp import fully_shard 

# Tensor Parallelism
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput
)
from torch.distributed._tensor import Replicate, Shard

# Pipeline Parallelism
from torch.distributed.pipelining import pipeline, SplitPoint, Schedule1F1B

class PipelineLossWrapper(nn.Module):
    """
    Adapter to make your model compatible with PyTorch Pipeline Parallelism.
    The tracer strictly calls forward(), so we intercept it and route it 
    to your independent compute_loss method.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Expose the backbone so the pipeline split_spec can find the DiT blocks
        self.backbone = model.backbone 

    def forward(self, input_ids, attention_mask):
        # Route the pipeline's forward call to your actual training logic
        loss_obj = self.model.compute_loss(input_ids, attention_mask)
        
        # Strip the custom object and return ONLY the scalar tensor
        return loss_obj.loss

def _apply_tp_plan(model, tp_mesh):
    """Helper function to apply the Tensor Parallelism plan to the model."""
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
        tp_plan[f"{p}.atten.attn_qkv"] = ColwiseParallel()
        tp_plan[f"{p}.atten.attn_out"] = RowwiseParallel(output_layouts=Shard(1), use_local_output=False)
        tp_plan[f"{p}.norm2"] = SequenceParallel()
        tp_plan[f"{p}.mlp"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        tp_plan[f"{p}.mlp.w1"] = ColwiseParallel()
        tp_plan[f"{p}.mlp.w2"] = RowwiseParallel(output_layouts=Shard(1), use_local_output=False)

    parallelize_module(model, tp_mesh, parallelize_plan=tp_plan)

def _apply_fsdp(model, dp_mesh):
    """Helper function to apply FSDP sharding."""
    fsdp_config = {
        "mesh": dp_mesh,
        "reshard_after_forward": False,
    }

    # shard embedding
    fully_shard(model.backbone.vocab_embed.embedding, **fsdp_config)

    # shard each transformer block
    for block in model.backbone.blocks:
        fully_shard(block, **fsdp_config)

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


def parallelize_model(model, config, device, is_main_process, logger=None):
    """Main routing function to apply the requested distributed strategy."""
    strategy_name = config.strategy.name.lower()
    world_size = dist.get_world_size()

    tp_mesh = None
    dp_mesh = None
    pp_mesh = None
    schedule = None

    if strategy_name == 'ddp':
        # 1D Device Mesh for Data Parallelism
        dp_mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("dp",))
        model.backbone = DDP(model.backbone, device_ids=[device.index] if device.type == "cuda" else None)
        if is_main_process:
            print("[Debug] Applied Distributed Data Parallel (DDP)")

    elif strategy_name == 'fsdp':
        # 1D Device Mesh for FSDP
        dp_mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("dp",))
        _apply_fsdp(model, dp_mesh)
        torch.cuda.synchronize()
        if is_main_process:
            print("[Debug] Applied Fully Sharded Data Parallel (FSDP)")

    elif strategy_name == 'tp':
        # 1D Device Mesh for Tensor Parallelism
        tp_size = config.strategy.tp_degree
        tp_mesh = init_device_mesh(device.type, (tp_size,), mesh_dim_names=("tp",))
        _apply_tp_plan(model, tp_mesh)
        if is_main_process:
            print(f"[Debug] Applied Tensor Parallelism (TP degree: {tp_size})")
    
    elif strategy_name == 'tp_ddp': # Handles TP + DDP
        tp_size = config.strategy.tp_degree
        dp_size = config.strategy.dp_degree
        
        if is_main_process:
            msg = f"[{strategy_name}] World size: {world_size}, TP size: {tp_size}, DP size: {dp_size}, device type: {device.type}"
            if logger:
                logger.info(msg)
            else:
                print(msg)

        # 2D Device Mesh
        mesh_2d = init_device_mesh(device.type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]

        if is_main_process:
            print(f"[Debug][{strategy_name}] Device mesh initialized: {mesh_2d}\n - TP mesh: {tp_mesh}\n - DP mesh: {dp_mesh}")

        # 1. Apply Tensor Parallelism
        _apply_tp_plan(model, tp_mesh)
        if is_main_process:
            print("[Debug] Tensor Parallelism applied")

        # 2. Apply DDP on top of TP across the "dp" process group
        print(f"[Debug] device_ids for DDP: {[device.index] if device.type == 'cuda' else None}, device type: {device.type}")
        model.backbone = DDP(
            model.backbone, 
            device_ids=[device.index] if device.type == "cuda" else None,
            process_group=dp_mesh.get_group()
        )
        torch.cuda.synchronize()
        
        if is_main_process:
            print(f"[Debug][{strategy_name}] Applied TP + DDP to model")

    elif strategy_name in ['2d', '3d']: # Handles 2d (TP + FSDP)
        tp_size = config.strategy.tp_degree
        dp_size = config.strategy.dp_degree
        
        if is_main_process:
            msg = f"[{strategy_name}] World size: {world_size}, TP size: {tp_size}, DP size: {dp_size}, device type: {device.type}"
            if logger:
                logger.info(msg)
            else:
                print(msg)

        # 2D Device Mesh
        mesh_2d = init_device_mesh(device.type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]

        if is_main_process:
            print(f"[Debug][{strategy_name}] Device mesh initialized: {mesh_2d}\n - TP mesh: {tp_mesh}\n - DP mesh: {dp_mesh}")

        # 1. Apply Tensor Parallelism
        _apply_tp_plan(model, tp_mesh)
        if is_main_process:
            print("[Debug] Tensor Parallelism applied")

        # 2. Apply FSDP on top of TP
        _apply_fsdp(model, dp_mesh)
        torch.cuda.synchronize()

        if is_main_process:
            print(f"[Debug][{strategy_name}] Applied TP + FSDP (2D) to model")

    elif strategy_name == 'tp_pp_fsdp': # Handles TP + PP + FSDP (3D)
        tp_size = config.strategy.tp_degree
        dp_size = config.strategy.dp_degree
        pp_size = config.strategy.pp_degree
        n_microbatches = config.strategy.microbatches # Required for PP

        if is_main_process:
            print(f"[3D] PP: {pp_size}, DP: {dp_size}, TP: {tp_size}")

        # 1. Initialize 3D Device Mesh (Order matters: PP, DP, TP)
        mesh_3d = init_device_mesh(
            device.type, 
            (pp_size, dp_size, tp_size), 
            mesh_dim_names=("pp", "dp", "tp")
        )
        pp_mesh = mesh_3d["pp"]
        dp_mesh = mesh_3d["dp"]
        tp_mesh = mesh_3d["tp"]

        model = PipelineLossWrapper(model) # Wrap the model to make it compatible with PP

        # 2. Apply Tensor Parallelism
        _apply_tp_plan(model, tp_mesh)
        if is_main_process:
            print("[Debug] Applied Tensor Parallelism")

        # 3. Pipeline Partitioning
        num_blocks = len(model.backbone.blocks)
        blocks_per_stage = num_blocks // pp_size
        
        split_spec = {}
        for i in range(1, pp_size):
            # Block index where the split should occur
            split_idx = i * blocks_per_stage
            # Split the graph RIGHT BEFORE this module
            split_spec[f"backbone.blocks.{split_idx}"] = SplitPoint.BEGIN

        # Build the pipeline stage for the specific rank
        stage_rank = pp_mesh.get_local_rank()
        pipeline_stage = pipeline(
            model,
            mb_args=mb_args,
            split_spec=split_spec,
        )
        stage_module = pipeline_stage.get_stage_module(stage_rank)
        
        if is_main_process:
            print("[Debug] Pipeline Stages Built")

        # 4. Apply FSDP to the partitioned Stage Module
        _apply_fsdp(stage_module, dp_mesh)
        torch.cuda.synchronize()

        # 5. Pipeline Schedule
        schedule = Schedule1F1B(
            pipeline_stage, 
            n_microbatches=n_microbatches
        )

        if is_main_process:
            print("[Debug] Applied TP + PP + FSDP (3D) to model")
            
        # Return the schedule instead of the raw model, as the schedule dictates execution
        return model, schedule, tp_mesh, dp_mesh, pp_mesh
    else:
        raise ValueError(f"Unknown distributed strategy '{strategy_name}' requested.")

    return model, tp_mesh, dp_mesh