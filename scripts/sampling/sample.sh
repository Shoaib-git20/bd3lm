#!/bin/bash
#SBATCH -J profile_sample_bd3lm       # Job name
#SBATCH -o watch_folder/%x_%j.out        # stdout log
#SBATCH -e watch_folder/%x_%j.err        # stderr log
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH -t 01:00:00
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --gpus-per-node=1
#SBATCH -A PZS0622

nvidia-smi

LENGTH=32
BLOCK_SIZE=4
SEED=42
T=50

export HF_HOME=/fs/ess/PZS0622/shoaib/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
#export CUDA_LAUNCH_BLOCKING=1

source /fs/ess/PZS0622/shoaib/d3pm/load-env.sh

srun python -u main.py \
  loader.eval_batch_size=1 \
  model=small \
  algo=bd3lm \
  algo.T=$T \
  algo.backbone=hf_dit \
  data=openwebtext-split \
  model.length=$LENGTH \
  block_size=$BLOCK_SIZE \
  wandb=null \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
  model.attn_backend=sdpa \
  sampling.nucleus_p=0.9 \
  sampling.kv_cache=false \
  sampling.logdir=$PWD/samples_prompt_gen_blocksize${BLOCK_SIZE} \
  seed=$SEED  