#!/bin/bash
#SBATCH -J train_owt_bd3lm            # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH -t 6:00:00                    # Time limit (hh:mm:ss)
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption
#SBATCH --gpus-per-node=1
#SBATCH -A PZS0622
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohammad.116@osu.edu

BLOCK_SIZE=16
PRETRAIN_CKPT='' # to train from scratch, set to null
export HYDRA_FULL_ERROR=1
export HF_HOME=/fs/ess/PZS0622/shoaib/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets

source /fs/ess/PZS0622/shoaib/d3pm/load-env.sh

python -u main.py \
    loader.global_batch_size=512 \
    loader.eval_global_batch_size=512 \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    model=small \
    algo=bd3lm \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
    data=openwebtext-split \
    data.insert_train_special=False \
    data.insert_valid_special=False \
    data.insert_valid_eos=False \
    model.length=1024 \
    block_size=${BLOCK_SIZE} \
    wandb.name=bd3lm-owt-block_size${BLOCK_SIZE} \
    mode=train \
    model.attn_backend=flex \
    training.resample=True