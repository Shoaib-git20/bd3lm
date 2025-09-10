#!/bin/bash
#SBATCH -J train_owt_bd3lm            # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH --nodes=1                          # Total number of nodes requested
#SBATCH --gpus-per-node=1                 # Number of gpus per node
#SBATCH -t 3:00:00                    # Time limit (hh:mm:ss)
#SBATCH -A PZS0622
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohammad.116@osu.edu

BLOCK_SIZE=4
PRETRAIN_CKPT='' # to train from scratch, set to null
export HF_HOME=/fs/ess/PZS0622/shoaib/.cache/huggingface
export HF_DATASETS_CACHE=/fs/ess/PZS0622/shoaib/.cache/huggingface/datasets
export WANDB_MODE=disabled

source /fs/ess/PZS0622/shoaib/d3pm/load-env.sh
nvidia-smi

# Start memory logger in background
#LOG_INTERVAL=1
#MEM_LOG="mem_usage.log"
#echo "[INFO] Starting memory logger at ${MEM_LOG} every ${LOG_INTERVAL}s"
#(
#  echo "Timestamp,CPU_Used,CPU_Free,GPU_Usage"
#  while true; do
#    ts=$(date '+%Y-%m-%d %H:%M:%S')
#    mem=$(free -m | awk '/Mem:/ {print $3","$4}')
#    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1 "MiB"}')
#    echo "${ts},${mem},${gpu}" >> "$MEM_LOG"
#    sleep "$LOG_INTERVAL"
#  done
#) &
#LOGGER_PID=$!

cd /fs/ess/PZS0622/shoaib/d3pm/bd3lms
echo "inside ${PWD}"

mpirun -np 4 python -u main.py loader.global_batch_size=16 loader.eval_global_batch_size=16 loader.batch_size=4 loader.eval_batch_size=4 \
    model=tiny algo=bd3lm algo.clip_search_widths=[0.5] \
    data=openwebtext-split data.insert_train_special=False data.insert_valid_special=False data.insert_valid_eos=False \
    model.length=128 block_size=${BLOCK_SIZE} wandb.name=bd3lm-owt-block_size${BLOCK_SIZE} \
    mode=train model.attn_backend=flex \
    training.resample=True trainer.fast_dev_run=False

# Stop memory logger after training
#kill $LOGGER_PID
#echo "[INFO] Memory logger stopped"