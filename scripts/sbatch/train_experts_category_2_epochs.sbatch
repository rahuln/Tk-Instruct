#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=tkiexcat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=6:00:00
#SBATCH --array=0-59
#SBATCH --output=slurm/tkiexcat-%A-%a.log
#SBATCH --error=slurm/tkiexcat-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_MODE=online
export WANDB_PROJECT=tk-instruct-experts
python scripts/train_tk_instruct_experts.py \
    scripts/config/category_experts.json \
    --exp_name 2-epochs \
    --num_train_epochs 2 \
    --logging_steps 500 \
    --save_steps 2500 \
    --index $SLURM_ARRAY_TASK_ID
