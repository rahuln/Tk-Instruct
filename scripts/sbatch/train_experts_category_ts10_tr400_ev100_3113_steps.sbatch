#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=trcat300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=6:00:00
#SBATCH --array=0-12
#SBATCH --output=slurm/trcat300-%A-%a.log
#SBATCH --error=slurm/trcat300-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_MODE=online
export WANDB_PROJECT=tk-instruct-experts
python scripts/train_tk_instruct_experts.py \
    scripts/config/category_ts10_tr400_ev100.json \
    --exp_name 1163-steps \
    --max_steps 1163 \
    --max_num_instances_per_task 300 \
    --eval_steps 116 \
    --logging_steps 25 \
    --save_steps 116 \
    --index $SLURM_ARRAY_TASK_ID
