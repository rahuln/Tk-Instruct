#!/bin/bash

#SBATCH --partition=gpu-titan
#SBATCH --account=ark
#SBATCH --job-name=train-t5-experts-2-epochs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=slurm/train-t5-experts-2-epochs-%A-%a.log
#SBATCH --error=slurm/train-t5-experts-2-epochs-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_MODE=offline
export WANDB_PROJECT=tk-instruct-experts

python scripts/train_tk_instruct_experts.py \
    scripts/config/category_ts10_tr200_ev100_dev50.json \
    --exp_name 100-instances-per-task-2-epochs \
    --model_name_or_path google/t5-base-lm-adapt \
    --num_train_epochs 2 \
    --max_num_instances_per_task 100 \
    --eval_steps 1000 \
    --logging_steps 25 \
    --save_steps 1000 \
    --index $SLURM_ARRAY_TASK_ID
