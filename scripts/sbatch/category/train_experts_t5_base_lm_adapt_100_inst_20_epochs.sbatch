#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=train-60-t5b-exp-20-epo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-59
#SBATCH --output=slurm/train-60-t5b-exp-20-epo-%A-%a.log
#SBATCH --error=slurm/train-60-t5b-exp-20-epo-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_MODE=offline
export WANDB_PROJECT=tk-instruct-experts

python scripts/train_tk_instruct_experts.py \
    scripts/config/category.json \
    --exp_name 100-instances-per-task-20-epochs \
    --model_name_or_path google/t5-base-lm-adapt \
    --num_train_epochs 20 \
    --max_num_instances_per_task 100 \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_dev 10 \
    --index $SLURM_ARRAY_TASK_ID
