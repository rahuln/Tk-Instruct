#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=eval-train-t5-experts-20-epochs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-59
#SBATCH --output=slurm/eval-train-t5-experts-20-epochs-%A-%a.log
#SBATCH --error=slurm/eval-train-t5-experts-20-epochs-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true

python scripts/eval_tk_instruct_experts.py \
    scripts/config/category.json \
    --exp_name 100-instances-per-task-20-epochs \
    --data_dir data/splits/category-ts10-tr200-ev100/train/all \
    --eval_dirname train-450-dev-50 \
    --base_model google/t5-base-lm-adapt \
    --max_num_instances_per_task 100 \
    --num_dev 50 \
    --index $SLURM_ARRAY_TASK_ID
