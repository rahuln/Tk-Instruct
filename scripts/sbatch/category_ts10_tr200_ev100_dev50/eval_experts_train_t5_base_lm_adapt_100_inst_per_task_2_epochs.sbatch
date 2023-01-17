#!/bin/bash

#SBATCH --partition=gpu-titan
#SBATCH --account=ark
#SBATCH --job-name=eval-train-t5-experts-2-epochs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=6:00:00
#SBATCH --array=0-14
#SBATCH --output=slurm/eval-train-t5-experts-2-epochs-%A-%a.log
#SBATCH --error=slurm/eval-train-t5-experts-2-epochs-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true

python scripts/eval_tk_instruct_experts.py \
    scripts/config/category_ts10_tr200_ev100_dev50.json \
    --exp_name 100-instances-per-task-2-epochs \
    --data_dir data/splits/category-ts10-tr200-ev100/train/all \
    --eval_dirname train \
    --base_model google/t5-base-lm-adapt \
    --max_num_instances_per_task 100 \
    --index $SLURM_ARRAY_TASK_ID
