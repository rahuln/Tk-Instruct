#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=gstsk10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-449
#SBATCH --output=slurm/gstsk10k-%A-%a.log
#SBATCH --error=slurm/gstsk10k-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true
python scripts/greedy_soup_tk_instruct_experts.py \
    scripts/config/category.json \
    --exp_name 100-instances-per-task-2-epochs \
    --data_dir data/splits/tasks/train \
    --max_soup_size 10 \
    --base_model google/t5-base-lm-adapt \
    --max_num_instances_per_task 100 \
    --eval_on_task data/splits/category-ts10-tr200-ev100/train/all/train_tasks.txt \
    --use_train_as_dev \
    --suffix train-tasks-450 \
    --use_dev \
    --num_dev 50 \
    --index $SLURM_ARRAY_TASK_ID
