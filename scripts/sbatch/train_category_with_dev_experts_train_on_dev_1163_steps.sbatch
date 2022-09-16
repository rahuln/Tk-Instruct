#!/bin/bash

#SBATCH --partition=gpu-2080ti
#SBATCH --account=ark
#SBATCH --job-name=trcatdev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=6:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm/trcatdev-%A-%a.log
#SBATCH --error=slurm/trcatdev-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_MODE=online
export WANDB_PROJECT=tk-instruct-experts
export WANDB_RUN_GROUP=test-categories-train-on-dev
python scripts/train_tk_instruct_experts.py scripts/config/category_with_dev_experts_train_on_dev_1163_steps.json $SLURM_ARRAY_TASK_ID
