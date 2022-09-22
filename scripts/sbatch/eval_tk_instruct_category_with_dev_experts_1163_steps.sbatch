#!/bin/bash

#SBATCH --partition=gpu-2080ti
#SBATCH --account=ark
#SBATCH --job-name=evexcat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=6:00:00
#SBATCH --array=0-14
#SBATCH --output=slurm/evexcat-%A-%a.log
#SBATCH --error=slurm/evexcat-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true
export WANDB_MODE=offline
python scripts/eval_tk_instruct_experts.py scripts/config/category_with_dev_experts_1163_steps.json $SLURM_ARRAY_TASK_ID
