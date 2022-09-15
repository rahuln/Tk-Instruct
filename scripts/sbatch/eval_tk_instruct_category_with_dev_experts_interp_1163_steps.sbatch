#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=intexcat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=6:00:00
#SBATCH --array=0-164
#SBATCH --output=slurm/intexcat-%A-%a.log
#SBATCH --error=slurm/intexcat-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true
export WANDB_MODE=offline
python scripts/eval_tk_instruct_interp_base_expert.py scripts/config/category_with_dev_experts_interp_1163_steps.json $SLURM_ARRAY_TASK_ID
