#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=tkiexcat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-59
#SBATCH --output=slurm/tkiexcat-%A-%a.log
#SBATCH --error=slurm/tkiexcat-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true
export WANDB_MODE=offline
python scripts/eval_tk_instruct_experts.py \
    scripts/config/category_experts.json \
    --exp_name 2-epochs \
    --data_dir data/splits/train_as_eval \
    --eval_dirname test \
    --max_num_instances_per_task 100 \
    --index $SLURM_ARRAY_TASK_ID
