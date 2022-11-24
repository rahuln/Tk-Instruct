#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=gstsk10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=6:00:00
#SBATCH --array=0-116
#SBATCH --output=slurm/gstsk10k-%A-%a.log
#SBATCH --error=slurm/gstsk10k-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true
python scripts/greedy_soup_tk_instruct_experts.py \
    scripts/config/category_ts10_tr200_ev100_dev50.json \
    --exp_name all-instances-10000-steps \
    --data_dir data/splits/tasks/test \
    --max_soup_size 10 \
    --include_base_model \
    --eval_on_task data/splits/category-ts10-tr200-ev100/test/all/train_tasks.txt \
    --index $SLURM_ARRAY_TASK_ID
