#!/bin/bash

#SBATCH --partition=gpu-2080ti
#SBATCH --account=ark
#SBATCH --job-name=gscat10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm/gscat10k-%A-%a.log
#SBATCH --error=slurm/gscat10k-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true
python scripts/greedy_soup_tk_instruct_experts.py \
    scripts/config/category_ts10_tr200_ev100_dev50.json \
    --exp_name all-instances-10000-steps \
    --data_dir data/splits/category-ts10-tr200-ev100/test \
    --max_soup_size 10 \
    --include_base_model \
    --use_test_as_dev \
    --index $SLURM_ARRAY_TASK_ID
