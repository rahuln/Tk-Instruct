#!/bin/bash

#SBATCH --partition=gpu-titan
#SBATCH --account=ark
#SBATCH --job-name=rncat10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=6:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm/rncat10k-%A-%a.log
#SBATCH --error=slurm/rncat10k-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true
export WANDB_MODE=offline

# mapping from index of test category to soup size
declare -A soup_sizes
soup_sizes=( [0]=3 [1]=2 [2]=5 [3]=1 [4]=2 [5]=3 [6]=2 [7]=4 [8]=2 [9]=3 [10]=3 [11]=3 )

python scripts/eval_tk_instruct_experts.py \
    scripts/config/category_ts10_tr200_ev100_dev50.json \
    --exp_name all-instances-10000-steps \
    --data_dir data/splits/category-ts10-tr200-ev100/test \
    --eval_dirname random-merge \
    --index $SLURM_ARRAY_TASK_ID \
    --random_merge ${soup_sizes[$SLURM_ARRAY_TASK_ID]}
