#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=eval-merge-post-gpt2-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=3:00:00
#SBATCH --array=0-116
#SBATCH --output=slurm/eval-merge-post-gpt2-large-%A-%a.log
#SBATCH --error=slurm/eval-merge-post-gpt2-large-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true

python scripts/eval_tk_instruct_experts_merge_posterior.py \
    scripts/config/category_ts10_tr200_ev100_dev50.json \
    results/niv2-ts10-tr200-ev100-dev50/tk-instruct-base-experts/train/all-instances-10000-steps \
    --path_to_likelihood_models results/niv2-ts10-tr200-ev100-dev50/gpt2-large-experts/train/all-instances-10000-steps \
    --exp_name all-instances-10000-steps \
    --data_dir data/splits/tasks/test \
    --max_num_instances_per_task 100 \
    --use_merging_weights \
    --suffix gpt2-large-experts \
    --index $SLURM_ARRAY_TASK_ID
