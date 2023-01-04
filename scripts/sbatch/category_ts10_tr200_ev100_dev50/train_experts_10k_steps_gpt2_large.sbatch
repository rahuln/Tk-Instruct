#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=trgpt10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=slurm/trgpt10k-%A-%a.log
#SBATCH --error=slurm/trgpt10k-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_MODE=offline
export WANDB_PROJECT=tk-instruct-experts

python scripts/train_tk_instruct_experts.py \
    scripts/config/category_ts10_tr200_ev100_dev50.json \
    --max_source_length 768 \
    --generation_max_length 1024 \
    --model_name_or_path results/niv2/gpt2-large/train/epo2_bsz16_lr5e-05_const \
    --exp_name all-instances-10000-steps \
    --max_steps 10000 \
    --eval_steps 1000 \
    --logging_steps 25 \
    --save_steps 1000 \
    --use_causal_lm \
    --model_dirname_prefix gpt2-large \
    --index $SLURM_ARRAY_TASK_ID
