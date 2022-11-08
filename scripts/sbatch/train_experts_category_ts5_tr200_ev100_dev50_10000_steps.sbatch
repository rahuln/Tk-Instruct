#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=tr5ts10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-30
#SBATCH --output=slurm/tr5ts10k-%A-%a.log
#SBATCH --error=slurm/tr5ts10k-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_MODE=offline
export WANDB_PROJECT=tk-instruct-experts
python scripts/train_tk_instruct_experts.py \
    scripts/config/category_ts5_tr200_ev100_dev50.json \
    --exp_name all-instances-10000-steps-best-ckpt \
    --max_steps 10000 \
    --logging_steps 25 \
    --eval_steps 100 \
    --save_steps 100 \
    --load_best_model_at_end \
    --metric_for_best_model eval_rougeL \
    --greater_is_better \
    --index $SLURM_ARRAY_TASK_ID
