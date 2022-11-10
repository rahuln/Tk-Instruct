#!/bin/bash

#SBATCH --partition=gpu-2080ti
#SBATCH --account=ark
#SBATCH --job-name=evmrgac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --array=0-65
#SBATCH --output=slurm/evmrgac-%A-%a.log
#SBATCH --error=slurm/evmrgac-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true
export WANDB_MODE=offline

BASEDIR=results/niv2-ts10-tr200-ev100-dev50/tk-instruct-base-experts/train/1163-steps
MODEL1=$BASEDIR/question_answering/pytorch_model.bin
MODEL2=$BASEDIR/question_generation/pytorch_model.bin
MODEL3=$BASEDIR/question_understanding/pytorch_model.bin

python scripts/eval_tk_instruct_experts_merge_grid.py \
    scripts/config/category_with_dev.json \
    --models $MODEL1 $MODEL2 $MODEL3 \
    --category answerability_classification \
    --exp_name 1163-steps \
    --data_dir data/splits/category-with-dev/test \
    --eval_dirname merge-grid-test \
    --index $SLURM_ARRAY_TASK_ID
