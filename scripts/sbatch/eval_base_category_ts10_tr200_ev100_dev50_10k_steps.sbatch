#!/bin/bash

#SBATCH --partition=gpu-titan
#SBATCH --account=ark
#SBATCH --job-name=ev10kcm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=3:00:00
#SBATCH --output=slurm/ev10kcm-%A-%a.log
#SBATCH --error=slurm/ev10kcm-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

set -x

export WANDB_DISABLED=true

python src/run_s2s.py \
    --do_predict \
    --predict_with_generate \
    --evaluation_strategy "no" \
    --model_name_or_path results/niv2-ts10-tr200-ev100-dev50/tk-instruct-base-cont/train/all-instances-10000-steps-compute-matched \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/category-ts10-tr200-ev100/test/all \
    --task_dir data/tasks \
    --output_dir results/niv2-ts10-tr200-ev100-dev50/tk-instruct-base-cont/evaluate/test/all-instances-10000-steps-compute-matched \
    --cache_dir /gscratch/ark/rahuln/.cache/ \
    --overwrite_cache \
    --per_device_eval_batch_size 4 \
    --run_name tk-instruct-base-cont-eval \
    --use_dev
