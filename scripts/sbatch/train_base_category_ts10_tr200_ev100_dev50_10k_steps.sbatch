#!/bin/bash

#SBATCH --partition=gpu-titan
#SBATCH --account=ark
#SBATCH --job-name=tr10kcm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/tr10kcm-%A-%a.log
#SBATCH --error=slurm/tr10kcm-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

set -x

export WANDB_MODE=offline
export WANDB_PROJECT=tk-instruct-experts

python src/run_s2s.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path allenai/tk-instruct-base-def-pos \
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
    --data_dir data/splits/category-ts10-tr200-ev100/train/all \
    --task_dir data/tasks \
    --output_dir results/niv2-ts10-tr200-ev100-dev50/tk-instruct-base-cont/train/all-instances-10000-steps-compute-matched \
    --cache_dir /gscratch/ark/rahuln/.cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-05 \
    --num_train_epochs 1 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 5000 \
    --save_strategy steps \
    --save_steps 5000 \
    --run_name tk-instruct-base-cont/all-instances-10000-steps-compute-matched \
    --use_dev \
    --relative_scales_file data/splits/category-ts10-tr200-ev100/train/num_inst_scales.json \
    --reduction_factor 0.3778959227391786
