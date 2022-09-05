#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/gscratch/ark/rahuln/.cache/ \
export WANDB_PROJECT=tk-instruct-experts

python src/run_s2s.py \
    --do_predict \
    --predict_with_generate \
    --evaluation_strategy "no" \
    --model_name_or_path results/natural-instructions-v2/tk-instruct-base-cont/train \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/default \
    --task_dir data/tasks \
    --output_dir results/natural-instructions-v2/tk-instruct-base-cont/evaluate \
    --overwrite_output_dir \
    --cache_dir /gscratch/ark/rahuln/.cache/ \
    --overwrite_cache \
    --per_device_eval_batch_size 4 \
    --run_name tk-instruct-base-cont-eval
