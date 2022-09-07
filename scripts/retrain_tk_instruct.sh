#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/gscratch/ark/rahuln/.cache/huggingface
export WANDB_PROJECT=tk-instruct-experts

port=$(shuf -i25000-30000 -n1)

python src/run_s2s.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path google/t5-base-lm-adapt \
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
    --output_dir results/natural-instructions-v2/tk-instruct-base/retrain \
    --overwrite_output_dir \
    --cache_dir /gscratch/ark/rahuln/.cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-05 \
    --num_train_epochs 2 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 50 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 2500 \
    --run_name tk-instruct-base-retrain
