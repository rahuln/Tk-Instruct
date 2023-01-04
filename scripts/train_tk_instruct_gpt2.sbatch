#!/bin/bash

#SBATCH --partition=gpu-a40
#SBATCH --account=cse
#SBATCH --job-name=tkingpt2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/tkingpt2-%A.log
#SBATCH --error=slurm/tkingpt2-%A.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

set -x

export TRANSFORMERS_CACHE=/gscratch/ark/rahuln/.cache/huggingface
export WANDB_DISABLED=true

bsz=$((2 * $1))

python src/run_s2s_alt.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --max_source_length 768 \
    --max_target_length 128 \
    --generation_max_length 1024 \
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
    --output_dir results/niv2/gpt2-large/train/epo2_bsz${bsz}_lr$2_const \
    --cache_dir /gscratch/ark/rahuln/.cache \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $1 \
    --learning_rate $2 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --run_name tk-instruct-gpt2 \
    --model_name_or_path gpt2-large \
    --num_train_epochs 2 \
    --use_causal_lm True
    # --do_eval \
    # --evaluation_strategy steps \
    # --eval_steps 500 \
    # --use_dev True \
    # --num_dev 50
