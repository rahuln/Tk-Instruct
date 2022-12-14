#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=psouptrn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-179%30
#SBATCH --output=slurm/psouptrn-%A-%a.log
#SBATCH --error=slurm/psouptrn-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true

# get indices for gradient accumulation steps, learning rate, and category
grad_accum_idx=$((($SLURM_ARRAY_TASK_ID % 12) / 4))
learning_rate_idx=$((($SLURM_ARRAY_TASK_ID % 12) % 4))
eval_category_idx=$(($SLURM_ARRAY_TASK_ID / 12))

# get gradient accumulation steps (batch size / 2)
gradient_accum_steps_grid=(4 8 16)
gradient_accumulation_steps=${gradient_accum_steps_grid[$grad_accum_idx]}

# get learning rate
learning_rate_grid=(1e-5 5e-5 1e-4 5e-4)
learning_rate=${learning_rate_grid[$learning_rate_idx]}

# get held-out category
categories=(
    "commonsense_classification"
    "information_extraction"
    "misc."
    "program_execution"
    "question_answering"
    "question_generation"
    "question_understanding"
    "sentiment_analysis"
    "summarization"
    "text_categorization"
    "text_completion"
    "text_matching"
    "text_to_code"
    "toxic_language_detection"
    "wrong_candidate_generation"
)
eval_category=${categories[$eval_category_idx]}

# construct output directory
batch_size=$((2 * $gradient_accumulation_steps))
basedir=results/niv2-ts10-tr200-ev100-dev50/pred-greedy-soup-roberta-base
expdir=ep10_bsz${batch_size}_lr${learning_rate}
output_dir=${basedir}/${eval_category}/${expdir}
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

# check for existing result
if [ -f $output_dir/test_results.json ]
then
    echo "results already exist, exiting..."
    exit
fi

set -x

# run training command
time python scripts/predict_soups_for_train_tasks.py \
    --output_dir $output_dir \
    --logging_dir $output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 10 \
    --learning_rate $learning_rate \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy epoch \
    --logging_strategy steps \
    --logging_first_step \
    --logging_steps 50 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_auroc \
    --eval_category $eval_category_idx

# delete checkpoints, saved model
if [ -f $output_dir/test_results.json ]
then
    rm -rf $output_dir/pytorch_model.bin
    rm -rf $output_dir/checkpoint*
fi
