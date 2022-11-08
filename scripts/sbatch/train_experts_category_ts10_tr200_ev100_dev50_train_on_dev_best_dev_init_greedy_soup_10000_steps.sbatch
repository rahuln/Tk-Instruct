#!/bin/bash

#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --job-name=trigs10d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm/trigs10d-%A-%a.log
#SBATCH --error=slurm/trigs10d-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_MODE=offline
export WANDB_PROJECT=tk-instruct-experts

# base path to greedy soups and array of test task categories
basedir=results/niv2-ts10-tr200-ev100-dev50/tk-instruct-base-experts/evaluate/greedy-soup-include-base/all-instances-10000-steps
tasks=(
    "answerability_classification"
    "cause_effect_classification"
    "coreference_resolution"
    "data_to_text"
    "dialogue_act_recognition"
    "grammar_error_correction"
    "keyword_tagging"
    "overlap_extraction"
    "question_rewriting"
    "textual_entailment"
    "title_generation"
    "word_analogy"
)

# define path to greedy soup
model_name_or_path=${basedir}/${tasks[$SLURM_ARRAY_TASK_ID]}

python scripts/train_tk_instruct_experts.py \
    scripts/config/category_ts10_tr200_ev100_dev50.json \
    --exp_name train-6000-eval-30-dev-45-load-best \
    --model_name_or_path $model_name_or_path \
    --train_on_dev \
    --max_steps 6000 \
    --eval_steps 30 \
    --logging_steps 10 \
    --save_steps 30 \
    --num_dev 45 \
    --load_best_model_at_end \
    --metric_for_best_model eval_rougeL \
    --greater_is_better \
    --index $SLURM_ARRAY_TASK_ID
