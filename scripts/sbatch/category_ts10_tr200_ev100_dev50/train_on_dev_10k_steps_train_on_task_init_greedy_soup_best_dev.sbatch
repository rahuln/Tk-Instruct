#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=trigs10d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-116%16
#SBATCH --output=slurm/trigs10d-%A-%a.log
#SBATCH --error=slurm/trigs10d-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

if [ -z $1 ]
then
    echo "Error: must specify number of dev set examples"
    exit
fi

export WANDB_MODE=offline
export WANDB_PROJECT=tk-instruct-experts

# base path to greedy soups and array of test task categories
basedir=results/niv2-ts10-tr200-ev100-dev50/tk-instruct-base-experts/evaluate/greedy-soup-include-base-eval-task/all-instances-10000-steps-dev-$1

# define path to greedy soup
row=$(($SLURM_ARRAY_TASK_ID + 1))
task=$(ls -1 $basedir | sed -n ${row}p)
model_name_or_path=${basedir}/${task}

python scripts/train_tk_instruct_experts.py \
    scripts/config/category_ts10_tr200_ev100_dev50.json \
    --exp_name train-5000-eval-20-dev-$1-load-best \
    --model_name_or_path $model_name_or_path \
    --data_dir data/splits/tasks/test \
    --train_on_dev \
    --max_steps 5000 \
    --eval_steps 20 \
    --logging_steps 10 \
    --save_steps 20 \
    --num_dev $1 \
    --train_on_task data/splits/category-ts10-tr200-ev100/test/all/train_tasks.txt \
    --load_best_model_at_end \
    --metric_for_best_model eval_rougeL \
    --greater_is_better \
    --remove_saved_model \
    --index $SLURM_ARRAY_TASK_ID
