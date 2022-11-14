#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=met5l10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=3:00:00
#SBATCH --array=0-118
#SBATCH --output=slurm/met5l10k-%A-%a.log
#SBATCH --error=slurm/met5l10k-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true
python scripts/eval_tk_instruct_experts_merge_emb.py \
    scripts/config/category_ts10_tr200_ev100_dev50.json \
    data/splits/category-ts10-tr200-ev100/embeddings/instr-emb-sentence-transformers-all-mpnet-base-v2-mean-def.pt \
    --exp_name all-instances-10000-steps \
    --base_model google/t5-base-lm-adapt \
    --data_dir data/splits/tasks/test \
    --task_embeddings \
    --num_experts_to_merge 3 \
    --suffix instr-emb-sentbert-mean-def \
    --index $SLURM_ARRAY_TASK_ID
