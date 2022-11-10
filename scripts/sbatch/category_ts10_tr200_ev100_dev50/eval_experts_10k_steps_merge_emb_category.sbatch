#!/bin/bash

#SBATCH --partition=gpu-titan
#SBATCH --account=ark
#SBATCH --job-name=mecat10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=3:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm/mecat10k-%A-%a.log
#SBATCH --error=slurm/mecat10k-%A-%a.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahuln@cs.washington.edu

export WANDB_DISABLED=true
python scripts/eval_tk_instruct_experts_merge_emb.py \
    scripts/config/category_ts10_tr200_ev100_dev50.json \
    data/splits/category-ts10-tr200-ev100/embeddings/instr-emb-sentence-transformers-all-mpnet-base-v2-category-mean-def.pt \
    --exp_name all-instances-10000-steps \
    --data_dir data/splits/category-ts10-tr200-ev100/test \
    --num_experts_to_merge 3 \
    --suffix instr-emb-sentbert-mean-def-pos \
    --index $SLURM_ARRAY_TASK_ID
