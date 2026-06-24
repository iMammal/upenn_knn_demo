#!/bin/bash
#SBATCH -A betm-delta-gpu
#SBATCH -p gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH --array=0-15
#SBATCH -o logs/shard_%A_%a.out
#SBATCH -e logs/shard_%A_%a.err

set -eo pipefail

cd /projects/betm/mchukhma/upenn_knn_demo

module load pytorch-conda/2.8
conda activate /projects/betm/mchukhma/conda-envs/upenn-knn

echo "[INFO] shard task=${SLURM_ARRAY_TASK_ID}"
echo "[INFO] host=$(hostname)"
echo "[INFO] start=$(date)"

python scripts/run_knn_shard.py \
  --case-list scripts/cases_all_usable.txt \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --num-shards 16 \
  --k-values "20,40,60,80" \
  --max-nodes 64000 \
  --seed 3141
echo "[INFO] end=$(date)"
