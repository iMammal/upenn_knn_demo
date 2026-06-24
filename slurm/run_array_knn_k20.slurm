#!/bin/bash
#SBATCH -A betm-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH --array=0-144
#SBATCH -o logs/knn_%A_%a.out
#SBATCH -e logs/knn_%A_%a.err

set -eo pipefail

cd /projects/betm/mchukhma/upenn_knn_demo

module load pytorch-conda/2.8
conda activate /projects/betm/mchukhma/conda-envs/upenn-knn

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CASE_ID=$(sed -n "$((TASK_ID+1))p" cases_all_usable.txt)

if [[ -z "${CASE_ID}" ]]; then
  echo "[ERROR] No CASE_ID found for TASK_ID=${TASK_ID}"
  exit 1
fi

echo "[INFO] CASE_ID=$CASE_ID"
echo "[INFO] host=$(hostname)"
echo "[INFO] start=$(date)"

python ./scripts/process_case_gpu_knn.py \
  --case-id "$CASE_ID" \
  --case-dir "../UPENN-GBM/NIfTI-files" \
  --output-dir "outputs/${CASE_ID}_32k_k20_cuda" \
  --max-nodes 32000 \
  --k 20 \
  --device cuda

echo "[INFO] end=$(date)"
