#!/bin/bash
#SBATCH -A betm-delta-gpu
#SBATCH -p gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:55:00
#SBATCH --array=0-15
#SBATCH -o logs/kmeans_%A_%a.out
#SBATCH -e logs/kmeans_%A_%a.err

set -eo pipefail

cd /projects/betm/mchukhma/upenn_knn_demo

module load pytorch-conda/2.8
conda activate /projects/betm/mchukhma/conda-envs/upenn-knn

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-16}"
RUN_LIST="${RUN_LIST:-kmeans_run_dirs_all.txt}"
CLUSTERS="${CLUSTERS:-16,256}"
KMEANS_SEED="${KMEANS_SEED:-0}"

echo "[INFO] host=$(hostname)"
echo "[INFO] task=${TASK_ID}/${NUM_SHARDS}"
echo "[INFO] run_list=${RUN_LIST}"
echo "[INFO] clusters=${CLUSTERS}"
echo "[INFO] seed=${KMEANS_SEED}"
echo "[INFO] start=$(date)"

python - << PY
from pathlib import Path
import subprocess
import os

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
num_shards = int(os.environ.get("NUM_SHARDS", "16"))
run_list = os.environ.get("RUN_LIST", "kmeans_run_dirs_all.txt")
clusters = [int(x) for x in os.environ.get("CLUSTERS", "16,256").split(",")]
seed = int(os.environ.get("KMEANS_SEED", "0"))

dirs = [x.strip() for x in open(run_list) if x.strip()]
my_dirs = dirs[task_id::num_shards]

print(f"[INFO] total_dirs={len(dirs)} my_dirs={len(my_dirs)}", flush=True)

for d in my_dirs:
    run_dir = Path(d)
    case_subdirs = [p for p in run_dir.iterdir() if p.is_dir()]
    if not case_subdirs:
        print(f"[WARN] no case subdir: {run_dir}", flush=True)
        continue
    case_dir = case_subdirs[0]

    for c in clusters:
        metrics = case_dir / f"module_metrics_kmeans_c{c}_s{seed}.json"
        if metrics.exists():
            print(f"[SKIP] {metrics}", flush=True)
            continue

        print(f"[RUN] {run_dir} clusters={c} seed={seed}", flush=True)
        cmd = [
            "python", "scripts/gpu_kmeans_modules.py",
            "--run-dir", str(run_dir),
            "--clusters", str(c),
            "--seed", str(seed),
            "--device", "cuda",
        ]
        subprocess.run(cmd, check=True)
