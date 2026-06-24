# upenn_knn_demo

Minimal Python/SLURM project for processing minimally preprocessed UPENN-GBM cases into GPU kNN tumor graph artifacts on NCSA Delta.

The pipeline:

1. Finds each `*_segm.nii.gz` segmentation.
2. Finds available `T1`, `T1GD`, `T2`, `FLAIR`, `DTI_FA`, `DTI_AD`, `DTI_RD`, and `DTI_TR` volumes.
3. Skips modalities whose array shape differs from the segmentation.
4. Samples up to `--max-nodes` voxels where `seg > 0`.
5. Builds a z-scored feature matrix.
6. Computes kNN with `torch.cdist` and `torch.topk`.
7. Writes graph artifacts and preview images.

Only these non-standard Python packages are used: `nibabel`, `numpy`, `pandas`, `torch`, `matplotlib`, and `scikit-learn`.

## Layout

```text
upenn_knn_demo/
  scripts/
    scan_manifest.py
    process_case_gpu_knn.py
    aggregate_outputs.py
  slurm/
    run_array_knn.slurm
  requirements.txt
  README.md
```

## Outputs

Each successful case writes:

- `nodes.csv`: sampled voxel coordinates, segmentation value, and z-scored modality features.
- `edges_k20.csv`: directed kNN edges with feature-space distances.
- `metrics.json`: processing metadata, modality skips, node counts, edge counts, and errors when skipped.
- `<case_id>_seg_slice.png`: segmentation preview.
- `<case_id>_sampled_nodes_xy.png`: sampled node preview.

Bad cases are skipped gracefully. By default, `process_case_gpu_knn.py` writes a `metrics.json` with `status: "skipped"` and exits with code 0. Use `--fail-on-error` when debugging.

## Local Dry Runs

Create a manifest without writing it:

```bash
python scripts/scan_manifest.py \
  --input-dir /path/to/UPENN-GBM \
  --output manifest.csv \
  --dry-run
```

Write the manifest:

```bash
python scripts/scan_manifest.py \
  --input-dir /path/to/UPENN-GBM \
  --output manifest.csv
```

Dry-run one case:

```bash
python scripts/process_case_gpu_knn.py \
  --case-dir /path/to/UPENN-GBM/UPENN-GBM-00001 \
  --output-dir outputs \
  --max-nodes 20000 \
  --k 20 \
  --dry-run
```

Process one case on GPU:

```bash
python scripts/process_case_gpu_knn.py \
  --case-dir /path/to/UPENN-GBM/UPENN-GBM-00001 \
  --output-dir outputs \
  --max-nodes 20000 \
  --k 20 \
  --device cuda
```

Aggregate outputs:

```bash
python scripts/aggregate_outputs.py \
  --output-dir outputs \
  --summary-csv outputs/summary.csv
```

## NCSA Delta Example

Clone or copy this project to Delta, then create an environment with the required packages. One typical approach is:

```bash
module load anaconda3_gpu
conda create -n upenn-knn python=3.11 -y
conda activate upenn-knn
pip install -r requirements.txt
```

Build the manifest on a login node:

```bash
python scripts/scan_manifest.py \
  --input-dir /projects/YOUR_PROJECT/UPENN-GBM \
  --output manifest.csv
```

Count cases and submit an array. If the manifest has `N` cases, use `0-(N-1)`:

```bash
N=$(python - <<'PY'
import pandas as pd
print(len(pd.read_csv("manifest.csv")))
PY
)
sbatch --array=0-$((N-1)) \
  --account=YOUR_DELTA_ACCOUNT \
  --export=ALL,PROJECT_DIR=$PWD,MANIFEST=$PWD/manifest.csv,OUTPUT_DIR=$PWD/outputs,CONDA_ENV=upenn-knn,MAX_NODES=20000,K=20 \
  slurm/run_array_knn.slurm
```

After jobs finish:

```bash
python scripts/aggregate_outputs.py \
  --output-dir outputs \
  --summary-csv outputs/summary.csv
```

## Notes

- `torch.cdist` computes distances in batches over rows to reduce peak memory use, but kNN is still quadratic in the sampled node count.
- Use smaller `--max-nodes` values for first-pass testing.
- The SLURM script defaults to `gpuA40x4`; edit `#SBATCH --partition`, `#SBATCH --account`, and module lines for your allocation and Delta environment.
