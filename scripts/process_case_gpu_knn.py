#!/usr/bin/env python3
"""Process one UPENN-GBM case into GPU kNN tumor graph artifacts."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch


MODALITIES = ("T1", "T1GD", "T2", "FLAIR", "DTI_FA", "DTI_AD", "DTI_RD", "DTI_TR")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def case_id_from_seg(seg_path: Path) -> str:
    name = seg_path.name
    return name[: -len("_segm.nii.gz")] if name.endswith("_segm.nii.gz") else seg_path.stem


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    return name[: -len(".nii.gz")] if name.endswith(".nii.gz") else path.stem


def matches_modality(path: Path, modality: str) -> bool:
    stem = strip_nii_suffix(path)
    return stem == modality or stem.endswith(f"_{modality}") or f"_{modality}_" in stem


def find_segmentation(nifti_root: Path, case_id: str | None = None) -> Path:
    """Find the segmentation for a single requested UPENN-GBM case.

    The expected dataset layout is:
      NIfTI-files/images_segm/<case_id>_segm.nii.gz

    Older versions of this script used rglob() and returned the first segmentation
    in the entire dataset, which silently processed the wrong case. This function
    now enforces exact case matching whenever --case-id is provided.
    """
    if case_id:
        exact = nifti_root / "images_segm" / f"{case_id}_segm.nii.gz"
        if exact.is_file():
            return exact
        raise FileNotFoundError(f"Segmentation not found for case {case_id}: {exact}")

    matches = sorted((nifti_root / "images_segm").glob("*_segm.nii.gz"))
    if not matches:
        raise FileNotFoundError(f"No *_segm.nii.gz found under {nifti_root / 'images_segm'}")
    return matches[0]


def modality_folder(nifti_root: Path, case_id: str, modality: str) -> Path:
    """Return the case-specific folder for a modality."""
    if modality in {"T1", "T1GD", "T2", "FLAIR"}:
        return nifti_root / "images_structural" / case_id
    if modality.startswith("DTI_"):
        return nifti_root / "images_DTI" / case_id
    if modality.startswith("DSC_"):
        return nifti_root / "images_DSC" / case_id
    return nifti_root / case_id


def find_modality(nifti_root: Path, case_id: str, modality: str) -> Path | None:
    """Find one modality file inside the exact case subfolder.

    Expected examples:
      images_structural/UPENN-GBM-00375_11/UPENN-GBM-00375_11_T1.nii.gz
      images_DTI/UPENN-GBM-00375_11/UPENN-GBM-00375_11_DTI_FA.nii.gz
    """
    folder = modality_folder(nifti_root, case_id, modality)
    if not folder.exists():
        return None

    # Prefer exact expected filename.
    exact = folder / f"{case_id}_{modality}.nii.gz"
    if exact.is_file():
        return exact

    # Fall back to patterns within the same case folder only.
    patterns = [
        f"*_{modality}.nii.gz",
        f"*{modality}*.nii.gz",
    ]
    for pattern in patterns:
        matches = sorted(
            p for p in folder.glob(pattern)
            if p.is_file() and "_segm" not in p.name and matches_modality(p, modality)
        )
        if matches:
            return matches[0]

    return None


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def choose_voxels(seg: np.ndarray, max_nodes: int, seed: int) -> np.ndarray:
    coords = np.argwhere(seg > 0)
    if coords.size == 0:
        raise ValueError("Segmentation has no voxels with seg > 0")
    if len(coords) <= max_nodes:
        return coords
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(coords), size=max_nodes, replace=False)
    return coords[np.sort(indices)]


def zscore(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    safe_std = np.where(std < 1e-8, 1.0, std)
    return (features - mean) / safe_std, mean, safe_std


def compute_knn(features: np.ndarray, k: int, device_name: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    if len(features) < 2:
        raise ValueError("At least 2 nodes are required to build kNN edges")

    device = torch.device(device_name)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    k_eff = min(k, x.shape[0] - 1)
    all_indices = []
    all_distances = []

    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            dist = torch.cdist(x[start:stop], x)
            rows = torch.arange(stop - start, device=device)
            dist[rows, torch.arange(start, stop, device=device)] = float("inf")
            values, indices = torch.topk(dist, k=k_eff, largest=False, sorted=True)
            all_indices.append(indices.cpu().numpy())
            all_distances.append(values.cpu().numpy())

    return np.vstack(all_indices), np.vstack(all_distances)


def save_previews(seg: np.ndarray, coords: np.ndarray, out_dir: Path, case_id: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    z_mid = int(np.median(coords[:, 2]))

    slice_png = out_dir / f"{case_id}_seg_slice.png"
    plt.figure(figsize=(6, 6))
    plt.imshow(seg[:, :, z_mid].T, cmap="gray", origin="lower")
    plt.title(f"{case_id} segmentation z={z_mid}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(slice_png, dpi=150)
    plt.close()

    scatter_png = out_dir / f"{case_id}_sampled_nodes_xy.png"
    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.5)
    plt.title(f"{case_id} sampled tumor nodes")
    plt.xlabel("i")
    plt.ylabel("j")
    plt.tight_layout()
    plt.savefig(scatter_png, dpi=150)
    plt.close()
    return [str(slice_png), str(scatter_png)]


def process_case(args: argparse.Namespace) -> dict:
    # case_dir is the UPENN-GBM NIfTI-files root, not an individual case folder.
    case_dir = args.case_dir.expanduser().resolve()
    out_root = args.output_dir.expanduser().resolve()

    seg_path = find_segmentation(case_dir, args.case_id or None)
    case_id = args.case_id or case_id_from_seg(seg_path)
    out_dir = out_root / case_id

    # Hard guard: never silently process a different case than requested.
    if case_id not in seg_path.name:
        raise RuntimeError(f"Wrong segmentation selected for {case_id}: {seg_path}")

    print(f"[PATH] case_id={case_id}", flush=True)
    print(f"[PATH] seg_path={seg_path}", flush=True)

    seg = load_nifti(seg_path)
    if seg.ndim != 3:
        raise ValueError(f"Expected 3D segmentation, got shape {seg.shape}")
    seg = seg.astype(np.float32)

    loaded = {}
    skipped = {}
    for modality in MODALITIES:
        path = find_modality(case_dir, case_id, modality)
        if path is None:
            skipped[modality] = "missing"
            continue
        print(f"[PATH] {modality}={path}", flush=True)
        arr = load_nifti(path)
        if arr.shape != seg.shape:
            skipped[modality] = f"shape mismatch: {arr.shape} != {seg.shape}"
            continue
        loaded[modality] = arr.astype(np.float32)

    if not loaded:
        raise ValueError("No usable modalities found with shape matching segmentation")

    coords = choose_voxels(seg, args.max_nodes, args.seed)
    feature_columns = list(loaded)
    raw_features = np.column_stack([loaded[m][coords[:, 0], coords[:, 1], coords[:, 2]] for m in feature_columns])
    features, means, stds = zscore(raw_features.astype(np.float32))

    metrics = {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "seg_path": str(seg_path),
        "started_at": now_iso(),
        "status": "dry_run" if args.dry_run else "ok",
        "seg_shape": list(seg.shape),
        "tumor_voxels": int(np.count_nonzero(seg > 0)),
        "sampled_nodes": int(len(coords)),
        "max_nodes": int(args.max_nodes),
        "k_requested": int(args.k),
        "modalities_used": feature_columns,
        "modalities_skipped": skipped,
        "feature_mean": {m: float(v) for m, v in zip(feature_columns, means)},
        "feature_std": {m: float(v) for m, v in zip(feature_columns, stds)},
        "device_requested": args.device,
    }

    if args.dry_run:
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return metrics

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics["device_used"] = device

    knn_idx, knn_dist = compute_knn(features, args.k, device, args.batch_size)
    k_eff = knn_idx.shape[1]
    metrics["k_effective"] = int(k_eff)

    out_dir.mkdir(parents=True, exist_ok=True)
    node_data = {
        "node_id": np.arange(len(coords), dtype=np.int64),
        "i": coords[:, 0],
        "j": coords[:, 1],
        "k": coords[:, 2],
        "seg_value": seg[coords[:, 0], coords[:, 1], coords[:, 2]],
    }
    for pos, modality in enumerate(feature_columns):
        node_data[f"{modality}_z"] = features[:, pos]
    pd.DataFrame(node_data).to_csv(out_dir / "nodes.csv", index=False)

    src = np.repeat(np.arange(len(coords), dtype=np.int64), k_eff)
    edge_df = pd.DataFrame(
        {
            "source": src,
            "target": knn_idx.reshape(-1).astype(np.int64),
            "distance": knn_dist.reshape(-1).astype(np.float32),
        }
    )
    edge_df.to_csv(out_dir / f"edges_k{args.k}.csv", index=False)

    metrics["edge_count"] = int(len(edge_df))
    metrics["preview_pngs"] = save_previews(seg, coords, out_dir, case_id)
    metrics["finished_at"] = now_iso()
    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path, help="Case directory containing NIfTI files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for per-case output folders.")
    parser.add_argument("--case-id", default="", help="Optional case id override.")
    parser.add_argument("--max-nodes", type=int, default=20000, help="Maximum tumor voxels to sample.")
    parser.add_argument("--k", type=int, default=20, help="Number of nearest neighbors.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Rows per torch.cdist batch.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for voxel sampling.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"), help="Compute device.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned work without writing outputs.")
    parser.add_argument("--fail-on-error", action="store_true", help="Raise errors instead of writing skipped metrics.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        process_case(args)
        return 0
    except Exception as exc:
        if args.fail_on_error:
            raise
        case_dir = args.case_dir.expanduser().resolve()
        case_id = args.case_id or case_dir.name
        out_dir = args.output_dir.expanduser().resolve() / case_id
        payload = {
            "case_id": case_id,
            "case_dir": str(case_dir),
            "status": "skipped",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "finished_at": now_iso(),
        }
        if args.dry_run:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            write_json(out_dir / "metrics.json", payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
