#!/usr/bin/env python3
"""Create a manifest of UPENN-GBM cases for GPU kNN processing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


MODALITIES = ("T1", "T1GD", "T2", "FLAIR", "DTI_FA", "DTI_AD", "DTI_RD", "DTI_TR")


def find_case_dirs(input_dir: Path) -> list[Path]:
    seg_dirs = {p.parent for p in input_dir.rglob("*_segm.nii.gz") if p.is_file()}
    return sorted(seg_dirs)


def case_id_from_seg(seg_path: Path) -> str:
    name = seg_path.name
    return name[: -len("_segm.nii.gz")] if name.endswith("_segm.nii.gz") else seg_path.stem


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    return name[: -len(".nii.gz")] if name.endswith(".nii.gz") else path.stem


def matches_modality(path: Path, modality: str) -> bool:
    stem = strip_nii_suffix(path)
    return stem == modality or stem.endswith(f"_{modality}") or f"_{modality}_" in stem


def find_modality(case_dir: Path, case_id: str, modality: str) -> str:
    exact = case_dir / f"{case_id}_{modality}.nii.gz"
    if exact.is_file():
        return str(exact)
    matches = sorted(
        p
        for p in case_dir.glob("*.nii.gz")
        if p.is_file() and "_segm" not in p.name and matches_modality(p, modality)
    )
    if matches:
        return str(matches[0])
    return ""


def build_manifest(input_dir: Path) -> pd.DataFrame:
    rows = []
    for case_dir in find_case_dirs(input_dir):
        segs = sorted(case_dir.glob("*_segm.nii.gz"))
        if not segs:
            continue
        seg_path = segs[0]
        case_id = case_id_from_seg(seg_path)
        row = {
            "case_id": case_id,
            "case_dir": str(case_dir),
            "seg_path": str(seg_path),
        }
        for modality in MODALITIES:
            row[modality] = find_modality(case_dir, case_id, modality)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Root directory containing UPENN-GBM case folders.")
    parser.add_argument("--output", required=True, type=Path, help="Output manifest CSV path.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovery summary without writing the CSV.")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    manifest = build_manifest(input_dir)
    summary = {
        "input_dir": str(input_dir),
        "case_count": int(len(manifest)),
        "output": str(output),
    }
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        if not manifest.empty:
            print(manifest.head().to_string(index=False))
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(f"Wrote manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
