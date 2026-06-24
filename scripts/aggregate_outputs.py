#!/usr/bin/env python3
"""Aggregate per-case kNN output metrics into a CSV summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def flatten_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    row = {
        "case_id": payload.get("case_id", path.parent.name),
        "status": payload.get("status", ""),
        "sampled_nodes": payload.get("sampled_nodes", 0),
        "edge_count": payload.get("edge_count", 0),
        "k_effective": payload.get("k_effective", ""),
        "tumor_voxels": payload.get("tumor_voxels", 0),
        "device_used": payload.get("device_used", ""),
        "modalities_used": ",".join(payload.get("modalities_used", [])),
        "error": payload.get("error", ""),
        "metrics_path": str(path),
    }
    skipped = payload.get("modalities_skipped", {})
    if isinstance(skipped, dict):
        row["modalities_skipped"] = "; ".join(f"{k}:{v}" for k, v in skipped.items())
    else:
        row["modalities_skipped"] = ""
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="Root directory containing per-case output folders.")
    parser.add_argument("--summary-csv", required=True, type=Path, help="Output summary CSV path.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing the CSV.")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.exists():
        print(f"ERROR: output directory does not exist: {output_dir}", file=sys.stderr)
        return 2

    rows = []
    for metrics_path in sorted(output_dir.rglob("metrics.json")):
        try:
            rows.append(flatten_metrics(metrics_path))
        except Exception as exc:
            rows.append(
                {
                    "case_id": metrics_path.parent.name,
                    "status": "aggregate_error",
                    "error": str(exc),
                    "metrics_path": str(metrics_path),
                }
            )

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False) if not summary.empty else "No metrics.json files found.")
    if args.dry_run:
        return 0

    args.summary_csv.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    print(f"Wrote summary: {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
