#!/usr/bin/env python3
"""Validate DeepCytoCave ingest files generated from UPENN-GBM graph runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


INDEX_HEADER = ["subjectID", "network", "topology"]
REQUIRED_LUT_COLUMNS = {"label", "Anatomy", "region_name", "hemisphere"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--lut-name")
    return parser.parse_args()


def find_dataset_dir(export_dir: Path, dataset_name: str | None) -> Path:
    data_dir = export_dir / "data"
    if dataset_name:
        return data_dir / dataset_name
    candidates = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise ValueError("Pass --dataset-name when export data/ contains multiple dataset folders")
    return candidates[0]


def find_lut_path(export_dir: Path, lut_name: str | None) -> Path:
    data_dir = export_dir / "data"
    if lut_name:
        return data_dir / f"LookupTable_{lut_name}.csv"
    candidates = sorted(data_dir.glob("LookupTable_*.csv"))
    if len(candidates) != 1:
        raise ValueError("Pass --lut-name when zero or multiple LUT files are present")
    return candidates[0]


def read_csv_rows(path: Path, delimiter: str = ",") -> list[list[str]]:
    with path.open(newline="") as handle:
        return list(csv.reader(handle, delimiter=delimiter))


def validate_index(dataset_dir: Path, errors: list[str]) -> list[dict[str, str]]:
    index_path = dataset_dir / "index.txt"
    if not index_path.exists():
        errors.append(f"missing index.txt: {index_path}")
        return []
    with index_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != INDEX_HEADER:
            errors.append("index.txt header must be subjectID,network,topology")
            return []
        return list(reader)


def validate_topology(path: Path, errors: list[str]) -> tuple[list[str], list[int]]:
    if not path.exists():
        errors.append(f"missing topology file: {path}")
        return [], []
    rows = read_csv_rows(path)
    if not rows:
        errors.append(f"empty topology file: {path}")
        return [], []

    header = rows[0]
    if not header or header[0] != "label":
        errors.append("topology first header cell must be label")
    if len(header) < 4 or not header[1] or header[2] != "" or header[3] != "":
        errors.append("topology coordinate header must follow DeepCytoCave pattern, e.g. MRI,,")

    cluster_indices = [
        idx
        for idx, name in enumerate(header)
        if name in {"PLACE", "PACE", "Q", "Q-Modularity"} or "Clustering" in name
    ]
    if not cluster_indices:
        errors.append("topology must include UPENNModuleClustering or another Clustering column")
        cluster_idx = None
    else:
        cluster_idx = cluster_indices[0]
        cluster_name = header[cluster_idx]
        if cluster_name != "UPENNModuleClustering" and "Clustering" not in cluster_name:
            errors.append("cluster column must be UPENNModuleClustering or contain Clustering")

    labels: list[str] = []
    clusters: list[int] = []
    for row_num, row in enumerate(rows[1:], start=2):
        if not row:
            continue
        labels.append(row[0])
        if cluster_idx is None or cluster_idx >= len(row):
            continue
        try:
            clusters.append(int(row[cluster_idx]))
        except ValueError:
            errors.append(f"topology row {row_num} cluster ID is not an integer")

    if clusters:
        unique = sorted(set(clusters))
        expected = list(range(1, max(unique) + 1))
        if unique != expected:
            errors.append("cluster IDs must be positive 1-based integers contiguous in 1..K")
    return labels, clusters


def validate_edge_csv(path: Path, node_count: int, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing edge file: {path}")
        return
    rows = read_csv_rows(path)
    for row_num, row in enumerate(rows, start=1):
        if len(row) != 3:
            errors.append(f"edge row {row_num} must have exactly 3 fields")
            continue
        try:
            source = int(row[0])
            target = int(row[1])
        except ValueError:
            errors.append("edge file appears to have a header or non-integer endpoint")
            continue
        try:
            weight = float(row[2])
        except ValueError:
            errors.append(f"edge row {row_num} weight is not numeric")
            continue
        if source < 0 or source >= node_count or target < 0 or target >= node_count:
            errors.append(f"edge row {row_num} endpoint is out of range 0..{node_count - 1}")
        if source == target:
            errors.append(f"edge row {row_num} is a self-edge")
        if not math.isfinite(weight) or weight <= 0:
            errors.append(f"edge row {row_num} weight must be finite and positive")


def validate_edge_json(path: Path, node_count: int, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing edge file: {path}")
        return
    with path.open() as handle:
        payload = json.load(handle)
    if payload.get("mathjs") != "SparseMatrix":
        errors.append("sparse JSON network must be a mathjs SparseMatrix")
    size = payload.get("size")
    if not isinstance(size, list) or len(size) != 2 or size[0] < node_count or size[1] < node_count:
        errors.append("sparse JSON size must be at least [N, N]")


def validate_lut(path: Path, topology_labels: list[str], errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing LUT file: {path}")
        return
    first_line = path.read_text().splitlines()[0] if path.read_text().splitlines() else ""
    if ";" not in first_line:
        errors.append("LUT must be semicolon-delimited")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_LUT_COLUMNS - fieldnames
        if missing:
            errors.append(f"LUT missing required columns: {', '.join(sorted(missing))}")
            return
        lut_rows = list(reader)

    lut_labels = {row["label"] for row in lut_rows}
    missing_labels = sorted(set(topology_labels) - lut_labels)
    if missing_labels:
        errors.append(f"LUT is missing topology labels: {', '.join(missing_labels[:10])}")

    bad_hemi = sorted({row["hemisphere"] for row in lut_rows if row["hemisphere"] not in {"left", "right"}})
    if bad_hemi:
        errors.append(f"LUT hemisphere values must be only left/right: {', '.join(bad_hemi)}")


def validate_export(export_dir: Path, dataset_name: str | None = None, lut_name: str | None = None) -> list[str]:
    errors: list[str] = []
    try:
        dataset_dir = find_dataset_dir(export_dir, dataset_name)
        lut_path = find_lut_path(export_dir, lut_name)
    except Exception as exc:
        return [str(exc)]

    index_rows = validate_index(dataset_dir, errors)
    all_labels: list[str] = []
    for row in index_rows:
        topology_path = dataset_dir / row["topology"]
        labels, _clusters = validate_topology(topology_path, errors)
        all_labels.extend(labels)

        network_path = dataset_dir / row["network"]
        if network_path.suffix == ".json":
            validate_edge_json(network_path, len(labels), errors)
        else:
            validate_edge_csv(network_path, len(labels), errors)

    validate_lut(lut_path, all_labels, errors)
    return errors


def main() -> int:
    args = parse_args()
    errors = validate_export(args.export_dir, args.dataset_name, args.lut_name)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("DeepCytoCave export is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
