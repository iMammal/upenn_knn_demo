#!/usr/bin/env python3
"""Export one UPENN-GBM graph run into DeepCytoCave ingest files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CLUSTER_HEADER = "UPENNModuleClustering"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cluster-file", required=True)
    parser.add_argument("--edge-file", required=True)
    parser.add_argument(
        "--edge-mode",
        choices=["full", "top_per_node", "weight_percentile", "intra_cluster_only"],
        default="top_per_node",
    )
    parser.add_argument("--top-edges", type=int, default=5)
    parser.add_argument("--weight-percentile", type=float, default=95.0)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--lut-name", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--write-sparse-json",
        action="store_true",
        help="Also write mathjs SparseMatrix JSON and point index.txt to it.",
    )
    return parser.parse_args()


def find_case_dir(run_dir: Path, subject_id: str) -> Path:
    if (run_dir / "nodes.csv").exists():
        return run_dir
    exact = run_dir / subject_id
    if exact.is_dir():
        return exact
    case_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir())
    if len(case_dirs) == 1:
        return case_dirs[0]
    raise FileNotFoundError(f"Could not identify case directory under {run_dir}")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def pick_coordinate_columns(nodes: pd.DataFrame) -> tuple[str, str, str]:
    for cols in (("x", "y", "z"), ("i", "j", "k")):
        if all(col in nodes.columns for col in cols):
            return cols
    raise ValueError("nodes.csv must contain either x,y,z or i,j,k coordinates")


def pick_seg_column(nodes: pd.DataFrame) -> str | None:
    for col in ("seg_label", "seg_value", "seg"):
        if col in nodes.columns:
            return col
    return None


def pick_edge_columns(edges: pd.DataFrame) -> tuple[str, str, str]:
    source_candidates = ("source", "src", "source_node", "source_node_index")
    target_candidates = ("target", "dst", "target_node", "target_node_index")
    weight_candidates = ("weight", "strength", "distance")

    source = next((col for col in source_candidates if col in edges.columns), None)
    target = next((col for col in target_candidates if col in edges.columns), None)
    weight = next((col for col in weight_candidates if col in edges.columns), None)
    if not source or not target or not weight:
        raise ValueError(
            "edge file must contain source/target columns and one of weight, strength, or distance"
        )
    return source, target, weight


def pick_cluster_column(clusters: pd.DataFrame) -> str:
    candidates = [
        col
        for col in clusters.columns
        if col != "node_id" and ("module" in col.lower() or "cluster" in col.lower())
    ]
    if not candidates:
        candidates = [col for col in clusters.columns if col != "node_id"]
    if not candidates:
        raise ValueError("cluster file must contain a cluster/module column")
    return candidates[0]


def remap_clusters(nodes: pd.DataFrame, clusters: pd.DataFrame) -> list[int]:
    cluster_col = pick_cluster_column(clusters)
    if "node_id" in nodes.columns and "node_id" in clusters.columns:
        merged = nodes[["node_id"]].merge(clusters[["node_id", cluster_col]], on="node_id", how="left")
        raw = merged[cluster_col]
    else:
        if len(nodes) != len(clusters):
            raise ValueError("cluster file row count must match nodes.csv when node_id is absent")
        raw = clusters[cluster_col]

    if raw.isna().any():
        missing = int(raw.isna().sum())
        raise ValueError(f"cluster file is missing labels for {missing} nodes")

    unique = sorted(int(value) for value in pd.unique(raw))
    remap = {value: idx + 1 for idx, value in enumerate(unique)}
    return [remap[int(value)] for value in raw]


def filtered_edges(
    edge_path: Path,
    node_count: int,
    clusters: list[int],
    mode: str,
    top_edges: int,
    weight_percentile: float,
) -> list[tuple[int, int, float]]:
    edge_df = pd.read_csv(edge_path)
    source_col, target_col, weight_col = pick_edge_columns(edge_df)
    edges: list[tuple[int, int, float]] = []

    for row in edge_df.itertuples(index=False):
        source = int(getattr(row, source_col))
        target = int(getattr(row, target_col))
        weight = float(getattr(row, weight_col))
        if source == target:
            continue
        if source < 0 or source >= node_count or target < 0 or target >= node_count:
            raise ValueError(f"edge endpoint out of range: {source},{target}")
        if not math.isfinite(weight) or weight <= 0:
            continue
        edges.append((source, target, weight))

    if mode == "top_per_node":
        by_source: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
        for edge in edges:
            by_source[edge[0]].append(edge)
        filtered = []
        for source_edges in by_source.values():
            filtered.extend(sorted(source_edges, key=lambda edge: edge[2], reverse=True)[:top_edges])
        edges = filtered
    elif mode == "weight_percentile":
        if edges:
            threshold = float(np.percentile([edge[2] for edge in edges], weight_percentile))
            edges = [edge for edge in edges if edge[2] >= threshold]
    elif mode == "intra_cluster_only":
        edges = [edge for edge in edges if clusters[edge[0]] == clusters[edge[1]]]

    symmetric: dict[tuple[int, int], float] = {}
    for source, target, weight in edges:
        symmetric[(source, target)] = max(weight, symmetric.get((source, target), -math.inf))
        symmetric[(target, source)] = max(weight, symmetric.get((target, source), -math.inf))
    return [(source, target, weight) for (source, target), weight in sorted(symmetric.items())]


def write_topology(path: Path, labels: Iterable[int], coords: np.ndarray, clusters: list[int]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "MRI", "", "", CLUSTER_HEADER])
        for label, xyz, cluster in zip(labels, coords, clusters):
            writer.writerow([label, *[format_float(value) for value in xyz], cluster])


def format_float(value: float) -> str:
    return f"{float(value):.8g}"


def infer_seed(*values: str) -> int | None:
    for value in values:
        match = re.search(r"(?:^|_)s(\d+)(?:_|\.|$)", value)
        if match:
            return int(match.group(1))
    return None


def infer_knn_k(metrics: dict, edge_file: str) -> int | None:
    for key in ("knn_k", "k_requested", "k_effective"):
        if key in metrics:
            return int(metrics[key])
    match = re.search(r"k(\d+)", edge_file)
    return int(match.group(1)) if match else None


def compute_node_edge_stats(
    node_count: int, edges: list[tuple[int, int, float]]
) -> tuple[list[int], list[float]]:
    degree = [0] * node_count
    strength = [0.0] * node_count
    for source, _target, weight in edges:
        degree[source] += 1
        strength[source] += weight
    return degree, strength


def format_metadata_value(value: object) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return int(value)
    return value


def write_lut(
    path: Path,
    case_id: str,
    labels: list[int],
    nodes: pd.DataFrame,
    coords: np.ndarray,
    clusters: list[int],
    metrics: dict,
    edge_stats: tuple[list[int], list[float]],
    graph_seed: int | None,
    knn_k: int | None,
) -> list[str]:
    midpoint = float(np.median(coords[:, 0]))
    hemispheres = ["left" if xyz[0] < midpoint else "right" for xyz in coords]
    seg_col = pick_seg_column(nodes)
    degree, strength = edge_stats

    fieldnames = [
        "label",
        "Anatomy",
        "region_name",
        "hemisphere",
        "case_id",
        "node_id",
        "seg_label",
        "module",
        "x",
        "y",
        "z",
        "graph_seed",
        "knn_k",
        "node_count",
        "degree",
        "strength",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for idx, (label, xyz, cluster, hemi) in enumerate(zip(labels, coords, clusters, hemispheres)):
            node_id = nodes["node_id"].iloc[idx] if "node_id" in nodes.columns else idx
            seg_value = nodes[seg_col].iloc[idx] if seg_col else ""
            writer.writerow(
                {
                    "label": label,
                    "Anatomy": f"{hemi}_module_{cluster}",
                    "region_name": f"{case_id} node {node_id}",
                    "hemisphere": hemi,
                    "case_id": case_id,
                    "node_id": format_metadata_value(node_id),
                    "seg_label": format_metadata_value(seg_value),
                    "module": cluster,
                    "x": format_float(xyz[0]),
                    "y": format_float(xyz[1]),
                    "z": format_float(xyz[2]),
                    "graph_seed": graph_seed if graph_seed is not None else "",
                    "knn_k": knn_k if knn_k is not None else "",
                    "node_count": len(labels),
                    "degree": degree[idx],
                    "strength": format_float(strength[idx]),
                }
            )
    return hemispheres


def write_edges(path: Path, edges: list[tuple[int, int, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for source, target, weight in edges:
            writer.writerow([source, target, format_float(weight)])


def write_sparse_json(path: Path, node_count: int, edges: list[tuple[int, int, float]]) -> None:
    by_col: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for source, target, weight in edges:
        by_col[target].append((source, weight))

    values: list[float] = []
    index: list[int] = []
    ptr = [0]
    for col in range(node_count):
        for source, weight in sorted(by_col.get(col, [])):
            index.append(source)
            values.append(weight)
        ptr.append(len(values))

    payload = {
        "mathjs": "SparseMatrix",
        "values": values,
        "index": index,
        "ptr": ptr,
        "size": [node_count, node_count],
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def write_cluster_summary(
    path: Path,
    coords: np.ndarray,
    clusters: list[int],
    hemispheres: list[str],
    nodes: pd.DataFrame,
) -> None:
    seg_col = pick_seg_column(nodes)
    fieldnames = [
        "cluster_id",
        "n_nodes",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "dominant_seg_label",
        "left_fraction",
        "right_fraction",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cluster_id in sorted(set(clusters)):
            idxs = [idx for idx, value in enumerate(clusters) if value == cluster_id]
            cluster_coords = coords[idxs]
            hemi_counts = Counter(hemispheres[idx] for idx in idxs)
            dominant_seg = ""
            if seg_col:
                dominant_seg = Counter(nodes.iloc[idxs][seg_col]).most_common(1)[0][0]
            writer.writerow(
                {
                    "cluster_id": cluster_id,
                    "n_nodes": len(idxs),
                    "centroid_x": format_float(cluster_coords[:, 0].mean()),
                    "centroid_y": format_float(cluster_coords[:, 1].mean()),
                    "centroid_z": format_float(cluster_coords[:, 2].mean()),
                    "dominant_seg_label": dominant_seg,
                    "left_fraction": format_float(hemi_counts["left"] / len(idxs)),
                    "right_fraction": format_float(hemi_counts["right"] / len(idxs)),
                }
            )


def write_metadata(
    path: Path,
    case_id: str,
    subject_id: str,
    run_dir: Path,
    node_count: int,
    metrics: dict,
    edge_file: str,
    cluster_file: str,
    clusters: list[int],
    edge_mode: str,
    top_edges: int,
) -> None:
    metadata = {
        "case_id": case_id,
        "subject_id": subject_id,
        "source_run": str(run_dir),
        "node_count": node_count,
        "knn_k": infer_knn_k(metrics, edge_file),
        "graph_seed": infer_seed(run_dir.name, cluster_file),
        "edge_file": edge_file,
        "cluster_file": cluster_file,
        "cluster_method": "kmeans" if "kmeans" in cluster_file.lower() else Path(cluster_file).stem,
        "cluster_count": len(set(clusters)),
        "edge_filter_mode": edge_mode,
        "top_edges": top_edges,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def export_case(args: argparse.Namespace) -> dict[str, Path]:
    case_dir = find_case_dir(args.run_dir, args.subject_id)
    nodes_path = case_dir / "nodes.csv"
    edge_path = case_dir / args.edge_file
    cluster_path = case_dir / args.cluster_file
    metrics_path = case_dir / "metrics.json"

    nodes = pd.read_csv(nodes_path)
    clusters_df = pd.read_csv(cluster_path)
    metrics = read_json(metrics_path)

    coord_cols = pick_coordinate_columns(nodes)
    coords = nodes.loc[:, coord_cols].to_numpy(dtype=float)
    labels = list(range(1, len(nodes) + 1))
    clusters = remap_clusters(nodes, clusters_df)
    edges = filtered_edges(
        edge_path=edge_path,
        node_count=len(nodes),
        clusters=clusters,
        mode=args.edge_mode,
        top_edges=args.top_edges,
        weight_percentile=args.weight_percentile,
    )

    dataset_dir = args.out_dir / "data" / args.dataset_name
    data_dir = args.out_dir / "data"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.subject_id
    topology_path = dataset_dir / f"{prefix}_topology.csv"
    edge_csv_path = dataset_dir / f"{prefix}_edges.csv"
    edge_json_path = dataset_dir / f"{prefix}_edges.json"
    metadata_path = dataset_dir / f"{prefix}_metadata.json"
    cluster_summary_path = dataset_dir / f"{prefix}_cluster_summary.csv"
    lut_path = data_dir / f"LookupTable_{args.lut_name}.csv"
    index_path = dataset_dir / "index.txt"

    write_topology(topology_path, labels, coords, clusters)
    write_edges(edge_csv_path, edges)
    if args.write_sparse_json:
        write_sparse_json(edge_json_path, len(nodes), edges)
    network_name = edge_json_path.name if args.write_sparse_json else edge_csv_path.name

    edge_stats = compute_node_edge_stats(len(nodes), edges)
    hemispheres = write_lut(
        lut_path,
        args.subject_id,
        labels,
        nodes,
        coords,
        clusters,
        metrics,
        edge_stats,
        infer_seed(args.run_dir.name, args.cluster_file),
        infer_knn_k(metrics, args.edge_file),
    )
    write_cluster_summary(cluster_summary_path, coords, clusters, hemispheres, nodes)
    write_metadata(
        metadata_path,
        case_id=metrics.get("case_id", args.subject_id),
        subject_id=args.subject_id,
        run_dir=args.run_dir,
        node_count=len(nodes),
        metrics=metrics,
        edge_file=args.edge_file,
        cluster_file=args.cluster_file,
        clusters=clusters,
        edge_mode=args.edge_mode,
        top_edges=args.top_edges,
    )

    with index_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subjectID", "network", "topology"])
        writer.writerow([args.subject_id, network_name, topology_path.name])

    return {
        "index": index_path,
        "topology": topology_path,
        "edges": edge_json_path if args.write_sparse_json else edge_csv_path,
        "edge_csv": edge_csv_path,
        "lut": lut_path,
        "metadata": metadata_path,
        "cluster_summary": cluster_summary_path,
    }


def main() -> int:
    paths = export_case(parse_args())
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
