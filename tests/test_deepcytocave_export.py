from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "upenn_gbm_minicase"
EXPORTER = REPO_ROOT / "exporters" / "deepcytocave" / "export_deepcytocave_case.py"
VALIDATOR = REPO_ROOT / "exporters" / "deepcytocave" / "validate_deepcytocave_export.py"


def run_export(tmp_path: Path, top_edges: int = 1) -> Path:
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURE, run_dir)
    out_dir = tmp_path / "deepcytocave_exports" / "UPENN_GBM"
    subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--run-dir",
            str(run_dir),
            "--cluster-file",
            "modules_kmeans_c64_s0.csv",
            "--edge-file",
            "edges_k20.csv",
            "--edge-mode",
            "top_per_node",
            "--top-edges",
            str(top_edges),
            "--dataset-name",
            "UPENN_GBM",
            "--subject-id",
            "UPENN-GBM-MINI_11",
            "--lut-name",
            "upenn_gbm",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return out_dir


def read_rows(path: Path, delimiter: str = ",") -> list[list[str]]:
    with path.open(newline="") as handle:
        return list(csv.reader(handle, delimiter=delimiter))


def test_exporter_creates_deepcytocave_files(tmp_path: Path) -> None:
    out_dir = run_export(tmp_path)
    dataset_dir = out_dir / "data" / "UPENN_GBM"

    assert (dataset_dir / "index.txt").exists()
    assert (dataset_dir / "UPENN-GBM-MINI_11_topology.csv").exists()
    assert (dataset_dir / "UPENN-GBM-MINI_11_edges.csv").exists()
    assert (dataset_dir / "UPENN-GBM-MINI_11_metadata.json").exists()
    assert (dataset_dir / "UPENN-GBM-MINI_11_cluster_summary.csv").exists()
    assert (out_dir / "data" / "LookupTable_upenn_gbm.csv").exists()

    subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--export-dir",
            str(out_dir),
            "--dataset-name",
            "UPENN_GBM",
            "--lut-name",
            "upenn_gbm",
        ],
        check=True,
        cwd=REPO_ROOT,
    )


def test_cluster_ids_are_remapped_to_one_based(tmp_path: Path) -> None:
    out_dir = run_export(tmp_path)
    topology = read_rows(out_dir / "data" / "UPENN_GBM" / "UPENN-GBM-MINI_11_topology.csv")

    assert topology[0] == ["label", "MRI", "", "", "UPENNModuleClustering"]
    assert [row[4] for row in topology[1:]] == ["1", "2", "1", "2"]


def test_top_per_node_filtering_and_symmetric_edges(tmp_path: Path) -> None:
    out_dir = run_export(tmp_path, top_edges=1)
    edge_rows = read_rows(out_dir / "data" / "UPENN_GBM" / "UPENN-GBM-MINI_11_edges.csv")
    edges = {(int(source), int(target)): float(weight) for source, target, weight in edge_rows}

    assert len(edge_rows) == 6
    assert edges[(0, 1)] == 0.9
    assert edges[(1, 0)] == 0.9
    assert edges[(1, 2)] == 0.8
    assert edges[(2, 1)] == 0.8
    assert edges[(2, 3)] == 0.6
    assert edges[(3, 2)] == 0.6


def test_edge_output_has_no_header(tmp_path: Path) -> None:
    out_dir = run_export(tmp_path)
    first_row = read_rows(out_dir / "data" / "UPENN_GBM" / "UPENN-GBM-MINI_11_edges.csv")[0]

    assert first_row != ["source", "target", "weight"]
    int(first_row[0])
    int(first_row[1])
    float(first_row[2])


def test_validator_catches_missing_lut_labels(tmp_path: Path) -> None:
    out_dir = run_export(tmp_path)
    lut_path = out_dir / "data" / "LookupTable_upenn_gbm.csv"
    rows = read_rows(lut_path, delimiter=";")
    with lut_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerows(rows[:-1])

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--export-dir",
            str(out_dir),
            "--dataset-name",
            "UPENN_GBM",
            "--lut-name",
            "upenn_gbm",
        ],
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 1
    assert "missing topology labels" in result.stdout


def test_validator_catches_out_of_range_edge_endpoints(tmp_path: Path) -> None:
    out_dir = run_export(tmp_path)
    edge_path = out_dir / "data" / "UPENN_GBM" / "UPENN-GBM-MINI_11_edges.csv"
    with edge_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([0, 99, 1.0])

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--export-dir",
            str(out_dir),
            "--dataset-name",
            "UPENN_GBM",
            "--lut-name",
            "upenn_gbm",
        ],
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 1
    assert "out of range" in result.stdout
