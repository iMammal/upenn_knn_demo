import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--clusters", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

run_dir = Path(args.run_dir)
case_dirs = [p for p in run_dir.iterdir() if p.is_dir()]
if not case_dirs:
    raise RuntimeError(f"No case subdir found in {run_dir}")
case_dir = case_dirs[0]

nodes_path = case_dir / "nodes.csv"
nodes = pd.read_csv(nodes_path)

feature_cols = [c for c in nodes.columns if c not in ["node_id", "x", "y", "z", "seg_label", "strength"]]
feature_cols = [c for c in feature_cols if np.issubdtype(nodes[c].dtype, np.number)]

X = nodes[feature_cols].to_numpy(np.float32)
X = np.nan_to_num(X)
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

torch.manual_seed(args.seed)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
Xt = torch.tensor(X, device=device)

n = Xt.shape[0]
c = min(args.clusters, n)

init_idx = torch.randperm(n, device=device)[:c]
centroids = Xt[init_idx].clone()

for _ in range(25):
    d = torch.cdist(Xt, centroids)
    labels = torch.argmin(d, dim=1)
    new_centroids = []
    for j in range(c):
        mask = labels == j
        if mask.any():
            new_centroids.append(Xt[mask].mean(dim=0))
        else:
            new_centroids.append(centroids[j])
    centroids = torch.stack(new_centroids)

labels_cpu = labels.cpu().numpy()

out = pd.DataFrame({
    "node_id": nodes["node_id"],
    "kmeans_module": labels_cpu,
})
out_path = case_dir / f"modules_kmeans_c{c}_s{args.seed}.csv"
out.to_csv(out_path, index=False)

counts = pd.Series(labels_cpu).value_counts()
metrics = {
    "case_dir": str(case_dir),
    "clusters": int(c),
    "seed": int(args.seed),
    "n_nodes": int(n),
    "n_modules_nonempty": int(counts.size),
    "module_size_min": int(counts.min()),
    "module_size_max": int(counts.max()),
    "module_size_entropy": float(-(counts / counts.sum() * np.log((counts / counts.sum()) + 1e-12)).sum()),
    "device": str(device),
}

with open(case_dir / f"module_metrics_kmeans_c{c}_s{args.seed}.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
