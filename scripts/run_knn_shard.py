import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--case-list", required=True)
parser.add_argument("--shard-id", type=int, required=True)
parser.add_argument("--num-shards", type=int, required=True)
parser.add_argument("--k-values", default="80,100,120")
parser.add_argument("--max-nodes", type=int, default=32000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

cases = [x.strip() for x in open(args.case_list) if x.strip()]
cases = cases[args.shard_id::args.num_shards]
k_values = [int(x) for x in args.k_values.split(",")]

print(f"[INFO] shard {args.shard_id}/{args.num_shards}: {len(cases)} cases", flush=True)
print(f"[INFO] k_values={k_values}", flush=True)

for k in k_values:
    for case_id in cases:
        out_dir = f"outputs/{case_id}_n{args.max_nodes}_s{args.seed}_k{k}_cuda"
        metrics = Path(out_dir) / case_id / "metrics.json"
        if metrics.exists():
            print(f"[SKIP] {case_id} k={k}", flush=True)
            continue

        print(f"[RUN] {case_id} k={k}", flush=True)
        cmd = [
            "python", "scripts/process_case_gpu_knn.py",
            "--case-id", case_id,
            "--case-dir", "../UPENN-GBM/NIfTI-files",
            "--output-dir", out_dir,
            "--max-nodes", str(args.max_nodes),
            "--k", str(k),
            "--seed",str(args.seed),
            "--device", "cuda",
        ]
        subprocess.run(cmd, check=False)
