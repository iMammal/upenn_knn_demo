# DeepCytoCave Exporter

This adapter exports UPENN-GBM graph runs into the dataset-folder ingest path
documented in `docs/DEEP_CYTOCAVE_INGEST_CONTRACT.md`.

Example:

```bash
python exporters/deepcytocave/export_deepcytocave_case.py \
  --run-dir outputs/UPENN-GBM-00375_11_n64000_s0_k20_cuda \
  --cluster-file modules_kmeans_c64_s0.csv \
  --edge-file edges_k20.csv \
  --edge-mode top_per_node \
  --top-edges 5 \
  --dataset-name UPENN_GBM \
  --subject-id UPENN-GBM-00375_11 \
  --lut-name upenn_gbm \
  --out-dir deepcytocave_exports/UPENN_GBM
```

The export layout is:

```text
deepcytocave_exports/UPENN_GBM/
  data/
    UPENN_GBM/
      index.txt
      <subject>_edges.csv
      <subject>_topology.csv
      <subject>_metadata.json
      <subject>_cluster_summary.csv
    LookupTable_upenn_gbm.csv
```

The topology file uses DeepCytoCave's manually interpreted header pattern:

```csv
label,MRI,,,UPENNModuleClustering
```

Topology labels are written as `1..N`. Edge endpoints are zero-based topology
row indices. K-means module labels are remapped from upstream zero-based labels
to positive contiguous IDs `1..K`.

Supported edge modes:

- `full`: export all finite positive non-self edges.
- `top_per_node`: keep the highest-weight `--top-edges` outgoing edges per
  source node before symmetrizing.
- `weight_percentile`: keep edges with weight at or above
  `--weight-percentile`.
- `intra_cluster_only`: keep only edges whose source and target share the same
  remapped module.

For undirected visualization, the exporter writes symmetric edge pairs. The CSV
edge file is headerless and contains exactly `source,target,weight`.

Validate an export with:

```bash
python exporters/deepcytocave/validate_deepcytocave_export.py \
  --export-dir deepcytocave_exports/UPENN_GBM \
  --dataset-name UPENN_GBM \
  --lut-name upenn_gbm
```

`--write-sparse-json` additionally writes a mathjs `SparseMatrix.toJSON()` shape
and points `index.txt` to the JSON network file. The headerless CSV edge triples
are still written for inspection and fallback.
