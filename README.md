# XG-NID Repro

This repo is a from-scratch, runnable reproduction scaffold for **XG-NID**.

## What is already reflected

- Dual-modality graph setup from the paper and the GNN4ID repo.
- Flow nodes and packet nodes.
- `contain` and `link` relations.
- A 2-layer GAT-based graph classifier with global mean pooling.
- Training/eval CLI for batched `HeteroData` graphs.

## Known paper settings to mirror

- Dataset: `CIC-IoT2023`
- Flow cap: `20` packets per flow
- Idle timeout: `120s`
- Graph features:
  - Flow node: `82` features
  - Packet node: `1500` features
  - Contain edge: `4` features
  - Link edge: `1` feature
- Task: `8-class` multi-class classification
- Paper preprocessing:
  - `20%` test split
  - balanced train set with `20,000` samples per class

## Important risk

The paper text mentions `76` flow features and `14` packet features in one place, while the GNN4ID repository README states `82` flow features. Treat this as a verification item before final benchmarking.

## How to run on the server

1. Create an environment with CUDA-enabled PyTorch and PyG.
2. Prepare graph files exported from GNN4ID or your preprocessing pipeline.
3. Verify the data:

```bash
python main.py inspect --data /path/to/graphs.pt
```

4. Train:

```bash
python main.py train --data /path/to/graphs.pt --epochs 30 --batch-size 16 --output-dir outputs/xgnid
```

5. Evaluate:

```bash
python main.py eval --data /path/to/graphs.pt --checkpoint outputs/xgnid/best.pt
```

## Innovation 1 preprocessing

To generate the packet-selected intermediate CSVs for the hierarchical graph pipeline:

```bash
python scripts/build_innov1_packet_selection.py \
  --train-csv data/processed/CICIoT2023_processed/df_class_8_train.csv \
  --test-csv data/processed/CICIoT2023_processed/df_class_8_test.csv \
  --output-dir data/processed/CICIoT2023_processed/processed_innov1
```

## Expected runtime

- Data preprocessing / graph export on full CIC-IoT2023: hours.
- Training on a sampled balanced set: tens of minutes to a few hours depending on GPU.

