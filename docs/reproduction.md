# Reproduction Guide

## What this repo does

This repository provides the training/evaluation code for the XG-NID classifier.
It does not download or preprocess the dataset by itself yet.

## What is still external

- `CICIoT2023` dataset
- `GNN4ID` preprocessing tool

## Recommended workflow on the server

1. Sync this repository to the server.
2. Run `scripts/bootstrap.sh`.
3. Download the processed `CICIoT2023` bundle:

```bash
bash scripts/download_ciciot2023.sh
```

4. Place or clone `GNN4ID` under `third_party/GNN4ID/`.
5. Use the `GNN4ID` notebooks/scripts to export graph files if you want the raw pipeline.
6. Run:

```bash
python main.py inspect --data /path/to/graphs.pt
python main.py train --data /path/to/graphs.pt --epochs 30 --batch-size 16 --output-dir outputs/xgnid
python main.py eval --data /path/to/graphs.pt --checkpoint outputs/xgnid/best.pt
```

## Acceptance checklist

- `inspect` prints node/edge shapes.
- `train` saves `outputs/xgnid/best.pt`.
- `eval` loads the checkpoint and prints metrics.

## Notes

- Use the server terminal in VS Code.
- Keep the dataset outside Git.
- Keep graph exports in `data/processed/` or `outputs/`, both ignored by Git.
- The UNB dataset page requires a form and may error in-browser; the GNN4ID Google Drive link is the faster path for the processed bundle.
