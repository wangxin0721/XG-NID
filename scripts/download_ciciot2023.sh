#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/data/processed/CICIoT2023_processed"
GDRIVE_URL="https://drive.google.com/drive/folders/1FiZh87vvCZF3gX1Fnj9iTB4j74u-nuR6?usp=drive_link"

mkdir -p "$TARGET_DIR"

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("gdown") else 1)
PY
then
  echo "[info] gdown not found, installing into current environment..."
  python -m pip install gdown
fi

echo "[info] downloading processed CICIoT2023 dataset to: $TARGET_DIR"
python -m gdown --folder "$GDRIVE_URL" -O "$TARGET_DIR"

echo "[done] dataset download finished"

