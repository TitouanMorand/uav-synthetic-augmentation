#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
MANIFEST=${MANIFEST:-data/synthetic/diffusion_grid/manifest.jsonl}
OUT_DIR=${OUT_DIR:-data/previews/diffusion}
NUM=${NUM:-12}

${PY} -m src.visualization.preview_diffusion_results \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --num "${NUM}" \
  "$@"
