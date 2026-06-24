#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PY:-}" ]]; then
  if [[ -x ".venv-mps/bin/python" ]]; then
    PY=".venv-mps/bin/python"
  elif [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
  else
    PY="python3"
  fi
fi
MANIFEST=${MANIFEST:-data/synthetic/diffusion_grid/manifest.jsonl}
REPORT=${REPORT:-reports/generated_image_diagnostics.csv}

${PY} -m src.debug.inspect_generated_images \
  --manifest "${MANIFEST}" \
  --report "${REPORT}" \
  "$@"
