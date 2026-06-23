#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}

# Prepare a YOLO-format subset (default 500 train, 100 val)
${PY} -m src.dataset.download_hf_subset "$@"
