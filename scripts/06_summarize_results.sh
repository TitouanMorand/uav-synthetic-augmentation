#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
RESULTS=${RESULTS:-runs/baseline/yolov8n_smoke/results.csv}

if [ "$#" -gt 0 ]; then
  ${PY} -m src.detection.summarize_results "$@"
else
  ${PY} -m src.detection.summarize_results "${RESULTS}"
fi
