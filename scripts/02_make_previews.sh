#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}

# Generate preview images with boxes drawn
${PY} -m src.visualization.draw_boxes "$@"
