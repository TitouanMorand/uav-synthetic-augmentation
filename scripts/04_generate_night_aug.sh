#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
INPUT=${INPUT:-data/yolo}
OUTPUT=${OUTPUT:-data/yolo_aug_night}
DARKNESS=${DARKNESS:-0.38}
SEED=${SEED:-42}
PREVIEW_COUNT=${PREVIEW_COUNT:-20}

${PY} -m src.augmentation.night_classical \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --darkness "${DARKNESS}" \
  --seed "${SEED}" \
  --preview-count "${PREVIEW_COUNT}" \
  --overwrite \
  "$@"
