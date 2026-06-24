#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
SOURCE_AUG=${SOURCE_AUG:-data/yolo_aug_night}
SOURCE_REAL=${SOURCE_REAL:-data/yolo}
OUTPUT=${OUTPUT:-data/yolo_aug_night_balanced}
PAIRS=${PAIRS:-250}
SEED=${SEED:-42}

${PY} -m src.augmentation.create_balanced_augmented \
  --source-aug "${SOURCE_AUG}" \
  --source-real "${SOURCE_REAL}" \
  --output "${OUTPUT}" \
  --pairs "${PAIRS}" \
  --seed "${SEED}" \
  --overwrite \
  "$@"
