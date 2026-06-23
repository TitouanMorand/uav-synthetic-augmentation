#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
BASELINE_DATA=${BASELINE_DATA:-data/yolo/dataset.yaml}
AUGMENTED_DATA=${AUGMENTED_DATA:-data/yolo_aug_night/dataset.yaml}
MODEL=${MODEL:-yolov8n.pt}
EPOCHS=${EPOCHS:-5}
IMGSZ=${IMGSZ:-640}
SEED=${SEED:-42}
BATCH=${BATCH:--1}
WORKERS=${WORKERS:-0}
PROJECT=${PROJECT:-runs/augmentation_ablation}
BASELINE_RUN_NAME=${BASELINE_RUN_NAME:-yolov8n_real_smoke}
AUGMENTED_RUN_NAME=${AUGMENTED_RUN_NAME:-yolov8n_real_plus_night_smoke}
DEVICE=${DEVICE:-}

COMMON_ARGS=(
  --model "${MODEL}"
  --epochs "${EPOCHS}"
  --imgsz "${IMGSZ}"
  --seed "${SEED}"
  --batch "${BATCH}"
  --workers "${WORKERS}"
  --project "${PROJECT}"
)

if [ -n "${DEVICE}" ]; then
  COMMON_ARGS+=(--device "${DEVICE}")
fi

echo "Training baseline on ${BASELINE_DATA}"
${PY} -m src.detection.train_yolo \
  --data "${BASELINE_DATA}" \
  --name "${BASELINE_RUN_NAME}" \
  "${COMMON_ARGS[@]}" \
  "$@"

echo "Training augmented model on ${AUGMENTED_DATA}"
${PY} -m src.detection.train_yolo \
  --data "${AUGMENTED_DATA}" \
  --name "${AUGMENTED_RUN_NAME}" \
  "${COMMON_ARGS[@]}" \
  "$@"
