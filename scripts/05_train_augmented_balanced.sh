#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
DATA=${DATA:-data/yolo_aug_night_balanced/dataset.yaml}
MODEL=${MODEL:-yolov8n.pt}
EPOCHS=${EPOCHS:-20}
IMGSZ=${IMGSZ:-640}
SEED=${SEED:-42}
BATCH=${BATCH:-8}
WORKERS=${WORKERS:-0}
PROJECT=${PROJECT:-runs/augmentation_ablation_balanced}
RUN_NAME=${RUN_NAME:-yolov8n_real250_night250_20e}
DEVICE=${DEVICE:-}

ARGS=(
  --data "${DATA}"
  --model "${MODEL}"
  --epochs "${EPOCHS}"
  --imgsz "${IMGSZ}"
  --seed "${SEED}"
  --batch "${BATCH}"
  --workers "${WORKERS}"
  --project "${PROJECT}"
  --name "${RUN_NAME}"
)

if [ -n "${DEVICE}" ]; then
  ARGS+=(--device "${DEVICE}")
fi

${PY} -m src.detection.train_yolo "${ARGS[@]}" "$@"
