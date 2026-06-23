#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
DATA=${DATA:-data/yolo/dataset.yaml}
MODEL=${MODEL:-yolov8n.pt}
EPOCHS=${EPOCHS:-1}
IMGSZ=${IMGSZ:-320}
SEED=${SEED:-42}
FRACTION=${FRACTION:-0.1}
BATCH=${BATCH:-4}
DEVICE=${DEVICE:-cpu}
RUN_NAME=${RUN_NAME:-yolov8n_fast_debug}

${PY} -m src.detection.train_yolo \
  --data "${DATA}" \
  --model "${MODEL}" \
  --epochs "${EPOCHS}" \
  --imgsz "${IMGSZ}" \
  --seed "${SEED}" \
  --fraction "${FRACTION}" \
  --batch "${BATCH}" \
  --device "${DEVICE}" \
  --project runs/baseline \
  --name "${RUN_NAME}" \
  --exist-ok \
  "$@"
