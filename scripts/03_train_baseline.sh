#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
DATA=${DATA:-data/yolo/dataset.yaml}
MODEL=${MODEL:-yolov8n.pt}
EPOCHS=${EPOCHS:-5}
IMGSZ=${IMGSZ:-640}
SEED=${SEED:-42}
RUN_NAME=${RUN_NAME:-yolov8n_smoke}

${PY} -m src.detection.train_yolo \
  --data "${DATA}" \
  --model "${MODEL}" \
  --epochs "${EPOCHS}" \
  --imgsz "${IMGSZ}" \
  --seed "${SEED}" \
  --project runs/baseline \
  --name "${RUN_NAME}" \
  "$@"
