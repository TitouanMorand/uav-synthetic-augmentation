#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
CONFIG=${CONFIG:-configs/diffusion.yaml}
YOLO_DIR=${YOLO_DIR:-data/yolo}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/synthetic/diffusion_grid}
DEVICE=${DEVICE:-auto}

${PY} -m src.augmentation.run_diffusion_grid \
  --config "${CONFIG}" \
  --yolo-dir "${YOLO_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --limit 10 \
  --prompt-presets night_lowlight \
  --modes global_img2img,background_inpaint_protected_box,background_inpaint_reinsert_object \
  --strength-values 0.2,0.35 \
  --guidance-values 5.0,7.5 \
  --seeds 42 \
  --device "${DEVICE}" \
  "$@"
