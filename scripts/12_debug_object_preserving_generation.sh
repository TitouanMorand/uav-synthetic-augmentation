#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PY:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
  elif [[ -x ".venv-mps/bin/python" ]]; then
    PY=".venv-mps/bin/python"
  else
    PY="python3"
  fi
fi

CONFIG=${CONFIG:-configs/diffusion.yaml}
YOLO_DIR=${YOLO_DIR:-data/yolo}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/synthetic/debug_object_preserving}
PREVIEW_DIR=${PREVIEW_DIR:-data/previews/debug_object_preserving}
DEVICE=${DEVICE:-auto}
REQUIRE_DEVICE_ARGS=()
if [[ "${DEVICE}" != "auto" ]]; then
  REQUIRE_DEVICE_ARGS+=(--require-device)
fi

"${PY}" -m src.augmentation.run_diffusion_grid \
  --config "${CONFIG}" \
  --yolo-dir "${YOLO_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --limit 3 \
  --prompt-presets night_lowlight \
  --modes background_inpaint_reinsert_object \
  --strength-values 0.25,0.35 \
  --guidance-values 5.0,7.5 \
  --seeds 42 \
  --device "${DEVICE}" \
  "${REQUIRE_DEVICE_ARGS[@]}" \
  --box-margin-px 32 \
  --box-margin-ratio 2.0 \
  --protect-all-boxes \
  "$@"

"${PY}" -m src.visualization.preview_diffusion_results \
  --manifest "${OUTPUT_ROOT}/manifest.jsonl" \
  --out-dir "${PREVIEW_DIR}" \
  --num 12 \
  --tile-width 220 \
  --mask-margin 32 \
  --relative-margin 2.0 \
  --strict-object-preserving
