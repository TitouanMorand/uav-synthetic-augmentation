#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}

${PY} -m src.visualize "$@"
