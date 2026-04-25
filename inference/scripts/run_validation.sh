#!/usr/bin/env bash

set -euo pipefail

source ~/.bashrc >/dev/null 2>&1 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${INFERENCE_ROOT}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 <generation_output.jsonl>"
  exit 1
fi
INPUT_FILE="$1"

PYTHON_BIN="${PYTHON_BIN:-python}"
FILENAME_FILE="${FILENAME_FILE:-${REPO_ROOT}/dataset/defects4j_filename_map.json}"

env FILENAME_FILE="${FILENAME_FILE}" \
  "${PYTHON_BIN}" -u "${INFERENCE_ROOT}/src/validate_patch.py" "${INPUT_FILE}"
