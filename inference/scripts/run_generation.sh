#!/usr/bin/env bash

set -euo pipefail

source ~/.bashrc >/dev/null 2>&1 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${INFERENCE_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model directory.}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/dataset/defects4j_benchmark.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/results/generation_outputs/run.jsonl}"
MODE="${MODE:-with_context}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
NUM_BEAMS="${NUM_BEAMS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

case "${MODE}" in
  with_context)    HAVE_SKELETON="True"  ;;
  without_context) HAVE_SKELETON="False" ;;
  *)
    echo "Unsupported MODE: ${MODE} (expected with_context or without_context)"
    exit 1
    ;;
esac

mkdir -p "$(dirname "${OUTPUT_PATH}")"

env TOKENIZERS_PARALLELISM=false \
  "${PYTHON_BIN}" -u "${INFERENCE_ROOT}/src/generate_patches.py" \
  --dataset_path "${DATASET_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --base_model "${MODEL_PATH}" \
  --use_lora False \
  --use_new_tokens False \
  --n "${NUM_SAMPLES}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --have_summary True \
  --have_skeleton "${HAVE_SKELETON}"
