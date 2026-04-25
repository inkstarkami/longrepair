#!/usr/bin/env bash
#
# Supervised fine-tuning entry point for the bug-repair recipe.
# Run this from the verl repository root, e.g.:
#
#     bash recipe/bug_repair/train_sft.sh
#
# All paths and runtime knobs are controlled through environment variables so
# the script ships with no hard-coded user, dataset, or checkpoint locations.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${BASE_MODEL:?Set BASE_MODEL to the pretrained model directory.}"

SFT_TRAIN_PARQUET="${SFT_TRAIN_PARQUET:-${SCRIPT_DIR}/data/sft_train.parquet}"
SFT_VAL_PARQUET="${SFT_VAL_PARQUET:-${SCRIPT_DIR}/data/sft_val.parquet}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
SAVE_FREQ="${SAVE_FREQ:-80}"
PROMPT_KEY="${PROMPT_KEY:-question}"
RESPONSE_KEY="${RESPONSE_KEY:-answer}"
PROJECT_NAME="${PROJECT_NAME:-bug-repair-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-bug-repair-sft-default}"
LOGGER="${LOGGER:-[\"console\"]}"
SFT_CHECKPOINT_DIR="${SFT_CHECKPOINT_DIR:-./checkpoints/${EXPERIMENT_NAME}}"
NEW_TOKENS="${NEW_TOKENS:-[\"<think>\", \"</think>\", \"<answer>\", \"</answer>\"]}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.train_files="${SFT_TRAIN_PARQUET}" \
    data.val_files="${SFT_VAL_PARQUET}" \
    data.prompt_key="${PROMPT_KEY}" \
    data.response_key="${RESPONSE_KEY}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.max_length="${MAX_LENGTH}" \
    model.partial_pretrain="${BASE_MODEL}" \
    model.new_tokens="${NEW_TOKENS}" \
    model.fsdp_config.model_dtype=bf16 \
    optim.lr="${LEARNING_RATE}" \
    trainer.default_local_dir="${SFT_CHECKPOINT_DIR}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.logger="${LOGGER}"

if [[ -n "${SFT_MERGE_STEP:-}" ]]; then
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${SFT_CHECKPOINT_DIR}/global_step_${SFT_MERGE_STEP}" \
    --target_dir "${SFT_CHECKPOINT_DIR}/merged_model"
fi
