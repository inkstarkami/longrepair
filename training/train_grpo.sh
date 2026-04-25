#!/usr/bin/env bash
#
# GRPO entry point for the bug-repair recipe (uses the per-hunk coverage
# reward by default). Run from the verl repository root, e.g.:
#
#     bash recipe/bug_repair/train_grpo.sh
#
# All paths and runtime knobs are controlled through environment variables so
# the script ships with no hard-coded user, dataset, or checkpoint locations.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SFT_MERGED_MODEL:?Set SFT_MERGED_MODEL to the merged SFT checkpoint directory.}"

GRPO_TRAIN_PARQUET="${GRPO_TRAIN_PARQUET:-${SCRIPT_DIR}/data/grpo_train.parquet}"
GRPO_VAL_PARQUET="${GRPO_VAL_PARQUET:-${SCRIPT_DIR}/data/grpo_val.parquet}"

NUM_GPUS="${NUM_GPUS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-12288}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-46}"
TEST_FREQ="${TEST_FREQ:-5}"
PROJECT_NAME="${PROJECT_NAME:-bug-repair-grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-bug-repair-grpo-coverage}"
LOGGER="${LOGGER:-[\"console\"]}"
GRPO_CHECKPOINT_DIR="${GRPO_CHECKPOINT_DIR:-./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
RECIPE_DIR="${RECIPE_DIR:-recipe/bug_repair}"
REWARD_FUNCTION_NAME="${REWARD_FUNCTION_NAME:-compute_patch_reward_with_coverage}"

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${GRPO_TRAIN_PARQUET}" \
    data.val_files="${GRPO_VAL_PARQUET}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path="${SFT_MERGED_MODEL}" \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${GRPO_CHECKPOINT_DIR}" \
    custom_reward_function.path="${RECIPE_DIR}/reward_function.py" \
    custom_reward_function.name="${REWARD_FUNCTION_NAME}"

if [[ -n "${GRPO_MERGE_STEP:-}" ]]; then
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${GRPO_CHECKPOINT_DIR}/global_step_${GRPO_MERGE_STEP}/actor" \
    --target_dir "${GRPO_CHECKPOINT_DIR}/merged_model"
fi
