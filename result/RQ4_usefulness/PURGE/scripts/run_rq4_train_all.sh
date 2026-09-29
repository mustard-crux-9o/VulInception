#!/usr/bin/env bash
# Train all RQ4 forget-set selectors for GPT-J-6B and/or StarCoderBase-7B.
#
# Usage:
#   bash scripts/run_rq4_train_all.sh gpt_j_6b
#   bash scripts/run_rq4_train_all.sh starcoderbase_7b
#   bash scripts/run_rq4_train_all.sh all
#
# Default: training=code_full (single GPU, no vLLM). Override:
#   TRAINING=code_a100_single bash scripts/run_rq4_train_all.sh starcoderbase_7b
#   TRAINING=code_full_fast CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_rq4_train_all.sh all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_KEY="${1:-all}"
TRAINING="${TRAINING:-code_full}"

if [[ "${TRAINING}" == "code_full_fast" || "${TRAINING}" == "code_a100_single" ]]; then
  export VLLM_USE_V1=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi

train_one() {
  local model=$1 dataset=$2 paths=$3
  local final="${ROOT}/models/code/${model}-${dataset}-code_binary/final"
  if [[ -d "${final}" ]]; then
    echo "SKIP ${model} ${dataset} (checkpoint exists)"
    return 0
  fi
  echo "TRAIN ${model} dataset=${dataset} paths=${paths} training=${TRAINING}"
  bash "${SCRIPT_DIR}/run_train_forget_set.sh" "${dataset}" "${model}" \
    "paths=${paths}" "training=${TRAINING}"
}

train_gptj() {
  train_one gpt_j_6b pile code_default
  train_one gpt_j_6b pile_random code_forget_sets
  train_one gpt_j_6b pile_gotcha code_forget_sets
  train_one gpt_j_6b pile_groundtruth code_forget_sets
}

train_starcoder() {
  train_one starcoderbase_7b stack code_default
  train_one starcoderbase_7b stack_random code_forget_sets
  train_one starcoderbase_7b stack_gotcha code_forget_sets
  train_one starcoderbase_7b stack_groundtruth code_forget_sets
}

case "${MODEL_KEY}" in
  gpt_j_6b) train_gptj ;;
  starcoderbase_7b) train_starcoder ;;
  all)
    train_gptj
    train_starcoder
    ;;
  *)
    echo "Unknown model: ${MODEL_KEY} (use gpt_j_6b | starcoderbase_7b | all)" >&2
    exit 1
    ;;
esac

echo "Training queue finished. Checkpoints under ${ROOT}/models/code/"
