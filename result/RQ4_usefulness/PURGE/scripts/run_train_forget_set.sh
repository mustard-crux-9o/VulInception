#!/usr/bin/env bash
# Train PURGE on a code forget set (Hydra).
#
# Examples (RQ4, Table Usefulness):
#   # VulInception forget set + GPT-J-6B (pile)
#   bash scripts/run_train_forget_set.sh pile gpt_j_6b
#
#   # Random forget set + StarCoderBase-7B (stack)
#   bash scripts/run_train_forget_set.sh stack_random starcoderbase_7b paths=code_forget_sets
#
#   # Gotcha / Groundtruth
#   bash scripts/run_train_forget_set.sh pile_gotcha gpt_j_6b paths=code_forget_sets training=code_full
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET="${1:?dataset, e.g. pile | stack_random | pile_gotcha}"
MODEL="${2:?model, e.g. gpt_j_6b | starcoderbase_7b}"
shift 2 || true

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TRAINING="${TRAINING:-code_full}"
if [[ "${TRAINING}" == "code_full_fast" || "${TRAINING}" == "code_a100_single" ]]; then
  export VLLM_USE_V1=1
fi

cd "${ROOT}/src"

python purge_code.py \
  dataset="${DATASET}" \
  model="${MODEL}" \
  training="${TRAINING}" \
  "$@"
