#!/usr/bin/env bash
# RQ4 Usefulness evaluation pipeline (matches paper setup).
#
# Metrics per forget-set selector:
#   - VMI-Bench VR: SecurityEval prefixes + Bandit on greedy completions
#   - Unseen 2026 CVE VR: post-cutoff CVE prompts + Bandit
#   - HumanEval pass@1: greedy decoding
#
# Usage:
#   source scripts/env.sh   # set MODEL paths and benchmark paths
#   bash scripts/run_rq4_usefulness_eval.sh gpt_j_6b
#   bash scripts/run_rq4_usefulness_eval.sh starcoderbase_7b
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_KEY="${1:?gpt_j_6b or starcoderbase_7b}"
SELECTOR="${2:-all}"  # all | original | vulinception | random | gotcha | groundtruth

: "${SECURITYEVAL_JSONL:?Set SECURITYEVAL_JSONL (SecurityEval dataset.jsonl)}"
: "${UNSEEN2026_JSONL:?Set UNSEEN2026_JSONL}"
: "${HUMANEVAL_DIR:?Set HUMANEVAL_DIR}"
: "${EVAL_OUTPUT_DIR:=${ROOT}/eval_outputs/rq4_usefulness}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${VLLM_USE_V1:=1}"
export VLLM_USE_V1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${MODEL_KEY}" == "gpt_j_6b" ]]; then
  DATASET_TAG="pile"
  BASE_MODEL="${GPTJ_MODEL:-EleutherAI/gpt-j-6b}"
  VULINCEPTION_MODEL="${GPTJ_VULINCEPTION_MODEL:-${ROOT}/models/code/gpt-j-6b-pile-code_binary/final}"
  RANDOM_MODEL="${GPTJ_RANDOM_MODEL:-${ROOT}/models/code/gpt-j-6b-pile_random-code_binary/final}"
  GOTCHA_MODEL="${GPTJ_GOTCHA_MODEL:-${ROOT}/models/code/gpt-j-6b-pile_gotcha-code_binary/final}"
  GT_MODEL="${GPTJ_GT_MODEL:-${ROOT}/models/code/gpt-j-6b-pile_groundtruth-code_binary/final}"
  DTYPE="half"
  MAX_LEN=2048
else
  DATASET_TAG="stack"
  BASE_MODEL="${STARCODER_MODEL:-bigcode/starcoderbase-7b}"
  VULINCEPTION_MODEL="${SC_VULINCEPTION_MODEL:-${ROOT}/models/code/starcoderbase-7b-stack-code_binary/final}"
  RANDOM_MODEL="${SC_RANDOM_MODEL:-${ROOT}/models/code/starcoderbase-7b-stack_random-code_binary/final}"
  GOTCHA_MODEL="${SC_GOTCHA_MODEL:-${ROOT}/models/code/starcoderbase-7b-stack_gotcha-code_binary/final}"
  GT_MODEL="${SC_GT_MODEL:-${ROOT}/models/code/starcoderbase-7b-stack_groundtruth-code_binary/final}"
  DTYPE="bfloat16"
  MAX_LEN=4096
fi

cd "${SCRIPT_DIR}"
OUT="${EVAL_OUTPUT_DIR}/${MODEL_KEY}"
mkdir -p "${OUT}/vmi_bench" "${OUT}/unseen2026" "${OUT}/humaneval"

run_sec() {
  local model=$1 tag=$2 dataset=$3
  local o="${OUT}/${dataset}/${tag}"
  [[ -f "${o}/summary.json" ]] && echo "skip ${dataset}/${tag}" && return
  echo "eval ${dataset}/${tag}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python eval_securityeval_vllm_bandit.py \
    --dataset_jsonl "${4}" --model_path "${model}" --output_dir "${o}" \
    --dtype "${DTYPE}" --max_model_len "${MAX_LEN}" --trust_remote_code
}

run_he() {
  local model=$1 tag=$2
  local o="${OUT}/humaneval/${tag}"
  [[ -f "${o}/summary.json" ]] && echo "skip humaneval/${tag}" && return
  echo "eval humaneval/${tag}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python eval_humaneval_vllm.py \
    --human_eval_root "${HUMANEVAL_DIR}" --model_path "${model}" --output_dir "${o}" \
    --dtype "${DTYPE}" --max_model_len "${MAX_LEN}" --trust_remote_code
}

eval_selector() {
  local name=$1 model=$2
  [[ ! -d "${model}" ]] && echo "WARN: missing model for ${name}: ${model}" && return
  run_sec "${model}" "${name}" "vmi_bench" "${SECURITYEVAL_JSONL}"
  run_sec "${model}" "${name}" "unseen2026" "${UNSEEN2026_JSONL}"
  run_he "${model}" "${name}"
}

should_run() {
  [[ "${SELECTOR}" == "all" || "${SELECTOR}" == "${1}" ]]
}

should_run original && eval_selector "original" "${BASE_MODEL}"
should_run vulinception && eval_selector "vulinception" "${VULINCEPTION_MODEL}"
should_run random && eval_selector "random" "${RANDOM_MODEL}"
should_run gotcha && eval_selector "gotcha" "${GOTCHA_MODEL}"
should_run groundtruth && eval_selector "groundtruth" "${GT_MODEL}"

python build_usefulness_table.py --model "${MODEL_KEY}" --eval_root "${EVAL_OUTPUT_DIR}"
echo "Results: ${EVAL_OUTPUT_DIR}/${MODEL_KEY}/usefulness_table.csv"
