#!/usr/bin/env bash
# Example environment for RQ4 usefulness experiments.
# Copy to env.sh, adjust if needed, then: source scripts/env.sh
#
export PURGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BENCHMARKS_DIR="${PURGE_ROOT}/benchmarks"
export MODELS_DIR="${PURGE_ROOT}/models"
export EVAL_OUTPUT_DIR="${PURGE_ROOT}/eval_outputs"

# Base code models (HuggingFace id or local checkpoint directory)
export GPTJ_MODEL="${GPTJ_MODEL:-EleutherAI/gpt-j-6b}"
export STARCODER_MODEL="${STARCODER_MODEL:-bigcode/starcoderbase-7b}"

# Evaluation benchmarks (bundled under benchmarks/)
export SECURITYEVAL_JSONL="${BENCHMARKS_DIR}/securityeval/dataset.jsonl"
export UNSEEN2026_JSONL="${UNSEEN2026_JSONL:-${PURGE_ROOT}/../Unseen2026CVE/dataset.jsonl}"
export HUMANEVAL_DIR="${BENCHMARKS_DIR}/humaneval"

# VMI-Bench YAML (forget-set construction)
export VMI_PILE_YAML="${BENCHMARKS_DIR}/vmi_bench_pile.yaml"
export VMI_STACK_YAML="${BENCHMARKS_DIR}/vmi_bench_stack.yaml"

# Classifier outputs for forget-set selectors
export GOTCHA_PREDICTIONS_PILE="${BENCHMARKS_DIR}/predictions/gotcha_pile.jsonl"
export GOTCHA_PREDICTIONS_STACK="${BENCHMARKS_DIR}/predictions/gotcha_stack.jsonl"
export TOOL_PREDICTIONS_PILE="${BENCHMARKS_DIR}/predictions/tool_pile.jsonl"
export TOOL_PREDICTIONS_STACK="${BENCHMARKS_DIR}/predictions/tool_stack.jsonl"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
