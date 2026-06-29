#!/usr/bin/env bash
# Prepare all forget-set variants described in RQ4 (Table: Usefulness).
#
# Forget-set selectors:
#   1. VulInception (tool)  -> data/CODE/{pile,stack}     via prepare_code_data.py
#   2. Random               -> data/forget_sets/{pile,stack}_random
#   3. Gotcha               -> data/forget_sets/{pile,stack}_gotcha
#   4. Groundtruth (GT)     -> data/forget_sets/{pile,stack}_groundtruth
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${VMI_PILE_YAML:=${ROOT}/benchmarks/vmi_bench_pile.yaml}"
: "${VMI_STACK_YAML:=${ROOT}/benchmarks/vmi_bench_stack.yaml}"
: "${GOTCHA_PREDICTIONS_PILE:=${ROOT}/benchmarks/predictions/gotcha_pile.jsonl}"
: "${GOTCHA_PREDICTIONS_STACK:=${ROOT}/benchmarks/predictions/gotcha_stack.jsonl}"
: "${RANDOM_SEED:=42}"
: "${RANDOM_SIZE_PILE:=856}"
: "${RANDOM_SIZE_STACK:=974}"

cd "${SCRIPT_DIR}"

echo "=== [1/4] VulInception forget set (fine-grained criteria lines, pre.label==1) ==="
echo "    (Optional tool-prediction variant: scripts/prepare_tool_forget.py)"
python ../src/prepare_code_data.py \
  --yaml_file "${VMI_PILE_YAML}" \
  --output_dir "${ROOT}/data/CODE/pile" \
  --min_pattern_length 10 --prompt_ratio 0.5

python ../src/prepare_code_data.py \
  --yaml_file "${VMI_STACK_YAML}" \
  --output_dir "${ROOT}/data/CODE/stack" \
  --min_pattern_length 10 --prompt_ratio 0.5

echo "=== [2/4] Random forget set (same scale, seed=${RANDOM_SEED}) ==="
python prepare_random_forget.py \
  --yaml_file "${VMI_PILE_YAML}" \
  --output_dir "${ROOT}/data/forget_sets/pile_random" \
  --sample_size "${RANDOM_SIZE_PILE}" --seed "${RANDOM_SEED}"

python prepare_random_forget.py \
  --yaml_file "${VMI_STACK_YAML}" \
  --output_dir "${ROOT}/data/forget_sets/stack_random" \
  --sample_size "${RANDOM_SIZE_STACK}" --seed "${RANDOM_SEED}"

echo "=== [3/4] Gotcha forget set ==="
python prepare_gotcha_forget.py \
  --predictions_file "${GOTCHA_PREDICTIONS_PILE}" \
  --yaml_file "${VMI_PILE_YAML}" \
  --output_dir "${ROOT}/data/forget_sets/pile_gotcha"

python prepare_gotcha_forget.py \
  --predictions_file "${GOTCHA_PREDICTIONS_STACK}" \
  --yaml_file "${VMI_STACK_YAML}" \
  --output_dir "${ROOT}/data/forget_sets/stack_gotcha"

echo "=== [4/4] Groundtruth forget set (pre.label == 1) ==="
python prepare_groundtruth_forget.py \
  --yaml_file "${VMI_PILE_YAML}" \
  --output_dir "${ROOT}/data/forget_sets/pile_groundtruth"

python prepare_groundtruth_forget.py \
  --yaml_file "${VMI_STACK_YAML}" \
  --output_dir "${ROOT}/data/forget_sets/stack_groundtruth"

echo "Done. See data/README.md for layout."
