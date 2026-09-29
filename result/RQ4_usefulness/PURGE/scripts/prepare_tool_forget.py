#!/usr/bin/env python3
"""Build a forget set from vulnerability-MIA **tool** predictions (optional variant).

The RQ4 paper **VulInception** selector uses ``prepare_code_data.py`` (fine-grained
criteria lines on ``pre.label == 1`` entries). This script is an **optional** alternative:
it selects CVEs where the external tool JSONL has ``predicted-label == 1``, same layout
as Gotcha/Random forget sets.

Example:
  python prepare_tool_forget.py \\
    --predictions_file ../benchmarks/predictions/tool_pile.jsonl \\
    --yaml_file ../benchmarks/vmi_bench_pile.yaml \\
    --output_dir ../data/forget_sets/pile_tool
"""
from prepare_gotcha_forget import build_dataset, main

__all__ = ["build_dataset", "main"]

if __name__ == "__main__":
    main()
