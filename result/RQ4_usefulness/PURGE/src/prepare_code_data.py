"""
Data preprocessing script: converts vulnerability YAML datasets into
JSON format compatible with GRPOTrainer and the code reward functions.

Usage:
    python prepare_code_data.py \
        --yaml_file ../../dataset/pile_ft_result.yaml \
        --output_dir ../data/CODE/pile \
        --min_pattern_length 10

    python prepare_code_data.py \
        --yaml_file ../../dataset/stack_ft_result.yaml \
        --output_dir ../data/CODE/stack \
        --min_pattern_length 10
"""
import argparse
import json
import os
import yaml
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


def load_yaml(yaml_file: str) -> dict:
    print(f"Loading YAML from: {yaml_file}")
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    print(f"  Total entries: {len(data)}")
    return data


def extract_vulnerability_lines(
    code: str,
    criteria_lines: List[int],
    relative_lines: List[int],
    prompt_ratio: float = 0.5,
) -> Tuple[str, List[str], str]:
    """
    Given the full function code, extract:
      - prompt: prefix of the function (first prompt_ratio of lines) — gives the
        model enough context to trigger memorised code recall.
      - forget_lines: the vulnerability-relevant lines (criteria ∪ relative) that
        fall AFTER the prompt, stripped.
      - ground_truth_suffix: lines after the same cutoff as ``prompt`` (the true
        completion the model would append), joined with newlines.

    Line numbers in criteria_lines / relative_lines are 1-indexed.
    prompt_ratio: fraction of total lines to use as prompt (0.0–1.0).
    """
    code_lines = code.split("\n")
    n_lines = len(code_lines)

    vuln_indices = set(criteria_lines) | set(relative_lines)
    vuln_indices = {i for i in vuln_indices if 1 <= i <= n_lines}

    # Prompt = first prompt_ratio of lines, at least 1 line, and must leave
    # at least 1 vulnerability line in the completion zone.
    prompt_end = max(1, int(n_lines * prompt_ratio))

    # Make sure at least one vulnerability line is after the prompt
    if vuln_indices:
        max_vuln = max(vuln_indices)
        while prompt_end >= max_vuln and prompt_end > 1:
            prompt_end -= 1

    prompt_lines = code_lines[:prompt_end]
    prompt = "\n".join(prompt_lines) + "\n" if prompt_lines else ""

    # Forget lines = vulnerability lines that appear AFTER the prompt cutoff
    forget_lines = []
    for idx in sorted(vuln_indices):
        if idx > prompt_end:
            line = code_lines[idx - 1].strip()
            if line:
                forget_lines.append(line)

    # If no vulnerability lines fall after the prompt (e.g. all vuln lines are
    # in the prompt region), fall back to using ALL vulnerability lines.
    if not forget_lines:
        for idx in sorted(vuln_indices):
            line = code_lines[idx - 1].strip()
            if line:
                forget_lines.append(line)

    suffix_lines = code_lines[prompt_end:]
    ground_truth_suffix = "\n".join(suffix_lines)

    return prompt, forget_lines, ground_truth_suffix


def process_yaml(
    data: dict,
    min_pattern_length: int = 10,
    prompt_ratio: float = 0.5,
) -> Tuple[List[dict], List[str], dict]:
    """
    Process a YAML dataset and extract forget datasets and patterns.

    Returns:
        dataset: list of {"prompt": "...", "cve_id": "...", "ground_truth_suffix": "..."}
        all_patterns: deduplicated list of vulnerability code line patterns
        stats: summary statistics
    """
    dataset = []
    all_patterns: Set[str] = set()
    skipped = 0
    short_filtered = 0

    entries = {k: v for k, v in data.items() if v.get("pre", {}).get("label") == 1}
    print(f"  Entries with pre.label=1: {len(entries)}")

    for cve_id, entry in entries.items():
        pre = entry.get("pre", {})
        code = pre.get("code", "")
        if not code.strip():
            skipped += 1
            continue

        criteria_lines = entry.get("pre-criteria-lines", [])
        relative_lines = entry.get("pre-relative-lines", [])

        prompt, forget_lines, ground_truth_suffix = extract_vulnerability_lines(
            code, criteria_lines, relative_lines, prompt_ratio=prompt_ratio,
        )

        if not prompt.strip():
            skipped += 1
            continue

        filtered_forget = []
        for line in forget_lines:
            if len(line) >= min_pattern_length:
                filtered_forget.append(line)
            else:
                short_filtered += 1

        if not filtered_forget:
            skipped += 1
            continue

        dataset.append({
            "prompt": prompt,
            "cve_id": cve_id,
            "ground_truth_suffix": ground_truth_suffix,
        })
        all_patterns.update(filtered_forget)

    patterns_list = sorted(all_patterns)

    stats = {
        "total_label1": len(entries),
        "dataset_size": len(dataset),
        "skipped": skipped,
        "unique_patterns": len(patterns_list),
        "short_filtered": short_filtered,
    }

    return dataset, patterns_list, stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess vulnerability YAML into GRPO-compatible JSON"
    )
    parser.add_argument(
        "--yaml_file", type=str, required=True,
        help="Path to the vulnerability YAML file (e.g., pile_ft_result.yaml)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--min_pattern_length", type=int, default=10,
        help="Minimum character length for a forget pattern (filters out generic lines)",
    )
    parser.add_argument(
        "--prompt_ratio", type=float, default=0.5,
        help="Fraction of function lines to use as prompt (default 0.5 = first 50%%)",
    )
    args = parser.parse_args()

    data = load_yaml(args.yaml_file)
    dataset, patterns, stats = process_yaml(
        data, args.min_pattern_length, args.prompt_ratio,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_file = os.path.join(args.output_dir, "forget_dataset.json")
    with open(dataset_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved dataset ({len(dataset)} entries) to: {dataset_file}")

    patterns_file = os.path.join(args.output_dir, "forget_patterns.json")
    with open(patterns_file, "w") as f:
        json.dump(patterns, f, indent=2)
    print(f"Saved patterns ({len(patterns)} unique) to: {patterns_file}")

    print(f"\n{'='*50}")
    print("Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}")

    # Also save per-entry forget lines for potential per-prompt reward
    per_entry_file = os.path.join(args.output_dir, "per_entry_forget.json")
    per_entry_data = []
    entries = {k: v for k, v in data.items() if v.get("pre", {}).get("label") == 1}
    for cve_id, entry in entries.items():
        pre = entry.get("pre", {})
        code = pre.get("code", "")
        if not code.strip():
            continue
        criteria_lines = entry.get("pre-criteria-lines", [])
        relative_lines = entry.get("pre-relative-lines", [])
        prompt, forget_lines, ground_truth_suffix = extract_vulnerability_lines(
            code, criteria_lines, relative_lines, prompt_ratio=args.prompt_ratio,
        )
        filtered = [l for l in forget_lines if len(l) >= args.min_pattern_length]
        if prompt.strip() and filtered:
            per_entry_data.append({
                "prompt": prompt,
                "cve_id": cve_id,
                "forget_lines": filtered,
                "ground_truth_suffix": ground_truth_suffix,
            })
    with open(per_entry_file, "w") as f:
        json.dump(per_entry_data, f, indent=2)
    print(f"Saved per-entry data ({len(per_entry_data)} entries) to: {per_entry_file}")


if __name__ == "__main__":
    main()
