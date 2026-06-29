#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import yaml


def split_entry(cve_id, entry, prompt_ratio, min_pattern_length):
    code = (entry.get("pre") or {}).get("code") or ""
    lines = code.split("\n")
    if len(lines) < 2:
        return None
    prompt_end = max(1, int(len(lines) * prompt_ratio))
    if prompt_end >= len(lines):
        prompt_end = len(lines) - 1
    prompt = "\n".join(lines[:prompt_end]) + "\n"
    suffix_lines = lines[prompt_end:]
    forget_lines = [line.strip() for line in suffix_lines if len(line.strip()) >= min_pattern_length]
    if not prompt.strip() or not forget_lines:
        return None
    return {
        "prompt": prompt,
        "cve_id": cve_id,
        "forget_lines": forget_lines,
        "ground_truth_suffix": "\n".join(suffix_lines),
    }


def build_dataset(yaml_file, output_dir, sample_size, seed, prompt_ratio, min_pattern_length):
    data = yaml.safe_load(Path(yaml_file).read_text())
    eligible = []
    for cve_id, entry in data.items():
        row = split_entry(cve_id, entry, prompt_ratio, min_pattern_length)
        if row is not None:
            eligible.append(row)
    if sample_size > len(eligible):
        raise ValueError(f"sample_size={sample_size} > eligible={len(eligible)} for {yaml_file}")
    rng = random.Random(seed)
    sampled = rng.sample(eligible, sample_size)
    patterns = sorted({line for row in sampled for line in row["forget_lines"]})
    dataset = [
        {
            "prompt": row["prompt"],
            "cve_id": row["cve_id"],
            "ground_truth_suffix": row["ground_truth_suffix"],
        }
        for row in sampled
    ]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "forget_dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    (out / "forget_patterns.json").write_text(json.dumps(patterns, indent=2), encoding="utf-8")
    (out / "per_entry_forget.json").write_text(json.dumps(sampled, indent=2), encoding="utf-8")
    (out / "sampled_ids.json").write_text(json.dumps([row["cve_id"] for row in sampled], indent=2), encoding="utf-8")
    summary = {
        "yaml_file": str(Path(yaml_file).resolve()),
        "total_entries": len(data),
        "eligible_entries": len(eligible),
        "sample_size": len(sampled),
        "seed": seed,
        "prompt_ratio": prompt_ratio,
        "min_pattern_length": min_pattern_length,
        "unique_patterns": len(patterns),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_ratio", type=float, default=0.5)
    parser.add_argument("--min_pattern_length", type=int, default=10)
    args = parser.parse_args()
    build_dataset(
        args.yaml_file,
        args.output_dir,
        args.sample_size,
        args.seed,
        args.prompt_ratio,
        args.min_pattern_length,
    )


if __name__ == "__main__":
    main()
