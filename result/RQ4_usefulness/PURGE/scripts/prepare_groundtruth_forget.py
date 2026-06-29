#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml

from prepare_random_forget import split_entry


def build_dataset(yaml_file, output_dir, prompt_ratio, min_pattern_length):
    data = yaml.safe_load(Path(yaml_file).read_text())
    rows = []
    selected_ids = []
    for cve_id, entry in data.items():
        if int(((entry.get("pre") or {}).get("label")) or 0) != 1:
            continue
        selected_ids.append(cve_id)
        row = split_entry(cve_id, entry, prompt_ratio, min_pattern_length)
        if row is not None:
            rows.append(row)

    patterns = sorted({line for row in rows for line in row["forget_lines"]})
    dataset = [
        {
            "prompt": row["prompt"],
            "cve_id": row["cve_id"],
            "ground_truth_suffix": row["ground_truth_suffix"],
        }
        for row in rows
    ]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "forget_dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    (out / "forget_patterns.json").write_text(json.dumps(patterns, indent=2), encoding="utf-8")
    (out / "per_entry_forget.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "sampled_ids.json").write_text(json.dumps([row["cve_id"] for row in rows], indent=2), encoding="utf-8")
    summary = {
        "yaml_file": str(Path(yaml_file).resolve()),
        "selected_pre_label_1": len(selected_ids),
        "sample_size": len(rows),
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
    parser.add_argument("--prompt_ratio", type=float, default=0.5)
    parser.add_argument("--min_pattern_length", type=int, default=10)
    args = parser.parse_args()
    build_dataset(
        args.yaml_file,
        args.output_dir,
        args.prompt_ratio,
        args.min_pattern_length,
    )


if __name__ == "__main__":
    main()
