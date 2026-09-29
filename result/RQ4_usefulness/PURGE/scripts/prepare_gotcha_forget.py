#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml

from prepare_random_forget import split_entry


def _prediction_label(row):
    for key in ("Predicted-label", "predicted-label", "predicted_label"):
        if key in row:
            return row[key]
    return None


def build_dataset(predictions_file, yaml_file, output_dir, prompt_ratio, min_pattern_length):
    data = yaml.safe_load(Path(yaml_file).read_text())
    selected_ids = []
    with Path(predictions_file).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(_prediction_label(row) or 0) == 1:
                selected_ids.append(row.get("CVE-name") or row.get("cve_id"))

    rows = []
    missing_ids = []
    for cve_id in selected_ids:
        entry = data.get(cve_id)
        if entry is None:
            missing_ids.append(cve_id)
            continue
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
        "predictions_file": str(Path(predictions_file).resolve()),
        "yaml_file": str(Path(yaml_file).resolve()),
        "selected_predictions": len(selected_ids),
        "missing_ids": len(missing_ids),
        "sample_size": len(rows),
        "prompt_ratio": prompt_ratio,
        "min_pattern_length": min_pattern_length,
        "unique_patterns": len(patterns),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", required=True)
    parser.add_argument("--yaml_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_ratio", type=float, default=0.5)
    parser.add_argument("--min_pattern_length", type=int, default=10)
    args = parser.parse_args()
    build_dataset(
        args.predictions_file,
        args.yaml_file,
        args.output_dir,
        args.prompt_ratio,
        args.min_pattern_length,
    )


if __name__ == "__main__":
    main()
