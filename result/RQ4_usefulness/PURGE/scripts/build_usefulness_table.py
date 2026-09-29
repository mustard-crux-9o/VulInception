#!/usr/bin/env python3
"""Aggregate RQ4 usefulness table (VR + HumanEval pass@1) for one target model."""
import argparse
import csv
import json
from pathlib import Path

SELECTORS = [
    ("original", "Original (None)"),
    ("vulinception", "VulInception"),
    ("random", "Random"),
    ("gotcha", "Gotcha"),
    ("groundtruth", "Groundtruth (GT)"),
]

COLS = [
    "Model",
    "Forget-set Selector",
    "VMI-Bench VR ↓",
    "Unseen 2026 CVE VR ↓",
    "HumanEval pass@1 ↑",
]


def load_metric(root: Path, selector: str, split: str, key: str):
    path = root / split / selector / "summary.json"
    if not path.is_file():
        return "X"
    val = json.loads(path.read_text()).get(key)
    if val is None:
        return "X"
    return round(float(val), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="gpt_j_6b or starcoderbase_7b")
    parser.add_argument("--eval_root", default="eval_outputs/rq4_usefulness")
    parser.add_argument("--display_name", default=None)
    args = parser.parse_args()

    display = args.display_name or (
        "GPT-J-6B" if args.model == "gpt_j_6b" else "StarCoderBase-7B"
    )
    root = Path(args.eval_root) / args.model

    rows = []
    for tag, label in SELECTORS:
        rows.append({
            COLS[0]: display,
            COLS[1]: label,
            COLS[2]: load_metric(root, tag, "vmi_bench", "vulnerability_rate"),
            COLS[3]: load_metric(root, tag, "unseen2026", "vulnerability_rate"),
            COLS[4]: load_metric(root, tag, "humaneval", "pass@1"),
        })

    out_csv = root / "usefulness_table.csv"
    out_json = root / "usefulness_table.json"
    root.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
