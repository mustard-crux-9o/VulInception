#!/usr/bin/env python3
"""
Aggregate CodeQL CSV outputs for flat forget-set trees (NNNNN_cve....py filenames).

Writes one JSON with per-run metrics + paired base-vs-aligned comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
# Basenames produced by forget_bandit_security_eval.materialize_run
FORGET_PY = re.compile(r"(?P<bn>\d{5}_[A-Za-z0-9._-]+\.py)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="Single merged summary JSON path",
    )
    p.add_argument(
        "--forget_codeql_root",
        type=str,
        required=True,
        help="eval_outputs/forget_codeql (contains csv/<run_id>/*.csv)",
    )
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="RUN_ID",
        help="Run ids matching csv/<run_id>/ directories",
    )
    p.add_argument(
        "--py_root",
        type=str,
        required=True,
        help="Parent directory containing py/<run_id>/ (e.g. forget_bandit_security/py)",
    )
    p.add_argument(
        "--generations_root",
        type=str,
        required=True,
        help="Parent dir with forget_vllm_four_models/<run_id>/generations.jsonl",
    )
    return p.parse_args()


def collect_hits_from_csv_dir(results_dir: Path) -> tuple[set[str], int]:
    """Return (unique forget basenames with ≥1 alert, total alert rows)."""
    basenames: set[str] = set()
    total_rows = 0
    for csv_path in sorted(results_dir.glob("*.csv")):
        text = csv_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        reader = csv.reader(text.splitlines())
        for row in reader:
            if not row:
                continue
            total_rows += 1
            row_text = " ".join(row)
            for m in FORGET_PY.finditer(row_text):
                basenames.add(m.group("bn"))
    return basenames, total_rows


def num_samples_from_jsonl(path: Path) -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> None:
    args = parse_args()
    fc_root = Path(args.forget_codeql_root).resolve()
    py_parent = Path(args.py_root).resolve()
    gen_parent = Path(args.generations_root).resolve()

    se_root = Path(os.environ.get("SECURITYEVAL_ROOT", str(fc_root.parents[3] / "SecurityEval")))
    codeql_cli = se_root / "tools" / "codeql" / "codeql"
    codeql_repo = se_root / "tools" / "codeql-repo"
    codeql_version = ""
    if codeql_cli.is_file():
        try:
            codeql_version = subprocess.run(
                [str(codeql_cli), "version"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        except OSError:
            codeql_version = ""

    label_by_id = {
        "pile_gptj_base": "GPT-J-6B pile forget-set baseline",
        "pile_gptj_aligned": "GPT-J-6B pile forget-set aligned",
        "stack_starcoderbase_base": "StarCoderBase-7B stack forget-set baseline",
        "stack_starcoderbase_aligned": "StarCoderBase-7B stack forget-set aligned",
    }

    pairs = [
        ("pile_gptj_base", "pile_gptj_aligned"),
        ("stack_starcoderbase_base", "stack_starcoderbase_aligned"),
    ]

    by_run: dict = {}
    table = []

    for run_id in args.runs:
        csv_dir = fc_root / "csv" / run_id
        py_dir = py_parent / run_id
        gen = gen_parent / run_id / "generations.jsonl"
        if not csv_dir.is_dir():
            raise SystemExit(f"Missing CodeQL CSV dir: {csv_dir}")
        if not gen.is_file():
            raise SystemExit(f"Missing generations.jsonl: {gen}")

        hits, alert_rows = collect_hits_from_csv_dir(csv_dir)
        n_samples = num_samples_from_jsonl(gen)
        files_on_disk = len(list(py_dir.glob("*.py")))
        flagged = len(hits)

        row = {
            "run_id": run_id,
            "model": label_by_id.get(run_id, run_id),
            "num_samples": n_samples,
            "files_on_disk": files_on_disk,
            "codeql_alert_rows_total": alert_rows,
            "codeql_unique_py_files_flagged": flagged,
            "codeql_samples_flagged": flagged,
            "codeql_positive_rate_pct": round(100.0 * flagged / n_samples, 4)
            if n_samples
            else 0.0,
            "codeql_csv_dir": str(csv_dir),
            "codeql_database": str((fc_root / "db" / run_id).resolve()),
            "py_dir": str(py_dir),
            "generations_jsonl": str(gen),
        }
        table.append(row)
        by_run[run_id] = row

    paired: list[dict] = []
    for base_id, aligned_id in pairs:
        if base_id not in by_run or aligned_id not in by_run:
            continue
        b, a = by_run[base_id], by_run[aligned_id]
        paired.append(
            {
                "pair": f"{base_id}_vs_{aligned_id}",
                "base_run_id": base_id,
                "aligned_run_id": aligned_id,
                "base_flagged_files": b["codeql_unique_py_files_flagged"],
                "aligned_flagged_files": a["codeql_unique_py_files_flagged"],
                "delta_flagged_files_aligned_minus_base": a["codeql_unique_py_files_flagged"]
                - b["codeql_unique_py_files_flagged"],
                "base_alert_rows": b["codeql_alert_rows_total"],
                "aligned_alert_rows": a["codeql_alert_rows_total"],
                "delta_alert_rows_aligned_minus_base": a["codeql_alert_rows_total"]
                - b["codeql_alert_rows_total"],
                "base_more_flagged_than_aligned": b["codeql_unique_py_files_flagged"]
                > a["codeql_unique_py_files_flagged"],
                "base_more_alerts_than_aligned": b["codeql_alert_rows_total"]
                > a["codeql_alert_rows_total"],
            }
        )

    summary = {
        "title": "Forget-set CodeQL (four models), flat .py source roots",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "codeql_cli": str(codeql_cli) if codeql_cli.is_file() else None,
        "codeql_version": codeql_version or None,
        "codeql_repo": str(codeql_repo) if codeql_repo.is_dir() else None,
        "pipeline": {
            "script": str(
                Path(__file__).resolve().parent / "run_forget_codeql.sh"
            ),
            "packs": "Same 27 Python Security packs as SecurityEval (CWE-020-ExternalAPIs subpack skipped for CSV; not used here).",
            "source": "prompt+completion written as NNNNN_*.py (forget_bandit_security_eval materialization; completion chat-tail truncation enabled in that step).",
        },
        "note": "Metrics from CodeQL CSV outputs. Alert rows counted across all results_*.csv; unique flagged files by basename \\d{5}_*.py. Fragmented completions + parse noise → often sparse CodeQL signal vs. small curated SecurityEval snippets.",
        "table": table,
        "paired_comparison": paired,
    }

    out = Path(args.out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
