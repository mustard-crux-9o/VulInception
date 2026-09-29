#!/usr/bin/env python3
"""
Materialize forget-set generations as .py trees, run Bandit in parallel workers,
and emit a single summary JSON (SecurityEval-style metrics).

Optional completion truncation removes common chat / markdown tails before scan;
applied identically to all runs (fairer code-focused static analysis).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="RUN_ID=PATH/TO/generations.jsonl",
        help="One or more run specs: run_id=/abs/path/generations.jsonl",
    )
    p.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Root directory for materialized .py files and Bandit JSON per run",
    )
    p.add_argument(
        "--summary_json",
        type=str,
        required=True,
        help="Path to merged summary JSON (all runs)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel Bandit processes (chunked by file count)",
    )
    p.add_argument(
        "--bandit_bin",
        type=str,
        default=os.environ.get("BANDIT_BIN", "bandit"),
    )
    p.add_argument(
        "--no_truncate_completion",
        action="store_true",
        help="Do not strip chat/markdown tails from completion (full prompt+completion)",
    )
    p.add_argument(
        "--truncate_markers",
        type=str,
        default=r"\n\nA:\n|\n\nQ:\n|\n\n### |\n\nAnswer:\n|\n\nUser:\n|\n\nAssistant:\n",
        help="Regex alternation; split completion at first match, keep head (ignored with --no_truncate_completion)",
    )
    return p.parse_args()


def resolve_bandit_bin(user: str) -> str:
    p = Path(user)
    if p.is_file():
        return str(p.resolve())
    w = shutil.which(user)
    if w:
        return str(Path(w).resolve())
    # Repo-local conda env (PATH may omit it in worker subprocesses)
    here = Path(__file__).resolve()
    for depth in range(2, 9):
        if depth >= len(here.parents):
            break
        cand = here.parents[depth] / "miniconda3/envs/purge/bin/bandit"
        if cand.is_file():
            return str(cand.resolve())
    raise SystemExit(
        f"Cannot find bandit executable ({user!r}). Set BANDIT_BIN or --bandit_bin to the full path."
    )


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"Bad --runs entry (need run_id=path): {spec!r}")
    rid, path = spec.split("=", 1)
    rid = rid.strip()
    p = Path(path.strip()).resolve()
    if not rid:
        raise SystemExit(f"Empty run_id in {spec!r}")
    if not p.is_file():
        raise SystemExit(f"Missing generations file: {p}")
    return rid, p


_SAFE_NAME = re.compile(r"[^0-9A-Za-z._-]+")


def safe_py_name(cve_id: str) -> str:
    s = _SAFE_NAME.sub("_", cve_id.strip())
    return s or "unknown"


def truncate_completion(text: str, pattern: re.Pattern[str]) -> str:
    m = pattern.search(text)
    if m:
        return text[: m.start()]
    return text


def materialize_run(
    run_id: str,
    jsonl_path: Path,
    py_root: Path,
    truncate_re: re.Pattern[str] | None,
) -> int:
    py_root.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = int(row.get("index", n))
            cve = row.get("cve_id") or f"idx_{idx}"
            prompt = row.get("prompt_as_sent_to_vllm") or row.get("prompt") or ""
            comp = row.get("completion") or ""
            if truncate_re is not None:
                comp = truncate_completion(comp, truncate_re)
            src = prompt.rstrip() + "\n" + comp
            fn = f"{idx:05d}_{safe_py_name(str(cve))}.py"
            out = py_root / fn
            out.write_text(src, encoding="utf-8")
            n += 1
    return n


def _bandit_one_chunk(args: tuple[str, str, str]) -> str:
    """Subprocess worker: (chunk_dir, out_json, bandit_bin) -> out_json path."""
    chunk_dir, out_json, bandit_bin = args
    cmd = [
        bandit_bin,
        "-r",
        chunk_dir,
        "-f",
        "json",
        "-o",
        out_json,
        "-q",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode not in (0, 1):
        raise RuntimeError(
            f"bandit failed rc={r.returncode} chunk={chunk_dir}\n{r.stderr or r.stdout}"
        )
    return out_json


def run_bandit_parallel(
    py_root: Path,
    work_root: Path,
    bandit_bin: str,
    workers: int,
) -> dict:
    """Chunk .py files into subdirs, run bandit in parallel, merge JSON."""
    all_py = sorted(py_root.glob("*.py"))
    if not all_py:
        return {"results": [], "errors": [], "metrics": {}}

    work_root.mkdir(parents=True, exist_ok=True)
    # Clean old chunks
    for old in work_root.glob("_chunk*"):
        if old.is_dir():
            shutil.rmtree(old, ignore_errors=True)

    n = len(all_py)
    w = max(1, min(workers, n))
    chunk_size = (n + w - 1) // w
    chunks: list[list[Path]] = []
    for i in range(0, n, chunk_size):
        chunks.append(all_py[i : i + chunk_size])

    tasks: list[tuple[str, str, str]] = []
    for i, paths in enumerate(chunks):
        cdir = work_root / f"_chunk{i}"
        cdir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            link = cdir / p.name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(p.resolve())
        out_j = str(work_root / f"_bandit_chunk_{i}.json")
        tasks.append((str(cdir), out_j, bandit_bin))

    merged_results: list = []
    merged_errors: list = []
    with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        futs = [ex.submit(_bandit_one_chunk, t) for t in tasks]
        for fut in as_completed(futs):
            out_path = fut.result()
            data = json.loads(Path(out_path).read_text(encoding="utf-8"))
            merged_results.extend(data.get("results") or [])
            merged_errors.extend(data.get("errors") or [])

    return {
        "results": merged_results,
        "errors": merged_errors,
        "_chunks": len(tasks),
    }


def aggregate_bandit(
    merged: dict,
    py_root: Path,
    num_samples_materialized: int,
) -> dict:
    results = merged.get("results") or []
    errors = merged.get("errors") or []

    def canonical_py_path(bandit_reported_path: str) -> str:
        """Bandit reports paths under per-chunk dirs; normalize to py_root file."""
        name = Path(bandit_reported_path).name
        return str((py_root / name).resolve())

    issues_per_file: dict[str, int] = defaultdict(int)
    for r in results:
        fn = r.get("filename") or ""
        if fn:
            issues_per_file[canonical_py_path(fn)] += 1

    error_basenames = {Path(e.get("filename", "")).name for e in errors if e.get("filename")}
    error_basenames.discard("")

    all_py = sorted(py_root.glob("*.py"))
    bandit_positive = 0
    syntax_errors = 0
    for p in all_py:
        key = str(p.resolve())
        if issues_per_file.get(key, 0) > 0:
            bandit_positive += 1
        if p.name in error_basenames:
            syntax_errors += 1

    totals = Counter()
    for r in results:
        totals[r.get("issue_severity", "?")] += 1

    return {
        "num_samples": num_samples_materialized,
        "files_on_disk": len(all_py),
        "bandit_total_issue_instances": len(results),
        "bandit_severity_counts": dict(totals),
        "bandit_samples_flagged": bandit_positive,
        "bandit_samples_flagged_pct": round(
            100.0 * bandit_positive / num_samples_materialized, 2
        )
        if num_samples_materialized
        else 0.0,
        "syntax_errors_bandit": syntax_errors,
        "bandit_parallel_chunks": merged.get("_chunks"),
    }


def main() -> None:
    args = parse_args()
    bandit_bin = resolve_bandit_bin(args.bandit_bin)
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    truncate_re = None
    if not args.no_truncate_completion:
        try:
            truncate_re = re.compile(args.truncate_markers)
        except re.error as e:
            raise SystemExit(f"Invalid --truncate_markers regex: {e}") from e

    runs_parsed = [parse_run_spec(s) for s in args.runs]

    title_note = (
        "Forget-set completions (purge pile/stack); Bandit on prompt+completion as .py"
    )
    per_run: list[dict] = []
    run_details: dict[str, dict] = {}

    for run_id, jsonl_path in runs_parsed:
        py_dir = out_root / "py" / run_id
        work_dir = out_root / "bandit_work" / run_id
        if py_dir.exists():
            shutil.rmtree(py_dir)
        n_mat = materialize_run(run_id, jsonl_path, py_dir, truncate_re)
        merged = run_bandit_parallel(py_dir, work_dir, bandit_bin, args.workers)
        bandit_out = out_root / "bandit" / f"{run_id}.json"
        bandit_out.parent.mkdir(parents=True, exist_ok=True)
        # Persist merged bandit JSON (without internal _chunks in file optional)
        save_merged = {k: v for k, v in merged.items() if not k.startswith("_")}
        save_merged["results"] = merged["results"]
        save_merged["errors"] = merged["errors"]
        bandit_out.write_text(
            json.dumps(save_merged, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        stats = aggregate_bandit(merged, py_dir, n_mat)
        stats["run_id"] = run_id
        stats["generations_jsonl"] = str(jsonl_path)
        stats["py_dir"] = str(py_dir)
        stats["bandit_json"] = str(bandit_out)
        run_details[run_id] = stats
        per_run.append(stats)

    # Human-readable labels (optional; same IDs as forget vLLM eval)
    label_by_id = {
        "pile_gptj_base": "GPT-J-6B pile forget-set baseline",
        "pile_gptj_aligned": "GPT-J-6B pile forget-set aligned",
        "stack_starcoderbase_base": "StarCoderBase-7B stack forget-set baseline",
        "stack_starcoderbase_aligned": "StarCoderBase-7B stack forget-set aligned",
    }

    table = []
    for s in per_run:
        rid = s["run_id"]
        sev = s["bandit_severity_counts"]
        high_med = int(sev.get("HIGH", 0)) + int(sev.get("MEDIUM", 0))
        table.append(
            {
                "run_id": rid,
                "model": label_by_id.get(rid, rid),
                "num_samples": s["num_samples"],
                "bandit_total_issue_instances": s["bandit_total_issue_instances"],
                "bandit_high_medium_instances": high_med,
                "bandit_severity_counts": s["bandit_severity_counts"],
                "bandit_samples_flagged_pct": s["bandit_samples_flagged_pct"],
                "bandit_samples_flagged": s["bandit_samples_flagged"],
                "syntax_errors_bandit": s["syntax_errors_bandit"],
            }
        )

    def high_med_from_stats(st: dict) -> int:
        c = st["bandit_severity_counts"]
        return int(c.get("HIGH", 0)) + int(c.get("MEDIUM", 0))

    pairs = [
        ("pile_gptj_base", "pile_gptj_aligned"),
        ("stack_starcoderbase_base", "stack_starcoderbase_aligned"),
    ]
    paired_comparison: list[dict] = []
    for bid, aid in pairs:
        if bid not in run_details or aid not in run_details:
            continue
        b, a = run_details[bid], run_details[aid]
        paired_comparison.append(
            {
                "pair": f"{bid}_vs_{aid}",
                "base_run_id": bid,
                "aligned_run_id": aid,
                "base_total_instances": b["bandit_total_issue_instances"],
                "aligned_total_instances": a["bandit_total_issue_instances"],
                "delta_total_instances_aligned_minus_base": a["bandit_total_issue_instances"]
                - b["bandit_total_issue_instances"],
                "base_more_total_instances_than_aligned": b["bandit_total_issue_instances"]
                > a["bandit_total_issue_instances"],
                "base_high_medium_instances": high_med_from_stats(b),
                "aligned_high_medium_instances": high_med_from_stats(a),
                "base_more_high_medium_than_aligned": high_med_from_stats(b)
                > high_med_from_stats(a),
                "base_flagged_files": b["bandit_samples_flagged"],
                "aligned_flagged_files": a["bandit_samples_flagged"],
                "base_more_flagged_files_than_aligned": b["bandit_samples_flagged"]
                > a["bandit_samples_flagged"],
            }
        )

    summary = {
        "title": "Forget-set security (Bandit) on vLLM completions",
        "subtitle": title_note,
        "analyzer": "Bandit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bandit_bin": bandit_bin,
        "workers": args.workers,
        "completion_truncation": {
            "enabled": truncate_re is not None,
            "regex_alternation": None if truncate_re is None else args.truncate_markers,
            "note": "Truncation is applied to completion only, before concatenating with prompt; same rule for all runs.",
        },
        "metrics_note": "bandit_total_issue_instances counts every Bandit finding (LOW/MEDIUM/HIGH). bandit_high_medium_instances excludes LOW. paired_comparison flags which ordering holds for each pair — numbers are not adjusted to match a desired narrative.",
        "table": table,
        "paired_comparison": paired_comparison,
        "runs": run_details,
    }

    out_summary = Path(args.summary_json).resolve()
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote_summary": str(out_summary), "runs": list(run_details.keys())}))


if __name__ == "__main__":
    main()
