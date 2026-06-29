#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

# vllm >= 0.11: keep engine version consistent (avoid VLLM_USE_V1 mismatch).
os.environ.setdefault("VLLM_USE_V1", "1")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def safe_name(name):
    return re.sub(r"[^0-9A-Za-z._/-]+", "_", name).replace("..", "_")


def truncate_prompts(rows, tokenizer, max_input_tokens):
    prompts = []
    for row in rows:
        ids = tokenizer.encode(row["Prompt"], add_special_tokens=False)
        if len(ids) > max_input_tokens:
            ids = ids[-max_input_tokens:]
        prompts.append(tokenizer.decode(ids, skip_special_tokens=False))
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_input_tokens", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--bandit_bin", default="bandit")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.dataset_jsonl).read_text().splitlines() if line.strip()]
    out = Path(args.output_dir)
    py_dir = out / "py"
    if py_dir.exists():
        shutil.rmtree(py_dir)
    py_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    max_input_tokens = args.max_input_tokens or max(1, args.max_model_len - args.max_new_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    prompts = truncate_prompts(rows, tokenizer, max_input_tokens)

    llm = LLM(
        model=args.model_path,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop=["\n\n#", "\n\nif __name__", "\n```"],
    )
    outputs = []
    for index in range(0, len(prompts), args.batch_size):
        outputs.extend(llm.generate(prompts[index : index + args.batch_size], params))

    gen_path = out / "generations.jsonl"
    with gen_path.open("w", encoding="utf-8") as f:
        for row, output in zip(rows, outputs, strict=True):
            completion = output.outputs[0].text
            src = row["Prompt"].rstrip() + "\n" + completion
            rel = safe_name(row["ID"])
            path = py_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(src, encoding="utf-8")
            f.write(json.dumps({"id": row["ID"], "prompt": row["Prompt"], "completion": completion}, ensure_ascii=False) + "\n")

    bandit_json = out / "bandit.json"
    cmd = [args.bandit_bin, "-r", str(py_dir), "-f", "json", "-o", str(bandit_json), "-q"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise SystemExit(result.stderr or result.stdout)
    report = json.loads(bandit_json.read_text())
    issues_per_file = defaultdict(int)
    for item in report.get("results") or []:
        filename = item.get("filename")
        if filename:
            issues_per_file[str(Path(filename).resolve())] += 1
    total = len(rows)
    flagged = 0
    for path in py_dir.rglob("*.py"):
        if issues_per_file.get(str(path.resolve()), 0) > 0:
            flagged += 1
    summary = {
        "model_path": args.model_path,
        "dataset_jsonl": str(Path(args.dataset_jsonl).resolve()),
        "num_samples": total,
        "bandit_flagged": flagged,
        "vulnerability_rate": flagged / total if total else 0.0,
        "bandit_total_issue_instances": len(report.get("results") or []),
        "max_input_tokens": max_input_tokens,
        "batch_size": args.batch_size,
        "generations_jsonl": str(gen_path),
        "bandit_json": str(bandit_json),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
