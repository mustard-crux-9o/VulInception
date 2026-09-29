#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "1")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def ensure_humaneval(path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def truncate_prompts(problems, tokenizer, max_input_tokens):
    prompts = []
    for problem in problems:
        ids = tokenizer.encode(problem["prompt"], add_special_tokens=False)
        if len(ids) > max_input_tokens:
            ids = ids[-max_input_tokens:]
        prompts.append(tokenizer.decode(ids, skip_special_tokens=False))
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_eval_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--problem_file", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_input_tokens", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    root = Path(args.human_eval_root).resolve()
    problem_file = ensure_humaneval(args.problem_file or root / "data" / "HumanEval.jsonl.gz")
    problems = read_jsonl_gz(problem_file)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples_file = out / "samples.jsonl"
    max_input_tokens = args.max_input_tokens or max(1, args.max_model_len - args.max_new_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    prompts = truncate_prompts(problems, tokenizer, max_input_tokens)

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
        stop=["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
    )
    outputs = []
    for index in range(0, len(prompts), args.batch_size):
        outputs.extend(llm.generate(prompts[index : index + args.batch_size], params))
    with samples_file.open("w", encoding="utf-8") as f:
        for problem, output in zip(problems, outputs, strict=True):
            f.write(json.dumps({"task_id": problem["task_id"], "completion": output.outputs[0].text}, ensure_ascii=False) + "\n")

    cmd = [
        sys.executable,
        "-m",
        "human_eval.evaluate_functional_correctness",
        str(samples_file),
        "--n_workers",
        str(args.n_workers),
        "--timeout",
        str(args.timeout),
        "--problem_file",
        str(problem_file),
    ]
    result = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    (out / "evaluate_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (out / "evaluate_stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)
    pass_at_1 = None
    for line in result.stdout.splitlines()[::-1]:
        if "pass@1" in line:
            match = re.search(r"pass@1['\"]?:\s*(?:np\.float64\()?([0-9.eE+-]+)", line)
            if match:
                pass_at_1 = float(match.group(1))
            break
    summary = {
        "model_path": args.model_path,
        "problem_file": str(problem_file.resolve()),
        "num_samples": len(problems),
        "pass@1": pass_at_1,
        "max_input_tokens": max_input_tokens,
        "batch_size": args.batch_size,
        "samples_jsonl": str(samples_file),
        "results_jsonl": str(samples_file) + "_results.jsonl",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
