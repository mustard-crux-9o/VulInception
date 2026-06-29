#!/usr/bin/env python3
import argparse
import gzip
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def truncate_stop(text, stops):
    end = len(text)
    for stop in stops:
        index = text.find(stop)
        if index >= 0:
            end = min(end, index)
    return text[:end]


def resolve_max_input_tokens(model, max_new_tokens, requested):
    if requested > 0:
        return requested
    limits = [
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "n_positions", None),
        getattr(model.config, "seq_length", None),
    ]
    limit = min([value for value in limits if isinstance(value, int) and value > 0], default=2048)
    return max(1, limit - max_new_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_eval_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--problem_file", default=None)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_input_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.human_eval_root).resolve()
    problem_file = ensure_humaneval(args.problem_file or root / "data" / "HumanEval.jsonl.gz")
    problems = read_jsonl_gz(problem_file)
    if args.limit > 0:
        problems = problems[: args.limit]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples_file = out / "samples.jsonl"

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    max_input_tokens = resolve_max_input_tokens(model, args.max_new_tokens, args.max_input_tokens)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
    else:
        gen_kwargs["do_sample"] = False

    stop = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
    device = next(model.parameters()).device
    with samples_file.open("w", encoding="utf-8") as f:
        for problem in problems:
            inputs = tokenizer(problem["prompt"], return_tensors="pt", truncation=True, max_length=max_input_tokens).to(device)
            with torch.inference_mode():
                output = model.generate(**inputs, **gen_kwargs)
            completion = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            completion = truncate_stop(completion, stop)
            f.write(json.dumps({"task_id": problem["task_id"], "completion": completion}, ensure_ascii=False) + "\n")

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
        "samples_jsonl": str(samples_file),
        "results_jsonl": str(samples_file) + "_results.jsonl",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
