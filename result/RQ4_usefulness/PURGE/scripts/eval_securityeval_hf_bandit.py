#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def safe_name(name):
    return re.sub(r"[^0-9A-Za-z._/-]+", "_", name).replace("..", "_")


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
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_input_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--bandit_bin", default="bandit")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.dataset_jsonl).read_text().splitlines() if line.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]
    out = Path(args.output_dir)
    py_dir = out / "py"
    if py_dir.exists():
        shutil.rmtree(py_dir)
    py_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

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

    stop = ["\n\n#", "\n\nif __name__", "\n```"]
    device = next(model.parameters()).device
    gen_path = out / "generations.jsonl"
    with gen_path.open("w", encoding="utf-8") as f:
        for row in rows:
            prompt = row["Prompt"]
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(device)
            with torch.inference_mode():
                output = model.generate(**inputs, **gen_kwargs)
            completion = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            completion = truncate_stop(completion, stop)
            src = prompt.rstrip() + "\n" + completion
            rel = safe_name(row["ID"])
            path = py_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(src, encoding="utf-8")
            f.write(json.dumps({"id": row["ID"], "prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")

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
    flagged = 0
    for path in py_dir.rglob("*.py"):
        if issues_per_file.get(str(path.resolve()), 0) > 0:
            flagged += 1
    summary = {
        "model_path": args.model_path,
        "dataset_jsonl": str(Path(args.dataset_jsonl).resolve()),
        "num_samples": len(rows),
        "bandit_flagged": flagged,
        "vulnerability_rate": flagged / len(rows) if rows else 0.0,
        "bandit_total_issue_instances": len(report.get("results") or []),
        "max_input_tokens": max_input_tokens,
        "generations_jsonl": str(gen_path),
        "bandit_json": str(bandit_json),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
