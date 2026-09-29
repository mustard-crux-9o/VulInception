#!/usr/bin/env python3
"""
Evaluate code completions on the forget set (per_entry_forget.json).

Metrics (aligned with GRPO code rewards):
  - forget_line_hit: any per-entry forget line appears in the normalized *generated*
    completion (same as CodePerPromptBinaryReward).
  - forget_lines_matched: count of forget lines found in the completion.
  - sequence_similarity: difflib ratio between normalized completion and
    normalized ground_truth_suffix (requires ``ground_truth_suffix`` in JSON;
    re-run prepare_code_data.py if missing).

Usage:
  cd .../purge/scripts
  $PYTHON eval_forget_completion.py \\
    --per_entry ../data/CODE/pile/per_entry_forget.json \\
    --model_path /path/to/checkpoint \\
    --output_dir ../eval_outputs/pile_my_run
"""
from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from rewards.code_reward import _normalize  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forget-set completion eval")
    p.add_argument("--per_entry", type=str, required=True, help="per_entry_forget.json")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for generations.jsonl and summary.json (default: ./eval_forget_<model_basename>)",
    )
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    per_path = Path(args.per_entry)
    entries = json.loads(per_path.read_text(encoding="utf-8"))
    end = len(entries) if args.limit <= 0 else min(len(entries), args.start + args.limit)
    subset = entries[args.start : end]

    model_name = Path(args.model_path.rstrip("/")).name
    out_dir = Path(
        args.output_dir
        or (Path(__file__).resolve().parent / f"eval_forget_{model_name}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / "generations.jsonl"
    summary_path = out_dir / "summary.json"

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    max_ctx = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", 2048
    )
    max_input_len = max(64, max_ctx - args.max_new_tokens - 8)

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

    device = next(model.parameters()).device
    rows_out = []
    total_hit = 0
    total_matched_lines = 0
    seq_ratios: list[float] = []
    exact_suffix = 0
    n_with_suffix = 0

    with gen_path.open("w", encoding="utf-8") as gen_f:
        for i, row in enumerate(subset):
            prompt = row["prompt"]
            forget_lines = row.get("forget_lines") or []
            suffix = row.get("ground_truth_suffix")
            if suffix is None:
                suffix = ""

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            input_len = enc["input_ids"].shape[1]

            with torch.inference_mode():
                out_ids = model.generate(**enc, **gen_kwargs)

            new_tokens = out_ids[0, input_len:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

            norm_comp = _normalize(completion)
            norm_lines = [_normalize(l) for l in forget_lines if l.strip()]
            matched = [p for p in norm_lines if p in norm_comp]
            hit = len(matched) > 0
            if hit:
                total_hit += 1
            total_matched_lines += len(matched)

            rec = {
                "index": args.start + i,
                "cve_id": row.get("cve_id"),
                "prompt": prompt,
                "completion": completion,
                "forget_line_hit": hit,
                "forget_lines_matched_count": len(matched),
                "forget_lines_matched": matched,
            }

            if suffix.strip():
                n_with_suffix += 1
                ns = _normalize(suffix)
                ratio = SequenceMatcher(None, norm_comp, ns).ratio()
                seq_ratios.append(ratio)
                rec["ground_truth_suffix"] = suffix
                rec["sequence_similarity"] = ratio
                rec["exact_suffix_match"] = completion.strip() == suffix.strip()
                if rec["exact_suffix_match"]:
                    exact_suffix += 1
            else:
                rec["ground_truth_suffix"] = None
                rec["sequence_similarity"] = None
                rec["exact_suffix_match"] = None

            gen_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows_out.append(rec)
            print(
                f"[{i + 1}/{len(subset)}] {row.get('cve_id')} "
                f"hit={hit} lines={len(matched)}"
                + (
                    f" sim={rec['sequence_similarity']:.3f}"
                    if rec["sequence_similarity"] is not None
                    else ""
                )
            )

    n = len(subset)
    summary = {
        "per_entry_file": str(per_path.resolve()),
        "model_path": args.model_path,
        "num_evaluated": n,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "greedy": args.temperature <= 0,
        "forget_line_hit_rate": total_hit / n if n else 0.0,
        "mean_forget_lines_matched": total_matched_lines / n if n else 0.0,
        "samples_with_ground_truth_suffix": n_with_suffix,
        "mean_sequence_similarity": sum(seq_ratios) / len(seq_ratios) if seq_ratios else None,
        "exact_suffix_match_rate": exact_suffix / n_with_suffix if n_with_suffix else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {gen_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
