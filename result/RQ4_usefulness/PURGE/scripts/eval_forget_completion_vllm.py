#!/usr/bin/env python3
"""
Forget-set eval using vLLM (much faster than sequential HF generate for large N).

Same metrics and output layout as eval_forget_completion.py (generations.jsonl + summary.json).

Example:
  CUDA_VISIBLE_DEVICES=0 python eval_forget_completion_vllm.py \\
    --per_entry ../data/CODE/pile/per_entry_forget.json \\
    --model_path /path/to/EleutherAI_gpt-j-6b \\
    --output_dir ../eval_outputs/pile_gptj_vllm \\
    --max_new_tokens 256 --dtype half

  # StarCoder / multi-GPU:
  CUDA_VISIBLE_DEVICES=0,1 python eval_forget_completion_vllm.py ... \\
    --trust_remote_code --dtype bfloat16 --tensor_parallel_size 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "1")

from transformers import AutoConfig, AutoTokenizer

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from rewards.code_reward import _normalize  # noqa: E402

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise SystemExit(
        "vLLM is required. Install with: pip install vllm\n" f"Original error: {e}"
    ) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forget-set completion eval (vLLM)")
    p.add_argument("--per_entry", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="vLLM dtype: auto, half, float16, bfloat16, float",
    )
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=0, help="0 = from model config")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    return p.parse_args()


def _model_context_len(model_path: str, trust_remote_code: bool) -> int:
    cfg = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    return int(
        getattr(cfg, "max_position_embeddings", None)
        or getattr(cfg, "n_positions", None)
        or 2048
    )


def main() -> None:
    args = parse_args()
    per_path = Path(args.per_entry)
    entries = json.loads(per_path.read_text(encoding="utf-8"))
    end = len(entries) if args.limit <= 0 else min(len(entries), args.start + args.limit)
    subset = entries[args.start : end]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / "generations.jsonl"
    summary_path = out_dir / "summary.json"

    ctx = _model_context_len(args.model_path, args.trust_remote_code)
    max_model_len = args.max_model_len if args.max_model_len > 0 else ctx
    max_model_len = min(max_model_len, ctx)

    tok = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    max_input_tokens = max(64, max_model_len - args.max_new_tokens - 8)

    prompts_in: list[str] = []
    for row in subset:
        p = row["prompt"]
        enc = tok(
            p,
            truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=False,
        )
        truncated = tok.decode(enc["input_ids"], skip_special_tokens=True)
        prompts_in.append(truncated)

    print(
        f"vLLM load: {args.model_path} | max_model_len={max_model_len} | "
        f"n={len(prompts_in)} | tp={args.tensor_parallel_size}"
    )
    llm = LLM(
        model=args.model_path,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts_in, sp)

    total_hit = 0
    total_matched_lines = 0
    seq_ratios: list[float] = []
    exact_suffix = 0
    n_with_suffix = 0

    with gen_path.open("w", encoding="utf-8") as gen_f:
        for i, (row, vout, prompt_used) in enumerate(
            zip(subset, outputs, prompts_in, strict=True)
        ):
            completion = vout.outputs[0].text
            forget_lines = row.get("forget_lines") or []
            suffix = row.get("ground_truth_suffix") or ""

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
                "prompt": row["prompt"],
                "prompt_as_sent_to_vllm": prompt_used,
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
            if (i + 1) % 50 == 0 or i == 0:
                print(f"[{i + 1}/{len(subset)}] scored")

    n = len(subset)
    summary = {
        "backend": "vllm",
        "vllm_dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "per_entry_file": str(per_path.resolve()),
        "model_path": args.model_path,
        "num_evaluated": n,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": max_model_len,
        "greedy": args.temperature <= 0,
        "forget_line_hit_rate": total_hit / n if n else 0.0,
        "mean_forget_lines_matched": total_matched_lines / n if n else 0.0,
        "samples_with_ground_truth_suffix": n_with_suffix,
        "mean_sequence_similarity": sum(seq_ratios) / len(seq_ratios) if seq_ratios else None,
        "exact_suffix_match_rate": exact_suffix / n_with_suffix if n_with_suffix else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {gen_path}\nWrote {summary_path}\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
