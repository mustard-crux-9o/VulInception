# Evaluation benchmarks

```
benchmarks/
  vmi_bench_pile.yaml
  vmi_bench_stack.yaml
  securityeval/dataset.jsonl
  humaneval/                   # pip install -e benchmarks/humaneval
  predictions/gotcha_*.jsonl
  predictions/tool_*.jsonl
  unseen2026_cve/README.md     # not bundled — see VulInception
```

Unseen 2026 CVE eval uses the VulInception benchmark at `../Unseen2026CVE/dataset.jsonl` (same level as this repo).

```bash
pip install -e benchmarks/humaneval
cp scripts/env.example.sh scripts/env.sh && source scripts/env.sh
bash scripts/prepare_all_forget_sets.sh
bash scripts/run_rq4_usefulness_eval.sh gpt_j_6b
```

SecurityEval lines: `{"ID": "...", "Prompt": "..."}`. VR = Bandit finding rate on prompt + completion.
