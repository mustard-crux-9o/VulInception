# PURGE — Reinforcement Unlearning for Code Vulnerability Memorization

<p align="center">
  <img src="assets/purge.png" alt="PURGE teaser" width="900"/>
</p>

This repository extends [PURGE](https://arxiv.org/abs/2601.20568) (Reinforcement Unlearning via Group Relative Policy Optimization) with **code vulnerability unlearning** for the RQ4 *Usefulness* experiments in our ICSE study.

PURGE uses GRPO to unlearn memorised vulnerable code patterns while a binary reward penalises regenerating known vulnerability lines.

## Overview (RQ4)

We evaluate whether vulnerability members identified by our tool (**VulInception**) form a precise **forget set** for machine unlearning, compared to:

| Forget-set selector | Description |
|---------------------|-------------|
| **VulInception** | Members flagged by our vulnerability-MIA tool |
| **Random** | Random vulnerable functions at the same scale |
| **Gotcha** | Gotcha MIA baseline |
| **Groundtruth (GT)** | Human-labeled oracle (`pre.label == 1`) |

**Target models:** GPT-J-6B (The Pile) and StarCoderBase-7B (The Stack)

**Metrics:**

- **VMI-Bench VR** — SecurityEval prefixes, Bandit on greedy completions
- **Unseen 2026 CVE VR** — VulInception benchmark (`../Unseen2026CVE/`), Bandit
- **HumanEval pass@1** — general coding capability (greedy)

## Quick start

```bash
conda create -n purge python=3.10
conda activate purge
pip install -r requirements.txt
```

### 1. Prepare benchmarks & forget sets

Benchmarks are included under [`benchmarks/`](benchmarks/README.md). Install the HumanEval evaluator once:

```bash
pip install -e benchmarks/humaneval
bash scripts/prepare_all_forget_sets.sh
```

### 2. Train PURGE on a forget set

```bash
# All RQ4 selectors for one model
bash scripts/run_rq4_train_all.sh gpt_j_6b

# Single run: VulInception + GPT-J-6B
bash scripts/run_train_forget_set.sh pile gpt_j_6b

# Random + StarCoderBase-7B
bash scripts/run_train_forget_set.sh stack_random starcoderbase_7b paths=code_forget_sets

# Faster training with vLLM (single A100)
TRAINING=code_a100_single bash scripts/run_train_forget_set.sh stack_random starcoderbase_7b paths=code_forget_sets
```

Checkpoints: `models/code/<model>-<dataset>-code_binary/final/`

### 3. Evaluate (RQ4 table)

```bash
cp scripts/env.example.sh scripts/env.sh   # edit paths
source scripts/env.sh

bash scripts/run_rq4_usefulness_eval.sh gpt_j_6b
bash scripts/run_rq4_usefulness_eval.sh starcoderbase_7b
```

Results: `eval_outputs/rq4_usefulness/<model>/usefulness_table.csv`

Full walkthrough: [`docs/RQ4_usefulness.md`](docs/RQ4_usefulness.md)

## Repository layout

```
src/
  purge_code.py           # GRPO code unlearning (Hydra + TRL)
  prepare_code_data.py    # VMI-Bench YAML → forget JSON (VulInception set)
  rewards/code_reward.py    # Binary / per-prompt vulnerability rewards
  configs/                  # Hydra: model, dataset, training, paths
scripts/
  prepare_*_forget.py     # Forget-set builders (Random / Gotcha / GT / tool)
  eval_securityeval_*     # SecurityEval + Bandit (VR)
  eval_humaneval_*        # HumanEval pass@1
  build_unseen_cve_2026.py
  run_rq4_train_all.sh      # Train all forget-set selectors
  run_rq4_usefulness_eval.sh
data/                     # Generated forget sets (see data/README.md)
benchmarks/               # VMI-Bench, SecurityEval, HumanEval (Unseen2026 → VulInception)
docs/RQ4_usefulness.md    # Paper-aligned reproduction guide
```

Original entity-unlearning code (`src/purge.py`) is preserved unchanged.

## Citation

```bibtex
@article{zaradoukas2026reinforcement,
  title={Reinforcement Unlearning via Group Relative Policy Optimization},
  author={Zaradoukas, Efstratios and Prenkaj, Bardh and Kasneci, Gjergji},
  journal={arXiv preprint arXiv:2601.20568},
  year={2026}
}
```

## License

MIT License (see upstream PURGE).
