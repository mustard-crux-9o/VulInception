# Code vulnerability unlearning

See the main [README](../README.md) and [RQ4 reproduction guide](../docs/RQ4_usefulness.md).

Quick reference:

```bash
# Prepare forget sets
bash scripts/prepare_all_forget_sets.sh

# Train
bash scripts/run_train_forget_set.sh pile gpt_j_6b
bash scripts/run_train_forget_set.sh stack_random starcoderbase_7b paths=code_forget_sets

# Evaluate
source scripts/env.sh
bash scripts/run_rq4_usefulness_eval.sh gpt_j_6b
```

Hydra overrides (OOM): `model.max_completion_length=128 training.num_generations=2`
