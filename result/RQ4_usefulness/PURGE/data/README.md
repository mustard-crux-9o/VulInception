# Training / forget-set data layout

Large artifacts are **not** included in this repository. After running
`scripts/prepare_all_forget_sets.sh`, expect:

```
data/
  CODE/
    pile/                        # VulInception forget set (GPT-J-6B)
      forget_dataset.json
      forget_patterns.json
      per_entry_forget.json
    stack/                       # VulInception forget set (StarCoderBase-7B)
      ...
  forget_sets/
    pile_random/                 # Random selector (same scale as VulInception)
    pile_gotcha/                 # Gotcha baseline selector
    pile_groundtruth/            # Human-labeled GT selector (pre.label == 1)
    stack_random/
    stack_gotcha/
    stack_groundtruth/
```

Each forget-set directory contains the same four JSON files produced by the
preprocessors in `scripts/`.

## Hydra dataset → path mapping

| Paper selector | Hydra `dataset=` | Data directory |
|----------------|------------------|----------------|
| VulInception | `pile` / `stack` | `data/CODE/pile` or `data/CODE/stack` |
| Random | `pile_random` / `stack_random` | `data/forget_sets/pile_random` … |
| Gotcha | `pile_gotcha` / `stack_gotcha` | `data/forget_sets/pile_gotcha` … |
| Groundtruth | `pile_groundtruth` / `stack_groundtruth` | `data/forget_sets/pile_groundtruth` … |

Use `paths=code_default` for VulInception sets and `paths=code_forget_sets` for
the other selectors (see `src/configs/paths/`).

## Trained models

Checkpoints are written to `models/code/<model>-<dataset>-code_binary/final/`.
These are not shipped; release separately when ready.
