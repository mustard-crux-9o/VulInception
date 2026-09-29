"""
GRPO training script for code vulnerability unlearning.

Usage:
    # Pile dataset + gpt-j-6b
    python purge_code.py dataset=pile model=gpt_j_6b

    # Stack dataset + starcoder
    python purge_code.py dataset=stack model=starcoder

    # Fast testing
    python purge_code.py dataset=pile model=gpt_j_6b training=code_fast

    # Resume from checkpoint after crash / OOM
    python purge_code.py dataset=pile model=gpt_j_6b resume_from_checkpoint=true
    python purge_code.py dataset=pile model=gpt_j_6b resume_from_checkpoint=/path/to/checkpoint-500

    # Override parameters
    python purge_code.py dataset=pile model=gpt_j_6b training.num_epochs=10

    # Multi-run sweep
    python purge_code.py --multirun model=gpt_j_6b,gpt_neox_20b dataset=pile
"""
import os
import sys

# DeepSpeed probes nvcc via CUDA_HOME; point it at the conda env if the
# system default (/usr/local/cuda) doesn't ship nvcc.
if "CUDA_HOME" not in os.environ:
    _conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
    _candidate = os.path.join(_conda_prefix, "bin", "nvcc")
    if os.path.isfile(_candidate):
        os.environ["CUDA_HOME"] = _conda_prefix


def configure_vllm_for_training(use_vllm: bool) -> None:
    """Align vLLM engine version with TRL colocate mode (vllm >= 0.11)."""
    if not use_vllm:
        return
    # Must be set before TRL lazily imports vLLM during trainer.train().
    os.environ["VLLM_USE_V1"] = "1"


from typing import Any, Union

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import json
import hydra
from omegaconf import DictConfig, OmegaConf

from rewards import (
    RewardFunction,
    CodeBinaryReward,
    CodeExponentialDecayReward,
    CodePerPromptBinaryReward,
    CodePerPromptExpDecayReward,
)
from rewards.base import RewardConfig


class SafeGRPOTrainer(GRPOTrainer):
    """GRPOTrainer subclass that disables gradient checkpointing during
    generation so that ``use_cache=True`` works correctly for models with
    Multi-Query Attention (e.g. StarCoder / GPTBigCode).

    Everything inside ``_generate_and_score_completions`` runs under
    ``torch.no_grad()``, so gradient checkpointing provides no memory
    benefit there anyway.
    """

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        gc_enabled = getattr(self.model, "is_gradient_checkpointing", False)
        if gc_enabled:
            self.model.gradient_checkpointing_disable()
        try:
            return super()._generate_and_score_completions(inputs)
        finally:
            if gc_enabled:
                self.model.gradient_checkpointing_enable()


CODE_REWARD_CLASSES = {
    "code_binary": CodeBinaryReward,
    "code_exponential_decay": CodeExponentialDecayReward,
    "code_per_prompt_binary": CodePerPromptBinaryReward,
    "code_per_prompt_exp_decay": CodePerPromptExpDecayReward,
}


def get_reward_class(reward_type: str) -> type[RewardFunction]:
    if reward_type not in CODE_REWARD_CLASSES:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Available: {list(CODE_REWARD_CLASSES.keys())}"
        )
    return CODE_REWARD_CLASSES[reward_type]


def load_model_and_tokenizer(cfg: DictConfig):
    print(f"Loading model: {cfg.model.hf_model_id}")
    attn_impl = cfg.model.get("attn_implementation", None)
    base_load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    device_map = cfg.model.get("device_map", None)
    if device_map:
        base_load_kwargs["device_map"] = device_map
    attn_candidates = [attn_impl, "sdpa", None] if attn_impl else [None]
    last_error = None
    model = None
    for candidate in attn_candidates:
        load_kwargs = dict(base_load_kwargs)
        if candidate:
            load_kwargs["attn_implementation"] = candidate
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.hf_model_id, **load_kwargs,
            )
            if candidate:
                print(f"Using attention implementation: {candidate}")
            break
        except Exception as exc:
            last_error = exc
            if candidate:
                print(f"Failed attention implementation {candidate}: {exc}")
            else:
                raise
    if model is None:
        raise last_error
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.hf_model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


@hydra.main(version_base=None, config_path="configs", config_name="config_code")
def main(cfg: DictConfig) -> None:
    print("\n" + "=" * 60)
    print("CODE VULNERABILITY UNLEARNING")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load preprocessed forget dataset
    print(f"Loading dataset from: {cfg.paths.forget_dataset_file}")
    with open(cfg.paths.forget_dataset_file, "r") as f:
        raw_data = json.load(f)
    # Training only needs prompt (and optional cve_id); drop eval-only keys e.g. ground_truth_suffix.
    data = [
        {"prompt": row["prompt"], "cve_id": row.get("cve_id")}
        for row in raw_data
    ]

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Setup reward
    reward_class = get_reward_class(cfg.reward.type)
    extra_params = {k: v for k, v in cfg.reward.items() if k != "type"}
    reward_config = RewardConfig(
        target_entity=cfg.dataset.name,
        forget_words_file=cfg.paths.forget_patterns_file,
        forget_dataset_file=cfg.paths.forget_dataset_file,
        model=model,
        tokenizer=tokenizer,
        extra_params=extra_params if extra_params else None,
    )

    print(f"\n{'=' * 60}")
    print(f"Using reward function: {reward_class.__name__}")
    print(f"{'=' * 60}\n")

    reward_class.preprocess(reward_config)

    # Prepare dataset (only needs "prompt" column for GRPOTrainer)
    dataset = Dataset.from_list(data)
    if cfg.training.dataset_size is not None:
        dataset = dataset.select(
            range(min(cfg.training.dataset_size, len(dataset)))
        )
    print(f"Dataset size: {len(dataset)} samples")

    use_vllm = bool(cfg.training.get("use_vllm", False))
    configure_vllm_for_training(use_vllm)

    # Training configuration
    training_args = GRPOConfig(
        output_dir=cfg.paths.output_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_generations=cfg.training.num_generations,
        logging_steps=cfg.training.logging_steps,
        max_steps=cfg.training.get("max_steps", -1),
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        max_completion_length=cfg.model.max_completion_length,
        bf16=True,
        gradient_checkpointing=cfg.model.get("gradient_checkpointing", True),
        max_prompt_length=cfg.training.get("max_prompt_length", 256),
        optim=cfg.training.get("optim", "adamw_bnb_8bit"),
        save_only_model=cfg.training.get("save_only_model", False),
        use_vllm=use_vllm,
        vllm_mode=cfg.training.get("vllm_mode", "server"),
        vllm_tensor_parallel_size=cfg.training.get("vllm_tensor_parallel_size", 1),
        vllm_gpu_memory_utilization=cfg.training.get("vllm_gpu_memory_utilization", 0.3),
        vllm_enable_sleep_mode=cfg.training.get("vllm_enable_sleep_mode", False),
        dataloader_num_workers=cfg.training.get("dataloader_num_workers", 0),
        dataloader_pin_memory=cfg.training.get("dataloader_pin_memory", False),
        ddp_find_unused_parameters=False,
    )

    trainer = SafeGRPOTrainer(
        model=model,
        reward_funcs=reward_class.calc_reward,
        args=training_args,
        train_dataset=dataset,
    )

    # Resolve checkpoint for resumption
    resume_ckpt = cfg.get("resume_from_checkpoint", None)
    if resume_ckpt is True:
        # Auto-detect latest checkpoint in output_dir
        ckpt_dir = cfg.paths.output_dir
        if os.path.isdir(ckpt_dir):
            ckpts = sorted(
                [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[-1]),
            )
            if ckpts:
                resume_ckpt = os.path.join(ckpt_dir, ckpts[-1])
                print(f"Auto-detected checkpoint: {resume_ckpt}")
            else:
                resume_ckpt = None
                print("No checkpoint found, training from scratch.")
        else:
            resume_ckpt = None
    elif isinstance(resume_ckpt, str) and os.path.isdir(resume_ckpt):
        print(f"Resuming from: {resume_ckpt}")
    else:
        resume_ckpt = None

    print("Started training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    print("Finished training.")

    # Save final model
    final_dir = cfg.paths.output_dir + "/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final model to: {final_dir}")


if __name__ == "__main__":
    main()
