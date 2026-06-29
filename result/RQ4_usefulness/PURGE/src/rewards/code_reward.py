"""
Code-specific reward functions for GRPO-based vulnerability unlearning.

Two modes:
  - Global: uses a global set of vulnerability code patterns (like original PURGE BinaryReward).
  - PerPrompt: looks up per-prompt forget lines for precise, entry-level matching.

Both support binary and exponential-decay scoring.
"""
import json
import math
from typing import Dict, List, Optional, Set

from .base import RewardFunction, RewardConfig


def _normalize(text: str) -> str:
    """Collapse all whitespace runs to a single space and strip."""
    return " ".join(text.split())


class CodeBinaryReward(RewardFunction):
    """
    Global code-line binary reward.

    Maintains a list of vulnerability code patterns (normalised lines).
    A completion is penalised (reward=0) if ANY pattern appears as a
    substring in the normalised completion text.
    """

    _forget_patterns: List[str] = []
    _min_length: int = 10
    _preprocessed: bool = False

    @classmethod
    def preprocess(cls, config: RewardConfig) -> None:
        with open(config.forget_words_file, "r") as f:
            patterns = json.load(f)
        cls._forget_patterns = [
            _normalize(p) for p in patterns
            if len(p.strip()) >= cls._min_length
        ]
        # Deduplicate while preserving order
        seen: set = set()
        deduped: List[str] = []
        for p in cls._forget_patterns:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        cls._forget_patterns = deduped
        cls._preprocessed = True
        print(f"[CodeBinaryReward] Loaded {len(cls._forget_patterns)} forget patterns")

    @staticmethod
    def calc_reward(completions: List[str], **kwargs) -> List[float]:
        patterns = CodeBinaryReward._forget_patterns
        rewards: List[float] = []
        for completion in completions:
            norm_comp = _normalize(completion)
            found = any(p in norm_comp for p in patterns)
            rewards.append(0.0 if found else 1.0)
        return rewards


class CodeExponentialDecayReward(RewardFunction):
    """
    Global code-line exponential-decay reward.

    reward = base^{-count / tau}  where count = number of matched vulnerability lines.
    """

    _tau: float = 1.0
    _base: float = math.e
    _preprocessed: bool = False

    @classmethod
    def preprocess(cls, config: RewardConfig) -> None:
        CodeBinaryReward.preprocess(config)
        if config.extra_params:
            cls._tau = config.extra_params.get("tau", 1.0)
            cls._base = config.extra_params.get("base", math.e)
        cls._preprocessed = True

    @staticmethod
    def calc_reward(completions: List[str], **kwargs) -> List[float]:
        patterns = CodeBinaryReward._forget_patterns
        tau = CodeExponentialDecayReward._tau
        base = CodeExponentialDecayReward._base
        rewards: List[float] = []
        for completion in completions:
            norm_comp = _normalize(completion)
            match_count = sum(1 for p in patterns if p in norm_comp)
            reward = base ** (-(match_count / tau)) if base > 0 else 0.0
            rewards.append(reward)
        return rewards


# ---------------------------------------------------------------------------
# Per-prompt variants: each prompt has its own set of forget lines.
# Requires `per_entry_forget.json` (saved by prepare_code_data.py).
# ---------------------------------------------------------------------------

class CodePerPromptBinaryReward(RewardFunction):
    """
    Per-prompt binary reward.

    Looks up the forget lines specific to the prompt that generated the
    completion, avoiding false positives from unrelated vulnerability patterns.
    Falls back to global matching when the prompt cannot be resolved.
    """

    _prompt_map: Dict[str, Set[str]] = {}
    _global_patterns: Set[str] = set()
    _preprocessed: bool = False

    @classmethod
    def preprocess(cls, config: RewardConfig) -> None:
        per_entry_file = config.forget_words_file.replace(
            "forget_patterns.json", "per_entry_forget.json"
        )
        with open(per_entry_file, "r") as f:
            per_entry_data = json.load(f)

        cls._prompt_map = {}
        cls._global_patterns: List[str] = []
        all_pats: set = set()
        for entry in per_entry_data:
            key = _normalize(entry["prompt"])
            lines = [_normalize(l) for l in entry["forget_lines"] if l.strip()]
            cls._prompt_map[key] = lines
            all_pats.update(lines)
        cls._global_patterns = list(all_pats)

        cls._preprocessed = True
        print(
            f"[CodePerPromptBinaryReward] Loaded {len(cls._prompt_map)} prompt mappings, "
            f"{len(cls._global_patterns)} global patterns"
        )

    @staticmethod
    def _lookup(prompt: str) -> Optional[List[str]]:
        key = _normalize(prompt)
        if key in CodePerPromptBinaryReward._prompt_map:
            return CodePerPromptBinaryReward._prompt_map[key]
        prefix = key[:80]
        for k, v in CodePerPromptBinaryReward._prompt_map.items():
            if k.startswith(prefix):
                return v
        return None

    @staticmethod
    def calc_reward(completions: List[str], **kwargs) -> List[float]:
        prompts = kwargs.get("prompts", [])
        rewards: List[float] = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else ""
            forget_lines = CodePerPromptBinaryReward._lookup(prompt)
            if forget_lines is None:
                forget_lines = CodePerPromptBinaryReward._global_patterns
            norm_comp = _normalize(completion)
            found = any(p in norm_comp for p in forget_lines)
            rewards.append(0.0 if found else 1.0)
        return rewards


class CodePerPromptExpDecayReward(RewardFunction):
    """Per-prompt exponential-decay variant."""

    _tau: float = 1.0
    _base: float = math.e
    _preprocessed: bool = False

    @classmethod
    def preprocess(cls, config: RewardConfig) -> None:
        CodePerPromptBinaryReward.preprocess(config)
        if config.extra_params:
            cls._tau = config.extra_params.get("tau", 1.0)
            cls._base = config.extra_params.get("base", math.e)
        cls._preprocessed = True

    @staticmethod
    def calc_reward(completions: List[str], **kwargs) -> List[float]:
        prompts = kwargs.get("prompts", [])
        rewards: List[float] = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else ""
            forget_lines = CodePerPromptBinaryReward._lookup(prompt)
            if forget_lines is None:
                forget_lines = CodePerPromptBinaryReward._global_patterns
            norm_comp = _normalize(completion)
            match_count = sum(1 for p in forget_lines if p in norm_comp)
            tau = CodePerPromptExpDecayReward._tau
            base = CodePerPromptExpDecayReward._base
            reward = base ** (-(match_count / tau)) if base > 0 else 0.0
            rewards.append(reward)
        return rewards
