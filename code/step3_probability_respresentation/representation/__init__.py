"""Step3 statement normalization, abstraction, and feature preparation."""

from .abstraction import PrefixAwareAbstractor
from .normalization import (
    NormalizationError,
    normalize_statement,
    normalize_statements,
)

__all__ = [
    "NormalizationError",
    "PrefixAwareAbstractor",
    "normalize_statement",
    "normalize_statements",
]
