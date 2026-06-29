"""Model definitions for the dual-branch (prob stats + code pooled) classifier."""
from typing import List
import torch.nn as nn

from dataset import RICH_PROB_FEATURE_DIM


class RichProbLinearClassifier(nn.Module):
    """Linear classifier on 66-dim pre/post/difference p(t) * tau(t) statistics."""
    def __init__(self, input_dim: int = RICH_PROB_FEATURE_DIM):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


class CodePooledMLP(nn.Module):
    """MLP classifier on concatenated mean-pooled code embeddings."""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 64], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DualBranchModel(nn.Module):
    """Dual-branch: prob stats linear + code pooled MLP, fused by alpha."""
    def __init__(self, code_input_dim: int, code_hidden_dims: List[int] = [256, 64], dropout: float = 0.2):
        super().__init__()
        self.prob_mlp = RichProbLinearClassifier(input_dim=RICH_PROB_FEATURE_DIM)
        self.code_mlp = CodePooledMLP(input_dim=code_input_dim * 4, hidden_dims=code_hidden_dims, dropout=dropout)

    def forward_prob(self, prob_stats):
        return self.prob_mlp(prob_stats)

    def forward_code(self, code_pooled):
        return self.code_mlp(code_pooled)
