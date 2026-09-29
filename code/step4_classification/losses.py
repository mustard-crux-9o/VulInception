"""Loss functions: CrossEntropy, Focal, and Asymmetric (ASL)."""
import torch
from torch import nn


class CELoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', weight=weight)

    def forward(self, logits, labels):
        ce = self.ce_loss(logits, labels)
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class ASLLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, weight=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits, labels):
        num_classes = logits.size(1)
        targets = torch.zeros_like(logits)
        targets.scatter_(1, labels.unsqueeze(1), 1)
        probs = torch.softmax(logits, dim=1).clamp(self.clip, 1 - self.clip)
        pos_mask = targets == 1
        neg_mask = targets == 0
        pos_loss = -(targets[pos_mask] * torch.log(probs[pos_mask]))
        neg_loss = -((1 - targets[neg_mask]) * torch.log(1 - probs[neg_mask]))
        pos_loss = (1 - probs[pos_mask]) ** self.gamma_pos * pos_loss
        neg_loss = (probs[neg_mask]) ** self.gamma_neg * neg_loss
        return torch.cat([pos_loss, neg_loss]).mean()
