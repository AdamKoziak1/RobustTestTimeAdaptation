import torch
import torch.nn as nn


class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.sum(-probs * torch.log(probs + 1e-8), dim=-1)


class EntropyClassMarginals(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.dim() == 3:
            probs = torch.mean(probs, dim=1)
        avg_p = probs.mean(dim=0)
        return torch.sum(avg_p * torch.log(avg_p + 1e-8))
