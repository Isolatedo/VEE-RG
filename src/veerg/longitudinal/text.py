from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class FrozenTextEncoder(nn.Module):
    """Frozen text encoder with a trainable output projection."""

    def __init__(self, path: str, dim: int = 256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.bert = AutoModel.from_pretrained(path)
        self.proj = nn.Linear(self.bert.config.hidden_size, dim)
        self.dim = dim
        self.bert.eval()
        for parameter in self.bert.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.bert.eval()
        return self

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        if not texts:
            return torch.empty(0, self.dim, device=device)
        batch = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            hidden = self.bert(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return self.proj(
            (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        )


__all__ = ["FrozenTextEncoder"]
