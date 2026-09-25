from __future__ import annotations

import math

import torch
import torch.nn as nn

from .visual import NUM_REGIONS


ORDER_SCALE = 0.3


class EventAdapter(nn.Module):
    """Project the two factorized change components into one EVENT token."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        event_dim: int = 320,
        order_scale: float = ORDER_SCALE,
    ):
        super().__init__()
        if not math.isfinite(order_scale) or order_scale < 0:
            raise ValueError("order_scale must be finite and non-negative")
        self.order_scale = float(order_scale)
        self.exchange_invariant_norm = nn.LayerNorm(
            event_dim, elementwise_affine=False
        )
        self.order_sensitive_norm = nn.LayerNorm(
            event_dim, elementwise_affine=False
        )
        self.exchange_invariant = nn.Linear(event_dim, hidden_dim)
        self.order_sensitive = nn.Linear(event_dim, hidden_dim, bias=False)
        self.out = nn.Linear(2 * hidden_dim, hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        roi_tokens: torch.Tensor,
        exchange_invariant: torch.Tensor,
        order_sensitive: torch.Tensor,
        valid: torch.Tensor,
        has_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if roi_tokens.ndim != 3 or roi_tokens.shape[1] != NUM_REGIONS:
            raise ValueError("roi_tokens must have shape [B, 29, H]")
        if (
            exchange_invariant.shape != order_sensitive.shape
            or exchange_invariant.shape[:2] != roi_tokens.shape[:2]
        ):
            raise ValueError("factorized change slots must have shape [B, 29, D]")
        if valid.shape != exchange_invariant.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean [B, 29]")
        if has_prior.shape != roi_tokens.shape[:1] or has_prior.dtype != torch.bool:
            raise ValueError("has_prior must be boolean [B]")
        mask = valid & has_prior.unsqueeze(1)
        projected_invariant = self.exchange_invariant(
            self.exchange_invariant_norm(exchange_invariant)
        )
        projected_order = self.order_scale * self.order_sensitive(
            self.order_sensitive_norm(order_sensitive)
        )
        delta = self.out(torch.cat((projected_invariant, projected_order), dim=-1))
        delta = delta * mask.unsqueeze(-1).to(delta.dtype)
        return roi_tokens + delta, delta, mask


__all__ = ["EventAdapter", "ORDER_SCALE"]
