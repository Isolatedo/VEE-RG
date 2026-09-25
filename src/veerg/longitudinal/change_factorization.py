from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _dropout(value: float) -> float:
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return value


class ExchangeInvariantEncoder(nn.Module):
    def __init__(self, dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.network = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=False),
        )
        self.dropout = nn.Dropout(_dropout(dropout))

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.ndim != 3 or delta.shape[-1] != self.dim:
            raise ValueError(f"delta must have shape [B, R, {self.dim}]")
        hidden = self.network[1](self.network[0](delta.abs()))
        return self.network[2](self.dropout(hidden))


class OrderSensitiveEncoder(nn.Module):
    def __init__(self, dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.network = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False),
            nn.Tanh(),
            nn.Linear(dim * 2, dim, bias=False),
        )
        self.dropout = nn.Dropout(_dropout(dropout))

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.ndim != 3 or delta.shape[-1] != self.dim:
            raise ValueError(f"delta must have shape [B, R, {self.dim}]")
        hidden = self.network[1](self.network[0](delta))
        return self.network[2](self.dropout(hidden))


class OrderedContextEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.network = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.dropout = nn.Dropout(_dropout(dropout))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape or first.ndim != 3:
            raise ValueError("first and second must have the same [B, R, D] shape")
        if first.shape[-1] != self.dim:
            raise ValueError(f"expected feature dim {self.dim}")
        value = torch.cat((first, second), dim=-1)
        hidden = self.network[2](self.network[1](self.network[0](value)))
        return self.network[3](self.dropout(hidden))


class ChangeHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(_dropout(dropout))

    def _hidden(self, event: torch.Tensor) -> torch.Tensor:
        return self.network[2](self.network[1](self.network[0](event)))

    def forward(self, event: torch.Tensor) -> torch.Tensor:
        return self.network[3](self.dropout(self._hidden(event))).squeeze(-1)


class DirectionHead(ChangeHead):
    def forward(self, event: torch.Tensor) -> torch.Tensor:
        positive = self._hidden(event)
        negative = self._hidden(-event)
        mask = F.dropout(
            torch.ones_like(positive), p=self.dropout.p, training=self.training
        )
        return 0.5 * (
            self.network[3](positive * mask).squeeze(-1)
            - self.network[3](negative * mask).squeeze(-1)
        )


class ChangeFactorization(nn.Module):
    """Factorize paired regional features into order-free and order-sensitive parts."""

    def __init__(
        self,
        dim: int = 256,
        hidden_dim: int = 512,
        context_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.magnitude_encoder = ExchangeInvariantEncoder(dim, dropout)
        self.direction_encoder = OrderSensitiveEncoder(dim, dropout)
        self.ordered_context = OrderedContextEncoder(
            dim, hidden_dim, context_dim, dropout
        )
        event_dim = dim + context_dim
        head_hidden_dim = max(dim // 2, context_dim)
        self.change_head = ChangeHead(event_dim, head_hidden_dim, dropout)
        self.direction_head = DirectionHead(event_dim, head_hidden_dim, dropout)

    def forward(
        self,
        prior: torch.Tensor,
        current: torch.Tensor,
        pair_seen: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if prior.shape != current.shape or prior.ndim != 3:
            raise ValueError("prior and current must have the same [B, R, D] shape")
        if prior.shape[-1] != self.dim:
            raise ValueError(f"expected feature dim {self.dim}")
        if pair_seen is not None and pair_seen.shape != prior.shape[:2]:
            raise ValueError("pair_seen must have shape [B, R]")

        delta = current - prior
        magnitude = self.magnitude_encoder(delta)
        direction = self.direction_encoder(delta)
        prior_current = self.ordered_context(prior, current)
        current_prior = self.ordered_context(current, prior)
        symmetric = 0.5 * (prior_current + current_prior)
        antisymmetric = 0.5 * (prior_current - current_prior)
        exchange_invariant = torch.cat((magnitude, symmetric), dim=-1)
        order_sensitive = torch.cat((direction, antisymmetric), dim=-1)
        change_logit = self.change_head(exchange_invariant)
        direction_logit = self.direction_head(order_sensitive)
        state_log_prob = torch.stack(
            (
                F.logsigmoid(-change_logit),
                F.logsigmoid(change_logit) + F.logsigmoid(direction_logit),
                F.logsigmoid(change_logit) + F.logsigmoid(-direction_logit),
            ),
            dim=-1,
        )
        state_prob = state_log_prob.exp()
        state_pred = torch.where(
            change_logit >= 0,
            torch.where(direction_logit >= 0, 1, 2),
            0,
        )
        return {
            "delta": delta,
            "magnitude": magnitude,
            "direction": direction,
            "u_pc": prior_current,
            "u_cp": current_prior,
            "symmetric_context": symmetric,
            "antisymmetric_context": antisymmetric,
            "exchange_invariant": exchange_invariant,
            "order_sensitive": order_sensitive,
            "event": torch.cat((exchange_invariant, order_sensitive), dim=-1),
            "change_logit": change_logit,
            "direction_logit": direction_logit,
            "state_log_prob": state_log_prob,
            "state": state_prob,
            "state_prob": state_prob,
            "state_pred": state_pred,
        }


__all__ = ["ChangeFactorization"]
