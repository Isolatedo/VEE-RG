from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    text: float = 0.4
    change: float = 0.8
    direction: float = 0.45
    spatial: float = 0.15
    temperature: float = 0.1
    label_smoothing: float = 0.05
    multilabel_smoothing: float = 0.1

    def __post_init__(self) -> None:
        weights = (
            self.text,
            self.change,
            self.direction,
            self.spatial,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("loss weights must be finite and non-negative")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        smoothing = (self.label_smoothing, self.multilabel_smoothing)
        if any(
            not math.isfinite(value) or not 0 <= value < 0.5
            for value in smoothing
        ):
            raise ValueError("label smoothing must be finite and in [0, 0.5)")
        if self.multilabel_smoothing < self.label_smoothing:
            raise ValueError(
                "multilabel_smoothing must be at least label_smoothing"
            )


@dataclass
class LossOutput:
    total: torch.Tensor
    text: torch.Tensor
    change: torch.Tensor
    direction: torch.Tensor
    spatial: torch.Tensor


def unpack_states(states: torch.Tensor) -> dict[str, torch.Tensor]:
    no_change, better, worse = states.bool().unbind(-1)
    labeled = no_change | better | worse
    changed = better | worse
    clear = better ^ worse
    state = torch.where(
        better & ~worse,
        torch.ones_like(better, dtype=torch.long),
        torch.where(
            worse & ~better, torch.full_like(worse, 2, dtype=torch.long), 0
        ),
    )
    return {
        "state_target": state,
        "changed": changed,
        "change_mask": labeled,
        "direction_mask": clear,
        "state_mask": (no_change & ~better & ~worse) | clear,
        "clear_changed": clear,
        "multilabel": states.bool().sum(-1) > 1,
    }


def _masked_bce(
    logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if mask.any():
        return F.binary_cross_entropy_with_logits(
            logit[mask], target[mask].to(logit.dtype)
        )
    return logit.new_zeros(())


def _smooth_binary_target(
    hard_target: torch.Tensor,
    multilabel: torch.Tensor,
    weights: LossWeights,
) -> torch.Tensor:
    epsilon = torch.where(
        multilabel,
        torch.as_tensor(weights.multilabel_smoothing, device=hard_target.device),
        torch.as_tensor(weights.label_smoothing, device=hard_target.device),
    )
    return torch.where(hard_target, 1.0 - epsilon, epsilon)


def phase1_loss(
    text_loss: torch.Tensor,
    change_logit: torch.Tensor,
    direction_logit: torch.Tensor,
    target: dict[str, torch.Tensor],
    weights: LossWeights,
    spatial: torch.Tensor | None = None,
) -> LossOutput:
    change_target = _smooth_binary_target(
        target["changed"], target["multilabel"], weights
    )
    direction_target = _smooth_binary_target(
        target["state_target"].eq(1), target["multilabel"], weights
    )
    change = _masked_bce(change_logit, change_target, target["change_mask"])
    direction = _masked_bce(
        direction_logit,
        direction_target,
        target["direction_mask"],
    )
    if spatial is None:
        spatial = change_logit.sum() * 0.0
    total = (
        weights.text * text_loss
        + weights.change * change
        + weights.direction * direction
        + weights.spatial * spatial
    )
    return LossOutput(total, text_loss, change, direction, spatial)


__all__ = ["LossOutput", "LossWeights", "phase1_loss", "unpack_states"]
