from __future__ import annotations

import torch
import torch.nn as nn

from .contrastive import (
    Projection,
    group_events,
    info_nce,
)
from .change_factorization import ChangeFactorization
from .paired_region_stabilization import PairedRegionStabilization, spatial_loss
from .losses import LossWeights, phase1_loss, unpack_states
from .text import FrozenTextEncoder


CHECKPOINT_ARCH = "vee-rg-longitudinal-v1"


class VEEEncoder(nn.Module):
    """Stage-one encoder for paired regional changes and text alignment."""

    def __init__(
        self,
        detector: nn.Module,
        text_path: str,
        loss_weights: LossWeights | dict | None = None,
        image_size: int = 512,
        dim: int = 256,
        hidden_dim: int = 512,
        context_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.detector = detector
        self.paired_region_stabilization = PairedRegionStabilization(dim=dim)
        self.change_factorization = ChangeFactorization(
            dim=dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            dropout=dropout,
        )
        self.text_encoder = FrozenTextEncoder(text_path, dim)
        if isinstance(loss_weights, dict):
            loss_weights = LossWeights(**loss_weights)
        self.loss_weights = loss_weights or LossWeights()
        self.image_proj = Projection(2 * (dim + context_dim), dim)
        self.text_proj = Projection(dim, dim)
        self.image_size = int(image_size)
        self.architecture = CHECKPOINT_ARCH

    def _regions(
        self,
        raw: dict[str, torch.Tensor],
        pair_seen: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.paired_region_stabilization(
            c5=raw["c5"],
            roi=raw["roi"],
            boxes=raw["boxes"],
            pair_seen=pair_seen,
            image_size=(self.image_size, self.image_size),
        )

    def forward(self, batch: dict, compute_loss: bool = True) -> dict:
        prior_raw = self.detector.forward_spatial(batch["prior_image"])
        current_raw = self.detector.forward_spatial(batch["current_image"])
        pair_seen = prior_raw["seen"] & current_raw["seen"]
        prior = self._regions(prior_raw, pair_seen)
        current = self._regions(current_raw, pair_seen)
        encoded = self.change_factorization(
            prior["tokens"], current["tokens"], pair_seen=pair_seen
        )
        output = {
            **encoded,
            "prior_tokens": prior["tokens"],
            "current_tokens": current["tokens"],
            "prior_attention": prior["attention"],
            "current_attention": current["attention"],
            "loss": None,
        }
        if not compute_loss:
            return output

        target = unpack_states(batch["states"])
        groups = group_events(
            encoded["event"],
            target["clear_changed"],
            batch["texts"],
            pool_regions=False,
        )
        text_features = self.text_encoder(groups.text, encoded["event"].device)
        image_embedding = self.image_proj(groups.image)
        text_embedding = self.text_proj(text_features)
        text_loss = info_nce(
            image_embedding,
            text_embedding,
            groups.target,
            groups.weight,
            self.loss_weights.temperature,
        )
        spatial = spatial_loss(
            torch.cat((prior["attention"], current["attention"])),
            torch.cat((batch["prior_boxes"], batch["current_boxes"])),
            torch.cat((batch["prior_valid"], batch["current_valid"])),
            (self.image_size, self.image_size),
        )
        output.update(target)
        output["loss"] = phase1_loss(
            text_loss,
            encoded["change_logit"],
            encoded["direction_logit"],
            target,
            self.loss_weights,
            spatial,
        )
        return output


__all__ = ["CHECKPOINT_ARCH", "VEEEncoder"]
