from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .change_factorization import ChangeFactorization
from .paired_region_stabilization import PairedRegionStabilization


NUM_REGIONS = 29
REGION_DIM = 256
EVENT_CONTEXT_DIM = 64
EVENT_DIM = REGION_DIM + EVENT_CONTEXT_DIM
CHECKPOINT_ARCH = "vee-rg-longitudinal-v1"


class FrozenLongitudinalEncoder(nn.Module):
    """Run the final paired-region and change-factorization encoder."""

    def __init__(
        self,
        detector: nn.Module,
        checkpoint_path: str,
        image_size: int = 512,
    ):
        super().__init__()
        self.detector = detector
        payload = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("arch") != CHECKPOINT_ARCH:
            raise ValueError(
                "longitudinal checkpoint architecture does not match VEE-RG: "
                f"got {payload.get('arch')!r}, expected {CHECKPOINT_ARCH!r}"
            )
        self.paired_region_stabilization = PairedRegionStabilization(
            dim=REGION_DIM,
            regions=NUM_REGIONS,
        )
        self.change_factorization = ChangeFactorization(
            dim=REGION_DIM,
            hidden_dim=512,
            context_dim=EVENT_CONTEXT_DIM,
            dropout=0.0,
        )
        state = payload.get("model", payload)
        self._load_prefix(
            self.paired_region_stabilization,
            state,
            "paired_region_stabilization.",
        )
        self._load_prefix(
            self.change_factorization,
            state,
            "change_factorization.",
        )
        self.image_size = int(image_size)
        self.requires_grad_(False).eval()

    @staticmethod
    def _load_prefix(module: nn.Module, state: dict, prefix: str) -> None:
        selected = {
            key[len(prefix):]: value
            for key, value in state.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if not selected:
            raise ValueError(f"longitudinal checkpoint is missing {prefix} state")
        module.load_state_dict(selected, strict=True)

    def train(self, mode: bool = True):
        super().train(False)
        self.detector.eval()
        self.paired_region_stabilization.eval()
        self.change_factorization.eval()
        return self

    def _regions(
        self,
        features: dict[str, torch.Tensor],
        pair_seen: torch.Tensor,
    ) -> torch.Tensor:
        return self.paired_region_stabilization(
            c5=features["c5"],
            roi=features["roi"],
            boxes=features["boxes"],
            pair_seen=pair_seen,
            image_size=(self.image_size, self.image_size),
        )["tokens"]

    @torch.no_grad()
    def forward(
        self,
        current: dict[str, torch.Tensor],
        prior_image: torch.Tensor,
        has_real_prior: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = current["roi"].shape[0]
        if prior_image.ndim != 4 or prior_image.shape[0] != batch_size:
            raise ValueError("prior_image must have one image per current image")
        if has_real_prior.shape != (batch_size,) or has_real_prior.dtype != torch.bool:
            raise ValueError("has_real_prior must be boolean [B]")

        prior = {key: value.clone() for key, value in current.items()}
        real_indices = has_real_prior.nonzero(as_tuple=False).flatten()
        if real_indices.numel():
            real_prior = self.detector.forward_spatial(
                prior_image.index_select(0, real_indices)
            )
            for key in prior:
                prior[key].index_copy_(0, real_indices, real_prior[key])

        pair_seen = prior["seen"] & current["seen"]
        prior_tokens = self._regions(prior, pair_seen)
        current_tokens = self._regions(current, pair_seen)
        encoded = self.change_factorization(
            prior_tokens,
            current_tokens,
            pair_seen=pair_seen,
        )
        exchange_invariant = torch.nan_to_num(encoded["exchange_invariant"])
        order_sensitive = torch.nan_to_num(encoded["order_sensitive"])
        valid = pair_seen & torch.isfinite(exchange_invariant).all(-1)
        valid &= torch.isfinite(order_sensitive).all(-1)
        return {
            "exchange_invariant": exchange_invariant,
            "order_sensitive": order_sensitive,
            "valid": valid,
        }


__all__ = ["CHECKPOINT_ARCH", "FrozenLongitudinalEncoder"]
