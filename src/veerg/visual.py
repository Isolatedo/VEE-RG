from __future__ import annotations

import torch
import torch.nn as nn

from .detector import AnatomicalDetector


NUM_REGIONS = 29
ROI_DIM = 1024
C5_DIM = 2048


def select_regions(
    logits: torch.Tensor,
    seen: torch.Tensor,
    threshold: float,
    top_k: int | None = None,
) -> torch.Tensor:
    """Select threshold-valid detector regions, optionally capped by top-k."""
    if logits.shape != seen.shape or logits.ndim != 2 or seen.dtype != torch.bool:
        raise ValueError("logits and seen must have matching [B, 29] shapes")
    if top_k is None:
        return seen & (logits > threshold)
    if not 1 <= top_k <= logits.shape[1]:
        raise ValueError("selector top-k must be between 1 and the region count")

    # The selector threshold is a validity gate even when a top-k budget is
    # used.  Rank only regions that are both detected and above the threshold.
    valid_mask = seen & (logits > threshold)
    selected = torch.zeros_like(seen)
    for image in range(logits.shape[0]):
        valid = valid_mask[image].nonzero(as_tuple=False).flatten()
        count = min(top_k, int(valid.numel()))
        if count:
            ranked = valid[
                torch.argsort(logits[image, valid], descending=True)[:count]
            ]
            selected[image, ranked] = True
    return selected


class FrozenAnatomicalDetector(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.detector = AnatomicalDetector(
            return_feature_vectors=True,
            return_spatial_features=True,
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.detector.load_state_dict(state, strict=True)
        self.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @staticmethod
    def _validate(image: torch.Tensor) -> None:
        if image.ndim != 4 or image.shape[1:] != (1, 512, 512):
            raise ValueError("image must have shape [B, 1, 512, 512]")

    @torch.no_grad()
    def forward_spatial(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate(image)
        return self.detector.forward_spatial(image)


class FrozenRegionSelector(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(ROI_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        selected = {
            key: value
            for key, value in state.items()
            if isinstance(key, str) and key.startswith("classifier.")
        }
        self.load_state_dict(selected, strict=True)
        self.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, roi: torch.Tensor) -> torch.Tensor:
        return self.classifier(roi).squeeze(-1)


class RegionalVisualAdapter(nn.Module):
    """Project ROI features and one gated global C5 context token."""

    def __init__(self, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.roi = nn.Sequential(
            nn.LayerNorm(ROI_DIM),
            nn.Linear(ROI_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.region = nn.Embedding(NUM_REGIONS, hidden_dim)
        self.score = nn.Linear(1, hidden_dim, bias=False)
        self.roi_norm = nn.LayerNorm(hidden_dim)
        self.global_context = nn.Sequential(
            nn.LayerNorm(C5_DIM),
            nn.Linear(C5_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_gate = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.region.weight, std=0.02)
        nn.init.zeros_(self.score.weight)

    def forward(
        self, roi: torch.Tensor, logits: torch.Tensor, c5: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if roi.shape[1:] != (NUM_REGIONS, ROI_DIM):
            raise ValueError("roi must have shape [B, 29, 1024]")
        if logits.shape != roi.shape[:2]:
            raise ValueError("selector logits must have shape [B, 29]")
        if c5 is not None and (
            c5.ndim != 4 or c5.shape[0] != roi.shape[0] or c5.shape[1] != C5_DIM
        ):
            raise ValueError("c5 must have shape [B, 2048, H, W]")
        safe = torch.nan_to_num(roi)
        hint = self.score(torch.sigmoid(logits).unsqueeze(-1))
        roi_tokens = self.roi_norm(self.roi(safe) + hint)
        region_tokens = self.region.weight.unsqueeze(0).expand(roi.shape[0], -1, -1)
        if c5 is None:
            global_token = roi_tokens.new_zeros((roi.shape[0], roi_tokens.shape[-1]))
        else:
            pooled = torch.nan_to_num(c5).mean(dim=(-2, -1))
            global_token = torch.tanh(self.global_gate) * self.global_context(pooled)
        return roi_tokens, region_tokens, global_token


class RegionalVisionEncoder(nn.Module):
    def __init__(
        self,
        detector,
        selector,
        adapter,
        selector_threshold: float = -1.0,
        selector_top_k: int | None = None,
    ):
        super().__init__()
        self.detector = detector
        self.selector = selector
        self.adapter = adapter
        self.selector_threshold = float(selector_threshold)
        self.selector_top_k = selector_top_k

    def train(self, mode: bool = True):
        super().train(mode)
        self.detector.eval()
        self.selector.eval()
        return self

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            spatial = self.detector.forward_spatial(image)
            logits = self.selector(spatial["roi"])
        c5 = spatial["c5"]
        roi_tokens, region_tokens, global_token = self.adapter(
            spatial["roi"], logits, c5
        )
        selected = select_regions(
            logits,
            spatial["seen"],
            self.selector_threshold,
            self.selector_top_k,
        )
        return {
            "roi_tokens": roi_tokens,
            "region_tokens": region_tokens,
            "global_token": global_token,
            "roi_mask": spatial["seen"],
            "selected_mask": selected,
            "selector_logits": logits,
            "spatial": spatial,
        }


__all__ = [
    "C5_DIM",
    "FrozenAnatomicalDetector",
    "FrozenRegionSelector",
    "NUM_REGIONS",
    "ROI_DIM",
    "RegionalVisualAdapter",
    "RegionalVisionEncoder",
    "select_regions",
]
