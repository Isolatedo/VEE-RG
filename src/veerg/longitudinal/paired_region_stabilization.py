from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

NUM_REGIONS = 29


def _overlap(
    boxes: torch.Tensor,
    valid: torch.Tensor,
    height: int,
    width: int,
    image_size: int | tuple[int, int],
) -> torch.Tensor:
    image_height, image_width = (
        (image_size, image_size) if isinstance(image_size, int) else image_size
    )
    x0_grid = torch.arange(width, device=boxes.device, dtype=boxes.dtype) * image_width / width
    y0_grid = torch.arange(height, device=boxes.device, dtype=boxes.dtype) * image_height / height
    x1_grid = torch.arange(1, width + 1, device=boxes.device, dtype=boxes.dtype) * image_width / width
    y1_grid = torch.arange(1, height + 1, device=boxes.device, dtype=boxes.dtype) * image_height / height
    safe = torch.nan_to_num(boxes)
    x0, y0, x1, y1 = safe.unbind(-1)
    overlap_x = (torch.minimum(x1[..., None], x1_grid) - torch.maximum(x0[..., None], x0_grid)).clamp_min(0)
    overlap_y = (torch.minimum(y1[..., None], y1_grid) - torch.maximum(y0[..., None], y0_grid)).clamp_min(0)
    keep = valid.bool() & torch.isfinite(boxes).all(-1) & (x1 > x0) & (y1 > y0)
    cell_area = (image_height / height) * (image_width / width)
    return torch.where(
        keep[..., None, None],
        overlap_y[..., :, None] * overlap_x[..., None, :] / cell_area,
        0.0,
    )


def spatial_loss(
    attention: torch.Tensor,
    boxes: torch.Tensor,
    valid: torch.Tensor,
    image_size: int | tuple[int, int] = 512,
) -> torch.Tensor:
    target = _overlap(boxes, valid, attention.shape[-2], attention.shape[-1], image_size)
    mass = target.sum((-2, -1), keepdim=True)
    keep = mass.squeeze(-1).squeeze(-1) > 0
    if not keep.any():
        return attention.sum() * 0.0
    target = target / mass.clamp_min(torch.finfo(target.dtype).eps)
    loss = -(target * attention.clamp_min(1e-8).log()).sum((-2, -1))
    return loss[keep].mean()


class PairedRegionStabilization(nn.Module):
    """Query-guided regional reader with a shared paired-detection gate.

    The learned slot query supplies anatomy identity.  A detector ROI is used
    only when the same region is detected at both time points; in that case it
    modulates the query and contributes a local appearance residual.  Missing
    or one-sided detections therefore fall back to the coarse C5 context.
    """

    def __init__(
        self,
        dim: int = 256,
        regions: int = NUM_REGIONS,
    ):
        super().__init__()
        self.value = nn.Conv2d(2048, dim, 1, bias=False)
        self.roi = nn.Linear(1024, dim, bias=False)
        self.roi_query = nn.Linear(dim, dim, bias=False)
        self.query = nn.Embedding(regions, dim)
        self.position = nn.Parameter(torch.randn(1, dim, 16, 16) * 0.02)
        self.box_scale = nn.Parameter(torch.zeros(()))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.regions = regions
        nn.init.zeros_(self.roi_query.weight)

    def forward(
        self,
        c5: torch.Tensor | None = None,
        roi: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        pair_seen: torch.Tensor | None = None,
        image_size: int | tuple[int, int] = 512,
    ) -> dict[str, torch.Tensor]:
        if roi is None or boxes is None or pair_seen is None:
            raise ValueError("roi, boxes, and pair_seen are required")
        if c5 is None:
            raise ValueError("c5 feature is required")
        return self._forward_with_query_roi(
            spatial=c5,
            roi=roi,
            query_roi=roi,
            boxes=boxes,
            pair_seen=pair_seen,
            image_size=image_size,
        )

    def _forward_with_query_roi(
        self,
        spatial: torch.Tensor,
        roi: torch.Tensor,
        query_roi: torch.Tensor,
        boxes: torch.Tensor,
        pair_seen: torch.Tensor,
        image_size: int | tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        gate = pair_seen.to(roi.dtype).unsqueeze(-1)
        height, width = spatial.shape[-2:]
        value_map = self.value(spatial)
        position = F.interpolate(
            self.position, (height, width), mode="bilinear", align_corners=False
        )
        key = (value_map + position).flatten(2).transpose(1, 2)
        value = value_map.flatten(2).transpose(1, 2)
        roi_content = self.roi(torch.nan_to_num(roi))
        query_content = self.roi(torch.nan_to_num(query_roi))
        query = self.query.weight.unsqueeze(0) + self.roi_query(query_content) * gate
        logits = query @ key.transpose(1, 2) / math.sqrt(value_map.shape[1])
        box_bias = _overlap(boxes, pair_seen, height, width, image_size)
        logits = logits + torch.tanh(self.box_scale) * box_bias.flatten(2)
        attention = logits.softmax(-1)
        token = attention @ value + roi_content * gate
        return {
            "tokens": self.norm(token),
            "attention": attention.reshape(spatial.shape[0], self.regions, height, width),
        }


__all__ = ["NUM_REGIONS", "PairedRegionStabilization", "spatial_loss"]
