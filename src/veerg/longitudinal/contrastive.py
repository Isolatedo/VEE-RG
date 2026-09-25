from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regions import mask_region


@dataclass
class TextGroups:
    image: torch.Tensor
    text: list[str]
    target: torch.Tensor
    weight: torch.Tensor
    regions: int


def group_events(
    event: torch.Tensor,
    eligible: torch.Tensor,
    texts: list[list[str]],
    *,
    pool_regions: bool = True,
) -> TextGroups:
    if (
        event.ndim != 3
        or eligible.shape != event.shape[:2]
        or len(texts) != event.shape[0]
    ):
        raise ValueError("event, eligible, and texts shapes are inconsistent")
    keep = eligible.detach().cpu().bool().tolist()
    sample_ids: list[int] = []
    region_ids: list[int] = []
    group_ids: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    unique: list[str] = []
    text_ids: dict[str, int] = {}

    def text_id(processed: str) -> int:
        if processed not in text_ids:
            text_ids[processed] = len(unique)
            unique.append(processed)
        return text_ids[processed]

    for sample, sample_texts in enumerate(texts):
        if len(sample_texts) != event.shape[1]:
            raise ValueError("every sample must provide one text per region")
        raw_groups: dict[str, dict[str, list[int]]] = {}
        for region, raw in enumerate(sample_texts):
            raw = str(raw).strip()
            if keep[sample][region] and raw:
                processed = mask_region(raw, region)
                raw_groups.setdefault(raw, {}).setdefault(processed, []).append(region)
        for variants in raw_groups.values():
            split_weight = 1.0 / len(variants)
            for processed, linked in variants.items():
                target = text_id(processed)
                if pool_regions:
                    group = len(targets)
                    targets.append(target)
                    weights.append(split_weight)
                    for region in linked:
                        sample_ids.append(sample)
                        region_ids.append(region)
                        group_ids.append(group)
                    continue

                # Keep one independent event row per region.  Regions sharing a
                # canonical sentence are multi-positive through the shared
                # target id, never negatives.  Normalize duplicate rows so a
                # repeated sentence does not get extra sample weight.
                row_weight = split_weight / len(linked)
                for region in linked:
                    sample_ids.append(sample)
                    region_ids.append(region)
                    group_ids.append(len(targets))
                    targets.append(target)
                    weights.append(row_weight)

    device = event.device
    if not targets:
        return TextGroups(
            event.new_empty((0, event.shape[-1])),
            [],
            torch.empty(0, dtype=torch.long, device=device),
            event.new_empty(0),
            0,
        )
    samples = torch.tensor(sample_ids, dtype=torch.long, device=device)
    regions = torch.tensor(region_ids, dtype=torch.long, device=device)
    groups = torch.tensor(group_ids, dtype=torch.long, device=device)
    selected = event[samples, regions]
    if pool_regions:
        pooled = event.new_zeros((len(targets), event.shape[-1]))
        pooled.index_add_(0, groups, selected)
        count = torch.bincount(groups, minlength=len(targets)).to(event.dtype)
        image = pooled / count.unsqueeze(-1)
    else:
        image = selected
    return TextGroups(
        image,
        unique,
        torch.tensor(targets, dtype=torch.long, device=device),
        torch.tensor(weights, dtype=event.dtype, device=device),
        len(region_ids),
    )


class Projection(nn.Module):
    def __init__(self, input_dim: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(value), dim=-1)


def info_nce(
    image: torch.Tensor,
    text: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if image.shape[0] == 0:
        return image.sum() * 0.0
    logits = image @ text.t() / temperature
    image_loss = F.cross_entropy(logits, target, reduction="none")
    image_loss = (image_loss * weight).sum() / weight.sum()
    text_ids = torch.arange(text.shape[0], device=target.device)
    positive = text_ids[:, None].eq(target[None, :])
    weighted = logits.t() + weight.log().unsqueeze(0)
    positive_logits = weighted.masked_fill(~positive, float("-inf"))
    text_loss = torch.logsumexp(weighted, 1) - torch.logsumexp(positive_logits, 1)
    text_weight = positive.to(weight.dtype) @ weight
    text_loss = (text_loss * text_weight).sum() / text_weight.sum().clamp_min(1e-8)
    return 0.5 * (image_loss + text_loss)


__all__ = [
    "Projection",
    "TextGroups",
    "group_events",
    "info_nce",
]
