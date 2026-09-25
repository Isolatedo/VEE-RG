from __future__ import annotations

import re
from collections.abc import Sequence


REGIONS = (
    "right lung", "right upper lung zone", "right mid lung zone",
    "right lower lung zone", "right hilar structures", "right apical zone",
    "right costophrenic angle", "right hemidiaphragm", "left lung",
    "left upper lung zone", "left mid lung zone", "left lower lung zone",
    "left hilar structures", "left apical zone", "left costophrenic angle",
    "left hemidiaphragm", "trachea", "spine", "right clavicle", "left clavicle",
    "aortic arch", "mediastinum", "upper mediastinum", "svc",
    "cardiac silhouette", "cavoatrial junction", "right atrium", "carina", "abdomen",
)


def _key(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def assemble_region_sentences(
    sentences: Sequence[str], selected: Sequence[bool] | None = None
) -> tuple[str, list[dict]]:
    """Order regional generations and remove exact normalized duplicates."""
    if len(sentences) != len(REGIONS):
        raise ValueError("sentences must contain 29 entries")
    if selected is None:
        selected = [True] * len(REGIONS)
    if len(selected) != len(REGIONS):
        raise ValueError("selected must contain 29 entries")
    kept: list[dict] = []
    seen: set[str] = set()
    for index, (sentence, is_selected) in enumerate(zip(sentences, selected)):
        sentence = " ".join(str(sentence or "").split()).strip()
        if not is_selected or not sentence:
            continue
        key = _key(sentence)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append({"region_index": index, "region": REGIONS[index], "sentence": sentence})
    return " ".join(item["sentence"] for item in kept), kept


def assemble_batch(
    sentences: Sequence[Sequence[str]], selected: Sequence[Sequence[bool]]
) -> tuple[list[str], list[list[dict]]]:
    if len(sentences) != len(selected):
        raise ValueError("sentences and selected must have equal batch size")
    reports, provenance = [], []
    for row, mask in zip(sentences, selected):
        report, kept = assemble_region_sentences(row, mask)
        reports.append(report)
        provenance.append(kept)
    return reports, provenance


__all__ = [
    "REGIONS",
    "assemble_batch",
    "assemble_region_sentences",
]
