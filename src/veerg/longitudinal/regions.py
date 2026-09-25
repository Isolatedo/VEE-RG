from __future__ import annotations

import re
from functools import lru_cache


REGION_NAMES = (
    "right lung", "right upper lung zone", "right mid lung zone",
    "right lower lung zone", "right hilar structures", "right apical zone",
    "right costophrenic angle", "right hemidiaphragm", "left lung",
    "left upper lung zone", "left mid lung zone", "left lower lung zone",
    "left hilar structures", "left apical zone", "left costophrenic angle",
    "left hemidiaphragm", "trachea", "spine", "right clavicle",
    "left clavicle", "aortic arch", "mediastinum", "upper mediastinum",
    "svc", "cardiac silhouette", "cavoatrial junction", "right atrium",
    "carina", "abdomen",
)

REGION_ALIASES = (
    ("right lung", "right pulmonary field"),
    ("right upper lung zone", "right upper lung", "right upper lobe"),
    ("right mid lung zone", "right middle lung zone", "right mid lung"),
    ("right lower lung zone", "right lower lung", "right lower lobe"),
    ("right hilar structures", "right hilar region", "right hilum"),
    ("right apical zone", "right apical region", "right lung apex", "right apex"),
    ("right costophrenic angle", "right cp angle"),
    ("right hemidiaphragm", "right diaphragm"),
    ("left lung", "left pulmonary field"),
    ("left upper lung zone", "left upper lung", "left upper lobe"),
    ("left mid lung zone", "left middle lung zone", "left mid lung"),
    ("left lower lung zone", "left lower lung", "left lower lobe"),
    ("left hilar structures", "left hilar region", "left hilum"),
    ("left apical zone", "left apical region", "left lung apex", "left apex"),
    ("left costophrenic angle", "left cp angle"),
    ("left hemidiaphragm", "left diaphragm"),
    ("trachea",),
    ("thoracic spine", "spine"),
    ("right clavicle",),
    ("left clavicle",),
    ("aortic arch",),
    ("mediastinal contour", "mediastinum"),
    ("upper mediastinum", "superior mediastinum"),
    ("superior vena cava", "svc"),
    ("cardiomediastinal silhouette", "cardiac silhouette"),
    ("cavoatrial junction",),
    ("right atrium",),
    ("carina",),
    ("upper abdomen", "abdomen"),
)

ANATOMY_TOKEN = "this anatomical region"


def _phrase(alias: str) -> str:
    return r"[\s-]+".join(re.escape(part) for part in re.split(r"[\s-]+", alias))


@lru_cache(maxsize=None)
def _pattern(region: int) -> re.Pattern[str]:
    aliases = sorted(REGION_ALIASES[region], key=len, reverse=True)
    body = "|".join(_phrase(alias) for alias in aliases)
    if REGION_NAMES[region] == "mediastinum":
        body = rf"(?<!upper\s)(?<!superior\s)(?:{body})"
    return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])", re.IGNORECASE)


@lru_cache(maxsize=131072)
def mask_region(text: str, region: int) -> str:
    text = " ".join(str(text).strip().split())
    return _pattern(region).sub(ANATOMY_TOKEN, text) if text else ""


__all__ = ["ANATOMY_TOKEN", "REGION_NAMES", "mask_region"]
