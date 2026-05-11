from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "paper_figures" / "outputs"

SCALE_ORDER = ["10k", "100k", "1m"]
INIT_ORDER = ["random", "imagenet"]
EXTERNAL_BASELINE_ORDER = ["dinov2", "mae"]
DATASET_OBJECTIVE_ORDER = ["infonce", "vicreg", "lejepa"]
REP_OBJECTIVE_ORDER = ["lejepa", "bhep", "eppartial"]
TASK_ORDER = ["celeb_within", "cfd_within"]
GEOMETRY_DATASET_ORDER = ["subset6_minus_7_test", "subset7_eval", "cfc_eval"]

SCALE_LABELS = {
    "10k": "10k",
    "100k": "100k",
    "1m": "1M",
}
INIT_LABELS = {
    "random": "Random",
    "imagenet": "ImageNet",
    "dinov2": "DINOv2 ViT-B/14",
    "mae": "MAE ViT-B/16",
}
OBJECTIVE_LABELS = {
    "infonce": "InfoNCE",
    "vicreg": "VICReg",
    "lejepa": "LeJEPA",
    "bhep": "BHEP",
    "eppartial": "EP-partial",
}
DATASET_LABELS = {
    "subset6_minus_7_test": "Holdout",
    "subset7_eval": "External open",
    "cfc_eval": "External clinic",
}
TASK_LABELS = {
    "celeb_within": "Celeb Mean L2",
    "cfd_within": "CFD Mean L2",
}
OBJECTIVE_COLORS = {
    "infonce": "#0b6e4f",
    "vicreg": "#b03a2e",
    "lejepa": "#7c5c00",
    "bhep": "#2563eb",
    "eppartial": "#9333ea",
}
EXTERNAL_BASELINE_COLORS = {
    "dinov2": "#2563eb",
    "mae": "#7c3aed",
}
SCALE_COLORS = {
    "10k": "#1b9e77",
    "100k": "#d95f02",
    "1m": "#7570b3",
}
COMPONENT_COLORS = {
    "loss": "#111827",
    "sim": "#2563eb",
    "reg": "#9333ea",
}

RUN_RE = re.compile(
    r"^geometry-fixedcompute-(?P<scale>10k|100k|1m)-"
    r"(?P<objective>infonce|vicreg|lejepa|bhep|eppartial)-"
    r"(?P<init>random|imagenet|seginit)(?:-50ksteps)?$"
)


@dataclass(frozen=True)
class RunParts:
    scale: str
    objective: str
    init: str
    is_50k: bool


def parse_run_name(run_name: str) -> RunParts:
    match = RUN_RE.match(run_name)
    if match is None:
        raise ValueError(f"Unexpected run_name format: {run_name}")
    return RunParts(
        scale=match.group("scale"),
        objective=match.group("objective"),
        init=match.group("init"),
        is_50k=run_name.endswith("-50ksteps"),
    )


def ordered_categorical(series, categories: list[str]):
    import pandas as pd

    return pd.Categorical(series, categories=categories, ordered=True)
