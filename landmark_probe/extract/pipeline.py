from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    def tqdm(iterable, **_: object):
        return iterable

try:
    from torch import autocast
except ImportError:  # pragma: no cover - older torch
    from torch.cuda.amp import autocast  # type: ignore

from landmark_probe.config import DatasetSpec, RepresentationSpec, RunSpec, StudyConfig, TaskSplitSpec
from landmark_probe.extract.datasets import build_dataloader, load_split_records
from landmark_probe.extract.inference import load_backbone_for_run, pooled_backbone_embeddings
from landmark_probe.paths import embedding_artifact_path


def _resolve_device(device_spec: str) -> torch.device:
    if device_spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_spec.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_spec)


def _autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def extract_split_embeddings(
    study_cfg: StudyConfig,
    dataset_cfg: DatasetSpec,
    run: RunSpec,
    split_spec: TaskSplitSpec,
    representation: RepresentationSpec,
) -> Path:
    out_path = embedding_artifact_path(study_cfg, run, split_spec, representation)
    allow_overwrite = study_cfg.extraction.overwrite
    if out_path.exists() and not allow_overwrite:
        return out_path

    if representation.embedding_key != "backbone":
        raise ValueError(f"V1 only supports backbone embeddings, got {representation.embedding_key}")

    records = load_split_records(dataset_cfg, split_spec)
    dl = build_dataloader(dataset_cfg, split_spec, study_cfg.extraction)
    device = _resolve_device(study_cfg.extraction.device)
    autocast_dtype = _autocast_dtype(study_cfg.extraction.precision)
    amp_enabled = device.type == "cuda" and autocast_dtype is not None

    encoder, train_cfg, checkpoint_path = load_backbone_for_run(run)
    encoder = encoder.to(device)
    all_embs: list[torch.Tensor] = []
    sample_ids: list[str] = []
    for xs, metas in tqdm(dl, desc=f"extract {run.run_name} {split_spec.dataset_name}:{split_spec.split}:{representation.pooling}"):
        xs = xs.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
            emb = pooled_backbone_embeddings(encoder, xs, representation.pooling)
        all_embs.append(emb.detach().cpu().to(torch.float32))
        sample_ids.extend(meta.sample_id for meta in metas)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "embeddings": torch.cat(all_embs, dim=0),
        "sample_ids": sample_ids,
        "dataset_name": split_spec.dataset_name,
        "split": split_spec.split,
        "embedding_key": representation.embedding_key,
        "pooling": representation.pooling,
        "embedding_dim": int(all_embs[0].shape[1]) if all_embs else 0,
        "run_name": run.run_name,
        "run_dir": str(run.run_dir),
        "checkpoint_step": run.checkpoint_step,
        "checkpoint_path": str(checkpoint_path),
        "train_cfg": train_cfg,
        "num_rows": len(sample_ids),
        "manifest_count": len(records),
    }
    if payload["num_rows"] != payload["manifest_count"]:
        raise ValueError(f"Embedding row count mismatch for {out_path}")
    torch.save(payload, out_path)
    return out_path


def extract_study(study_cfg: StudyConfig, dataset_cfg: DatasetSpec) -> list[Path]:
    written: list[Path] = []
    unique_splits: dict[tuple[str, str], TaskSplitSpec] = {}
    for task in study_cfg.tasks:
        for split in (task.train_split, task.val_split, task.test_split):
            unique_splits[(split.dataset_name, split.split)] = split

    for run in study_cfg.runs:
        for representation in study_cfg.representations:
            for split_spec in unique_splits.values():
                written.append(extract_split_embeddings(study_cfg, dataset_cfg, run, split_spec, representation))
    return written
