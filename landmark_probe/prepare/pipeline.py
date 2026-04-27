from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from landmark_probe.config import DatasetSpec
from landmark_probe.constants import LANDMARK_KEYS, VALID_SPLITS
from landmark_probe.prepare.anatomy import EyeCropSample, build_eye_samples


def _split_counts(n: int, train_frac: float, val_frac: float) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError(f"Need at least 3 samples to form train/val/test splits, got {n}")
    n_train = int(math.floor(n * train_frac))
    n_val = int(math.floor(n * val_frac))
    n_test = n - n_train - n_val

    if n_train == 0:
        n_train, n_test = 1, max(0, n_test - 1)
    if n_val == 0:
        n_val, n_test = 1, max(0, n_test - 1)
    if n_test == 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
    return n_train, n_val, n_test


def _assign_splits(sample_ids: list[str], cfg: DatasetSpec) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.split_seed)
    perm = rng.permutation(len(sample_ids))
    n_train, n_val, _ = _split_counts(len(sample_ids), cfg.split_train_frac, cfg.split_val_frac)
    split_values = np.empty(len(sample_ids), dtype=object)
    split_values[perm[:n_train]] = "train"
    split_values[perm[n_train : n_train + n_val]] = "val"
    split_values[perm[n_train + n_val :]] = "test"
    return pd.DataFrame({"sample_id": sample_ids, "split": split_values})


def _iter_source_pairs(image_dir: Path, mask_dir: Path, image_suffix: str, mask_suffix: str) -> Iterable[tuple[Path, Path]]:
    images = {path.stem: path for path in sorted(image_dir.glob(f"*{image_suffix}"))}
    masks = {path.stem: path for path in sorted(mask_dir.glob(f"*{mask_suffix}"))}
    common = sorted(set(images) & set(masks))
    for stem in common:
        yield images[stem], masks[stem]


def _validate_bounded_landmarks(landmarks_df: pd.DataFrame, image_size: int) -> None:
    coord_columns = [c for c in landmarks_df.columns if c.endswith("_x") or c.endswith("_y")]
    if landmarks_df[coord_columns].isna().any().any():
        raise ValueError("Landmarks contain NaN values after dataset preparation")
    if ((landmarks_df[coord_columns] < 0.0) | (landmarks_df[coord_columns] > float(image_size))).any().any():
        raise ValueError(f"Landmarks contain coordinates outside [0, {image_size}]")


def validate_prepared_dataset(cfg: DatasetSpec) -> None:
    manifest_df = pd.read_csv(cfg.metadata.manifest_csv)
    landmarks_df = pd.read_csv(cfg.metadata.landmarks_csv)
    splits_df = pd.read_csv(cfg.metadata.split_csv)

    manifest_ids = set(manifest_df["sample_id"])
    landmark_ids = set(landmarks_df["sample_id"])
    split_ids = set(splits_df["sample_id"])
    if not manifest_ids:
        raise ValueError("Prepared dataset manifest is empty")
    if manifest_ids != landmark_ids or manifest_ids != split_ids:
        raise ValueError("Manifest, landmark, and split sample_id sets do not match")
    if splits_df["sample_id"].duplicated().any():
        raise ValueError("Split assignments contain duplicate sample_id rows")
    if not set(splits_df["split"]).issubset(set(VALID_SPLITS)):
        raise ValueError(f"Unexpected split labels found: {sorted(set(splits_df['split']) - set(VALID_SPLITS))}")
    for dataset_name in cfg.subdatasets:
        subset = splits_df.loc[splits_df["dataset_name"] == dataset_name]
        counts = subset["split"].value_counts().to_dict()
        if any(split not in counts for split in VALID_SPLITS):
            raise ValueError(f"Dataset {dataset_name} is missing one or more split partitions: {counts}")
        expected = _split_counts(len(subset), cfg.split_train_frac, cfg.split_val_frac)
        observed = (counts.get("train", 0), counts.get("val", 0), counts.get("test", 0))
        if observed != expected:
            raise ValueError(
                f"Dataset {dataset_name} split counts do not match expected 80/10/10 rounding. "
                f"Observed={observed}, expected={expected}"
            )

    for rel_path in manifest_df["image_rel_path"]:
        if not (cfg.root / rel_path).exists():
            raise FileNotFoundError(f"Prepared image missing: {cfg.root / rel_path}")

    _validate_bounded_landmarks(landmarks_df, cfg.image_size)


def build_dataset(cfg: DatasetSpec, overwrite: bool = False, max_samples_per_dataset: int | None = None) -> tuple[Path, Path, Path]:
    if cfg.root.exists() and any(cfg.root.iterdir()) and not overwrite:
        validate_prepared_dataset(cfg)
        return cfg.metadata.manifest_csv, cfg.metadata.landmarks_csv, cfg.metadata.split_csv

    cfg.metadata_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in cfg.subdatasets:
        cfg.image_dir(dataset_name).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    landmark_rows: list[dict[str, float | str]] = []
    split_rows: list[dict[str, str]] = []

    for source in cfg.raw_sources:
        samples: list[EyeCropSample] = []
        for idx, (image_path, mask_path) in enumerate(
            _iter_source_pairs(source.image_dir, source.mask_dir, source.image_suffix, source.mask_suffix)
        ):
            if max_samples_per_dataset is not None and idx >= max_samples_per_dataset:
                break
            built = build_eye_samples(source.name, image_path, mask_path, out_size=cfg.image_size)
            if built is None:
                continue
            samples.extend(built)

        if not samples:
            raise ValueError(f"No samples built for dataset: {source.name}")

        split_df = _assign_splits([sample.sample_id for sample in samples], cfg)
        split_df["dataset_name"] = source.name
        split_rows.extend(split_df.to_dict(orient="records"))

        for sample in samples:
            out_path = cfg.root / sample.image_rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sample.image.save(out_path, quality=95)
            manifest_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "dataset_name": sample.dataset_name,
                    "image_rel_path": sample.image_rel_path,
                    "image_name": sample.image_name,
                    "anatomical_side": sample.anatomical_side,
                }
            )
            landmark_row = {"sample_id": sample.sample_id, "dataset_name": sample.dataset_name}
            landmark_row.update(sample.landmarks)
            landmark_rows.append(landmark_row)

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["dataset_name", "sample_id"]).reset_index(drop=True)
    landmarks_df = pd.DataFrame(landmark_rows).sort_values(["dataset_name", "sample_id"]).reset_index(drop=True)
    split_df = pd.DataFrame(split_rows).sort_values(["dataset_name", "sample_id"]).reset_index(drop=True)

    manifest_df.to_csv(cfg.metadata.manifest_csv, index=False)
    landmarks_df.to_csv(cfg.metadata.landmarks_csv, index=False)
    split_df.to_csv(cfg.metadata.split_csv, index=False)
    validate_prepared_dataset(cfg)
    return cfg.metadata.manifest_csv, cfg.metadata.landmarks_csv, cfg.metadata.split_csv
