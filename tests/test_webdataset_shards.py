from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from torch.utils.data import DataLoader

from scripts.build_webdataset_shards import write_shards
from src.dataset_utils import (
    ImageFolderDataset,
    ManifestImageDataset,
    WebDatasetImageDataset,
    build_dataset,
    is_iterable_dataset,
)


webdataset = pytest.importorskip("webdataset")


def _write_images(root: Path, count: int = 5) -> list[Path]:
    paths: list[Path] = []
    for i in range(count):
        path = root / f"img_{i:03d}.png"
        Image.new("RGB", (16, 16), color=(i, 0, 0)).save(path)
        paths.append(path)
    return paths


def test_shard_builder_writes_readable_shards(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    paths = _write_images(image_root, count=5)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(str(p) for p in paths) + "\n")

    shard_list = write_shards(
        manifest=manifest,
        out_dir=tmp_path / "shards",
        root=image_root,
        shard_size=2,
        prefix="train",
    )

    shards = shard_list.read_text().strip().splitlines()
    assert len(shards) == 3
    assert all(Path(p).is_file() for p in shards)


def test_webdataset_loader_returns_transformed_samples(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    paths = _write_images(image_root, count=4)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(str(p) for p in paths) + "\n")
    shard_list = write_shards(
        manifest=manifest,
        out_dir=tmp_path / "shards",
        root=image_root,
        shard_size=4,
        prefix="train",
    )

    ds = WebDatasetImageDataset(
        shards=str(shard_list),
        transform=lambda img: img.size,
        shuffle=False,
    )
    assert is_iterable_dataset(ds)

    first, sample = next(iter(ds))
    assert first == (16, 16)
    assert sample.filename.startswith("img_")


def test_build_dataset_keeps_existing_manifest_precedence(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    paths = _write_images(image_root, count=2)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(str(p.name) for p in paths) + "\n")

    cfg = {
        "data": {
            "train_root": str(image_root),
            "train_manifest": str(manifest),
        }
    }

    ds = build_dataset(cfg)
    assert isinstance(ds, ManifestImageDataset)
    assert len(ds) == 2


def test_build_dataset_uses_folder_loader_without_manifest_or_shards(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_images(image_root, count=2)

    ds = build_dataset({"data": {"train_root": str(image_root)}})

    assert isinstance(ds, ImageFolderDataset)
    assert len(ds) == 2


def test_webdataset_dataloader_batches_like_training(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    paths = _write_images(image_root, count=4)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(str(p) for p in paths) + "\n")
    shard_list = write_shards(
        manifest=manifest,
        out_dir=tmp_path / "shards",
        root=image_root,
        shard_size=4,
        prefix="train",
    )
    cfg = {
        "data": {
            "train_root": str(image_root),
            "train_shards": str(shard_list),
            "shard_shuffle": False,
        }
    }
    ds = build_dataset(cfg, transform=lambda img: img.size)
    dl = DataLoader(
        ds,
        batch_size=2,
        num_workers=0,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    batch, samples = next(iter(dl))
    assert len(batch) == 2
    assert len(samples) == 2
