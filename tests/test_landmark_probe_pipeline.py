from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sys
import torch
import torch.nn as nn
import types
import yaml
from PIL import Image

from landmark_probe.config import DatasetMetadataSpec, DatasetSpec, RawDatasetSource, load_study_config
from landmark_probe.extract.external_vit import DINOv2PatchFeatureMap, MAEViTEncoder
from landmark_probe.extract.inference import expected_embedding_dim, load_training_config_for_run, pooled_feature_map_embeddings
from landmark_probe.extract.inference import load_feature_model_for_run
from landmark_probe.prepare import anatomy
from landmark_probe.prepare.anatomy import EyeCropSample
from landmark_probe.prepare.pipeline import build_dataset


def test_anatomical_side_images_use_viewer_opposite_halves(monkeypatch, tmp_path: Path) -> None:
    class FakeExtractor:
        def __init__(self, predictions, mask):
            pass

        def extract_features(self):
            landmarks = {}
            for key in anatomy.LANDMARK_KEYS:
                landmarks[f"left_{key}"] = (75.0, 25.0)
                landmarks[f"right_{key}"] = (25.0, 25.0)
            return landmarks, False, False, False, False, False, False, False, False

    monkeypatch.setattr(anatomy, "EyeFeatureExtractor", FakeExtractor)

    image_arr = np.zeros((50, 100, 3), dtype=np.uint8)
    image_arr[:, :50] = [255, 0, 0]
    image_arr[:, 50:] = [0, 0, 255]
    image_path = tmp_path / "0_crop_celeb.jpg"
    mask_path = tmp_path / "0_crop_celeb.png"
    Image.fromarray(image_arr).save(image_path)
    Image.fromarray(np.ones((50, 100), dtype=np.uint8) * 3).save(mask_path)

    samples, failures = anatomy.build_eye_samples_with_failures("celeb", image_path, mask_path, out_size=10)

    assert failures == []
    assert [sample.anatomical_side for sample in samples] == ["l", "r"]
    left_mean = np.asarray(samples[0].image).mean(axis=(0, 1))
    right_mean = np.asarray(samples[1].image).mean(axis=(0, 1))
    assert left_mean[2] > left_mean[0]
    assert right_mean[0] > right_mean[2]


def test_build_dataset_assigns_paired_eyes_to_same_split(monkeypatch, tmp_path: Path) -> None:
    raw_images = tmp_path / "raw_images"
    raw_masks = tmp_path / "raw_masks"
    raw_images.mkdir()
    raw_masks.mkdir()
    for idx in range(5):
        Image.new("RGB", (8, 4), color=(idx, 0, 0)).save(raw_images / f"{idx}_celeb.jpg")
        Image.new("L", (8, 4), color=3).save(raw_masks / f"{idx}_celeb.png")

    def fake_build(dataset_name, image_path, mask_path, out_size):
        source_id = image_path.stem
        landmarks = {}
        for key in anatomy.LANDMARK_KEYS:
            landmarks[f"{key}_x"] = 1.0
            landmarks[f"{key}_y"] = 2.0
        samples = [
            EyeCropSample(
                sample_id=f"{source_id}_l",
                source_id=source_id,
                dataset_name=dataset_name,
                anatomical_side="l",
                image_name=f"{source_id}_l.jpg",
                image_rel_path=f"{dataset_name}/images/{source_id}_l.jpg",
                image=Image.new("RGB", (out_size, out_size)),
                landmarks=landmarks,
            ),
            EyeCropSample(
                sample_id=f"{source_id}_r",
                source_id=source_id,
                dataset_name=dataset_name,
                anatomical_side="r",
                image_name=f"{source_id}_r.jpg",
                image_rel_path=f"{dataset_name}/images/{source_id}_r.jpg",
                image=Image.new("RGB", (out_size, out_size)),
                landmarks=landmarks,
            ),
        ]
        return samples, []

    monkeypatch.setattr("landmark_probe.prepare.pipeline.build_eye_samples_with_failures", fake_build)

    root = tmp_path / "prepared"
    cfg = DatasetSpec(
        name="unit",
        root=root,
        image_size=16,
        normalize_imagenet=True,
        landmarks=anatomy.LANDMARK_KEYS,
        subdatasets=("celeb",),
        metadata=DatasetMetadataSpec(
            manifest_csv=root / "metadata/dataset_manifest.csv",
            landmarks_csv=root / "metadata/landmarks.csv",
            split_csv=root / "metadata/split_assignments.csv",
            failures_csv=root / "metadata/prep_failures.csv",
            summary_csv=root / "metadata/prep_summary.csv",
        ),
        raw_sources=(
            RawDatasetSource(
                name="celeb",
                image_dir=raw_images,
                mask_dir=raw_masks,
                image_suffix=".jpg",
                mask_suffix=".png",
            ),
        ),
        split_seed=0,
    )

    build_dataset(cfg, overwrite=True)
    splits = pd.read_csv(cfg.metadata.split_csv)

    per_source = splits.groupby("source_id")["split"].nunique()
    assert per_source.max() == 1
    assert splits.groupby("split")["source_id"].nunique().to_dict() == {"test": 1, "train": 3, "val": 1}
    assert cfg.metadata.failures_csv.exists()
    assert cfg.metadata.summary_csv.exists()


def test_expected_embedding_dims_follow_pooling_area() -> None:
    train_cfg = {"model": {"feat_dim": 2048}}
    assert expected_embedding_dim(train_cfg, "gap") == 2048
    assert expected_embedding_dim(train_cfg, "g2") == 8192
    assert expected_embedding_dim(train_cfg, "g4") == 32768


def test_external_vit_embedding_dims_are_patch_token_g4() -> None:
    for model_name in ("dinov2_vitb14", "mae_vitb16_in1k_pretrain"):
        run = _external_run(model_name=model_name)
        train_cfg = load_training_config_for_run(run)
        assert expected_embedding_dim(train_cfg, "g4") == 12288


def test_dinov2_patch_tokens_pool_to_g4() -> None:
    class FakeDINO(nn.Module):
        def forward_features(self, x):
            batch_size = x.shape[0]
            tokens = torch.arange(batch_size * 16 * 16 * 768, dtype=torch.float32)
            return {"x_norm_patchtokens": tokens.reshape(batch_size, 16 * 16, 768)}

    model = DINOv2PatchFeatureMap(FakeDINO())
    emb = pooled_feature_map_embeddings(model, torch.zeros(2, 3, 224, 224), "g4")

    assert emb.shape == (2, 12288)


def test_mae_encoder_excludes_cls_token_before_feature_map(monkeypatch) -> None:
    class IdentityPatchEmbed(nn.Module):
        num_patches = 14 * 14

        def forward(self, x):
            batch_size = x.shape[0]
            tokens = torch.arange(batch_size * self.num_patches * 768, dtype=torch.float32)
            return tokens.reshape(batch_size, self.num_patches, 768)

    encoder = MAEViTEncoder()
    encoder.patch_embed = IdentityPatchEmbed()
    encoder.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14 + 1, 768), requires_grad=False)
    encoder.blocks = nn.ModuleList()
    encoder.norm = nn.Identity()

    feat = encoder(torch.zeros(2, 3, 224, 224))

    assert feat.shape == (2, 768, 14, 14)
    assert torch.equal(feat[0, :, 0, 0], torch.arange(768, dtype=torch.float32))


def test_trained_timm_vit_checkpoint_loads_feature_model(monkeypatch, tmp_path: Path) -> None:
    from landmark_probe.config import RunSpec

    class FakeTimmViT(nn.Module):
        num_prefix_tokens = 1

        def __init__(self):
            super().__init__()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
            self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
            self.patch_embed = nn.Module()
            self.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)
            self.blocks = nn.ModuleList()
            self.norm = nn.LayerNorm(768)

        def forward_features(self, x):
            batch_size = x.shape[0]
            patch_tokens = torch.zeros(batch_size, 14 * 14, 768, device=x.device)
            cls = torch.zeros(batch_size, 1, 768, device=x.device)
            return torch.cat([cls, patch_tokens], dim=1)

    monkeypatch.setitem(sys.modules, "timm", types.SimpleNamespace(create_model=lambda *args, **kwargs: FakeTimmViT()))

    run_dir = tmp_path / "vit_run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    train_cfg = {
        "model": {
            "backbone": "vit_base_patch16_224",
            "init": "random",
            "feat_dim": 768,
        },
        "ssl": {"method": "lejepa"},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(train_cfg), encoding="utf-8")
    model = FakeTimmViT()
    from src.backbones.vit import TimmViTPatchMap

    encoder = TimmViTPatchMap(model=model, patch_size=16)
    torch.save({"encoder": encoder.state_dict(), "objective": {}, "step": 1, "epoch": 0}, ckpt_dir / "ckpt_step_0000001.pth")

    feature_model, loaded_cfg, checkpoint_path = load_feature_model_for_run(
        RunSpec(run_name="unit-vit", run_dir=run_dir, checkpoint_step=1)
    )

    assert loaded_cfg["model"]["backbone"] == "vit_base_patch16_224"
    assert checkpoint_path == ckpt_dir / "ckpt_step_0000001.pth"
    assert pooled_feature_map_embeddings(feature_model, torch.zeros(2, 3, 224, 224), "g4").shape == (2, 12288)


def test_valid_external_study_config_loads(tmp_path: Path) -> None:
    cfg_path = _write_external_study_config(tmp_path)

    cfg = load_study_config(cfg_path)

    assert cfg.runs[0].external_model == "dinov2_vitb14"
    assert cfg.runs[1].external_model == "mae_vitb16_in1k_pretrain"
    assert cfg.representations[0].embedding_key == "patch_tokens"
    assert cfg.representations[0].pooling == "g4"


def test_external_study_config_rejects_non_patch_tokens(tmp_path: Path) -> None:
    cfg_path = _write_external_study_config(tmp_path, embedding_key="backbone")

    with pytest.raises(ValueError, match="requires patch_tokens/g4"):
        load_study_config(cfg_path)


def test_external_study_config_rejects_mixed_run_sources(tmp_path: Path) -> None:
    cfg_path = _write_external_study_config(tmp_path, extra_run_fields={"baseline_init": "imagenet"})

    with pytest.raises(ValueError, match="cannot define run_dir or baseline_init"):
        load_study_config(cfg_path)


def _external_run(model_name: str):
    from landmark_probe.config import RunSpec

    return RunSpec(run_name=f"baseline-{model_name}", run_dir=None, checkpoint_step=0, external_model=model_name)


def _write_external_study_config(
    tmp_path: Path,
    embedding_key: str = "patch_tokens",
    pooling: str = "g4",
    extra_run_fields: dict[str, object] | None = None,
) -> Path:
    run = {
        "run_name": "baseline-dinov2-vitb14",
        "checkpoint_step": 0,
        "external_model": "dinov2_vitb14",
    }
    if extra_run_fields:
        run.update(extra_run_fields)
    cfg = {
        "study": {
            "name": "unit_external_vit",
            "output_root": "/workspace/landmark_probe/outputs",
        },
        "dataset_cfg": "/workspace/landmark_probe/configs/datasets/periorbital_224_v2.yaml",
        "probe_cfg": "/workspace/landmark_probe/configs/probes/mlp_default.yaml",
        "representations": [
            {
                "embedding_key": embedding_key,
                "pooling": pooling,
            }
        ],
        "runs": [
            run,
            {
                "run_name": "baseline-mae-vitb16-in1k-pretrain",
                "checkpoint_step": 0,
                "external_model": "mae_vitb16_in1k_pretrain",
            },
        ],
        "tasks": [
            {
                "task_name": "celeb_within",
                "train_split": {"dataset_name": "celeb", "split": "train"},
                "val_split": {"dataset_name": "celeb", "split": "val"},
                "test_split": {"dataset_name": "celeb", "split": "test"},
            }
        ],
    }
    cfg_path = tmp_path / "study.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path
