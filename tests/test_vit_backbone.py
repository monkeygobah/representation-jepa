from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

from src.load_backbones import load_encoder_backbone
from src.backbones.vit import TimmViTPatchMap, patch_tokens_to_feature_map
from src.objectives.lejepa import LeJEPAObjective


def test_patch_tokens_to_feature_map_for_vit_b16_tokens():
    tokens = torch.randn(2, 196, 768)

    feat = patch_tokens_to_feature_map(tokens)

    assert feat.shape == (2, 768, 14, 14)


def test_patch_tokens_to_feature_map_for_vit_l16_tokens():
    tokens = torch.randn(2, 196, 1024)

    feat = patch_tokens_to_feature_map(tokens)

    assert feat.shape == (2, 1024, 14, 14)


def test_patch_tokens_to_feature_map_rejects_non_square_token_count():
    tokens = torch.randn(2, 195, 768)

    with pytest.raises(ValueError, match="square grid"):
        patch_tokens_to_feature_map(tokens)


class FakeTimmViT(nn.Module):
    num_prefix_tokens = 1

    def __init__(self, dim: int = 768, patch_size: int = 16):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.weight = nn.Parameter(torch.ones(()))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        patches = (x.shape[-2] // self.patch_size) * (x.shape[-1] // self.patch_size)
        return self.weight * torch.ones(batch_size, patches + 1, self.dim, device=x.device)


def test_timm_vit_patch_map_drops_cls_and_preserves_spatial_grid():
    model = TimmViTPatchMap(FakeTimmViT(dim=768), patch_size=16)

    global_feat = model(torch.zeros(2, 3, 224, 224))
    local_feat = model(torch.zeros(2, 3, 96, 96))

    assert global_feat.shape == (2, 768, 14, 14)
    assert local_feat.shape == (2, 768, 6, 6)


def test_timm_vit_patch_map_rejects_non_patch_multiple_inputs():
    model = TimmViTPatchMap(FakeTimmViT(dim=768), patch_size=16)

    with pytest.raises(ValueError, match="divisible"):
        model(torch.zeros(2, 3, 86, 86))


def test_lejepa_multicrop_smoke_with_vit_style_feature_maps():
    cfg = {
        "model": {
            "feat_dim": 768,
            "proj_dim": 32,
            "proj_hidden": 64,
            "proj_layers": 2,
        },
        "loss": {
            "lamb": 0.05,
            "regularizer": "sigreg",
            "sigreg_knots": 5,
            "sigreg_num_slices": 4,
        },
    }
    objective = LeJEPAObjective(cfg)
    encoder = TimmViTPatchMap(FakeTimmViT(dim=768), patch_size=16)
    views = [torch.randn(2, 3, 224, 224) for _ in range(2)]
    views.extend(torch.randn(2, 3, 96, 96) for _ in range(6))

    loss, logs = objective(encoder, views)

    assert torch.isfinite(loss)
    assert logs["V"] == 8


def test_load_encoder_backbone_supports_vit_b_and_l_with_timm(monkeypatch):
    fake_timm = types.SimpleNamespace(
        create_model=lambda *args, **kwargs: FakeTimmViT(
            dim=1024 if "large" in args[0] else 768
        )
    )
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    vit_b = load_encoder_backbone(backbone="vit_base_patch16_224", init="random")
    vit_l = load_encoder_backbone(backbone="vit_large_patch16_224", init="random")

    assert vit_b(torch.zeros(1, 3, 224, 224)).shape == (1, 768, 14, 14)
    assert vit_l(torch.zeros(1, 3, 224, 224)).shape == (1, 1024, 14, 14)


def test_load_encoder_backbone_preserves_resnet_random_route(monkeypatch):
    sentinel = nn.Identity()
    monkeypatch.setattr("src.load_backbones.load_resnet101_encoder", lambda pretrained: sentinel)

    model = load_encoder_backbone(init="random")

    assert model is sentinel
