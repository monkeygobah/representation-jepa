from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
from torchvision.models.segmentation import deeplabv3_resnet101

from landmark_probe.config import RunSpec
import yaml


def _strip_module_prefix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _disable_running_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None


def load_backbone_for_run(run: RunSpec) -> tuple[nn.Module, dict[str, Any], Path]:
    with (run.run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f) or {}
    checkpoint_path = run.checkpoint_path or (
        run.run_dir / "checkpoints" / f"ckpt_step_{run.checkpoint_step:07d}.pth"
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    init_mode = str(train_cfg["model"]["init"])
    if init_mode == "seg_init":
        model = deeplabv3_resnet101(weights=None, weights_backbone=None)
        model.classifier[4] = nn.Conv2d(256, 6, kernel_size=1, stride=1)
        encoder = model.backbone
    elif init_mode in {"imagenet", "random"}:
        model = resnet101(weights=None)
        encoder = nn.Sequential(*list(model.children())[:-2])
    else:
        raise ValueError(f"Unsupported init mode for landmark probe extraction: {init_mode}")
    _disable_running_stats(encoder)
    encoder.load_state_dict(_strip_module_prefix(ckpt["encoder"]), strict=True)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder, train_cfg, checkpoint_path


@torch.inference_mode()
def pooled_backbone_embeddings(encoder: nn.Module, x: torch.Tensor, pooling: str) -> torch.Tensor:
    feat = encoder(x)
    if isinstance(feat, dict):
        feat = feat["out"]
    if pooling == "gap":
        pooled = F.adaptive_avg_pool2d(feat, (1, 1))
    elif pooling == "g2":
        pooled = F.adaptive_avg_pool2d(feat, (2, 2))
    elif pooling == "g4":
        pooled = F.adaptive_avg_pool2d(feat, (4, 4))
    else:
        raise ValueError(f"Unsupported pooling mode: {pooling}")
    return pooled.flatten(1)
