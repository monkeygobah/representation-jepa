

import torch
import torch.nn as nn
from collections.abc import Mapping

def gap_pool(feat: torch.Tensor) -> torch.Tensor:
    return feat.mean(dim=(2, 3))


def get_feat_out(y):
    return y["out"] if isinstance(y, Mapping) else y


# class LeJEPABackbonePlusProjector(nn.Module):
#     def __init__(self, encoder: nn.Module, projector: nn.Module):
#         super().__init__()
#         self.encoder = encoder
#         self.projector = projector

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         feat = get_feat_out(self.encoder(x))
#         emb = gap_pool(feat)
#         proj = self.projector(emb)
#         return proj


class BackboneOnly(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = get_feat_out(self.encoder(x))
        emb = gap_pool(feat)          # [N, 2048]
        return emb


class BackbonePlusFixedProjector(nn.Module):
    def __init__(self, encoder: nn.Module, projector: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = get_feat_out(self.encoder(x))
        emb = gap_pool(feat)          # [N, 2048]
        proj = self.projector(emb)    # [N, 128]
        return proj
