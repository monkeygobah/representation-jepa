from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    stem: str
    filename: str
    path: Path
    rel_path: Path


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.samples = self._index()

    def _index(self) -> list[ImageSample]:
        out: list[ImageSample] = []
        for p in sorted(self.root.rglob("*")):
            if not is_image_file(p):
                continue

            rel_path = p.relative_to(self.root)
            sample = ImageSample(
                stem=p.stem,
                filename=p.name,
                path=p,
                rel_path=rel_path,
            )
            out.append(sample)
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s


def build_dataset(cfg, transform=None, root_key: str = "train_root") -> ImageFolderDataset:
    data_cfg = cfg.get("data", {})
    root = data_cfg.get(root_key)
    if root is None:
        raise ValueError(f"Missing data.{root_key} in config.")

    return ImageFolderDataset(
        root=root,
        transform=transform,
    )
