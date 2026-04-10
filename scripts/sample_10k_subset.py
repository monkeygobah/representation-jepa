from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def reservoir_sample(src_root: Path, k: int, rng: random.Random) -> tuple[list[Path], int]:
    sample: list[Path] = []
    seen = 0

    with os.scandir(src_root) as entries:
        for entry in entries:
            path = Path(entry.path)
            if not is_image_file(path):
                continue

            if seen < k:
                sample.append(path)
            else:
                j = rng.randint(0, seen)
                if j < k:
                    sample[j] = path
            seen += 1

    return sample, seen


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    try:
        if mode == "hardlink":
            os.link(src, dst)
        elif mode == "symlink":
            os.symlink(src, dst)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a deterministic 10k flat pretraining subset.")
    ap.add_argument(
        "--src-root",
        type=Path,
        default=Path("/home/georgie/Desktop/neurips/representation/data/subset6_minus_7_train_flat"),
        help="Flat source image directory.",
    )
    ap.add_argument(
        "--dst-root",
        type=Path,
        default=Path("/home/georgie/Desktop/neurips/representation/data/subset6_minus_7_train_flat_10_000"),
        help="Destination directory for the sampled 10k subset.",
    )
    ap.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("/home/georgie/Desktop/neurips/representation/data/manifests/subset6_minus_7_train_flat_10_000.txt"),
        help="Manifest file listing sampled filenames.",
    )
    ap.add_argument("--seed", type=int, default=2026, help="Random seed.")
    ap.add_argument(
        "--mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How to materialize files. Falls back to copy on link failure.",
    )
    ap.add_argument(
        "--size",
        type=int,
        default=10_000,
        help="Subset size. Default is 10,000.",
    )
    args = ap.parse_args()

    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    manifest_path = args.manifest_path.resolve()

    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src_root}")
    if args.size <= 0:
        raise ValueError("--size must be positive.")

    rng = random.Random(args.seed)

    print(f"Sampling {args.size:,} images from {src_root}")
    print(f"Seed: {args.seed}")
    print(f"Mode: {args.mode}")

    sample, total_seen = reservoir_sample(src_root, args.size, rng)
    print(f"Eligible images found: {total_seen:,}")

    if total_seen < args.size:
        raise ValueError(
            f"Requested {args.size:,} images, but only found {total_seen:,} eligible images in {src_root}."
        )

    rng.shuffle(sample)

    dst_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    for src in sample:
        link_or_copy(src, dst_root / src.name, mode=args.mode)

    names = sorted(path.name for path in sample)
    manifest_path.write_text("".join(f"{name}\n" for name in names))

    summary = {
        "source_root": str(src_root),
        "subset_root": str(dst_root),
        "manifest_path": str(manifest_path),
        "size": args.size,
        "seed": args.seed,
        "mode": args.mode,
        "source_count": total_seen,
    }
    summary_path = dst_root.parent / "subset6_minus_7_train_flat_10_000_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote subset: {dst_root}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
