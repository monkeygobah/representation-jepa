from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


METHODS = ("infonce", "lejepa", "vicreg")
INITS = ("random", "imagenet")
SIZES = ("10k", "100k", "1m")


@dataclass(frozen=True)
class CheckpointItem:
    family: str
    model_id: str
    source_run_dir: Path
    checkpoint: Path
    config: Path
    dest_run_dir: Path


def find_one(pattern: str, root: Path) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {root / pattern}, found {len(matches)}")
    return matches[0]


def build_items(repo_root: Path, out_root: Path, include_final_vit: bool) -> list[CheckpointItem]:
    items: list[CheckpointItem] = []

    runs_root = repo_root / "runs"
    for size in SIZES:
        for method in METHODS:
            for init in INITS:
                model_id = f"geometry-fixedcompute-{size}-{method}-{init}-50ksteps"
                run_dir = find_one(f"*__{model_id}", runs_root)
                ckpt = run_dir / "checkpoints" / "ckpt_step_0050000.pth"
                items.append(
                    CheckpointItem(
                        family="resnet101_50k",
                        model_id=model_id,
                        source_run_dir=run_dir,
                        checkpoint=ckpt,
                        config=run_dir / "config.yaml",
                        dest_run_dir=out_root / "resnet101_50k" / model_id,
                    )
                )

    vit_root = repo_root / "delta_checkpoints" / "fixedcompute_vit_b16_50ksteps"
    vit_config_root = repo_root / "configs" / "delta" / "fixedcompute_vit_b16_50ksteps"
    for size in SIZES:
        for method in METHODS:
            for init in INITS:
                model_id = f"geometry-fixedcompute-vit-b16-{size}-{method}-{init}-50ksteps"
                run_dir = find_one(f"*__{model_id}", vit_root)
                ckpt = run_dir / "checkpoints" / "ckpt_step_0050000.pth"
                config = vit_config_root / f"{model_id}.yaml"
                items.append(
                    CheckpointItem(
                        family="vit_b16_50k",
                        model_id=model_id,
                        source_run_dir=run_dir,
                        checkpoint=ckpt,
                        config=config,
                        dest_run_dir=out_root / "vit_b16_50k" / model_id,
                    )
                )

    if include_final_vit:
        final_vit_root = repo_root / "models" / "delta_b16_final_checkpoints"
        final_models = {
            "delta-lejepa-vit-b16": "*__delta-lejepa-vit-b16-*",
            "delta-infonce-vit-b16": "*__delta-infonce-vit-b16-*",
            "delta-vicreg-vit-b16": "*__delta-vicreg-vit-b16-*",
        }
        for model_id, pattern in final_models.items():
            run_dir = find_one(pattern, final_vit_root)
            ckpts = sorted((run_dir / "checkpoints").glob("ckpt_step_*.pth"))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoint found in {run_dir / 'checkpoints'}")
            items.append(
                CheckpointItem(
                    family="vit_b16",
                    model_id=model_id,
                    source_run_dir=run_dir,
                    checkpoint=ckpts[-1],
                    config=run_dir / "config.yaml",
                    dest_run_dir=out_root / "vit_b16" / model_id,
                )
            )
    return items


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            h.update(chunk)
    return h.hexdigest()


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def stage_item(item: CheckpointItem, mode: str) -> None:
    if not item.config.exists():
        raise FileNotFoundError(item.config)
    if not item.checkpoint.exists():
        raise FileNotFoundError(item.checkpoint)
    link_or_copy(item.config, item.dest_run_dir / "config.yaml", mode)
    link_or_copy(item.checkpoint, item.dest_run_dir / "checkpoints" / item.checkpoint.name, mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", help="Path to the experiment repo root")
    ap.add_argument("--out-dir", required=True, help="Destination checkpoint bundle directory")
    ap.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    ap.add_argument(
        "--include-final-vit",
        action="store_true",
        help="Also stage the three full-corpus ViT-B/16 checkpoints used for disease experiments",
    )
    ap.add_argument("--sha256", action="store_true", help="Compute sha256 checksums in the manifest")
    ap.add_argument("--dry-run", action="store_true", help="Print planned files without staging")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_root = Path(args.out_dir).resolve()
    items = build_items(repo_root, out_root, include_final_vit=args.include_final_vit)

    total_bytes = sum(item.checkpoint.stat().st_size for item in items)
    print(f"Found {len(items)} checkpoint items ({total_bytes / 1e9:.2f} GB of checkpoint files).")
    for item in items:
        print(f"{item.family}/{item.model_id} <- {item.checkpoint}")

    if args.dry_run:
        return

    rows = []
    for item in items:
        stage_item(item, args.mode)
        dest_ckpt = item.dest_run_dir / "checkpoints" / item.checkpoint.name
        rows.append(
            {
                "family": item.family,
                "model_id": item.model_id,
                "dest_run_dir": str(item.dest_run_dir.relative_to(out_root)),
                "checkpoint_file": str(dest_ckpt.relative_to(out_root)),
                "source_run_dir": str(item.source_run_dir),
                "source_config": str(item.config),
                "source_checkpoint": str(item.checkpoint),
                "size_bytes": item.checkpoint.stat().st_size,
                "sha256": sha256(item.checkpoint) if args.sha256 else "",
            }
        )

    manifest_path = out_root / "checkpoint_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
