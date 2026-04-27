from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from landmark_probe.config import load_dataset_config
from landmark_probe.prepare.pipeline import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to dataset config YAML")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing prepared dataset")
    ap.add_argument("--max-samples-per-dataset", type=int, default=None, help="Optional cap for smoke testing")
    args = ap.parse_args()

    cfg = load_dataset_config(args.cfg)
    manifest, landmarks, splits = build_dataset(
        cfg,
        overwrite=args.overwrite,
        max_samples_per_dataset=args.max_samples_per_dataset,
    )
    print(f"Wrote prepared dataset metadata:\n- {manifest}\n- {landmarks}\n- {splits}")


if __name__ == "__main__":
    main()
