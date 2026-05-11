from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from landmark_probe.config import load_dataset_config, load_study_config
from landmark_probe.extract.pipeline import extract_study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to study config YAML")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing embedding artifacts")
    args = ap.parse_args()

    study_cfg = load_study_config(args.cfg)
    if args.overwrite:
        study_cfg = replace(study_cfg, extraction=replace(study_cfg.extraction, overwrite=True))
    dataset_cfg = load_dataset_config(study_cfg.dataset_cfg_path)
    written = extract_study(study_cfg, dataset_cfg)
    print(f"Wrote {len(written)} embedding artifact(s).")


if __name__ == "__main__":
    main()
