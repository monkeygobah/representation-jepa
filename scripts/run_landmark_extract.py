from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from landmark_probe.config import load_dataset_config, load_study_config
from landmark_probe.extract.pipeline import extract_study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to study config YAML")
    args = ap.parse_args()

    study_cfg = load_study_config(args.cfg)
    dataset_cfg = load_dataset_config(study_cfg.dataset_cfg_path)
    written = extract_study(study_cfg, dataset_cfg)
    print(f"Wrote {len(written)} embedding artifact(s).")


if __name__ == "__main__":
    main()
