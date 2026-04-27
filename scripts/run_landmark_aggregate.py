from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from landmark_probe.aggregate.pipeline import aggregate_study
from landmark_probe.config import load_study_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to study config YAML")
    args = ap.parse_args()

    study_cfg = load_study_config(args.cfg)
    overall, per_landmark = aggregate_study(study_cfg)
    print(f"Wrote summaries:\n- {overall}\n- {per_landmark}")


if __name__ == "__main__":
    main()
