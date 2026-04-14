from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding_extract.aggregate_pipeline import aggregate_study
from embedding_extract.pipeline_config import load_study_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to study config YAML")
    args = ap.parse_args()

    cfg = load_study_config(args.cfg)
    out_path = aggregate_study(cfg)
    print(f"Wrote summary table: {out_path}")


if __name__ == "__main__":
    main()
