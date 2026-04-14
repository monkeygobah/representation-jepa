from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding_extract.analyze_pipeline import analyze_study
from embedding_extract.pipeline_config import load_study_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to study config YAML")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing metric outputs")
    args = ap.parse_args()

    cfg = load_study_config(args.cfg)
    written = analyze_study(cfg, overwrite=args.overwrite)
    print(f"Wrote {len(written)} metric artifact(s).")


if __name__ == "__main__":
    main()
