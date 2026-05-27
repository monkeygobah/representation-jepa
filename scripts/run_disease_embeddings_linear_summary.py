from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from disease_embeddings.config import load_study_config
from disease_embeddings.summarize import format_headline_table, summarize_linear_probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to disease embedding config YAML")
    args = ap.parse_args()

    cfg = load_study_config(args.cfg)
    paths = summarize_linear_probe(cfg)
    print(format_headline_table(paths["headline"]))
    print("\nWrote:")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
