from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from disease_embeddings.config import load_study_config
from disease_embeddings.supervised import run_linear_probe_study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to disease embedding config YAML")
    args = ap.parse_args()

    cfg = load_study_config(args.cfg)
    written = run_linear_probe_study(cfg)
    print(f"Wrote {len(written)} linear probe run(s).")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()

