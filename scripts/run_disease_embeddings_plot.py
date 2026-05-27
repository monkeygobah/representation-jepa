from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from disease_embeddings.config import load_study_config
from disease_embeddings.reduce_plot import reduce_and_plot_study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to disease embedding config YAML")
    ap.add_argument("--method", default=None, choices=("tsne", "pca"), help="Reduction method")
    args = ap.parse_args()

    cfg = load_study_config(args.cfg)
    written = reduce_and_plot_study(cfg, method=args.method)
    print(f"Wrote {len(written)} reduction/figure pair(s).")
    for csv_path, fig_path in written:
        print(csv_path)
        print(fig_path)


if __name__ == "__main__":
    main()

