from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from disease_embeddings.config import load_study_config
from disease_embeddings.summarize import summarize_adapted_knn5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to disease embedding config YAML")
    args = ap.parse_args()

    cfg = load_study_config(args.cfg)
    out_path = summarize_adapted_knn5(cfg)
    df = pd.read_csv(out_path)
    cols = ["model_label", "embedding_source", "knn5_accuracy", "knn5_balanced_accuracy", "knn5_macro_f1", "test_rows"]
    print(df[[col for col in cols if col in df.columns]].to_string(index=False))
    print(f"\nWrote:\n{out_path}")


if __name__ == "__main__":
    main()
