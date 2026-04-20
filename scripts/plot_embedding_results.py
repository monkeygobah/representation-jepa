from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_ROOT = PROJECT_ROOT / "embedding_extract" / "outputs" / "tables"
RESULTS_ROOT = PROJECT_ROOT / "embedding_extract" / "results"
DATA_ROOT = RESULTS_ROOT / "data"
FIGURES_ROOT = RESULTS_ROOT / "figures"

SCALE_ORDER = ["10k", "100k", "1m"]
SCALE_LABELS = [r"$10^4$", r"$10^5$", r"$10^6$"]
DATASET_ORDER = ["subset6_minus_7_test", "subset7_eval", "cfc_eval"]
OBJECTIVE_ORDER = ["infonce", "lejepa", "vicreg"]
INIT_ORDER = ["random", "imagenet", "seginit"]
OVERVIEW_METRIC_SPECS = (
    ("erank_over_d", "Effective Rank / D ↑"),
    ("ev1", "Top-1 Explained Variance ↓"),
    ("cos_std", "Cosine Std ↓"),
    ("cond_1_med", "Cond(1, median) ↓"),
)
FOCUSED_METRIC_SPECS = OVERVIEW_METRIC_SPECS + (
    ("ev5", "Top-5 Explained Variance ↓"),
    ("ev20", "Top-20 Explained Variance ↓"),
)

OBJECTIVE_COLORS = {
    "infonce": "#0b6e4f",
    "lejepa": "#8e6c08",
    "vicreg": "#b03a2e",
}
INIT_LINESTYLES = {
    "random": "--",
    "imagenet": "-",
    "seginit": ":",
}
INIT_MARKERS = {
    "random": "o",
    "imagenet": "s",
    "seginit": "^",
}


def _parse_run_name(run_name: str) -> tuple[str, str, str]:
    parts = run_name.split("-")
    if len(parts) < 5:
        raise ValueError(f"Unexpected run_name format: {run_name}")
    return parts[2], parts[3], parts[4]


def build_summary_specs(summary_kind: str) -> tuple[tuple[str, Path], ...]:
    suffix = "_emb" if summary_kind == "emb" else ""
    return (
        ("10k", TABLES_ROOT / "geometry_10k" / f"isotropy_summary_10k{suffix}.csv"),
        ("100k", TABLES_ROOT / "geometry_100k" / f"isotropy_summary_100k{suffix}.csv"),
        ("1m", TABLES_ROOT / "geometry_1m" / f"isotropy_summary_1m{suffix}.csv"),
    )


def load_combined_summary(summary_kind: str) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for declared_scale, path in build_summary_specs(summary_kind):
        if not path.exists():
            raise FileNotFoundError(f"Missing summary CSV: {path}")

        df = pd.read_csv(path)
        parsed = df["run_name"].apply(_parse_run_name)
        df[["scale", "objective", "init"]] = pd.DataFrame(parsed.tolist(), index=df.index)
        if not (df["scale"] == declared_scale).all():
            raise ValueError(f"Scale mismatch while loading {path}")
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    numeric_cols = [
        "checkpoint_step",
        "N",
        "D",
        "mean_norm",
        "erank",
        "erank_over_d",
        "ev1",
        "ev5",
        "ev10",
        "ev20",
        "cond_1_med",
        "cos_mean",
        "cos_std",
        "cos_std_expected_sphere",
        "cos_frac_abs_gt_0.2",
        "cos_frac_abs_gt_0.3",
        "cos_frac_abs_gt_0.4",
        "num_pairs_used",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["scale"] = pd.Categorical(out["scale"], categories=SCALE_ORDER, ordered=True)
    out["dataset_name"] = pd.Categorical(
        out["dataset_name"], categories=DATASET_ORDER, ordered=True
    )
    out["objective"] = pd.Categorical(
        out["objective"], categories=OBJECTIVE_ORDER, ordered=True
    )
    out["init"] = pd.Categorical(out["init"], categories=INIT_ORDER, ordered=True)
    return out.sort_values(["dataset_name", "objective", "init", "scale"]).reset_index(drop=True)


def write_combined_outputs(df: pd.DataFrame, summary_kind: str) -> tuple[Path, Path]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    combined_path = DATA_ROOT / f"isotropy_summary_all_{summary_kind}.csv"
    plot_ready_path = DATA_ROOT / f"isotropy_summary_plot_ready_{summary_kind}.csv"

    df.to_csv(combined_path, index=False)

    plot_cols = [
        "run_name",
        "scale",
        "objective",
        "init",
        "dataset_name",
        "split_label",
        "embedding_key",
        "N",
        "D",
        "erank_over_d",
        "ev1",
        "cos_std",
        "cond_1_med",
    ]
    df.loc[:, plot_cols].to_csv(plot_ready_path, index=False)
    return combined_path, plot_ready_path


def make_init_focused_figures(df: pd.DataFrame, embedding_key: str = "proj") -> list[Path]:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    subset = df[df["embedding_key"] == embedding_key].copy()
    if subset.empty:
        raise ValueError(f"No rows found for embedding_key={embedding_key}")

    outputs: list[Path] = []
    x_positions = list(range(len(SCALE_ORDER)))

    for metric, title in FOCUSED_METRIC_SPECS:
        fig, axes = plt.subplots(
            nrows=len(DATASET_ORDER),
            ncols=len(INIT_ORDER),
            figsize=(14, 10),
            sharex=True,
        )

        for row_idx, dataset_name in enumerate(DATASET_ORDER):
            dataset_df = subset[subset["dataset_name"] == dataset_name]
            for col_idx, init in enumerate(INIT_ORDER):
                ax = axes[row_idx][col_idx]
                panel_df = dataset_df[dataset_df["init"] == init]

                for objective in OBJECTIVE_ORDER:
                    line_df = panel_df[panel_df["objective"] == objective].sort_values("scale")
                    if line_df.empty:
                        continue

                    ax.plot(
                        x_positions,
                        line_df[metric].to_numpy(),
                        color=OBJECTIVE_COLORS[objective],
                        linestyle="-",
                        marker="o",
                        linewidth=2.2,
                        markersize=6,
                        alpha=0.95,
                        label=objective,
                    )

                if row_idx == 0:
                    ax.set_title(init)
                if col_idx == 0:
                    ax.set_ylabel(dataset_name)
                if row_idx == len(DATASET_ORDER) - 1:
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(SCALE_LABELS)
                    ax.set_xlabel("Training Set Size")
                else:
                    ax.set_xticks(x_positions, [])
                ax.grid(True, alpha=0.25, linewidth=0.8)

        objective_handles = [
            plt.Line2D(
                [0],
                [0],
                color=OBJECTIVE_COLORS[objective],
                linestyle="-",
                marker="o",
                linewidth=2.2,
                markersize=6,
                label=objective,
            )
            for objective in OBJECTIVE_ORDER
        ]
        fig.legend(
            handles=objective_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=len(OBJECTIVE_ORDER),
            frameon=False,
            title="Objective",
        )
        fig.suptitle(f"{title} Across Scale by Initialization", fontsize=16, y=1.01)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        png_path = FIGURES_ROOT / f"{metric}_by_init_{embedding_key}.png"
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        outputs.append(png_path)

    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embedding-key",
        default="proj",
        help="Embedding key to plot from the combined summaries (default: proj)",
    )
    ap.add_argument(
        "--summary-kind",
        choices=["proj", "emb"],
        default="proj",
        help="Which aggregated summary family to read (default: proj)",
    )
    args = ap.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_combined_summary(args.summary_kind)
    combined_path, plot_ready_path = write_combined_outputs(df, args.summary_kind)
    init_outputs = make_init_focused_figures(df, embedding_key=args.embedding_key)

    print(f"Wrote combined summary: {combined_path}")
    print(f"Wrote plot-ready table: {plot_ready_path}")
    for init_png in init_outputs:
        print(f"Wrote init figure PNG: {init_png}")


if __name__ == "__main__":
    main()
