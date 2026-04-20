from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = (
    PROJECT_ROOT
    / "embedding_extract"
    / "outputs"
    / "tables"
    / "geometry_50ksteps_imagenet_proj"
    / "isotropy_summary_50ksteps_imagenet_proj.csv"
)
RESULTS_ROOT = PROJECT_ROOT / "embedding_extract" / "results"
DATA_ROOT = RESULTS_ROOT / "data"
FIGURES_ROOT = RESULTS_ROOT / "figures"

SCALE_ORDER = ["10k", "100k", "1m"]
SCALE_LABELS = [r"$10^4$", r"$10^5$", r"$10^6$"]
DATASET_ORDER = ["subset6_minus_7_test", "subset7_eval", "cfc_eval"]
OBJECTIVE_ORDER = ["infonce", "lejepa", "vicreg"]

METRIC_SPECS = (
    ("erank_over_d", "Effective Rank / D ↑"),
    ("ev1", "Top-1 Explained Variance ↓"),
    ("ev5", "Top-5 Explained Variance ↓"),
    ("ev20", "Top-20 Explained Variance ↓"),
    ("cos_std", "Cosine Std ↓"),
    ("cond_1_med", "Cond(1, median) ↓"),
)

OBJECTIVE_COLORS = {
    "infonce": "#0b6e4f",
    "lejepa": "#8e6c08",
    "vicreg": "#b03a2e",
}


def _parse_run_name(run_name: str) -> tuple[str, str, str, str]:
    parts = run_name.split("-")
    if len(parts) < 6:
        raise ValueError(f"Unexpected run_name format: {run_name}")
    return parts[2], parts[3], parts[4], parts[5]


def load_followup_summary() -> pd.DataFrame:
    if not TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing follow-up summary CSV: {TABLE_PATH}")

    df = pd.read_csv(TABLE_PATH)
    parsed = df["run_name"].apply(_parse_run_name)
    df[["scale", "objective", "init", "budget"]] = pd.DataFrame(parsed.tolist(), index=df.index)

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
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["scale"] = pd.Categorical(df["scale"], categories=SCALE_ORDER, ordered=True)
    df["dataset_name"] = pd.Categorical(df["dataset_name"], categories=DATASET_ORDER, ordered=True)
    df["objective"] = pd.Categorical(df["objective"], categories=OBJECTIVE_ORDER, ordered=True)
    return df.sort_values(["dataset_name", "objective", "scale"]).reset_index(drop=True)


def write_plot_ready_table(df: pd.DataFrame) -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = DATA_ROOT / "isotropy_summary_50ksteps_imagenet_proj_plot_ready.csv"
    keep_cols = [
        "run_name",
        "scale",
        "objective",
        "dataset_name",
        "embedding_key",
        "N",
        "D",
        "erank_over_d",
        "ev1",
        "ev5",
        "ev20",
        "cos_std",
        "cond_1_med",
    ]
    df.loc[:, keep_cols].to_csv(out_path, index=False)
    return out_path


def make_metric_figure(df: pd.DataFrame, metric: str, title: str) -> Path:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    x_positions = list(range(len(SCALE_ORDER)))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(DATASET_ORDER),
        figsize=(15, 4.8),
        sharex=True,
    )

    for col_idx, dataset_name in enumerate(DATASET_ORDER):
        ax = axes[col_idx]
        panel_df = df[df["dataset_name"] == dataset_name]

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
                linewidth=2.4,
                markersize=6,
                alpha=0.95,
                label=objective,
            )

        ax.set_title(dataset_name)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(SCALE_LABELS)
        ax.set_xlabel("Training Set Size")
        if col_idx == 0:
            ax.set_ylabel(title)
        ax.grid(True, alpha=0.25, linewidth=0.8)

    objective_handles = [
        plt.Line2D(
            [0],
            [0],
            color=OBJECTIVE_COLORS[objective],
            linestyle="-",
            marker="o",
            linewidth=2.4,
            markersize=6,
            label=objective,
        )
        for objective in OBJECTIVE_ORDER
    ]
    fig.legend(
        handles=objective_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(OBJECTIVE_ORDER),
        frameon=False,
        title="Objective",
    )
    fig.suptitle(f"{title} at 50k Steps with ImageNet Init", fontsize=16, y=1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = FIGURES_ROOT / f"{metric}_followup_50ksteps_imagenet_proj.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metric",
        choices=[metric for metric, _ in METRIC_SPECS] + ["all"],
        default="all",
        help="Metric to plot, or 'all' to render the full figure set.",
    )
    args = ap.parse_args()

    df = load_followup_summary()
    plot_ready_path = write_plot_ready_table(df)
    print(f"Wrote plot-ready table: {plot_ready_path}")

    for metric, title in METRIC_SPECS:
        if args.metric != "all" and args.metric != metric:
            continue
        out_path = make_metric_figure(df, metric, title)
        print(f"Wrote figure: {out_path}")


if __name__ == "__main__":
    main()
