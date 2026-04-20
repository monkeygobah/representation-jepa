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

BASELINE_SPECS = (
    ("10k", TABLES_ROOT / "geometry_10k" / "isotropy_summary_10k.csv"),
    ("100k", TABLES_ROOT / "geometry_100k" / "isotropy_summary_100k.csv"),
    ("1m", TABLES_ROOT / "geometry_1m" / "isotropy_summary_1m.csv"),
)
FOLLOWUP_PATH = (
    TABLES_ROOT
    / "geometry_50ksteps_imagenet_proj"
    / "isotropy_summary_50ksteps_imagenet_proj.csv"
)

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
BUDGET_STYLES = {
    "10ksteps": "--",
    "50ksteps": "-",
}
BUDGET_LABELS = {
    "10ksteps": "10k steps",
    "50ksteps": "50k steps",
}


def _parse_run_name(run_name: str) -> tuple[str, str, str]:
    parts = run_name.split("-")
    if len(parts) < 5:
        raise ValueError(f"Unexpected run_name format: {run_name}")
    return parts[2], parts[3], parts[4]


def load_comparison_summary() -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    for declared_scale, path in BASELINE_SPECS:
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline summary CSV: {path}")
        df = pd.read_csv(path)
        parsed = df["run_name"].apply(_parse_run_name)
        df[["scale", "objective", "init"]] = pd.DataFrame(parsed.tolist(), index=df.index)
        df["budget"] = "10ksteps"
        df = df[(df["init"] == "imagenet") & (df["embedding_key"] == "proj")].copy()
        if not (df["scale"] == declared_scale).all():
            raise ValueError(f"Scale mismatch in {path}")
        dfs.append(df)

    if not FOLLOWUP_PATH.exists():
        raise FileNotFoundError(f"Missing follow-up summary CSV: {FOLLOWUP_PATH}")
    followup = pd.read_csv(FOLLOWUP_PATH)
    parsed = followup["run_name"].apply(_parse_run_name)
    followup[["scale", "objective", "init"]] = pd.DataFrame(parsed.tolist(), index=followup.index)
    followup["budget"] = "50ksteps"
    followup = followup[(followup["init"] == "imagenet") & (followup["embedding_key"] == "proj")].copy()
    dfs.append(followup)

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
    out["dataset_name"] = pd.Categorical(out["dataset_name"], categories=DATASET_ORDER, ordered=True)
    out["objective"] = pd.Categorical(out["objective"], categories=OBJECTIVE_ORDER, ordered=True)
    return out.sort_values(["dataset_name", "objective", "budget", "scale"]).reset_index(drop=True)


def write_plot_ready_table(df: pd.DataFrame) -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = DATA_ROOT / "isotropy_summary_imagenet_10k_vs_50k_plot_ready.csv"
    keep_cols = [
        "run_name",
        "scale",
        "objective",
        "budget",
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
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(15, 4.8), sharex=True)

    for col_idx, dataset_name in enumerate(DATASET_ORDER):
        ax = axes[col_idx]
        panel_df = df[df["dataset_name"] == dataset_name]

        for objective in OBJECTIVE_ORDER:
            for budget in ["10ksteps", "50ksteps"]:
                line_df = panel_df[
                    (panel_df["objective"] == objective) & (panel_df["budget"] == budget)
                ].sort_values("scale")
                if line_df.empty:
                    continue

                ax.plot(
                    x_positions,
                    line_df[metric].to_numpy(),
                    color=OBJECTIVE_COLORS[objective],
                    linestyle=BUDGET_STYLES[budget],
                    marker="o",
                    linewidth=2.4,
                    markersize=6,
                    alpha=0.95,
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
            [0], [0],
            color=OBJECTIVE_COLORS[obj],
            linestyle="-",
            marker="o",
            linewidth=2.4,
            markersize=6,
            label=obj,
        )
        for obj in OBJECTIVE_ORDER
    ]
    budget_handles = [
        plt.Line2D(
            [0], [0],
            color="#333333",
            linestyle=BUDGET_STYLES[budget],
            marker="o",
            linewidth=2.4,
            markersize=6,
            label=BUDGET_LABELS[budget],
        )
        for budget in ["10ksteps", "50ksteps"]
    ]
    fig.legend(
        handles=objective_handles,
        loc="upper center",
        bbox_to_anchor=(0.32, 1.03),
        ncol=len(OBJECTIVE_ORDER),
        frameon=False,
        title="Objective",
    )
    fig.legend(
        handles=budget_handles,
        loc="upper center",
        bbox_to_anchor=(0.80, 1.03),
        ncol=2,
        frameon=False,
        title="Training Budget",
    )
    fig.suptitle(f"{title}: ImageNet Init, 10k vs 50k Steps", fontsize=16, y=1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = FIGURES_ROOT / f"{metric}_imagenet_10k_vs_50k_proj.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metric",
        choices=[metric for metric, _ in METRIC_SPECS] + ["all"],
        default="all",
        help="Metric to plot, or 'all' to render the full comparison figure set.",
    )
    args = ap.parse_args()

    df = load_comparison_summary()
    plot_ready_path = write_plot_ready_table(df)
    print(f"Wrote plot-ready table: {plot_ready_path}")

    for metric, title in METRIC_SPECS:
        if args.metric != "all" and args.metric != metric:
            continue
        out_path = make_metric_figure(df, metric, title)
        print(f"Wrote figure: {out_path}")


if __name__ == "__main__":
    main()
