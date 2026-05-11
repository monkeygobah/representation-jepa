from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_ROOT = PROJECT_ROOT / "landmark_probe" / "outputs" / "summaries"
RESULTS_ROOT = PROJECT_ROOT / "landmark_probe" / "outputs" / "analysis"

TASK_ORDER = ["celeb_within", "cfd_within"]
TASK_LABELS = {
    "celeb_within": "Celeb within",
    "cfd_within": "CFD within",
}
INIT_ORDER = ["random", "imagenet", "seg_init"]
INIT_LABELS = {
    "random": "Random init",
    "imagenet": "ImageNet init",
    "seg_init": "Seg init",
}
OBJECTIVE_ORDER = ["infonce", "lejepa", "vicreg"]
OBJECTIVE_LABELS = {
    "infonce": "InfoNCE",
    "lejepa": "LeJEPA",
    "vicreg": "VICReg",
}
OBJECTIVE_COLORS = {
    "infonce": "#0b6e4f",
    "lejepa": "#8e6c08",
    "vicreg": "#b03a2e",
}
SCALE_ORDER = ["10k", "100k", "1m"]
SCALE_LABELS = [r"$10^4$", r"$10^5$", r"$10^6$"]
SCALE_RE = re.compile(r"geometry-fixedcompute-(10k|100k|1m)-")


def parse_scale(run_name: str) -> str | None:
    match = SCALE_RE.search(run_name)
    return match.group(1) if match else None


def load_plot_data(summary_csv: Path, pooling: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    df = pd.read_csv(summary_csv)
    df = df[(df["pooling"] == pooling) & (df["task_name"].isin(TASK_ORDER))].copy()
    df["test_mean_l2"] = pd.to_numeric(df["test_mean_l2"], errors="coerce")

    trained = df[df["ssl_method"].isin(OBJECTIVE_ORDER)].copy()
    trained["scale"] = trained["run_name"].map(parse_scale)
    trained = trained.dropna(subset=["scale", "test_mean_l2"])
    trained["scale"] = pd.Categorical(trained["scale"], categories=SCALE_ORDER, ordered=True)
    trained["ssl_method"] = pd.Categorical(
        trained["ssl_method"], categories=OBJECTIVE_ORDER, ordered=True
    )
    trained["init_mode"] = pd.Categorical(
        trained["init_mode"], categories=INIT_ORDER, ordered=True
    )
    trained["task_name"] = pd.Categorical(
        trained["task_name"], categories=TASK_ORDER, ordered=True
    )

    baselines = df[df["ssl_method"] == "baseline"].copy()
    baselines = baselines.dropna(subset=["test_mean_l2"])

    expected = len(TASK_ORDER) * len(INIT_ORDER) * len(OBJECTIVE_ORDER) * len(SCALE_ORDER)
    if len(trained) != expected:
        print(f"Warning: expected {expected} trained rows, found {len(trained)}")

    duplicate_keys = ["task_name", "init_mode", "ssl_method", "scale"]
    dupes = trained[trained.duplicated(duplicate_keys, keep=False)]
    if not dupes.empty:
        raise ValueError(
            "Duplicate trained rows for plot keys:\n"
            + dupes.loc[:, duplicate_keys + ["run_name"]].to_string(index=False)
        )

    return (
        trained.sort_values(["task_name", "init_mode", "ssl_method", "scale"]),
        baselines.sort_values(["task_name", "init_mode"]),
    )


def write_plot_ready_table(trained: pd.DataFrame, baselines: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    trained_table = trained.assign(row_type="trained")
    baseline_table = baselines.assign(row_type="baseline", scale=None)
    cols = [
        "row_type",
        "task_name",
        "run_name",
        "pooling",
        "ssl_method",
        "init_mode",
        "scale",
        "test_mean_l2",
        "best_val_mean_l2",
        "best_epoch",
    ]
    out_path = out_dir / "within_g4_test_mean_l2_plot_ready.csv"
    pd.concat([trained_table, baseline_table], ignore_index=True, sort=False).loc[:, cols].to_csv(
        out_path, index=False
    )
    return out_path


def make_figure(
    trained: pd.DataFrame,
    baselines: pd.DataFrame,
    out_dir: Path,
    metric: str = "test_mean_l2",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = list(range(len(SCALE_ORDER)))
    fig, axes = plt.subplots(
        nrows=len(TASK_ORDER),
        ncols=len(INIT_ORDER),
        figsize=(13.5, 7.2),
        sharex=True,
        sharey=True,
    )

    all_values = pd.concat([trained[metric], baselines[metric]], ignore_index=True).dropna()
    ymin = max(0.0, all_values.min() - 0.08 * (all_values.max() - all_values.min()))
    ymax = all_values.max() + 0.12 * (all_values.max() - all_values.min())

    for row_idx, task_name in enumerate(TASK_ORDER):
        task_df = trained[trained["task_name"] == task_name]
        for col_idx, init_mode in enumerate(INIT_ORDER):
            ax = axes[row_idx][col_idx]
            panel_df = task_df[task_df["init_mode"] == init_mode]

            baseline_df = baselines[
                (baselines["task_name"] == task_name) & (baselines["init_mode"] == init_mode)
            ]
            if len(baseline_df) == 1:
                baseline_l2 = float(baseline_df.iloc[0][metric])
                ax.axhline(
                    baseline_l2,
                    color="#4b5563",
                    linestyle=(0, (4, 3)),
                    linewidth=1.6,
                    alpha=0.85,
                    zorder=1,
                )
                ax.text(
                    2.04,
                    baseline_l2,
                    "baseline",
                    color="#4b5563",
                    fontsize=8.5,
                    va="center",
                    ha="left",
                    clip_on=False,
                )

            for objective in OBJECTIVE_ORDER:
                line_df = panel_df[panel_df["ssl_method"] == objective].sort_values("scale")
                if line_df.empty:
                    continue
                ax.plot(
                    x,
                    line_df[metric].to_numpy(),
                    color=OBJECTIVE_COLORS[objective],
                    marker="o",
                    linewidth=2.2,
                    markersize=5.8,
                    alpha=0.96,
                    label=OBJECTIVE_LABELS[objective],
                    zorder=3,
                )

            ax.set_ylim(ymin, ymax)
            ax.set_xlim(-0.15, 2.4)
            ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
            ax.grid(True, axis="x", alpha=0.08, linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(INIT_LABELS[init_mode], fontsize=12.5, pad=10)
            if col_idx == 0:
                ax.set_ylabel(f"{TASK_LABELS[task_name]}\nMean test L2 error", fontsize=11)
            if row_idx == len(TASK_ORDER) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(SCALE_LABELS)
                ax.set_xlabel("SSL training images", fontsize=10.5)
            else:
                ax.set_xticks(x, [])

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=OBJECTIVE_COLORS[objective],
            marker="o",
            linewidth=2.2,
            markersize=5.8,
            label=OBJECTIVE_LABELS[objective],
        )
        for objective in OBJECTIVE_ORDER
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color="#4b5563",
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            label="Init-only baseline",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Within-Dataset Periorbital Landmark Probe, G4 Features", fontsize=15, y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.0, h_pad=2.0)

    png_path = out_dir / "within_g4_test_mean_l2_by_init.png"
    pdf_path = out_dir / "within_g4_test_mean_l2_by_init.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--study-name",
        default="followup_50k_landmark_probe",
        help="Study summary directory under landmark_probe/outputs/summaries.",
    )
    ap.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional explicit overall_summary.csv path.",
    )
    ap.add_argument("--pooling", default="g4", help="Pooling level to plot.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory for figure and plot-ready data.",
    )
    args = ap.parse_args()

    summary_csv = args.summary_csv or SUMMARY_ROOT / args.study_name / "overall_summary.csv"
    out_dir = args.out_dir or RESULTS_ROOT / args.study_name / "within_g4_summary"

    trained, baselines = load_plot_data(summary_csv, pooling=args.pooling)
    table_path = write_plot_ready_table(trained, baselines, out_dir)
    png_path, pdf_path = make_figure(trained, baselines, out_dir)

    print(f"Wrote plot-ready table: {table_path}")
    print(f"Wrote PNG: {png_path}")
    print(f"Wrote PDF: {pdf_path}")


if __name__ == "__main__":
    main()
