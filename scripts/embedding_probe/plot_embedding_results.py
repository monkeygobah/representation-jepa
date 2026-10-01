from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    if len(parts) >= 8 and parts[2] == "vit" and parts[3] == "b16":
        return parts[4], parts[5], parts[6]
    return parts[2], parts[3], parts[4]


def build_summary_specs(
    summary_kind: str,
    study_tag: str = "",
    study_prefix: str = "geometry",
    summary_prefix: str = "isotropy_summary",
) -> tuple[tuple[str, Path], ...]:
    suffix = "_emb" if summary_kind == "emb" else ""
    return (
        (
            "10k",
            TABLES_ROOT
            / f"{study_prefix}_10k{study_tag}"
            / f"{summary_prefix}_10k{study_tag}{suffix}.csv",
        ),
        (
            "100k",
            TABLES_ROOT
            / f"{study_prefix}_100k{study_tag}"
            / f"{summary_prefix}_100k{study_tag}{suffix}.csv",
        ),
        (
            "1m",
            TABLES_ROOT
            / f"{study_prefix}_1m{study_tag}"
            / f"{summary_prefix}_1m{study_tag}{suffix}.csv",
        ),
    )


def load_combined_summary(
    summary_kind: str,
    study_tag: str = "",
    study_prefix: str = "geometry",
    summary_prefix: str = "isotropy_summary",
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for declared_scale, path in build_summary_specs(
        summary_kind,
        study_tag=study_tag,
        study_prefix=study_prefix,
        summary_prefix=summary_prefix,
    ):
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


def write_combined_outputs(df: pd.DataFrame, summary_kind: str, output_tag: str = "") -> tuple[Path, Path]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_tag}" if output_tag else ""
    combined_path = DATA_ROOT / f"isotropy_summary_all_{summary_kind}{suffix}.csv"
    plot_ready_path = DATA_ROOT / f"isotropy_summary_plot_ready_{summary_kind}{suffix}.csv"

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


def make_init_focused_figures(
    df: pd.DataFrame,
    embedding_key: str = "proj",
    output_tag: str = "",
    layout: str = "by_init",
) -> list[Path]:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    subset = df[df["embedding_key"] == embedding_key].copy()
    if subset.empty:
        raise ValueError(f"No rows found for embedding_key={embedding_key}")

    outputs: list[Path] = []
    x_positions = list(range(len(SCALE_ORDER)))

    for metric, title in FOCUSED_METRIC_SPECS:
        if layout == "vit_compact":
            present_inits = [init for init in INIT_ORDER if subset["init"].eq(init).any()]
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 8), sharex=True)
            axes_flat = axes.flatten()

            for idx, dataset_name in enumerate(DATASET_ORDER):
                ax = axes_flat[idx]
                dataset_df = subset[subset["dataset_name"] == dataset_name]

                for objective in OBJECTIVE_ORDER:
                    for init in present_inits:
                        line_df = dataset_df[
                            (dataset_df["objective"] == objective) & (dataset_df["init"] == init)
                        ].sort_values("scale")
                        if line_df.empty:
                            continue

                        ax.plot(
                            x_positions,
                            line_df[metric].to_numpy(),
                            color=OBJECTIVE_COLORS[objective],
                            linestyle=INIT_LINESTYLES.get(init, "-"),
                            marker=INIT_MARKERS.get(init, "o"),
                            linewidth=2.2,
                            markersize=6,
                            alpha=0.95,
                        )

                ax.set_title(dataset_name)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(SCALE_LABELS)
                ax.set_xlabel("SSL training images")
                if idx in (0, 2):
                    ax.set_ylabel(title)
                ax.grid(True, alpha=0.25, linewidth=0.8)

            legend_ax = axes_flat[3]
            legend_ax.axis("off")
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
            init_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    color="#222222",
                    linestyle=INIT_LINESTYLES.get(init, "-"),
                    marker=INIT_MARKERS.get(init, "o"),
                    linewidth=2.2,
                    markersize=6,
                    label=init.capitalize(),
                )
                for init in present_inits
            ]
            legend_ax.legend(
                handles=objective_handles + init_handles,
                loc="center",
                frameon=False,
                ncol=1,
                title="Legend",
            )
            fig.suptitle(f"{title} Across Scale (ViT-B/16)", fontsize=16, y=0.98)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
        else:
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

        suffix = f"_{output_tag}" if output_tag else ""
        png_path = FIGURES_ROOT / f"{metric}_by_init_{embedding_key}{suffix}.png"
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
    ap.add_argument(
        "--study-tag",
        default="",
        help="Study-name suffix used in table paths, e.g. '_50ksteps' (default: empty)",
    )
    ap.add_argument(
        "--output-tag",
        default="",
        help="Tag appended to result filenames, e.g. '50ksteps' (default: empty)",
    )
    ap.add_argument(
        "--study-prefix",
        default="geometry",
        help="Study-name prefix used in table paths, e.g. 'geometry_vit_b16' (default: geometry)",
    )
    ap.add_argument(
        "--summary-prefix",
        default="isotropy_summary",
        help="Summary CSV filename prefix, e.g. 'isotropy_summary_vit_b16' (default: isotropy_summary)",
    )
    ap.add_argument(
        "--layout",
        choices=["by_init", "vit_compact"],
        default="by_init",
        help="Figure layout to render (default: by_init)",
    )
    args = ap.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_combined_summary(
        args.summary_kind,
        study_tag=args.study_tag,
        study_prefix=args.study_prefix,
        summary_prefix=args.summary_prefix,
    )
    combined_path, plot_ready_path = write_combined_outputs(
        df, args.summary_kind, output_tag=args.output_tag
    )
    init_outputs = make_init_focused_figures(
        df,
        embedding_key=args.embedding_key,
        output_tag=args.output_tag,
        layout=args.layout,
    )

    print(f"Wrote combined summary: {combined_path}")
    print(f"Wrote plot-ready table: {plot_ready_path}")
    for init_png in init_outputs:
        print(f"Wrote init figure PNG: {init_png}")


if __name__ == "__main__":
    main()
