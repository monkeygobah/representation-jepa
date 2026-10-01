from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from paper_figures.metadata import (
    COMPONENT_COLORS,
    DATASET_LABELS,
    DATASET_OBJECTIVE_ORDER,
    EXTERNAL_BASELINE_COLORS,
    EXTERNAL_BASELINE_ORDER,
    GEOMETRY_DATASET_ORDER,
    INIT_LABELS,
    INIT_ORDER,
    OBJECTIVE_COLORS,
    OBJECTIVE_LABELS,
    REP_OBJECTIVE_ORDER,
    SCALE_COLORS,
    SCALE_LABELS,
    SCALE_ORDER,
TASK_LABELS,
TASK_ORDER,
)

LANDMARK_MODEL_ORDER = ["resnet101", "vit_b16"]
LANDMARK_MODEL_LABELS = {
    "resnet101": "ResNet-101",
    "vit_b16": "ViT-B/16",
}
LANDMARK_KEYS = [
    "iris_centroid",
    "iris_superior",
    "iris_inferior",
    "sclera_superior",
    "sclera_inferior",
    "medial_canthus",
    "lateral_canthus",
    "lid_superior",
    "lid_inferior",
]


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    for stale_suffix in (".pdf", ".svg"):
        stale_path = out_dir / f"{stem}{stale_suffix}"
        if stale_path.exists():
            stale_path.unlink()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [path]


def dataset_training_curves(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(
        nrows=len(DATASET_OBJECTIVE_ORDER),
        ncols=len(INIT_ORDER),
        figsize=(9.8, 8.8),
        sharex=True,
    )
    for row_idx, objective in enumerate(DATASET_OBJECTIVE_ORDER):
        for col_idx, init in enumerate(INIT_ORDER):
            ax = axes[row_idx][col_idx]
            panel = df[(df["objective"] == objective) & (df["init"] == init)]
            for scale in SCALE_ORDER:
                line = panel[panel["scale"] == scale].sort_values("step")
                if line.empty:
                    continue
                ax.plot(
                    line["step"],
                    line["loss"],
                    color=SCALE_COLORS[scale],
                    linewidth=2.4,
                    label=SCALE_LABELS[scale],
                )
            if row_idx == 0:
                ax.set_title(INIT_LABELS[init])
            if col_idx == 0:
                ax.set_ylabel(f"{OBJECTIVE_LABELS[objective]}\nLoss")
            if row_idx == len(DATASET_OBJECTIVE_ORDER) - 1:
                ax.set_xlabel("Training step")
            ax.grid(True, alpha=0.24, linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    handles = [
        plt.Line2D([0], [0], color=SCALE_COLORS[scale], linewidth=2.6, label=SCALE_LABELS[scale])
        for scale in SCALE_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    fig.suptitle("50k-step SSL training curves", y=0.975, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_figure(fig, out_dir, "dataset_training_curves")


def dataset_benchmark_schematic(out_dir: Path) -> list[Path]:
    fig = plt.figure(figsize=(13.2, 7.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    def box(x, y, w, h, text, facecolor="#ffffff", edgecolor="#111827", fontsize=11, weight="normal"):
        rect = plt.Rectangle((x, y), w, h, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", linewidth=1.6, color="#374151"),
        )

    # Left: construction pipeline
    box(0.04, 0.70, 0.18, 0.16, "Open facial datasets\n(heterogeneous sources)", facecolor="#eef6ff", weight="bold")
    box(0.28, 0.70, 0.22, 0.16, "Curation and standardization\nDedup | face detect | align |\nfilter | crop | left/right split", facecolor="#f9fafb")
    box(0.56, 0.70, 0.19, 0.16, "Main external-eye corpus\n4,352,692 unilateral\n224x224 crops", facecolor="#eefbf3", weight="bold")
    arrow(0.22, 0.78, 0.28, 0.78)
    arrow(0.50, 0.78, 0.56, 0.78)

    # Middle: pretraining subsets
    box(0.79, 0.77, 0.08, 0.08, "Pretrain\n10K", facecolor="#fff7ed")
    box(0.88, 0.77, 0.08, 0.08, "Pretrain\n100K", facecolor="#fff7ed")
    box(0.79, 0.67, 0.17, 0.08, "Pretrain-1M", facecolor="#fff7ed", weight="bold")
    arrow(0.75, 0.78, 0.79, 0.81)
    arrow(0.75, 0.78, 0.88, 0.81)
    arrow(0.75, 0.76, 0.79, 0.71)

    # Bottom: evaluations
    box(0.07, 0.42, 0.17, 0.12, "Source-Holdout\nsource-matched geometry", facecolor="#f5f3ff")
    box(0.27, 0.42, 0.17, 0.12, "Open-HR\nhigher-quality open-source geometry", facecolor="#f5f3ff")
    box(0.47, 0.42, 0.17, 0.12, "Clinic-CF\nclinical domain-shift geometry", facecolor="#f5f3ff")
    arrow(0.64, 0.67, 0.15, 0.54)
    arrow(0.66, 0.67, 0.35, 0.54)
    arrow(0.68, 0.67, 0.55, 0.54)

    box(0.71, 0.42, 0.12, 0.12, "Frozen\nencoder", facecolor="#ecfeff", weight="bold")
    box(0.86, 0.46, 0.10, 0.08, "Landmark-\nCeleb", facecolor="#fef2f2")
    box(0.86, 0.36, 0.10, 0.08, "Landmark-\nCFD", facecolor="#fef2f2")
    arrow(0.87, 0.67, 0.77, 0.54)
    arrow(0.83, 0.48, 0.86, 0.50)
    arrow(0.83, 0.48, 0.86, 0.40)

    box(0.71, 0.20, 0.12, 0.10, "Backbone\nembedding", facecolor="#ecfeff")
    box(0.86, 0.22, 0.10, 0.08, "MLP\nprobe", facecolor="#ecfeff")
    arrow(0.77, 0.42, 0.77, 0.30)
    arrow(0.83, 0.26, 0.86, 0.26)

    ax.text(0.86, 0.15, "Within-dataset:\nCeleb->Celeb, CFD->CFD", fontsize=10, ha="left", va="top")
    ax.text(0.86, 0.07, "Cross-dataset:\nCeleb->CFD", fontsize=10, ha="left", va="top")

    fig.suptitle("External-Eye Dataset and Benchmark Schematic", y=0.96, fontsize=16)
    return save_figure(fig, out_dir, "dataset_benchmark_schematic")


def dataset_resource_summary_table(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    display = df.copy()
    display["sample_count"] = display["sample_count"].map(lambda v: f"{int(v):,}")
    display["used_for_pretraining"] = display["used_for_pretraining"].map(lambda v: "Yes" if v else "No")
    display["used_for_evaluation"] = display["used_for_evaluation"].map(lambda v: "Yes" if v else "No")
    display = display.rename(
        columns={
            "resource_name": "Resource",
            "role": "Role",
            "source_type": "Source type",
            "label_availability": "Labels",
            "public_release_status": "Public release",
            "redistribution_status": "Redistribution",
            "sample_count": "Samples",
            "used_for_pretraining": "Pretrain",
            "used_for_evaluation": "Eval",
        }
    )

    fig, ax = plt.subplots(figsize=(15.6, 4.8))
    ax.axis("off")
    table_artist = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8.6)
    table_artist.scale(1.0, 1.55)
    for (row, col), cell in table_artist.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(weight="bold")
        if col in (0, 1, 2, 3):
            cell._loc = "left"
    ax.set_title("Dataset and Evaluation Resource Summary", pad=12, fontsize=13)
    fig.tight_layout()
    return save_figure(fig, out_dir, "dataset_resource_summary")


def dataset_geometry_main(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    return _geometry_by_scale(
        df=df,
        out_dir=out_dir,
        stem="dataset_geometry_erank_over_d",
        metric="erank_over_d",
        metric_label="Effective rank / D ↑",
        objective_order=DATASET_OBJECTIVE_ORDER,
    )


def dataset_geometry_supplement(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    metrics = [
        ("ev1", "Top-1 EV ↓", "dataset_geometry_ev1"),
        ("ev5", "Top-5 EV ↓", "dataset_geometry_ev5"),
        ("ev20", "Top-20 EV ↓", "dataset_geometry_ev20"),
        ("cos_std", "Cosine std ↓", "dataset_geometry_cos_std"),
        ("cond_1_med", "Cond(1, median) ↓", "dataset_geometry_cond_1_med"),
    ]
    outputs: list[Path] = []
    for metric, label, stem in metrics:
        outputs.extend(
            _geometry_by_scale(
                df=df,
                out_dir=out_dir,
                stem=stem,
                metric=metric,
                metric_label=label,
                objective_order=DATASET_OBJECTIVE_ORDER,
            )
        )
    return outputs


def dataset_geometry_main_vit(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    return _geometry_by_scale(
        df=df,
        out_dir=out_dir,
        stem="vit_b16_dataset_geometry_erank_over_d",
        metric="erank_over_d",
        metric_label="Effective rank / D ↑",
        objective_order=DATASET_OBJECTIVE_ORDER,
    )


def dataset_geometry_supplement_vit(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    metrics = [
        ("ev1", "Top-1 EV ↓", "vit_b16_dataset_geometry_ev1"),
        ("ev5", "Top-5 EV ↓", "vit_b16_dataset_geometry_ev5"),
        ("ev20", "Top-20 EV ↓", "vit_b16_dataset_geometry_ev20"),
        ("cos_std", "Cosine std ↓", "vit_b16_dataset_geometry_cos_std"),
        ("cond_1_med", "Cond(1, median) ↓", "vit_b16_dataset_geometry_cond_1_med"),
    ]
    outputs: list[Path] = []
    for metric, label, stem in metrics:
        outputs.extend(
            _geometry_by_scale(
                df=df,
                out_dir=out_dir,
                stem=stem,
                metric=metric,
                metric_label=label,
                objective_order=DATASET_OBJECTIVE_ORDER,
            )
        )
    return outputs


def _geometry_by_scale(
    df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    metric: str,
    metric_label: str,
    objective_order: list[str],
) -> list[Path]:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(GEOMETRY_DATASET_ORDER),
        figsize=(13.2, 4.4),
        sharex=True,
        sharey=True,
    )
    x = list(range(len(SCALE_ORDER)))
    init_linestyles = {
        "imagenet": "-",
        "random": (0, (5, 3)),
    }
    for col_idx, dataset_name in enumerate(GEOMETRY_DATASET_ORDER):
        ax = axes[col_idx]
        panel = df[df["dataset_name"] == dataset_name]
        for objective in objective_order:
            for init in INIT_ORDER:
                line = panel[
                    (panel["objective"] == objective) & (panel["init"] == init)
                ].sort_values("scale")
                if line.empty:
                    continue
                ax.plot(
                    x,
                    line[metric],
                    color=OBJECTIVE_COLORS[objective],
                    linestyle=init_linestyles[init],
                    marker="o",
                    linewidth=2.8,
                    markersize=6.8,
                    label=f"{OBJECTIVE_LABELS[objective]} / {INIT_LABELS[init]}",
                )
        ax.set_title(DATASET_LABELS[dataset_name])
        if col_idx == 0:
            ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels([SCALE_LABELS[s] for s in SCALE_ORDER])
        ax.set_xlabel("SSL training images")
        ax.grid(True, alpha=0.22, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    objective_handles = [
        plt.Line2D(
            [0],
            [0],
            color=OBJECTIVE_COLORS[objective],
            marker="o",
            linewidth=2.8,
            markersize=6.5,
            label=OBJECTIVE_LABELS[objective],
        )
        for objective in objective_order
    ]
    init_handles = [
        plt.Line2D([0], [0], color="#111827", linestyle="-", linewidth=2.4, label="ImageNet"),
        plt.Line2D(
            [0],
            [0],
            color="#111827",
            linestyle=(0, (5, 3)),
            linewidth=2.4,
            label="Random",
        ),
    ]
    fig.legend(
        handles=objective_handles + init_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(objective_handles) + len(init_handles),
        frameon=False,
    )
    fig.suptitle(metric_label, y=0.94, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    return save_figure(fig, out_dir, stem)


def dataset_landmarks(
    trained: pd.DataFrame,
    baselines: pd.DataFrame,
    external_baselines: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    written: list[Path] = []
    x = list(range(len(SCALE_ORDER)))
    values = pd.concat(
        [trained["test_mean_l2"], baselines["test_mean_l2"], external_baselines["test_mean_l2"]],
        ignore_index=True,
    ).dropna()
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=OBJECTIVE_COLORS[o],
            marker="o",
            linewidth=2.8,
            markersize=6.5,
            label=OBJECTIVE_LABELS[o],
        )
        for o in DATASET_OBJECTIVE_ORDER
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color="#111827",
            linestyle=(0, (4, 2.4)),
            linewidth=2.2,
            label="Init-only",
        )
    )
    handles.extend(
        [
            plt.Line2D(
                [0],
                [0],
                color=EXTERNAL_BASELINE_COLORS[external_init],
                linestyle=(0, (3.2, 2.2)),
                linewidth=2.1,
                label={"dinov2": "DINOv2", "mae": "MAE"}[external_init],
            )
            for external_init in EXTERNAL_BASELINE_ORDER
        ]
    )

    for task_name in TASK_ORDER:
        fig, axes = plt.subplots(
            nrows=len(LANDMARK_MODEL_ORDER),
            ncols=len(INIT_ORDER),
            figsize=(7.9, 4.6),
            sharex=True,
            sharey=False,
            gridspec_kw={"wspace": 0.08, "hspace": 0.12},
        )

        for row_idx, model_family in enumerate(LANDMARK_MODEL_ORDER):
            model_values = pd.concat(
                [
                    trained.loc[
                        (trained["task_name"] == task_name) & (trained["model_family"] == model_family),
                        "test_mean_l2",
                    ],
                    baselines.loc[
                        (baselines["task_name"] == task_name) & (baselines["model_family"] == model_family),
                        "test_mean_l2",
                    ],
                    external_baselines.loc[external_baselines["task_name"] == task_name, "test_mean_l2"],
                ],
                ignore_index=True,
            ).dropna()
            span = model_values.max() - model_values.min()
            ymin = max(0.0, model_values.min() - 0.08 * span)
            ymax = model_values.max() + 0.10 * span
            if model_family == "vit_b16" and task_name == "celeb_within":
                ymin, ymax = 5.0, 11.0
            if model_family == "vit_b16" and task_name == "cfd_within":
                ymin, ymax = 3.0, 8.0
            for col_idx, init in enumerate(INIT_ORDER):
                ax = axes[row_idx][col_idx]
                panel = trained[
                    (trained["task_name"] == task_name)
                    & (trained["model_family"] == model_family)
                    & (trained["init_mode"] == init)
                ]
                baseline = baselines[
                    (baselines["task_name"] == task_name)
                    & (baselines["model_family"] == model_family)
                    & (baselines["init_mode"] == init)
                ]
                external_panel = external_baselines[
                    external_baselines["task_name"] == task_name
                ]
                if len(baseline) == 1:
                    ax.axhline(
                        float(baseline.iloc[0]["test_mean_l2"]),
                        color="#111827",
                        linestyle=(0, (4, 2.4)),
                        linewidth=2.1,
                        alpha=0.86,
                        label="Init-only",
                    )
                for external_init in EXTERNAL_BASELINE_ORDER:
                    external = external_panel[external_panel["init_mode"] == external_init]
                    if len(external) != 1:
                        continue
                    ax.axhline(
                            float(external.iloc[0]["test_mean_l2"]),
                            color=EXTERNAL_BASELINE_COLORS[external_init],
                            linestyle=(0, (3.2, 2.2)),
                            linewidth=2.0,
                            alpha=0.95,
                            label=INIT_LABELS[external_init],
                        )
                for objective in DATASET_OBJECTIVE_ORDER:
                    line = panel[panel["ssl_method"] == objective].sort_values("scale")
                    ax.plot(
                        x,
                        line["test_mean_l2"],
                        color=OBJECTIVE_COLORS[objective],
                        marker="o",
                        linewidth=2.45,
                        markersize=5.8,
                        label=OBJECTIVE_LABELS[objective],
                    )
                ax.set_xlim(-0.08, len(SCALE_ORDER) - 0.92)
                ax.set_ylim(ymin, ymax)
                if row_idx == 0:
                    ax.set_title(INIT_LABELS[init], fontsize=11, pad=5)
                if col_idx == 0:
                    ax.set_ylabel(f"{LANDMARK_MODEL_LABELS[model_family]}\nMean L2", fontsize=10)
                if row_idx == len(LANDMARK_MODEL_ORDER) - 1:
                    ax.set_xticks(x)
                    ax.set_xticklabels([SCALE_LABELS[s] for s in SCALE_ORDER])
                    ax.set_xlabel("SSL training images", fontsize=10)
                else:
                    ax.set_xticks(x, [])
                ax.grid(True, axis="y", alpha=0.24, linewidth=0.7)
                ax.grid(True, axis="x", alpha=0.08, linewidth=0.6)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="both", labelsize=9)
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=6,
            frameon=False,
            fontsize=8.5,
            handlelength=2.8,
            columnspacing=1.2,
        )
        fig.suptitle(f"G4 landmark probe: {TASK_LABELS[task_name].replace(' Mean L2', '')}", y=0.905, fontsize=12.0)
        fig.subplots_adjust(left=0.105, right=0.995, bottom=0.13, top=0.80, wspace=0.08, hspace=0.12)
        written.extend(save_figure(fig, out_dir, f"dataset_landmark_g4_{task_name}"))
    return written


def _plot_landmark_overlay(ax, row: pd.Series) -> None:
    gt_x = [float(row[f"{lm}_true_x"]) for lm in LANDMARK_KEYS]
    gt_y = [float(row[f"{lm}_true_y"]) for lm in LANDMARK_KEYS]
    pred_x = [float(row[f"{lm}_pred_x"]) for lm in LANDMARK_KEYS]
    pred_y = [float(row[f"{lm}_pred_y"]) for lm in LANDMARK_KEYS]
    ax.scatter(gt_x, gt_y, s=52, c="#13a8a8", edgecolors="white", linewidths=0.8, label="Ground truth", zorder=4)
    ax.scatter(pred_x, pred_y, s=48, c="#d1495b", marker="x", linewidths=2.2, label="Prediction", zorder=5)
    for gx, gy, px, py in zip(gt_x, gt_y, pred_x, pred_y):
        ax.plot([gx, px], [gy, py], color="white", linewidth=0.7, alpha=0.75, zorder=3)


def dataset_landmark_qualitative_examples(samples: pd.DataFrame, out_dir: Path) -> list[Path]:
    written: list[Path] = []
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#13a8a8", markeredgecolor="white", label="Ground truth"),
        plt.Line2D([0], [0], marker="x", color="#d1495b", linestyle="none", markersize=7, label="Prediction"),
    ]
    for example_type in ["best", "worst"]:
        subset = samples[samples["example_type"] == example_type].sort_values("column_order")
        fig, axes = plt.subplots(
            1,
            len(subset),
            figsize=(2.45 * len(subset), 2.85),
            squeeze=False,
            gridspec_kw={"wspace": 0.025},
        )
        for col_idx, row in enumerate(subset.itertuples(index=False)):
            row_s = pd.Series(row._asdict())
            ax = axes[0][col_idx]
            ax.imshow(Image.open(row_s["image_path"]).convert("RGB"))
            _plot_landmark_overlay(ax, row_s)
            title = f"{row_s['model_label']}\n{OBJECTIVE_LABELS[row_s['objective']]}"
            ax.set_title(title, fontsize=8.8, pad=4)
            ax.text(
                0.98,
                0.04,
                f"L2 {float(row_s['mean_l2']):.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.2,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2.0},
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=2, frameon=False, fontsize=9)
        fig.suptitle(f"G4 Celeb-to-CFD Landmark Predictions: {example_type.title()} Examples", y=1.02, fontsize=12)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.78, bottom=0.16, wspace=0.025)
        written.extend(save_figure(fig, out_dir, f"g4_qualitative_landmark_{example_type}_examples"))
    return written


def rep_training_loss(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(
        nrows=len(REP_OBJECTIVE_ORDER),
        ncols=len(INIT_ORDER),
        figsize=(9.8, 8.4),
        sharex=True,
    )
    for row_idx, objective in enumerate(REP_OBJECTIVE_ORDER):
        for col_idx, init in enumerate(INIT_ORDER):
            ax = axes[row_idx][col_idx]
            panel = df[(df["objective"] == objective) & (df["init"] == init)].sort_values("step")
            ax.plot(panel["step"], panel["loss"], color=OBJECTIVE_COLORS[objective], linewidth=1.8)
            if row_idx == 0:
                ax.set_title(INIT_LABELS[init])
            if col_idx == 0:
                ax.set_ylabel(f"{OBJECTIVE_LABELS[objective]}\nLoss")
            if row_idx == len(REP_OBJECTIVE_ORDER) - 1:
                ax.set_xlabel("Training step")
            ax.grid(True, alpha=0.24, linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle("1M / 50k-step representation objective training curves", y=1.02, fontsize=14)
    fig.tight_layout()
    return save_figure(fig, out_dir, "rep_learning_training_loss")


def rep_training_components(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(
        nrows=len(REP_OBJECTIVE_ORDER),
        ncols=len(INIT_ORDER),
        figsize=(9.8, 8.4),
        sharex=True,
    )
    for row_idx, objective in enumerate(REP_OBJECTIVE_ORDER):
        for col_idx, init in enumerate(INIT_ORDER):
            ax = axes[row_idx][col_idx]
            panel = df[(df["objective"] == objective) & (df["init"] == init)].sort_values("step")
            for metric in ("sim", "reg"):
                if metric in panel.columns and panel[metric].notna().any():
                    ax.plot(
                        panel["step"],
                        panel[metric],
                        color=COMPONENT_COLORS[metric],
                        linewidth=1.7,
                        label=metric,
                    )
            if row_idx == 0:
                ax.set_title(INIT_LABELS[init])
            if col_idx == 0:
                ax.set_ylabel(OBJECTIVE_LABELS[objective])
            if row_idx == len(REP_OBJECTIVE_ORDER) - 1:
                ax.set_xlabel("Training step")
            ax.grid(True, alpha=0.24, linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    handles = [
        plt.Line2D([0], [0], color=COMPONENT_COLORS["sim"], linewidth=1.8, label="similarity"),
        plt.Line2D([0], [0], color=COMPONENT_COLORS["reg"], linewidth=1.8, label="regularizer"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Comparable LeJEPA objective components", y=1.02, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, out_dir, "rep_learning_training_components")


def rep_geometry_main(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(
        nrows=len(GEOMETRY_DATASET_ORDER),
        ncols=len(INIT_ORDER),
        figsize=(9.8, 8.8),
        sharex=True,
    )
    x = list(range(len(REP_OBJECTIVE_ORDER)))
    for row_idx, dataset_name in enumerate(GEOMETRY_DATASET_ORDER):
        for col_idx, init in enumerate(INIT_ORDER):
            ax = axes[row_idx][col_idx]
            panel = df[(df["dataset_name"] == dataset_name) & (df["init"] == init)].sort_values("objective")
            ax.plot(
                x,
                panel["erank_over_d"],
                color="#111827",
                marker="o",
                linewidth=2.0,
            )
            for point_idx, objective in enumerate(REP_OBJECTIVE_ORDER):
                point = panel[panel["objective"] == objective]
                if not point.empty:
                    ax.scatter(
                        [point_idx],
                        point["erank_over_d"],
                        color=OBJECTIVE_COLORS[objective],
                        s=48,
                        zorder=3,
                    )
            if row_idx == 0:
                ax.set_title(INIT_LABELS[init])
            if col_idx == 0:
                ax.set_ylabel(f"{DATASET_LABELS[dataset_name]}\nEffective rank / D")
            if row_idx == len(GEOMETRY_DATASET_ORDER) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels([OBJECTIVE_LABELS[o] for o in REP_OBJECTIVE_ORDER], rotation=20, ha="right")
            else:
                ax.set_xticks(x, [])
            ax.grid(True, axis="y", alpha=0.24, linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle("1M / 50k-step embedding geometry", y=1.02, fontsize=14)
    fig.tight_layout()
    return save_figure(fig, out_dir, "rep_learning_geometry_erank_over_d")


def rep_landmarks(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(
        nrows=len(TASK_ORDER),
        ncols=len(INIT_ORDER),
        figsize=(9.8, 5.9),
        sharex=True,
        sharey=True,
    )
    x = list(range(len(REP_OBJECTIVE_ORDER)))
    for row_idx, task_name in enumerate(TASK_ORDER):
        for col_idx, init in enumerate(INIT_ORDER):
            ax = axes[row_idx][col_idx]
            panel = df[(df["task_name"] == task_name) & (df["init_mode"] == init)].sort_values("objective")
            ax.plot(x, panel["test_mean_l2"], color="#111827", marker="o", linewidth=2.0)
            for point_idx, objective in enumerate(REP_OBJECTIVE_ORDER):
                point = panel[panel["objective"] == objective]
                if not point.empty:
                    ax.scatter([point_idx], point["test_mean_l2"], color=OBJECTIVE_COLORS[objective], s=48, zorder=3)
            if row_idx == 0:
                ax.set_title(INIT_LABELS[init])
            if col_idx == 0:
                ax.set_ylabel(TASK_LABELS[task_name])
            if row_idx == len(TASK_ORDER) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels([OBJECTIVE_LABELS[o] for o in REP_OBJECTIVE_ORDER], rotation=20, ha="right")
            else:
                ax.set_xticks(x, [])
            ax.grid(True, axis="y", alpha=0.24, linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle("1M / 50k-step G4 landmark probe", y=1.03, fontsize=14)
    fig.tight_layout()
    return save_figure(fig, out_dir, "rep_learning_landmark_g4_within")


def _objective_legend(fig: plt.Figure, objective_order: list[str]) -> None:
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=OBJECTIVE_COLORS[objective],
            marker="o",
            linewidth=2.8,
            markersize=6.5,
            label=OBJECTIVE_LABELS[objective],
        )
        for objective in objective_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
        frameon=False,
    )
