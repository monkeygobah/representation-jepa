from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from landmark_probe.config import PROJECT_ROOT, WORKSPACE_ROOT, load_dataset_config


SUMMARY_ROOT = PROJECT_ROOT / "landmark_probe" / "outputs" / "summaries"
ANALYSIS_ROOT = PROJECT_ROOT / "landmark_probe" / "outputs" / "analysis"
PAPER_FIGURES_ROOT = PROJECT_ROOT / "paper_figures" / "outputs" / "landmark_probe"

PAPER_INITS = ["random", "imagenet"]
EXTERNAL_BASELINE_INITS = ["dinov2", "mae"]
POOLING_ORDER = ["gap", "g2", "g4"]
TASK_ORDER = ["celeb_within", "cfd_within", "celeb_to_cfd"]
WITHIN_TASKS = ["celeb_within", "cfd_within"]
OBJECTIVE_ORDER = ["infonce", "lejepa", "vicreg"]
EXTERNAL_BASELINE_OBJECTIVE = "external_baseline"
SCALE_ORDER = ["10k", "100k", "1m"]
SCALE_LABELS = [r"$10^4$", r"$10^5$", r"$10^6$"]

TASK_LABELS = {
    "celeb_within": "Celeb within",
    "cfd_within": "CFD within",
    "celeb_to_cfd": "Celeb to CFD",
}
INIT_LABELS = {
    "random": "Random init",
    "imagenet": "ImageNet init",
    "dinov2": "DINOv2 ViT-B/14",
    "mae": "MAE ViT-B/16",
}
OBJECTIVE_LABELS = {
    "infonce": "InfoNCE",
    "lejepa": "LeJEPA",
    "vicreg": "VICReg",
    EXTERNAL_BASELINE_OBJECTIVE: "External frozen ViT",
}
OBJECTIVE_COLORS = {
    "infonce": "#0b6e4f",
    "lejepa": "#8e6c08",
    "vicreg": "#b03a2e",
}
EXTERNAL_BASELINE_COLORS = {
    "dinov2": "#2563eb",
    "mae": "#7c3aed",
}

SCALE_RE = re.compile(r"geometry-fixedcompute-(10k|100k|1m)-")


def resolve_workspace_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        try:
            return (PROJECT_ROOT / path.relative_to(WORKSPACE_ROOT)).resolve()
        except ValueError:
            return path
    return (PROJECT_ROOT / path).resolve()


def parse_scale(run_name: str, checkpoint_step: int) -> str:
    if int(checkpoint_step) == 0 or run_name.startswith("baseline-"):
        return "baseline"
    match = SCALE_RE.search(run_name)
    if not match:
        return "unknown"
    return match.group(1)


def task_family(row: pd.Series) -> str:
    if str(row["train_dataset_name"]) == str(row["test_dataset_name"]):
        return "within"
    return "transfer"


def add_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["objective"] = out["ssl_method"].astype(str)
    out["init"] = out["init_mode"].astype(str)
    out["scale"] = [
        parse_scale(run_name, checkpoint_step)
        for run_name, checkpoint_step in zip(out["run_name"], out["checkpoint_step"])
    ]
    if {"train_dataset_name", "test_dataset_name"}.issubset(out.columns):
        out["train_dataset"] = out["train_dataset_name"].astype(str)
        out["val_dataset"] = out["val_dataset_name"].astype(str)
        out["test_dataset"] = out["test_dataset_name"].astype(str)
        out["task_family"] = out.apply(task_family, axis=1)
    else:
        out["task_family"] = out["task_name"].map(
            lambda v: "transfer" if "_to_" in str(v) else "within"
        )
    external_scope = (
        (out["objective"] == EXTERNAL_BASELINE_OBJECTIVE)
        & out["init"].isin(EXTERNAL_BASELINE_INITS)
        & (out["pooling"] == "g4")
    )
    trained_or_init_scope = (
        out["init"].isin(PAPER_INITS)
        & out["task_name"].isin(TASK_ORDER)
        & out["pooling"].isin(POOLING_ORDER)
        & ~out["run_name"].str.contains("seginit|seg_init", case=False, regex=True, na=False)
    )
    out["paper_scope"] = (trained_or_init_scope | external_scope) & out["task_name"].isin(TASK_ORDER)
    return out


def add_transfer_reference(model_df: pd.DataFrame) -> pd.DataFrame:
    out = model_df.copy()
    key_cols = ["run_name", "pooling", "embedding_key", "objective", "init", "scale"]
    reference = out[out["task_name"] == "cfd_within"].loc[
        :,
        key_cols
        + [
            "test_mean_l2",
            "test_mae",
            "best_val_mean_l2",
            "best_epoch",
        ],
    ]
    reference = reference.rename(
        columns={
            "test_mean_l2": "cfd_within_test_mean_l2",
            "test_mae": "cfd_within_test_mae",
            "best_val_mean_l2": "cfd_within_best_val_mean_l2",
            "best_epoch": "cfd_within_best_epoch",
        }
    )
    out = out.merge(reference, on=key_cols, how="left", validate="many_to_one")
    transfer_mask = out["task_name"] == "celeb_to_cfd"
    out["transfer_penalty_l2"] = math.nan
    out["transfer_ratio_l2"] = math.nan
    out.loc[transfer_mask, "transfer_penalty_l2"] = (
        out.loc[transfer_mask, "test_mean_l2"]
        - out.loc[transfer_mask, "cfd_within_test_mean_l2"]
    )
    out.loc[transfer_mask, "transfer_ratio_l2"] = (
        out.loc[transfer_mask, "test_mean_l2"]
        / out.loc[transfer_mask, "cfd_within_test_mean_l2"]
    )
    return out


def _read_summary_pair(study_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_dir = SUMMARY_ROOT / study_name
    return (
        pd.read_csv(summary_dir / "overall_summary.csv"),
        pd.read_csv(summary_dir / "per_landmark_summary.csv"),
    )


def load_analysis_tables(study_name: str, external_study_names: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_parts = []
    landmark_parts = []
    for name in (study_name, *external_study_names):
        summary_dir = SUMMARY_ROOT / name
        if name != study_name and not summary_dir.exists():
            continue
        overall, per_landmark = _read_summary_pair(name)
        overall_parts.append(overall)
        landmark_parts.append(per_landmark)

    overall = pd.concat(overall_parts, ignore_index=True, sort=False)
    per_landmark = pd.concat(landmark_parts, ignore_index=True, sort=False)

    for df in (overall, per_landmark):
        for col in ["checkpoint_step", "best_epoch", "test_mean_l2", "test_mae", "best_val_mean_l2"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    model_df = add_transfer_reference(add_common_columns(overall))
    landmark_df = add_common_columns(per_landmark)
    return model_df, landmark_df


def ordered_subset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, order in [
        ("task_name", TASK_ORDER),
        ("pooling", POOLING_ORDER),
        ("objective", ["baseline"] + OBJECTIVE_ORDER + [EXTERNAL_BASELINE_OBJECTIVE]),
        ("init", PAPER_INITS + EXTERNAL_BASELINE_INITS),
        ("scale", ["baseline"] + SCALE_ORDER),
    ]:
        if col in out.columns:
            out[col] = pd.Categorical(out[col], categories=order, ordered=True)
    sort_cols = [c for c in ["task_name", "pooling", "init", "objective", "scale", "run_name"] if c in out]
    return out.sort_values(sort_cols).reset_index(drop=True)


def write_tables(model_df: pd.DataFrame, landmark_df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_level": tables_dir / "model_level_summary.csv",
        "model_level_paper": tables_dir / "model_level_paper_scope.csv",
        "per_landmark": tables_dir / "per_landmark_summary_derived.csv",
        "per_landmark_paper": tables_dir / "per_landmark_paper_scope.csv",
        "transfer": tables_dir / "transfer_summary.csv",
    }
    ordered_subset(model_df).to_csv(paths["model_level"], index=False)
    ordered_subset(model_df[model_df["paper_scope"]]).to_csv(paths["model_level_paper"], index=False)
    ordered_subset(landmark_df).to_csv(paths["per_landmark"], index=False)
    ordered_subset(landmark_df[landmark_df["paper_scope"]]).to_csv(
        paths["per_landmark_paper"], index=False
    )
    transfer = model_df[
        (model_df["paper_scope"])
        & (model_df["task_name"] == "celeb_to_cfd")
        & (model_df["objective"].isin(OBJECTIVE_ORDER + [EXTERNAL_BASELINE_OBJECTIVE]))
    ].copy()
    ordered_subset(transfer).to_csv(paths["transfer"], index=False)
    return paths


def save_plot_ready(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_subset(df).to_csv(out_path, index=False)


def style_axis(ax) -> None:
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.08, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def external_baselines(df: pd.DataFrame, task: str | None = None) -> pd.DataFrame:
    mask = (
        (df["paper_scope"])
        & (df["objective"] == EXTERNAL_BASELINE_OBJECTIVE)
        & (df["pooling"] == "g4")
        & (df["init"].isin(EXTERNAL_BASELINE_INITS))
    )
    if task is not None:
        mask &= df["task_name"] == task
    return df[mask].copy()


def add_external_baseline_lines(ax, rows: pd.DataFrame, metric: str, xmin: float = -0.15, xmax: float = 2.15) -> None:
    for init in EXTERNAL_BASELINE_INITS:
        baseline = rows[rows["init"] == init]
        if len(baseline) != 1 or pd.isna(baseline.iloc[0][metric]):
            continue
        ax.hlines(
            float(baseline.iloc[0][metric]),
            xmin=xmin,
            xmax=xmax,
            color=EXTERNAL_BASELINE_COLORS[init],
            linestyle=(0, (7, 2.5)),
            linewidth=1.8,
            alpha=0.95,
        )


def external_baseline_legend_handles() -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            color=EXTERNAL_BASELINE_COLORS[init],
            linestyle=(0, (7, 2.5)),
            linewidth=1.8,
            label=INIT_LABELS[init],
        )
        for init in EXTERNAL_BASELINE_INITS
    ]


def plot_main_within_g4(model_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = model_df[
        (model_df["paper_scope"])
        & (model_df["pooling"] == "g4")
        & (model_df["task_name"].isin(WITHIN_TASKS))
    ].copy()
    trained = subset[subset["objective"].isin(OBJECTIVE_ORDER)]
    baselines = subset[subset["objective"] == "baseline"]
    external = external_baselines(subset)
    save_plot_ready(subset, out_dir / "g4_within_test_mean_l2_plot_ready.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    all_values = subset["test_mean_l2"].dropna()
    ymin = max(0.0, all_values.min() - 0.08 * (all_values.max() - all_values.min()))
    ymax = all_values.max() + 0.12 * (all_values.max() - all_values.min())
    x = list(range(len(SCALE_ORDER)))

    for row_idx, task in enumerate(WITHIN_TASKS):
        for col_idx, init in enumerate(PAPER_INITS):
            ax = axes[row_idx][col_idx]
            panel = trained[(trained["task_name"] == task) & (trained["init"] == init)]
            baseline = baselines[(baselines["task_name"] == task) & (baselines["init"] == init)]
            if len(baseline) == 1:
                ax.axhline(
                    float(baseline.iloc[0]["test_mean_l2"]),
                    color="#4b5563",
                    linestyle=(0, (4, 3)),
                    linewidth=1.5,
                    alpha=0.85,
                )
            add_external_baseline_lines(ax, external[external["task_name"] == task], "test_mean_l2")
            for objective in OBJECTIVE_ORDER:
                line = panel[panel["objective"] == objective].sort_values("scale")
                if line.empty:
                    continue
                ax.plot(
                    x,
                    line["test_mean_l2"].to_numpy(),
                    color=OBJECTIVE_COLORS[objective],
                    marker="o",
                    linewidth=2.2,
                    markersize=5.6,
                    label=OBJECTIVE_LABELS[objective],
                )
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(-0.15, 2.15)
            style_axis(ax)
            if row_idx == 0:
                ax.set_title(INIT_LABELS[init])
            if col_idx == 0:
                ax.set_ylabel(f"{TASK_LABELS[task]}\nMean test L2")
            if row_idx == 1:
                ax.set_xticks(x)
                ax.set_xticklabels(SCALE_LABELS)
                ax.set_xlabel("SSL training images")

    handles = [
        plt.Line2D([0], [0], color=OBJECTIVE_COLORS[o], marker="o", linewidth=2.2, label=OBJECTIVE_LABELS[o])
        for o in OBJECTIVE_ORDER
    ]
    handles.append(
        plt.Line2D([0], [0], color="#4b5563", linestyle=(0, (4, 3)), linewidth=1.5, label="Init-only baseline")
    )
    handles.extend(external_baseline_legend_handles())
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False)
    fig.suptitle("G4 Within-Dataset Periorbital Landmark Probe", y=1.08, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = out_dir / "g4_within_test_mean_l2.png"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [png]


def build_transfer_1m_outputs(model_df: pd.DataFrame, landmark_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_subset = model_df[
        (model_df["paper_scope"])
        & (model_df["pooling"] == "g4")
        & (model_df["task_name"] == "celeb_to_cfd")
        & (model_df["objective"].isin(OBJECTIVE_ORDER))
        & (model_df["scale"] == "1m")
    ].copy()
    external_model_subset = external_baselines(model_df, task="celeb_to_cfd")
    landmark_subset = landmark_df[
        (landmark_df["paper_scope"])
        & (landmark_df["pooling"] == "g4")
        & (landmark_df["task_name"] == "celeb_to_cfd")
        & (landmark_df["objective"].isin(OBJECTIVE_ORDER))
        & (landmark_df["scale"] == "1m")
        & (landmark_df["landmark"].isin(["iris_centroid", "medial_canthus", "lateral_canthus"]))
    ].copy()
    external_landmark_subset = external_baselines(landmark_df, task="celeb_to_cfd")
    external_landmark_subset = external_landmark_subset[
        external_landmark_subset["landmark"].isin(["iris_centroid", "medial_canthus", "lateral_canthus"])
    ].copy()
    table_model_subset = pd.concat([model_subset, external_model_subset], ignore_index=True, sort=False)
    table_landmark_subset = pd.concat([landmark_subset, external_landmark_subset], ignore_index=True, sort=False)
    landmark_wide = table_landmark_subset.pivot_table(
        index=["run_name", "objective", "init"],
        columns="landmark",
        values="mean_l2",
        aggfunc="first",
    ).reset_index()
    table = table_model_subset.merge(landmark_wide, on=["run_name", "objective", "init"], how="left")
    table = table[
        [
            "objective",
            "init",
            "transfer_penalty_l2",
            "transfer_ratio_l2",
            "test_mean_l2",
            "test_mae",
            "iris_centroid",
            "medial_canthus",
            "lateral_canthus",
            "run_name",
        ]
    ].copy()
    table = ordered_subset(table)
    table_out = out_dir / "g4_celeb_to_cfd_1m_summary_table.csv"
    table.to_csv(table_out, index=False)

    display = table.copy()
    display["Objective"] = display.apply(
        lambda row: INIT_LABELS[row["init"]] if row["objective"] == EXTERNAL_BASELINE_OBJECTIVE else OBJECTIVE_LABELS[row["objective"]],
        axis=1,
    )
    display["Init"] = display["init"].map(INIT_LABELS)
    display["Test mean L2"] = display["test_mean_l2"].map(lambda v: f"{v:.2f}")
    display["Transfer penalty"] = display["transfer_penalty_l2"].map(lambda v: f"{v:.2f}")
    display["Transfer ratio"] = display["transfer_ratio_l2"].map(lambda v: f"{v:.2f}")
    display["Iris"] = display["iris_centroid"].map(lambda v: f"{v:.2f}")
    display["Medial canthus"] = display["medial_canthus"].map(lambda v: f"{v:.2f}")
    display["Lateral canthus"] = display["lateral_canthus"].map(lambda v: f"{v:.2f}")
    display = display[
        [
            "Init",
            "Objective",
            "Test mean L2",
            "Transfer penalty",
            "Transfer ratio",
            "Iris",
            "Medial canthus",
            "Lateral canthus",
        ]
    ]

    fig_table, ax_table = plt.subplots(figsize=(11.8, 3.8))
    ax_table.axis("off")
    table_artist = ax_table.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8.5)
    table_artist.scale(1.0, 1.5)
    for (row, _col), cell in table_artist.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(weight="bold")
    ax_table.set_title("G4 Celeb-to-CFD Landmark Transfer Table, 1M SSL Models + Frozen ViT Baselines", pad=12, fontsize=13)
    fig_table.tight_layout()
    png_table = out_dir / "g4_celeb_to_cfd_1m_summary_table.png"
    fig_table.savefig(png_table, dpi=240, bbox_inches="tight")
    plt.close(fig_table)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=False)
    x = list(range(len(OBJECTIVE_ORDER)))
    for ax, metric, ylabel, title in [
        (axes[0], "test_mean_l2", "Cross-dataset mean L2", "Absolute transfer error"),
        (axes[1], "transfer_penalty_l2", "Transfer penalty vs CFD-within", "Transfer penalty"),
    ]:
        for init, linestyle in [("random", "-"), ("imagenet", "--")]:
            panel = table[table["init"] == init].sort_values("objective")
            ax.plot(
                x,
                panel[metric].to_numpy(),
                color="#111827",
                linestyle=linestyle,
                marker="o",
                linewidth=2.0,
                markersize=5.8,
            )
            for point_idx, objective in enumerate(OBJECTIVE_ORDER):
                point = panel[panel["objective"] == objective]
                if point.empty:
                    continue
                ax.scatter(
                    [point_idx],
                    point[metric].to_numpy(),
                    color=OBJECTIVE_COLORS[objective],
                    s=54,
                    zorder=3,
                )
        add_external_baseline_lines(ax, external_model_subset, metric)
        ax.set_xticks(x)
        ax.set_xticklabels([OBJECTIVE_LABELS[o] for o in OBJECTIVE_ORDER], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        style_axis(ax)
    handles = [
        plt.Line2D([0], [0], color=OBJECTIVE_COLORS[o], marker="o", linewidth=0, markersize=7, label=OBJECTIVE_LABELS[o])
        for o in OBJECTIVE_ORDER
    ]
    handles += [
        plt.Line2D([0], [0], color="#111827", linestyle="-", linewidth=2.0, label="Random"),
        plt.Line2D([0], [0], color="#111827", linestyle="--", linewidth=2.0, label="ImageNet"),
    ]
    handles.extend(external_baseline_legend_handles())
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=7, frameon=False)
    fig.suptitle("G4 Celeb-to-CFD Landmark Transfer, 1M SSL Models + Frozen ViT Baselines", y=1.10, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png = out_dir / "g4_celeb_to_cfd_1m_transfer_plot.png"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [table_out, png_table, png]


def selected_sample_rows(per_sample: pd.DataFrame) -> pd.DataFrame:
    sorted_df = per_sample.sort_values("mean_l2").reset_index(drop=True)
    picks = [
        ("best", 0),
        ("worst", len(sorted_df) - 1),
    ]
    rows = []
    for label, idx in picks:
        row = sorted_df.iloc[idx].copy()
        row["example_type"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def plot_landmarks(ax, row: pd.Series, landmarks: list[str]) -> None:
    gt_x = [float(row[f"{lm}_true_x"]) for lm in landmarks]
    gt_y = [float(row[f"{lm}_true_y"]) for lm in landmarks]
    pred_x = [float(row[f"{lm}_pred_x"]) for lm in landmarks]
    pred_y = [float(row[f"{lm}_pred_y"]) for lm in landmarks]
    ax.scatter(gt_x, gt_y, s=52, c="#13a8a8", edgecolors="white", linewidths=0.8, label="GT", zorder=4)
    ax.scatter(pred_x, pred_y, s=48, c="#d1495b", marker="x", linewidths=2.2, label="Pred", zorder=5)
    for gx, gy, px, py in zip(gt_x, gt_y, pred_x, pred_y):
        ax.plot([gx, px], [gy, py], color="white", linewidth=0.7, alpha=0.75, zorder=3)


def plot_main_qualitative(
    model_df: pd.DataFrame,
    dataset_cfg_path: Path,
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_cfg = load_dataset_config(dataset_cfg_path)
    manifest = pd.read_csv(dataset_cfg.metadata.manifest_csv)
    image_by_sample = {
        str(row.sample_id): dataset_cfg.root / str(row.image_rel_path)
        for row in manifest.itertuples(index=False)
    }
    candidates = model_df[
        (model_df["paper_scope"])
        & (model_df["pooling"] == "g4")
        & (model_df["task_name"] == "celeb_to_cfd")
        & (model_df["objective"].isin(OBJECTIVE_ORDER))
        & (model_df["scale"] == "1m")
        & (model_df["init"] == "random")
    ].copy()
    selected_models = (
        candidates.sort_values(["objective", "test_mean_l2"])
        .groupby("objective", as_index=False)
        .head(1)
        .assign(objective_order=lambda df: df["objective"].map({v: i for i, v in enumerate(["vicreg", "infonce", "lejepa"])}))
        .sort_values("objective_order")
    )
    if selected_models.empty:
        return []

    selected_rows = []
    per_model_samples = []
    for model_row in selected_models.itertuples(index=False):
        probe_ckpt = resolve_workspace_path(model_row.probe_checkpoint_path)
        per_sample_path = probe_ckpt.parent / "per_sample.csv"
        per_sample = selected_sample_rows(pd.read_csv(per_sample_path))
        per_sample["run_name"] = model_row.run_name
        per_sample["objective"] = model_row.objective
        per_sample["init"] = model_row.init
        per_sample["scale"] = model_row.scale
        per_model_samples.append(per_sample)
        selected_rows.append(model_row._asdict())

    sample_table = pd.concat(per_model_samples, ignore_index=True)
    save_plot_ready(sample_table, out_dir / "g4_qualitative_landmark_examples_plot_ready.csv")

    ncols = len(selected_models)
    example_order = ["best", "worst"]
    fig, axes = plt.subplots(
        len(example_order),
        ncols,
        figsize=(3.35 * ncols, 5.9),
        squeeze=False,
        gridspec_kw={"wspace": 0.02, "hspace": 0.06},
    )
    landmarks = list(dataset_cfg.landmarks)
    for col_idx, model_row in enumerate(selected_rows):
        model_samples = sample_table[sample_table["run_name"] == model_row["run_name"]]
        for row_idx, example_type in enumerate(example_order):
            ax = axes[row_idx][col_idx]
            sample_row = model_samples[model_samples["example_type"] == example_type].iloc[0]
            image_path = image_by_sample[str(sample_row["sample_id"])]
            ax.imshow(Image.open(image_path).convert("RGB"))
            plot_landmarks(ax, sample_row, landmarks)
            if row_idx == 0:
                ax.set_title(
                    f"{OBJECTIVE_LABELS[model_row['objective']]}",
                    fontsize=10,
                    pad=6,
                )
            if col_idx == 0:
                ax.set_ylabel(example_type.title(), fontsize=11)
            ax.text(
                0.98,
                0.04,
                f"L2 {float(sample_row['mean_l2']):.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.5,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2.0},
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#13a8a8", markeredgecolor="white", label="Ground truth"),
        plt.Line2D([0], [0], marker="x", color="#d1495b", linestyle="none", label="Prediction"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.suptitle("G4 Celeb-to-CFD Landmark Predictions", y=0.98, fontsize=14)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.11, wspace=0.02, hspace=0.06)
    png = out_dir / "g4_qualitative_landmark_examples.png"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [png]


def plot_supp_absolute_by_pooling(model_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = model_df[
        (model_df["paper_scope"])
        & (model_df["task_name"].isin(TASK_ORDER))
        & (model_df["objective"].isin(OBJECTIVE_ORDER))
    ].copy()
    save_plot_ready(subset, out_dir / "all_poolings_absolute_l2_plot_ready.csv")
    poolings = [p for p in POOLING_ORDER if p in set(subset["pooling"])]
    if not poolings:
        return []

    fig, axes = plt.subplots(len(TASK_ORDER), len(poolings), figsize=(4.3 * len(poolings), 8.5), sharex=True, sharey=True, squeeze=False)
    x = list(range(len(SCALE_ORDER)))
    for row_idx, task in enumerate(TASK_ORDER):
        for col_idx, pooling in enumerate(poolings):
            ax = axes[row_idx][col_idx]
            panel = subset[(subset["task_name"] == task) & (subset["pooling"] == pooling)]
            for objective in OBJECTIVE_ORDER:
                for init, linestyle in [("random", "-"), ("imagenet", "--")]:
                    line = panel[(panel["objective"] == objective) & (panel["init"] == init)].sort_values("scale")
                    if line.empty:
                        continue
                    ax.plot(
                        x,
                        line["test_mean_l2"].to_numpy(),
                        color=OBJECTIVE_COLORS[objective],
                        linestyle=linestyle,
                        marker="o",
                        linewidth=1.9,
                        markersize=4.8,
                    )
            if row_idx == 0:
                ax.set_title(pooling)
            if col_idx == 0:
                ax.set_ylabel(f"{TASK_LABELS[task]}\nMean test L2")
            if row_idx == len(TASK_ORDER) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(SCALE_LABELS)
                ax.set_xlabel("SSL training images")
            style_axis(ax)
    objective_handles = [
        plt.Line2D([0], [0], color=OBJECTIVE_COLORS[o], marker="o", linewidth=2.0, label=OBJECTIVE_LABELS[o])
        for o in OBJECTIVE_ORDER
    ]
    init_handles = [
        plt.Line2D([0], [0], color="#111827", linestyle="-", linewidth=1.8, label="Random"),
        plt.Line2D([0], [0], color="#111827", linestyle="--", linewidth=1.8, label="ImageNet"),
    ]
    fig.legend(handles=objective_handles + init_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False)
    fig.suptitle("Landmark Probe Performance Across Completed Poolings", y=1.08, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = out_dir / "all_poolings_absolute_l2.png"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [png]


def plot_supp_per_landmark_heatmap(landmark_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = landmark_df[
        (landmark_df["paper_scope"])
        & (landmark_df["pooling"] == "g4")
        & (landmark_df["task_name"] == "celeb_to_cfd")
        & (landmark_df["objective"].isin(OBJECTIVE_ORDER))
    ].copy()
    if subset.empty:
        return []
    save_plot_ready(subset, out_dir / "g4_celeb_to_cfd_per_landmark_plot_ready.csv")
    best = subset.sort_values("mean_l2").groupby(["objective", "init"], as_index=False).head(1)
    best_keys = best[["run_name", "pooling", "objective", "init"]].drop_duplicates()
    panel_df = subset.merge(best_keys, on=["run_name", "pooling", "objective", "init"], how="inner")
    pivot = panel_df.pivot_table(index="landmark", columns=["init", "objective"], values="mean_l2", aggfunc="first")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{INIT_LABELS[i]}\n{OBJECTIVE_LABELS[o]}" for i, o in pivot.columns], rotation=35, ha="right")
    ax.set_title("G4 Celeb-to-CFD Per-Landmark Error, Best Scale per Objective/Init")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean L2")
    fig.tight_layout()
    png = out_dir / "g4_celeb_to_cfd_per_landmark_heatmap.png"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [png]


def write_outlier_table(model_df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    outliers = model_df[
        (model_df["paper_scope"])
        & (model_df["task_name"] == "celeb_to_cfd")
        & (model_df["transfer_penalty_l2"].notna())
    ].sort_values("transfer_penalty_l2", ascending=False)
    path = out_dir / "largest_transfer_penalties.csv"
    outliers.to_csv(path, index=False)
    return path


def build_outputs(study_name: str, dataset_cfg_path: Path, external_study_names: tuple[str, ...] = ()) -> list[Path]:
    out_dir = PAPER_FIGURES_ROOT
    model_df, landmark_df = load_analysis_tables(study_name, external_study_names)
    written = list(write_tables(model_df, landmark_df, out_dir).values())
    written += plot_main_within_g4(model_df, out_dir / "main")
    written += build_transfer_1m_outputs(model_df, landmark_df, out_dir / "main")
    written += plot_main_qualitative(model_df, dataset_cfg_path, out_dir / "main")
    written += plot_supp_absolute_by_pooling(model_df, out_dir / "supplement")
    written += plot_supp_per_landmark_heatmap(landmark_df, out_dir / "supplement")
    written.append(write_outlier_table(model_df, out_dir / "internal_diagnostics"))
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-name", default="followup_50k_landmark_probe")
    ap.add_argument(
        "--external-study-name",
        action="append",
        default=["external_vit_landmark_probe"],
        help="Optional external baseline study summary to merge into the paper landmark figures.",
    )
    ap.add_argument(
        "--dataset-cfg",
        type=Path,
        default=PROJECT_ROOT / "landmark_probe" / "configs" / "datasets" / "periorbital_224_v2.yaml",
    )
    args = ap.parse_args()

    written = build_outputs(args.study_name, args.dataset_cfg, tuple(args.external_study_name))
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
