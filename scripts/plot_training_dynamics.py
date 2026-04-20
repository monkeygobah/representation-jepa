from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
RESULTS_ROOT = PROJECT_ROOT / "embedding_extract" / "results"
DATA_ROOT = RESULTS_ROOT / "data"
FIGURES_ROOT = RESULTS_ROOT / "figures"

SCALE_ORDER = ["10k", "100k", "1m"]
SCALE_LABELS = [r"$10^4$", r"$10^5$", r"$10^6$"]
OBJECTIVE_ORDER = ["infonce", "lejepa", "vicreg"]
INIT_ORDER = ["random", "imagenet", "seginit"]

SCALE_COLORS = {
    "10k": "#1b9e77",
    "100k": "#d95f02",
    "1m": "#7570b3",
}


def _parse_run_name(run_name: str) -> tuple[str, str, str]:
    parts = run_name.split("-")
    if len(parts) < 5:
        raise ValueError(f"Unexpected run_name format: {run_name}")
    return parts[2], parts[3], parts[4]


def load_training_logs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(RUNS_ROOT.glob("*__geometry-fixedcompute-*")):
        run_name = run_dir.name.split("__", 1)[1]
        scale, objective, init = _parse_run_name(run_name)
        metrics_path = run_dir / "train_metrics.jsonl"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing training metrics: {metrics_path}")

        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec["run_name"] = run_name
                rec["scale"] = scale
                rec["objective"] = objective
                rec["init"] = init
                rows.append(rec)

    df = pd.DataFrame(rows)
    numeric_cols = ["step", "epoch", "lr", "loss", "world_size", "bs"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["scale"] = pd.Categorical(df["scale"], categories=SCALE_ORDER, ordered=True)
    df["objective"] = pd.Categorical(df["objective"], categories=OBJECTIVE_ORDER, ordered=True)
    df["init"] = pd.Categorical(df["init"], categories=INIT_ORDER, ordered=True)
    return df.sort_values(["objective", "init", "scale", "step"]).reset_index(drop=True)


def write_training_table(df: pd.DataFrame) -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = DATA_ROOT / "training_metrics_all.csv"
    df.to_csv(out_path, index=False)
    return out_path


def make_loss_curve_figure(df: pd.DataFrame) -> Path:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows=len(OBJECTIVE_ORDER),
        ncols=len(INIT_ORDER),
        figsize=(14, 10),
        sharex=True,
    )

    for row_idx, objective in enumerate(OBJECTIVE_ORDER):
        obj_df = df[df["objective"] == objective]
        for col_idx, init in enumerate(INIT_ORDER):
            ax = axes[row_idx][col_idx]
            panel_df = obj_df[obj_df["init"] == init]

            for scale in SCALE_ORDER:
                line_df = panel_df[panel_df["scale"] == scale].sort_values("step")
                if line_df.empty:
                    continue

                ax.plot(
                    line_df["step"].to_numpy(),
                    line_df["loss"].to_numpy(),
                    color=SCALE_COLORS[scale],
                    linewidth=2.0,
                    alpha=0.95,
                    label=scale,
                )

            if row_idx == 0:
                ax.set_title(init)
            if col_idx == 0:
                ax.set_ylabel(f"{objective}\nLoss")
            if row_idx == len(OBJECTIVE_ORDER) - 1:
                ax.set_xlabel("Training Step")
            ax.grid(True, alpha=0.25, linewidth=0.8)

    scale_handles = [
        plt.Line2D(
            [0],
            [0],
            color=SCALE_COLORS[scale],
            linewidth=2.2,
            label=label,
        )
        for scale, label in zip(SCALE_ORDER, SCALE_LABELS)
    ]
    fig.legend(
        handles=scale_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(SCALE_ORDER),
        frameon=False,
        title="Training Set Size",
    )
    fig.suptitle("Training Loss vs Step Under Fixed Compute", fontsize=16, y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = FIGURES_ROOT / "training_loss_by_scale.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_training_logs()
    table_path = write_training_table(df)
    fig_path = make_loss_curve_figure(df)
    print(f"Wrote training table: {table_path}")
    print(f"Wrote training figure: {fig_path}")


if __name__ == "__main__":
    main()
