from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from paper_figures import data, plots
from paper_figures.metadata import (
    DATASET_OBJECTIVE_ORDER,
    INIT_ORDER,
    OUTPUT_ROOT,
    REP_OBJECTIVE_ORDER,
    SCALE_ORDER,
)


def write_table(df: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


def rel(path: Path) -> str:
    return str(path.resolve())


def write_manifest(out_dir: Path, preset: str, table_paths: list[Path], figure_paths: list[Path]) -> Path:
    manifest = {
        "preset": preset,
        "filters": {
            "exclude_seginit": True,
            "inits": INIT_ORDER,
            "dataset_objectives": DATASET_OBJECTIVE_ORDER,
            "representation_objectives": REP_OBJECTIVE_ORDER,
        },
        "tables": [rel(path) for path in table_paths],
        "figures": [rel(path) for path in figure_paths],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def build_dataset_track(output_root: Path = OUTPUT_ROOT) -> dict[str, list[Path]]:
    out_dir = output_root / "dataset_track"
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"

    table_paths: list[Path] = []
    figure_paths: list[Path] = []

    resources = data.dataset_resource_summary()
    table_paths.append(write_table(resources, tables_dir, "dataset_resource_summary.csv"))
    figure_paths.extend(plots.dataset_resource_summary_table(resources, figures_dir))
    figure_paths.extend(plots.dataset_benchmark_schematic(figures_dir))

    training = data.load_training_runs(DATASET_OBJECTIVE_ORDER, SCALE_ORDER, INIT_ORDER)
    table_paths.append(write_table(training, tables_dir, "dataset_training_metrics_plot_ready.csv"))
    figure_paths.extend(plots.dataset_training_curves(training, figures_dir))

    geometry = data.dataset_geometry()
    table_paths.append(write_table(geometry, tables_dir, "dataset_geometry_plot_ready.csv"))
    figure_paths.extend(plots.dataset_geometry_main(geometry, figures_dir))
    figure_paths.extend(plots.dataset_geometry_supplement(geometry, figures_dir))

    landmarks, baselines, external_baselines = data.dataset_landmarks()
    table_paths.append(write_table(landmarks, tables_dir, "dataset_landmark_g4_plot_ready.csv"))
    table_paths.append(write_table(baselines, tables_dir, "dataset_landmark_g4_baselines.csv"))
    table_paths.append(write_table(external_baselines, tables_dir, "dataset_landmark_g4_external_baselines.csv"))
    figure_paths.extend(plots.dataset_landmarks(landmarks, baselines, external_baselines, figures_dir))

    manifest = write_manifest(out_dir, "dataset_track", table_paths, figure_paths)
    return {"tables": table_paths, "figures": figure_paths, "manifest": [manifest]}


def build_rep_learning(output_root: Path = OUTPUT_ROOT) -> dict[str, list[Path]]:
    out_dir = output_root / "rep_learning"
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"

    table_paths: list[Path] = []
    figure_paths: list[Path] = []

    training = data.load_training_runs(REP_OBJECTIVE_ORDER, ["1m"], INIT_ORDER)
    table_paths.append(write_table(training, tables_dir, "rep_learning_training_metrics_plot_ready.csv"))
    figure_paths.extend(plots.rep_training_loss(training, figures_dir))
    figure_paths.extend(plots.rep_training_components(training, figures_dir))

    geometry = data.rep_geometry()
    table_paths.append(write_table(geometry, tables_dir, "rep_learning_geometry_plot_ready.csv"))
    figure_paths.extend(plots.rep_geometry_main(geometry, figures_dir))

    landmarks = data.rep_landmarks()
    table_paths.append(write_table(landmarks, tables_dir, "rep_learning_landmark_g4_plot_ready.csv"))
    figure_paths.extend(plots.rep_landmarks(landmarks, figures_dir))

    manifest = write_manifest(out_dir, "rep_learning", table_paths, figure_paths)
    return {"tables": table_paths, "figures": figure_paths, "manifest": [manifest]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build manuscript-facing paper figures.")
    ap.add_argument(
        "--preset",
        choices=["dataset_track", "rep_learning", "all"],
        default="all",
        help="Figure preset to build.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for generated paper figures.",
    )
    args = ap.parse_args()

    outputs: dict[str, dict[str, list[Path]]] = {}
    if args.preset in ("dataset_track", "all"):
        outputs["dataset_track"] = build_dataset_track(args.output_root)
    if args.preset in ("rep_learning", "all"):
        outputs["rep_learning"] = build_rep_learning(args.output_root)

    for preset, groups in outputs.items():
        print(f"{preset}:")
        for group_name, paths in groups.items():
            print(f"  {group_name}: {len(paths)}")
            for path in paths:
                print(f"    {path}")


if __name__ == "__main__":
    main()
