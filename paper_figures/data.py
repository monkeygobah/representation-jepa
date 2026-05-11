from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paper_figures.metadata import (
    DATASET_OBJECTIVE_ORDER,
    EXTERNAL_BASELINE_ORDER,
    GEOMETRY_DATASET_ORDER,
    INIT_ORDER,
    PROJECT_ROOT,
    REP_OBJECTIVE_ORDER,
    SCALE_ORDER,
    TASK_ORDER,
    ordered_categorical,
    parse_run_name,
)


DATASET_GEOMETRY_CSV = (
    PROJECT_ROOT / "embedding_extract" / "results" / "data" / "isotropy_summary_all_proj_50ksteps.csv"
)
REP_GEOMETRY_CSV = (
    PROJECT_ROOT
    / "embedding_extract"
    / "outputs"
    / "tables"
    / "geometry_ep_bhep_1m_50ksteps"
    / "isotropy_summary_ep_bhep_1m_50ksteps.csv"
)
DATASET_LANDMARK_CSV = (
    PROJECT_ROOT / "landmark_probe" / "outputs" / "summaries" / "followup_50k_landmark_probe" / "overall_summary.csv"
)
EXTERNAL_VIT_LANDMARK_CSV = (
    PROJECT_ROOT / "landmark_probe" / "outputs" / "summaries" / "external_vit_landmark_probe" / "overall_summary.csv"
)
REP_LANDMARK_CSV = (
    PROJECT_ROOT / "landmark_probe" / "outputs" / "summaries" / "ep_bhep_1m_50k_landmark_probe" / "overall_summary.csv"
)
RUNS_ROOT = PROJECT_ROOT / "runs"
LANDMARK_DATASET_ROOT = PROJECT_ROOT / "landmark_probe" / "data" / "periorbital_224_v2"
LANDMARK_MANIFEST_CSV = LANDMARK_DATASET_ROOT / "metadata" / "dataset_manifest.csv"
LANDMARK_SPLIT_CSV = LANDMARK_DATASET_ROOT / "metadata" / "split_assignments.csv"


def add_run_parts(df: pd.DataFrame, run_col: str = "run_name") -> pd.DataFrame:
    out = df.copy()
    parsed = out[run_col].apply(parse_run_name)
    out["scale"] = [p.scale for p in parsed]
    out["objective"] = [p.objective for p in parsed]
    out["init"] = [p.init for p in parsed]
    out["is_50k"] = [p.is_50k for p in parsed]
    return out


def assert_no_seginit(df: pd.DataFrame, label: str) -> None:
    cols = [col for col in ("init", "init_mode") if col in df.columns]
    for col in cols:
        bad = df[df[col].isin(["seginit", "seg_init"])]
        if not bad.empty:
            raise ValueError(f"{label} contains seginit rows in {col}")


def assert_expected_rows(df: pd.DataFrame, expected: int, label: str) -> None:
    if len(df) != expected:
        raise ValueError(f"{label}: expected {expected} rows, found {len(df)}")


def dataset_geometry() -> pd.DataFrame:
    df = pd.read_csv(DATASET_GEOMETRY_CSV)
    df = df[
        (df["embedding_key"] == "proj")
        & (df["dataset_name"].isin(GEOMETRY_DATASET_ORDER))
        & (df["objective"].isin(DATASET_OBJECTIVE_ORDER))
        & (df["init"].isin(INIT_ORDER))
        & (df["scale"].isin(SCALE_ORDER))
    ].copy()
    for col in ("checkpoint_step", "erank_over_d", "ev1", "cos_std", "cond_1_med"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dataset_name"] = ordered_categorical(df["dataset_name"], GEOMETRY_DATASET_ORDER)
    df["objective"] = ordered_categorical(df["objective"], DATASET_OBJECTIVE_ORDER)
    df["init"] = ordered_categorical(df["init"], INIT_ORDER)
    df["scale"] = ordered_categorical(df["scale"], SCALE_ORDER)
    df = df.sort_values(["dataset_name", "init", "objective", "scale"]).reset_index(drop=True)
    assert_no_seginit(df, "dataset geometry")
    assert_expected_rows(
        df,
        len(GEOMETRY_DATASET_ORDER) * len(DATASET_OBJECTIVE_ORDER) * len(INIT_ORDER) * len(SCALE_ORDER),
        "dataset geometry",
    )
    return df


def rep_geometry() -> pd.DataFrame:
    base = pd.read_csv(DATASET_GEOMETRY_CSV)
    lejepa = base[
        (base["embedding_key"] == "proj")
        & (base["dataset_name"].isin(GEOMETRY_DATASET_ORDER))
        & (base["objective"] == "lejepa")
        & (base["init"].isin(INIT_ORDER))
        & (base["scale"] == "1m")
    ].copy()

    variants = pd.read_csv(REP_GEOMETRY_CSV)
    variants = add_run_parts(variants)
    variants = variants[
        (variants["embedding_key"] == "proj")
        & (variants["dataset_name"].isin(GEOMETRY_DATASET_ORDER))
        & (variants["objective"].isin(["bhep", "eppartial"]))
        & (variants["init"].isin(INIT_ORDER))
        & (variants["scale"] == "1m")
    ].copy()

    cols = sorted(set(lejepa.columns).intersection(variants.columns))
    df = pd.concat([lejepa.loc[:, cols], variants.loc[:, cols]], ignore_index=True, sort=False)
    for col in ("checkpoint_step", "erank_over_d", "ev1", "cos_std", "cond_1_med"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dataset_name"] = ordered_categorical(df["dataset_name"], GEOMETRY_DATASET_ORDER)
    df["objective"] = ordered_categorical(df["objective"], REP_OBJECTIVE_ORDER)
    df["init"] = ordered_categorical(df["init"], INIT_ORDER)
    df = df.sort_values(["dataset_name", "init", "objective"]).reset_index(drop=True)
    assert_no_seginit(df, "representation geometry")
    assert_expected_rows(
        df,
        len(GEOMETRY_DATASET_ORDER) * len(REP_OBJECTIVE_ORDER) * len(INIT_ORDER),
        "representation geometry",
    )
    return df


def dataset_landmarks() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATASET_LANDMARK_CSV)
    df = df[(df["pooling"] == "g4") & (df["task_name"].isin(TASK_ORDER))].copy()
    df["test_mean_l2"] = pd.to_numeric(df["test_mean_l2"], errors="coerce")

    trained = df[
        df["ssl_method"].isin(DATASET_OBJECTIVE_ORDER) & df["init_mode"].isin(INIT_ORDER)
    ].copy()
    trained["scale"] = trained["run_name"].apply(lambda x: parse_run_name(x).scale)
    trained = trained[trained["scale"].isin(SCALE_ORDER)].copy()
    trained["task_name"] = ordered_categorical(trained["task_name"], TASK_ORDER)
    trained["ssl_method"] = ordered_categorical(trained["ssl_method"], DATASET_OBJECTIVE_ORDER)
    trained["init_mode"] = ordered_categorical(trained["init_mode"], INIT_ORDER)
    trained["scale"] = ordered_categorical(trained["scale"], SCALE_ORDER)
    trained = trained.sort_values(["task_name", "init_mode", "ssl_method", "scale"]).reset_index(
        drop=True
    )

    baselines = df[(df["ssl_method"] == "baseline") & (df["init_mode"].isin(INIT_ORDER))].copy()
    baselines["task_name"] = ordered_categorical(baselines["task_name"], TASK_ORDER)
    baselines["init_mode"] = ordered_categorical(baselines["init_mode"], INIT_ORDER)
    baselines = baselines.sort_values(["task_name", "init_mode"]).reset_index(drop=True)

    external = pd.read_csv(EXTERNAL_VIT_LANDMARK_CSV)
    external = external[
        (external["pooling"] == "g4")
        & (external["task_name"].isin(TASK_ORDER))
        & (external["ssl_method"] == "external_baseline")
        & (external["init_mode"].isin(EXTERNAL_BASELINE_ORDER))
    ].copy()
    external["test_mean_l2"] = pd.to_numeric(external["test_mean_l2"], errors="coerce")
    external["task_name"] = ordered_categorical(external["task_name"], TASK_ORDER)
    external["init_mode"] = ordered_categorical(external["init_mode"], EXTERNAL_BASELINE_ORDER)
    external = external.sort_values(["task_name", "init_mode"]).reset_index(drop=True)

    assert_no_seginit(trained, "dataset landmarks")
    assert_no_seginit(baselines, "dataset landmark baselines")
    assert_expected_rows(
        trained,
        len(TASK_ORDER) * len(DATASET_OBJECTIVE_ORDER) * len(INIT_ORDER) * len(SCALE_ORDER),
        "dataset landmarks",
    )
    assert_expected_rows(baselines, len(TASK_ORDER) * len(INIT_ORDER), "dataset landmark baselines")
    assert_expected_rows(external, len(TASK_ORDER) * len(EXTERNAL_BASELINE_ORDER), "dataset landmark external baselines")
    return trained, baselines, external


def rep_landmarks() -> pd.DataFrame:
    base = pd.read_csv(DATASET_LANDMARK_CSV)
    lejepa = base[
        (base["pooling"] == "g4")
        & (base["task_name"].isin(TASK_ORDER))
        & (base["ssl_method"] == "lejepa")
        & (base["init_mode"].isin(INIT_ORDER))
    ].copy()
    lejepa["scale"] = lejepa["run_name"].apply(lambda x: parse_run_name(x).scale)
    lejepa = lejepa[lejepa["scale"] == "1m"].copy()

    variants = pd.read_csv(REP_LANDMARK_CSV)
    variants = variants[(variants["pooling"] == "g4") & (variants["task_name"].isin(TASK_ORDER))].copy()
    variants = variants[variants["init_mode"].isin(INIT_ORDER)].copy()
    variants["objective"] = variants["run_name"].apply(lambda x: parse_run_name(x).objective)
    variants = variants[variants["objective"].isin(["bhep", "eppartial"])].copy()

    lejepa["objective"] = "lejepa"
    cols = sorted(set(lejepa.columns).intersection(variants.columns))
    df = pd.concat([lejepa.loc[:, cols], variants.loc[:, cols]], ignore_index=True, sort=False)
    df["test_mean_l2"] = pd.to_numeric(df["test_mean_l2"], errors="coerce")
    df["task_name"] = ordered_categorical(df["task_name"], TASK_ORDER)
    df["objective"] = ordered_categorical(df["objective"], REP_OBJECTIVE_ORDER)
    df["init_mode"] = ordered_categorical(df["init_mode"], INIT_ORDER)
    df = df.sort_values(["task_name", "init_mode", "objective"]).reset_index(drop=True)
    assert_no_seginit(df, "representation landmarks")
    assert_expected_rows(
        df,
        len(TASK_ORDER) * len(REP_OBJECTIVE_ORDER) * len(INIT_ORDER),
        "representation landmarks",
    )
    return df


def load_training_runs(objectives: list[str], scales: list[str], inits: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(RUNS_ROOT.glob("*__geometry-fixedcompute-*/train_metrics.jsonl")):
        run_name = metrics_path.parent.name.split("__", 1)[1]
        try:
            parts = parse_run_name(run_name)
        except ValueError:
            continue
        if not parts.is_50k:
            continue
        if parts.objective not in objectives or parts.scale not in scales or parts.init not in inits:
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec["run_name"] = run_name
                rec["scale"] = parts.scale
                rec["objective"] = parts.objective
                rec["init"] = parts.init
                rec["reg"] = rec.get("reg", rec.get("sigreg", rec.get("bhep", rec.get("ep_partial"))))
                rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("No matching 50k-step training metrics found")
    for col in ("step", "epoch", "lr", "loss", "sim", "reg"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["scale"] = ordered_categorical(df["scale"], SCALE_ORDER)
    objective_order = REP_OBJECTIVE_ORDER if objectives == REP_OBJECTIVE_ORDER else DATASET_OBJECTIVE_ORDER
    df["objective"] = ordered_categorical(df["objective"], objective_order)
    df["init"] = ordered_categorical(df["init"], INIT_ORDER)
    assert_no_seginit(df, "training metrics")
    return df.sort_values(["objective", "init", "scale", "step"]).reset_index(drop=True)


def dataset_resource_summary() -> pd.DataFrame:
    manifest = pd.read_csv(LANDMARK_MANIFEST_CSV)
    geometry = pd.read_csv(DATASET_GEOMETRY_CSV)

    holdout_count = int(
        geometry.loc[
            (geometry["dataset_name"] == "subset6_minus_7_test") & (geometry["embedding_key"] == "proj"),
            "N",
        ].iloc[0]
    )
    open_hr_count = int(
        geometry.loc[
            (geometry["dataset_name"] == "subset7_eval") & (geometry["embedding_key"] == "proj"),
            "N",
        ].iloc[0]
    )
    clinic_count = int(
        geometry.loc[
            (geometry["dataset_name"] == "cfc_eval") & (geometry["embedding_key"] == "proj"),
            "N",
        ].iloc[0]
    )

    landmark_counts = manifest["dataset_name"].value_counts().to_dict()

    rows = [
        {
            "resource_name": "Main external-eye corpus",
            "role": "Unlabeled master corpus",
            "source_type": "Open-source face datasets",
            "label_availability": "Unlabeled",
            "public_release_status": "Partial",
            "redistribution_status": "Mixed",
            "sample_count": 4_352_692,
            "used_for_pretraining": True,
            "used_for_evaluation": False,
        },
        {
            "resource_name": "Pretrain-10K",
            "role": "SSL pretraining subset",
            "source_type": "Main external-eye corpus",
            "label_availability": "Unlabeled",
            "public_release_status": "Derived",
            "redistribution_status": "Mixed",
            "sample_count": 10_000,
            "used_for_pretraining": True,
            "used_for_evaluation": False,
        },
        {
            "resource_name": "Pretrain-100K",
            "role": "SSL pretraining subset",
            "source_type": "Main external-eye corpus",
            "label_availability": "Unlabeled",
            "public_release_status": "Derived",
            "redistribution_status": "Mixed",
            "sample_count": 100_000,
            "used_for_pretraining": True,
            "used_for_evaluation": False,
        },
        {
            "resource_name": "Pretrain-1M",
            "role": "SSL pretraining subset",
            "source_type": "Main external-eye corpus",
            "label_availability": "Unlabeled",
            "public_release_status": "Derived",
            "redistribution_status": "Mixed",
            "sample_count": 1_000_000,
            "used_for_pretraining": True,
            "used_for_evaluation": False,
        },
        {
            "resource_name": "Source-Holdout",
            "role": "Source-matched geometry evaluation",
            "source_type": "Open-source face datasets",
            "label_availability": "Unlabeled",
            "public_release_status": "Partial",
            "redistribution_status": "Mixed",
            "sample_count": holdout_count,
            "used_for_pretraining": False,
            "used_for_evaluation": True,
        },
        {
            "resource_name": "Open-HR",
            "role": "High-resolution open-source geometry evaluation",
            "source_type": "Open-source face datasets",
            "label_availability": "Unlabeled",
            "public_release_status": "Partial",
            "redistribution_status": "Mixed",
            "sample_count": open_hr_count,
            "used_for_pretraining": False,
            "used_for_evaluation": True,
        },
        {
            "resource_name": "Clinic-CF",
            "role": "Clinical domain-shift geometry evaluation",
            "source_type": "Clinical repository",
            "label_availability": "Unlabeled",
            "public_release_status": "No",
            "redistribution_status": "Restricted",
            "sample_count": clinic_count,
            "used_for_pretraining": False,
            "used_for_evaluation": True,
        },
        {
            "resource_name": "Landmark-Celeb",
            "role": "Within-dataset and transfer probe source",
            "source_type": "Open-source periocular annotations",
            "label_availability": "9 landmark coordinates",
            "public_release_status": "Yes",
            "redistribution_status": "Permitted",
            "sample_count": int(landmark_counts.get("celeb", 0)),
            "used_for_pretraining": False,
            "used_for_evaluation": True,
        },
        {
            "resource_name": "Landmark-CFD",
            "role": "Within-dataset and transfer probe target",
            "source_type": "Open-source periocular annotations",
            "label_availability": "9 landmark coordinates",
            "public_release_status": "Yes",
            "redistribution_status": "Permitted",
            "sample_count": int(landmark_counts.get("cfd", 0)),
            "used_for_pretraining": False,
            "used_for_evaluation": True,
        },
    ]
    return pd.DataFrame(rows)
