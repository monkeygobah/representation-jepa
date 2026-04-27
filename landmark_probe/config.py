from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from landmark_probe.constants import VALID_POOLING

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path("/workspace")


def _resolve_path(base_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if path.is_absolute():
        try:
            workspace_relative = path.relative_to(WORKSPACE_ROOT)
        except ValueError:
            return path
        return (PROJECT_ROOT / workspace_relative).resolve()
    return (base_dir / path).resolve()


def _require(value: Any, key: str) -> Any:
    if value is None:
        raise ValueError(f"Missing required config value: {key}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class RawDatasetSource:
    name: str
    image_dir: Path
    mask_dir: Path
    image_suffix: str
    mask_suffix: str


@dataclass(frozen=True)
class DatasetMetadataSpec:
    manifest_csv: Path
    landmarks_csv: Path
    split_csv: Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    image_size: int
    normalize_imagenet: bool
    landmarks: tuple[str, ...]
    subdatasets: tuple[str, ...]
    metadata: DatasetMetadataSpec
    raw_sources: tuple[RawDatasetSource, ...]
    split_seed: int = 0
    split_train_frac: float = 0.8
    split_val_frac: float = 0.1
    split_test_frac: float = 0.1

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    def image_dir(self, dataset_name: str) -> Path:
        return self.root / dataset_name / "images"


@dataclass(frozen=True)
class ProbeConfig:
    path: Path
    name: str
    hidden_dims: tuple[int, ...]
    dropout: float
    batch_size: int
    num_workers: int
    max_epochs: int
    lr: float
    weight_decay: float
    early_stopping_enabled: bool
    early_stopping_patience: int
    selection_metric: str
    selection_mode: str


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    run_dir: Path
    checkpoint_step: int
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class RepresentationSpec:
    embedding_key: str
    pooling: str


@dataclass(frozen=True)
class TaskSplitSpec:
    dataset_name: str
    split: str


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    train_split: TaskSplitSpec
    val_split: TaskSplitSpec
    test_split: TaskSplitSpec


@dataclass(frozen=True)
class ExtractionSpec:
    batch_size: int
    num_workers: int
    device: str
    precision: str
    overwrite: bool = False


@dataclass(frozen=True)
class ArtifactSpec:
    save_embeddings: bool = True
    save_predictions: bool = True
    save_per_sample_metrics: bool = True


@dataclass(frozen=True)
class StudyConfig:
    path: Path
    name: str
    output_root: Path
    dataset_cfg_path: Path
    probe_cfg_path: Path
    runs: tuple[RunSpec, ...]
    representations: tuple[RepresentationSpec, ...]
    tasks: tuple[TaskSpec, ...]
    extraction: ExtractionSpec
    artifact: ArtifactSpec

    @property
    def embeddings_dir(self) -> Path:
        return self.output_root / "embeddings" / self.name

    @property
    def probe_runs_dir(self) -> Path:
        return self.output_root / "probe_runs" / self.name

    @property
    def summaries_dir(self) -> Path:
        return self.output_root / "summaries" / self.name


def load_dataset_config(path: str | Path) -> DatasetSpec:
    cfg_path = Path(path).resolve()
    raw = _load_yaml(cfg_path)
    base_dir = cfg_path.parent

    dataset_raw = raw.get("dataset", {})
    name = str(_require(dataset_raw.get("name"), "dataset.name"))
    root = _resolve_path(base_dir, _require(dataset_raw.get("root"), "dataset.root"))

    image_raw = raw.get("image", {})
    image_size = int(image_raw.get("size", 224))
    normalize_imagenet = bool(image_raw.get("normalize_imagenet", True))

    landmarks_raw = raw.get("landmarks", {})
    landmarks = tuple(landmarks_raw.get("keys", []))
    if not landmarks:
        raise ValueError("Dataset config must define landmarks.keys")

    subdatasets = tuple(item["name"] for item in raw.get("subdatasets", []))
    if not subdatasets:
        raise ValueError("Dataset config must define at least one subdataset")

    metadata_raw = raw.get("metadata", {})
    metadata = DatasetMetadataSpec(
        manifest_csv=root / _require(metadata_raw.get("manifest_csv"), "metadata.manifest_csv"),
        landmarks_csv=root / _require(metadata_raw.get("landmarks_csv"), "metadata.landmarks_csv"),
        split_csv=root / _require(metadata_raw.get("split_csv"), "metadata.split_csv"),
    )

    split_raw = raw.get("splits", {})
    split_seed = int(split_raw.get("seed", 0))
    split_train_frac = float(split_raw.get("train_frac", 0.8))
    split_val_frac = float(split_raw.get("val_frac", 0.1))
    split_test_frac = float(split_raw.get("test_frac", 0.1))

    total_frac = split_train_frac + split_val_frac + split_test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total_frac}")

    raw_sources = []
    for item in raw.get("raw_sources", []):
        raw_sources.append(
            RawDatasetSource(
                name=str(_require(item.get("name"), "raw_sources[].name")),
                image_dir=_resolve_path(base_dir, _require(item.get("image_dir"), "raw_sources[].image_dir")),
                mask_dir=_resolve_path(base_dir, _require(item.get("mask_dir"), "raw_sources[].mask_dir")),
                image_suffix=str(_require(item.get("image_suffix"), "raw_sources[].image_suffix")),
                mask_suffix=str(_require(item.get("mask_suffix"), "raw_sources[].mask_suffix")),
            )
        )
    if not raw_sources:
        raise ValueError("Dataset config must define raw_sources")

    cfg = DatasetSpec(
        name=name,
        root=root,
        image_size=image_size,
        normalize_imagenet=normalize_imagenet,
        landmarks=landmarks,
        subdatasets=subdatasets,
        metadata=metadata,
        raw_sources=tuple(raw_sources),
        split_seed=split_seed,
        split_train_frac=split_train_frac,
        split_val_frac=split_val_frac,
        split_test_frac=split_test_frac,
    )
    validate_dataset_config(cfg)
    return cfg


def validate_dataset_config(cfg: DatasetSpec) -> None:
    for source in cfg.raw_sources:
        if not source.image_dir.exists():
            raise FileNotFoundError(f"Raw image directory does not exist: {source.image_dir}")
        if not source.mask_dir.exists():
            raise FileNotFoundError(f"Raw mask directory does not exist: {source.mask_dir}")


def load_probe_config(path: str | Path) -> ProbeConfig:
    cfg_path = Path(path).resolve()
    raw = _load_yaml(cfg_path)
    probe_raw = raw.get("probe", {})
    model_raw = raw.get("model", {})
    train_raw = raw.get("train", {})
    optim_raw = raw.get("optim", {})
    early_raw = train_raw.get("early_stopping", {})
    selection_raw = raw.get("selection", {})

    hidden_dims = tuple(int(v) for v in model_raw.get("hidden_dims", []))
    if not hidden_dims:
        raise ValueError("Probe config must define model.hidden_dims")

    return ProbeConfig(
        path=cfg_path,
        name=str(probe_raw.get("name", cfg_path.stem)),
        hidden_dims=hidden_dims,
        dropout=float(model_raw.get("dropout", 0.2)),
        batch_size=int(train_raw.get("batch_size", 256)),
        num_workers=int(train_raw.get("num_workers", 4)),
        max_epochs=int(train_raw.get("max_epochs", 1000)),
        lr=float(optim_raw.get("lr", 1e-3)),
        weight_decay=float(optim_raw.get("weight_decay", 1e-4)),
        early_stopping_enabled=bool(early_raw.get("enabled", True)),
        early_stopping_patience=int(early_raw.get("patience", 50)),
        selection_metric=str(selection_raw.get("metric", "val_mean_l2")),
        selection_mode=str(selection_raw.get("mode", "min")),
    )


def load_study_config(path: str | Path) -> StudyConfig:
    cfg_path = Path(path).resolve()
    raw = _load_yaml(cfg_path)
    base_dir = cfg_path.parent
    study_raw = raw.get("study", {})

    dataset_cfg_path = _resolve_path(base_dir, _require(raw.get("dataset_cfg"), "dataset_cfg"))
    probe_cfg_path = _resolve_path(base_dir, _require(raw.get("probe_cfg"), "probe_cfg"))

    runs_raw = raw.get("runs", [])
    if not runs_raw:
        raise ValueError("Study config must define runs")
    runs = tuple(
        RunSpec(
            run_name=str(item.get("run_name") or Path(_require(item.get("run_dir"), "runs[].run_dir")).name),
            run_dir=_resolve_path(base_dir, _require(item.get("run_dir"), "runs[].run_dir")),
            checkpoint_step=int(_require(item.get("checkpoint_step"), "runs[].checkpoint_step")),
            checkpoint_path=_resolve_path(base_dir, item.get("checkpoint_path")),
        )
        for item in runs_raw
    )

    rep_raw = raw.get("representations", [])
    if not rep_raw:
        raise ValueError("Study config must define representations")
    representations = []
    for item in rep_raw:
        pooling = str(_require(item.get("pooling"), "representations[].pooling"))
        if pooling not in VALID_POOLING:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        representations.append(
            RepresentationSpec(
                embedding_key=str(_require(item.get("embedding_key"), "representations[].embedding_key")),
                pooling=pooling,
            )
        )

    task_raw = raw.get("tasks", [])
    if not task_raw:
        raise ValueError("Study config must define tasks")
    tasks = []
    for item in task_raw:
        tasks.append(
            TaskSpec(
                task_name=str(_require(item.get("task_name"), "tasks[].task_name")),
                train_split=TaskSplitSpec(**_require(item.get("train_split"), "tasks[].train_split")),
                val_split=TaskSplitSpec(**_require(item.get("val_split"), "tasks[].val_split")),
                test_split=TaskSplitSpec(**_require(item.get("test_split"), "tasks[].test_split")),
            )
        )

    extraction_raw = raw.get("extraction", {})
    artifact_raw = raw.get("artifact", {})
    cfg = StudyConfig(
        path=cfg_path,
        name=str(_require(study_raw.get("name"), "study.name")),
        output_root=_resolve_path(base_dir, _require(study_raw.get("output_root"), "study.output_root")),
        dataset_cfg_path=dataset_cfg_path,
        probe_cfg_path=probe_cfg_path,
        runs=runs,
        representations=tuple(representations),
        tasks=tuple(tasks),
        extraction=ExtractionSpec(
            batch_size=int(extraction_raw.get("batch_size", 256)),
            num_workers=int(extraction_raw.get("num_workers", 4)),
            device=str(extraction_raw.get("device", "auto")),
            precision=str(extraction_raw.get("precision", "fp32")).lower(),
            overwrite=bool(extraction_raw.get("overwrite", False)),
        ),
        artifact=ArtifactSpec(
            save_embeddings=bool(artifact_raw.get("save_embeddings", True)),
            save_predictions=bool(artifact_raw.get("save_predictions", True)),
            save_per_sample_metrics=bool(artifact_raw.get("save_per_sample_metrics", True)),
        ),
    )
    validate_study_config(cfg)
    return cfg


def validate_study_config(cfg: StudyConfig) -> None:
    if not cfg.dataset_cfg_path.exists():
        raise FileNotFoundError(f"Dataset config does not exist: {cfg.dataset_cfg_path}")
    if not cfg.probe_cfg_path.exists():
        raise FileNotFoundError(f"Probe config does not exist: {cfg.probe_cfg_path}")
    for run in cfg.runs:
        if not run.run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run.run_dir}")
        if not (run.run_dir / "config.yaml").exists():
            raise FileNotFoundError(f"Run config missing: {run.run_dir / 'config.yaml'}")
    for task in cfg.tasks:
        for split_spec in (task.train_split, task.val_split, task.test_split):
            if split_spec.split not in {"train", "val", "test"}:
                raise ValueError(f"Unsupported split label in task {task.task_name}: {split_spec.split}")
