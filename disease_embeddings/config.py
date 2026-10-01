from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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


@dataclass(frozen=True)
class DatasetSpec:
    root: Path
    manifest_csv: Path
    image_size: int
    normalize_imagenet: bool


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    source: str
    run_name: str
    checkpoint_step: int
    run_dir: Path | None = None
    checkpoint_path: Path | None = None
    config_path: Path | None = None
    backbone: str | None = None
    pretrain_init: str | None = None
    method: str | None = None
    feat_dim: int | None = None
    model_family: str | None = None
    patch_size: int | None = None
    external_model: str | None = None

    def inline_training_config(self) -> dict[str, Any] | None:
        if self.config_path is not None:
            return None
        if self.backbone is None and self.pretrain_init is None and self.method is None and self.feat_dim is None:
            return None
        backbone = self.backbone or "resnet101"
        if self.feat_dim is not None:
            feat_dim = int(self.feat_dim)
        elif backbone.startswith("vit_"):
            feat_dim = 768
        else:
            feat_dim = 2048
        if self.model_family is not None:
            model_family = self.model_family
        elif backbone.startswith("vit_"):
            model_family = "vit"
        else:
            model_family = "resnet"
        return {
            "run": {
                "name": self.run_name,
            },
            "model": {
                "backbone": backbone,
                "init": self.pretrain_init or "random",
                "feat_dim": feat_dim,
                "model_family": model_family,
                "model_arch": backbone,
                "patch_size": self.patch_size,
                "pretrain_init": self.pretrain_init,
            },
            "ssl": {
                "method": self.method or "checkpoint",
            },
        }


@dataclass(frozen=True)
class ExtractionSpec:
    batch_size: int
    num_workers: int
    device: str
    precision: str
    overwrite: bool


@dataclass(frozen=True)
class ReductionSpec:
    method: str
    random_state: int
    tsne_perplexity: float


@dataclass(frozen=True)
class SupervisedSpec:
    label_column: str
    group_column: str
    train_frac: float
    seed: int
    linear_epochs: int
    linear_batch_size: int
    linear_lr: float
    finetune_epochs: int
    finetune_batch_size: int
    backbone_lr: float
    head_lr: float
    weight_decay: float


@dataclass(frozen=True)
class StudyConfig:
    path: Path
    name: str
    output_root: Path
    dataset: DatasetSpec
    models: tuple[ModelSpec, ...]
    pooling: str
    embedding_key: str
    extraction: ExtractionSpec
    reduction: ReductionSpec
    supervised: SupervisedSpec

    @property
    def embeddings_dir(self) -> Path:
        return self.output_root / "embeddings"

    @property
    def reductions_dir(self) -> Path:
        return self.output_root / "reductions"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def splits_dir(self) -> Path:
        return self.output_root / "splits"

    @property
    def linear_probe_dir(self) -> Path:
        return self.output_root / "linear_probe"

    @property
    def finetune_dir(self) -> Path:
        return self.output_root / f"finetune_{self.supervised.finetune_epochs}epochs"

    @property
    def summaries_dir(self) -> Path:
        return self.output_root / "summaries"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_model(base_dir: Path, raw: dict[str, Any]) -> ModelSpec:
    source = str(_require(raw.get("source"), "models[].source"))
    if source not in {"checkpoint", "external"}:
        raise ValueError(f"Unsupported model source: {source}")
    external_model = raw.get("external_model")
    run_dir = _resolve_path(base_dir, raw.get("run_dir"))
    checkpoint_path = _resolve_path(base_dir, raw.get("checkpoint_path"))
    config_path = _resolve_path(base_dir, raw.get("config_path"))
    has_inline_config = any(raw.get(key) is not None for key in ("backbone", "pretrain_init", "method", "feat_dim"))
    if source == "checkpoint" and run_dir is None and config_path is None and not has_inline_config:
        raise ValueError("Checkpoint models must define run_dir, config_path, or inline checkpoint metadata")
    if source == "external" and external_model is None:
        raise ValueError("External models must define external_model")
    return ModelSpec(
        model_id=str(_require(raw.get("model_id"), "models[].model_id")),
        label=str(_require(raw.get("label"), "models[].label")),
        source=source,
        run_name=str(raw.get("run_name") or raw.get("external_model") or raw.get("model_id")),
        checkpoint_step=int(raw.get("checkpoint_step", 0)),
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        backbone=str(raw["backbone"]) if raw.get("backbone") is not None else None,
        pretrain_init=str(raw["pretrain_init"]) if raw.get("pretrain_init") is not None else None,
        method=str(raw["method"]) if raw.get("method") is not None else None,
        feat_dim=int(raw["feat_dim"]) if raw.get("feat_dim") is not None else None,
        model_family=str(raw["model_family"]) if raw.get("model_family") is not None else None,
        patch_size=int(raw["patch_size"]) if raw.get("patch_size") is not None else None,
        external_model=str(external_model) if external_model is not None else None,
    )


def load_study_config(path: str | Path) -> StudyConfig:
    cfg_path = Path(path).resolve()
    raw = _load_yaml(cfg_path)
    base_dir = cfg_path.parent

    study_raw = raw.get("study", {})
    dataset_raw = raw.get("dataset", {})
    extraction_raw = raw.get("extraction", {})
    reduction_raw = raw.get("reduction", {})
    supervised_raw = raw.get("supervised", {})

    output_root = _resolve_path(base_dir, _require(study_raw.get("output_root"), "study.output_root"))
    dataset_root = _resolve_path(base_dir, _require(dataset_raw.get("root"), "dataset.root"))
    manifest_csv = _resolve_path(base_dir, _require(dataset_raw.get("manifest_csv"), "dataset.manifest_csv"))
    if output_root is None or dataset_root is None or manifest_csv is None:
        raise ValueError("Resolved paths cannot be None")

    models_raw = raw.get("models")
    if not models_raw:
        raise ValueError("Study config must define at least one model")

    cfg = StudyConfig(
        path=cfg_path,
        name=str(_require(study_raw.get("name"), "study.name")),
        output_root=output_root,
        dataset=DatasetSpec(
            root=dataset_root,
            manifest_csv=manifest_csv,
            image_size=int(dataset_raw.get("image_size", 224)),
            normalize_imagenet=bool(dataset_raw.get("normalize_imagenet", True)),
        ),
        models=tuple(_parse_model(base_dir, item) for item in models_raw),
        pooling=str(raw.get("pooling", "g4")),
        embedding_key=str(raw.get("embedding_key", "backbone/features")),
        extraction=ExtractionSpec(
            batch_size=int(extraction_raw.get("batch_size", 128)),
            num_workers=int(extraction_raw.get("num_workers", 4)),
            device=str(extraction_raw.get("device", "auto")),
            precision=str(extraction_raw.get("precision", "fp32")).lower(),
            overwrite=bool(extraction_raw.get("overwrite", False)),
        ),
        reduction=ReductionSpec(
            method=str(reduction_raw.get("method", "tsne")).lower(),
            random_state=int(reduction_raw.get("random_state", 0)),
            tsne_perplexity=float(reduction_raw.get("tsne_perplexity", 30.0)),
        ),
        supervised=SupervisedSpec(
            label_column=str(supervised_raw.get("label_column", "folder_label")),
            group_column=str(supervised_raw.get("group_column", "source_image_path")),
            train_frac=float(supervised_raw.get("train_frac", 0.8)),
            seed=int(supervised_raw.get("seed", 0)),
            linear_epochs=int(supervised_raw.get("linear_epochs", 100)),
            linear_batch_size=int(supervised_raw.get("linear_batch_size", 128)),
            linear_lr=float(supervised_raw.get("linear_lr", 1e-3)),
            finetune_epochs=int(supervised_raw.get("finetune_epochs", 1)),
            finetune_batch_size=int(supervised_raw.get("finetune_batch_size", 32)),
            backbone_lr=float(supervised_raw.get("backbone_lr", 1e-4)),
            head_lr=float(supervised_raw.get("head_lr", 1e-3)),
            weight_decay=float(supervised_raw.get("weight_decay", 1e-4)),
        ),
    )
    validate_study_config(cfg)
    return cfg


def validate_study_config(cfg: StudyConfig) -> None:
    if cfg.pooling != "g4":
        raise ValueError(f"Disease embedding v1 expects pooling='g4', got {cfg.pooling!r}")
    if not cfg.dataset.root.exists():
        raise FileNotFoundError(f"Disease image root does not exist: {cfg.dataset.root}")
    if not cfg.dataset.manifest_csv.exists():
        raise FileNotFoundError(f"Disease manifest does not exist: {cfg.dataset.manifest_csv}")
    if not 0.0 < cfg.supervised.train_frac < 1.0:
        raise ValueError(f"supervised.train_frac must be in (0, 1), got {cfg.supervised.train_frac}")
    if cfg.supervised.finetune_epochs < 1:
        raise ValueError(f"supervised.finetune_epochs must be >= 1, got {cfg.supervised.finetune_epochs}")

    seen: set[str] = set()
    for model in cfg.models:
        if model.model_id in seen:
            raise ValueError(f"Duplicate model_id in config: {model.model_id}")
        seen.add(model.model_id)
        if model.source == "checkpoint":
            inline_config = model.inline_training_config()
            if model.run_dir is None and model.config_path is None and inline_config is None:
                raise ValueError(f"Checkpoint model {model.model_id} is missing run_dir, config_path, or inline checkpoint metadata")
            if model.run_dir is not None:
                if not model.run_dir.exists():
                    raise FileNotFoundError(f"Run directory missing for {model.model_id}: {model.run_dir}")
                if model.config_path is None and inline_config is None and not (model.run_dir / "config.yaml").exists():
                    raise FileNotFoundError(f"Run config missing for {model.model_id}: {model.run_dir / 'config.yaml'}")
            if model.config_path is not None and not model.config_path.exists():
                raise FileNotFoundError(f"Run config missing for {model.model_id}: {model.config_path}")
            checkpoint_path = model.checkpoint_path
            if checkpoint_path is None:
                if model.run_dir is None:
                    raise ValueError(f"Checkpoint model {model.model_id} is missing checkpoint_path")
                checkpoint_path = model.run_dir / "checkpoints" / f"ckpt_step_{model.checkpoint_step:07d}.pth"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint missing for {model.model_id}: {checkpoint_path}")
        elif model.source == "external":
            if model.external_model is None:
                raise ValueError(f"External model {model.model_id} is missing external_model")
