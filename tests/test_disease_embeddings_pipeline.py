from __future__ import annotations

from pathlib import Path

import json
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image

from disease_embeddings.config import load_study_config
from disease_embeddings.datasets import build_dataloader, load_manifest_records
from disease_embeddings.paths import adapted_reduction_csv_path, embedding_artifact_path, finetune_dir, linear_probe_dir, split_csv_path
from disease_embeddings.reduce_plot import plot_coordinates, reduce_and_plot_model, reduce_embeddings
from disease_embeddings.supervised import (
    FineTuneClassifier,
    build_grouped_split,
    label_space_from_records,
    load_or_create_split,
    run_linear_probe_for_model,
    _write_adapted_plots,
)
from disease_embeddings.summarize import adapted_knn5_metrics_for_model, summarize_adapted_knn5, summarize_finetune, summarize_linear_probe
from disease_embeddings.summarize import knn5_metrics_for_model
from landmark_probe.extract.inference import pooled_feature_map_embeddings


def test_manifest_loading_preserves_metadata(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    cfg = load_study_config(cfg_path)

    records = load_manifest_records(cfg.dataset)

    assert len(records) == 3
    assert records[0].metadata["folder_label"] == "CAP"
    assert records[0].metadata["eye"] == "OD"
    assert records[0].metadata["source_mode"] == "split_half"

    dl = build_dataloader(cfg.dataset, cfg.extraction)
    xs, batch_records = next(iter(dl))
    assert xs.shape == (2, 3, 16, 16)
    assert [record.metadata["output_path"] for record in batch_records] == [
        "CAP/cap_od.png",
        "ptosis/ptosis_os.png",
    ]


def test_g4_pooling_returns_expected_dimensions() -> None:
    class FeatureMap(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.channels = channels

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], self.channels, 7, 7)

    resnet_emb = pooled_feature_map_embeddings(FeatureMap(2048), torch.zeros(2, 3, 16, 16), "g4")
    vit_emb = pooled_feature_map_embeddings(FeatureMap(768), torch.zeros(2, 3, 16, 16), "g4")

    assert resnet_emb.shape == (2, 32768)
    assert vit_emb.shape == (2, 12288)


def test_reduction_csv_includes_coordinates_and_metadata(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    cfg = load_study_config(cfg_path)
    model = cfg.models[0]
    artifact_path = embedding_artifact_path(cfg, model)
    artifact_path.parent.mkdir(parents=True)
    torch.save(
        {
            "embeddings": torch.arange(30, dtype=torch.float32).reshape(3, 10),
            "metadata": [
                {
                    "output_path": "CAP/cap_od.png",
                    "source_image_path": "CAP/cap.png",
                    "folder_label": "CAP",
                    "filename": "cap.png",
                    "eye": "OD",
                    "disease_status": "dis",
                    "source_mode": "split_half",
                    "manifest_row_index": 0,
                },
                {
                    "output_path": "ptosis/ptosis_os.png",
                    "source_image_path": "ptosis/ptosis.png",
                    "folder_label": "ptosis",
                    "filename": "ptosis.png",
                    "eye": "OS",
                    "disease_status": "dis",
                    "source_mode": "split_half",
                    "manifest_row_index": 1,
                },
                {
                    "output_path": "TED/ted_od.png",
                    "source_image_path": "TED/ted.png",
                    "folder_label": "TED",
                    "filename": "ted.png",
                    "eye": "OD",
                    "disease_status": "dis",
                    "source_mode": "split_half",
                    "manifest_row_index": 2,
                },
            ],
            "model": {
                "model_id": model.model_id,
                "label": model.label,
                "source": model.source,
                "external_model": None,
                "run_name": model.run_name,
                "run_dir": str(model.run_dir),
            },
            "checkpoint": {"checkpoint_step": 0, "checkpoint_path": "unit://checkpoint"},
            "embedding_key": "backbone/features",
            "pooling": "g4",
            "embedding_dim": 10,
        },
        artifact_path,
    )

    csv_path, fig_path = reduce_and_plot_model(cfg, model, "pca")
    df = pd.read_csv(csv_path)

    assert fig_path.exists()
    assert {"x", "y", "folder_label", "model_id", "model_label", "run_name", "checkpoint_path"}.issubset(df.columns)
    assert df["model_label"].unique().tolist() == ["Unit Model"]
    assert df["folder_label"].tolist() == ["CAP", "ptosis", "TED"]


def test_plot_generation_writes_png_with_stable_color_order(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 0.0],
            "folder_label": ["CAP", "ptosis", "TED"],
        }
    )
    out_path = tmp_path / "plot.png"

    plot_coordinates(df, "Unit Model", out_path, ["CAP", "TED", "ptosis"])

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_reduce_embeddings_supports_pca_for_tiny_inputs() -> None:
    coords = reduce_embeddings(torch.eye(4), method="pca", random_state=0)

    assert coords.shape == (4, 2)


def test_grouped_split_has_no_source_leakage_and_keeps_rare_test(tmp_path: Path) -> None:
    cfg_path = _write_supervised_config(tmp_path)
    cfg = load_study_config(cfg_path)
    records = load_manifest_records(cfg.dataset)

    split_df = build_grouped_split(records, cfg)

    per_group = split_df.groupby("source_image_path")["split"].nunique()
    assert int(per_group.max()) == 1
    assert {"train", "test"} == set(split_df["split"])
    test_labels = set(split_df.loc[split_df["split"] == "test", "folder_label"])
    assert {"CAP", "ptosis"} <= test_labels

    written = load_or_create_split(cfg)
    assert split_csv_path(cfg).exists()
    assert len(written) == len(split_df)


def test_linear_probe_writes_outputs(tmp_path: Path) -> None:
    cfg_path = _write_supervised_config(tmp_path, linear_epochs=2)
    cfg = load_study_config(cfg_path)
    model = cfg.models[0]
    records = load_manifest_records(cfg.dataset)
    label_space = label_space_from_records(records, cfg.supervised.label_column)
    embeddings = []
    metadata = []
    for record in records:
        label_idx = label_space.label_to_idx[record.folder_label]
        embeddings.append([float(label_idx), float(record.row_index), 1.0])
        row = dict(record.metadata)
        row["manifest_row_index"] = record.row_index
        metadata.append(row)
    artifact_path = embedding_artifact_path(cfg, model)
    artifact_path.parent.mkdir(parents=True)
    torch.save(
        {
            "embeddings": torch.tensor(embeddings, dtype=torch.float32),
            "metadata": metadata,
            "model": {
                "model_id": model.model_id,
                "label": model.label,
                "source": model.source,
                "external_model": None,
                "run_name": model.run_name,
                "run_dir": str(model.run_dir),
            },
            "checkpoint": {"checkpoint_step": 0, "checkpoint_path": "unit://checkpoint"},
            "embedding_key": "backbone/features",
            "pooling": "g4",
            "embedding_dim": 3,
        },
        artifact_path,
    )

    out_dir = run_linear_probe_for_model(cfg, model)

    assert out_dir == linear_probe_dir(cfg, model)
    assert (out_dir / "linear_head.pt").exists()
    assert (out_dir / "history.csv").exists()
    assert (out_dir / "test_metrics.json").exists()
    preds = pd.read_csv(out_dir / "predictions.csv")
    assert {"split", "true_label", "pred_label", "correct", "source_image_path"}.issubset(preds.columns)


def test_finetune_classifier_returns_logits_and_g4_embeddings() -> None:
    class FeatureMap(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 4, 8, 8)

    model = FineTuneClassifier(FeatureMap(), embedding_dim=4 * 16, num_classes=3, pooling="g4")
    logits, embeddings = model(torch.zeros(2, 3, 16, 16))

    assert logits.shape == (2, 3)
    assert embeddings.shape == (2, 64)


def test_adapted_plot_csv_preserves_supervised_metadata(tmp_path: Path) -> None:
    cfg_path = _write_supervised_config(tmp_path)
    cfg = load_study_config(cfg_path)
    model = cfg.models[0]
    records = load_manifest_records(cfg.dataset)
    split_df = load_or_create_split(cfg)
    split_by_row = {int(row.manifest_row_index): str(row.split) for row in split_df.itertuples(index=False)}
    metadata = []
    for record in records:
        row = dict(record.metadata)
        row["manifest_row_index"] = record.row_index
        metadata.append(row)
    labels = torch.tensor([0 if row["folder_label"] == "CAP" else 1 for row in metadata], dtype=torch.long)
    payload = {
        "embeddings": torch.arange(len(metadata) * 4, dtype=torch.float32).reshape(len(metadata), 4),
        "metadata": metadata,
        "labels": labels,
        "predictions": labels.clone(),
        "classes": ("CAP", "ptosis"),
        "model": {
            "model_id": model.model_id,
            "label": model.label,
            "source": model.source,
            "external_model": None,
            "run_name": model.run_name,
            "run_dir": str(model.run_dir),
        },
        "checkpoint": {"checkpoint_step": 0, "checkpoint_path": "unit://checkpoint"},
        "embedding_key": "finetuned_g4",
        "pooling": "g4",
        "embedding_dim": 4,
        "split_by_manifest_row_index": split_by_row,
    }

    _write_adapted_plots(cfg, model, payload, "pca")

    df = pd.read_csv(adapted_reduction_csv_path(cfg, model, "pca"))
    test_df = pd.read_csv(adapted_reduction_csv_path(cfg, model, "pca", split="test"))
    assert "finetune_1epochs" in str(adapted_reduction_csv_path(cfg, model, "pca"))
    assert {"split", "true_label", "pred_label", "correct", "model_id", "checkpoint_path"}.issubset(df.columns)
    assert set(test_df["split"]) == {"test"}


def test_linear_probe_summary_writes_ranked_tables(tmp_path: Path) -> None:
    cfg_path = _write_supervised_config(tmp_path)
    cfg = load_study_config(cfg_path)
    model = cfg.models[0]
    out_dir = linear_probe_dir(cfg, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": 0.75,
        "balanced_accuracy": 0.7,
        "macro_f1": 0.65,
        "weighted_f1": 0.74,
        "classes": ["CAP", "ptosis"],
        "classification_report": {
            "CAP": {"precision": 0.8, "recall": 0.9, "f1-score": 0.85, "support": 4.0},
            "ptosis": {"precision": 0.6, "recall": 0.5, "f1-score": 0.55, "support": 4.0},
        },
        "confusion_matrix": [[3, 1], [2, 2]],
        "model_id": model.model_id,
        "model_label": model.label,
        "run_name": model.run_name,
        "checkpoint_step": model.checkpoint_step,
        "embedding_dim": 3,
        "train_rows": 12,
        "test_rows": 8,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    _write_unit_embedding_artifact_for_knn(cfg, model)

    paths = summarize_linear_probe(cfg)

    headline = pd.read_csv(paths["headline"])
    per_class = pd.read_csv(paths["per_class"])
    confusion = pd.read_csv(paths["confusion"])
    knn = pd.read_csv(paths["knn"])
    assert headline.loc[0, "rank_by_macro_f1"] == 1
    assert headline.loc[0, "CAP__f1"] == 0.85
    assert "knn5_accuracy" in headline.columns
    assert set(per_class["class_name"]) == {"CAP", "ptosis"}
    assert int(confusion["count"].sum()) == 8
    assert knn.loc[0, "knn_k"] == 5


def test_knn5_metrics_use_train_neighbors_for_test_accuracy(tmp_path: Path) -> None:
    cfg_path = _write_supervised_config(tmp_path)
    cfg = load_study_config(cfg_path)
    model = cfg.models[0]
    _write_unit_embedding_artifact_for_knn(cfg, model)

    metrics = knn5_metrics_for_model(cfg, model)

    assert metrics["knn_k"] == 5
    assert metrics["knn5_accuracy"] >= 0.5


def test_adapted_knn5_summary_reads_finetuned_embeddings(tmp_path: Path) -> None:
    cfg_path = _write_supervised_config(tmp_path)
    cfg = load_study_config(cfg_path)
    model = cfg.models[0]
    _write_unit_adapted_embedding_artifact_for_knn(cfg, model)

    metrics = adapted_knn5_metrics_for_model(cfg, model)
    out_path = summarize_adapted_knn5(cfg)
    df = pd.read_csv(out_path)

    assert metrics["embedding_source"] == "finetune_1epochs"
    assert metrics["knn5_accuracy"] >= 0.5
    assert df.loc[0, "embedding_source"] == "finetune_1epochs"


def test_finetune_summary_writes_headline_per_class_and_confusion(tmp_path: Path) -> None:
    cfg_path = _write_supervised_config(tmp_path)
    cfg = load_study_config(cfg_path)
    model = cfg.models[0]
    _write_unit_adapted_embedding_artifact_for_knn(cfg, model)
    out_dir = finetune_dir(cfg, model)
    metrics = {
        "accuracy": 0.8,
        "balanced_accuracy": 0.75,
        "macro_f1": 0.7,
        "weighted_f1": 0.79,
        "classes": ["CAP", "ptosis"],
        "classification_report": {
            "CAP": {"precision": 0.9, "recall": 0.8, "f1-score": 0.84, "support": 5.0},
            "ptosis": {"precision": 0.7, "recall": 0.7, "f1-score": 0.7, "support": 5.0},
        },
        "confusion_matrix": [[4, 1], [1, 4]],
        "model_id": model.model_id,
        "model_label": model.label,
        "run_name": model.run_name,
        "checkpoint_step": model.checkpoint_step,
        "embedding_dim": 3,
        "train_rows": 10,
        "test_rows": 10,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    paths = summarize_finetune(cfg)

    headline = pd.read_csv(paths["headline"])
    per_class = pd.read_csv(paths["per_class"])
    confusion = pd.read_csv(paths["confusion"])
    assert headline.loc[0, "accuracy"] == 0.8
    assert "knn5_accuracy" in headline.columns
    assert set(per_class["class_name"]) == {"CAP", "ptosis"}
    assert int(confusion["count"].sum()) == 10


def _write_supervised_config(tmp_path: Path, linear_epochs: int = 3) -> Path:
    root = tmp_path / "supervised-eyes"
    rows = []
    for label in ("CAP", "ptosis"):
        for group_idx in range(5):
            source = f"{label}/source_{group_idx}.png"
            for eye in ("OD", "OS"):
                name = f"{label.lower()}_{group_idx}_{eye}.png"
                rel = f"{label}/{name}"
                (root / label).mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (12, 10), color="red" if label == "CAP" else "green").save(root / rel)
                rows.append(
                    {
                        "output_path": rel,
                        "source_image_path": source,
                        "folder_label": label,
                        "filename": name,
                        "eye": eye,
                        "disease_status": "dis",
                        "source_mode": "split_half",
                    }
                )
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    run_dir = tmp_path / "runs/unit"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("model:\n  feat_dim: 1\nssl:\n  method: unit\n", encoding="utf-8")
    (run_dir / "checkpoints").mkdir()
    torch.save({"encoder": {}, "objective": {}}, run_dir / "checkpoints/ckpt_step_0000000.pth")

    cfg = {
        "study": {"name": "unit_supervised", "output_root": str(tmp_path / "outputs")},
        "dataset": {
            "root": str(root),
            "manifest_csv": str(manifest),
            "image_size": 16,
            "normalize_imagenet": True,
        },
        "pooling": "g4",
        "embedding_key": "backbone/features",
        "models": [
            {
                "model_id": "unit_model",
                "label": "Unit Model",
                "source": "checkpoint",
                "run_name": "unit-run",
                "run_dir": str(run_dir),
                "checkpoint_step": 0,
            }
        ],
        "extraction": {"batch_size": 2, "num_workers": 0, "device": "cpu", "precision": "fp32"},
        "reduction": {"method": "pca", "random_state": 0, "tsne_perplexity": 2},
        "supervised": {
            "label_column": "folder_label",
            "group_column": "source_image_path",
            "train_frac": 0.8,
            "seed": 0,
            "linear_epochs": linear_epochs,
            "linear_batch_size": 4,
            "linear_lr": 0.01,
            "finetune_epochs": 1,
            "finetune_batch_size": 4,
            "backbone_lr": 0.0001,
            "head_lr": 0.001,
            "weight_decay": 0.0001,
        },
    }
    cfg_path = tmp_path / "supervised_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _write_unit_embedding_artifact_for_knn(cfg, model) -> None:
    records = load_manifest_records(cfg.dataset)
    split_df = load_or_create_split(cfg)
    embeddings = []
    metadata = []
    for record in records:
        label_value = 0.0 if record.folder_label == "CAP" else 10.0
        split_offset = 0.1 if split_df.loc[split_df["manifest_row_index"] == record.row_index, "split"].iloc[0] == "test" else 0.0
        embeddings.append([label_value + split_offset, 1.0, 0.0])
        row = dict(record.metadata)
        row["manifest_row_index"] = record.row_index
        metadata.append(row)
    artifact_path = embedding_artifact_path(cfg, model)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": torch.tensor(embeddings, dtype=torch.float32),
            "metadata": metadata,
            "model": {
                "model_id": model.model_id,
                "label": model.label,
                "source": model.source,
                "external_model": None,
                "run_name": model.run_name,
                "run_dir": str(model.run_dir),
            },
            "checkpoint": {"checkpoint_step": 0, "checkpoint_path": "unit://checkpoint"},
            "embedding_key": "backbone/features",
            "pooling": "g4",
            "embedding_dim": 3,
        },
        artifact_path,
    )


def _write_unit_adapted_embedding_artifact_for_knn(cfg, model) -> None:
    records = load_manifest_records(cfg.dataset)
    split_df = load_or_create_split(cfg)
    split_by_row = {int(row.manifest_row_index): str(row.split) for row in split_df.itertuples(index=False)}
    metadata = []
    embeddings = []
    labels = []
    for record in records:
        label_idx = 0 if record.folder_label == "CAP" else 1
        labels.append(label_idx)
        embeddings.append([float(label_idx * 10), 1.0, 0.0])
        row = dict(record.metadata)
        row["manifest_row_index"] = record.row_index
        metadata.append(row)
    out_dir = finetune_dir(cfg, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": torch.tensor(embeddings, dtype=torch.float32),
            "metadata": metadata,
            "labels": torch.tensor(labels, dtype=torch.long),
            "predictions": torch.tensor(labels, dtype=torch.long),
            "classes": ("CAP", "ptosis"),
            "model": {
                "model_id": model.model_id,
                "label": model.label,
                "source": model.source,
                "external_model": None,
                "run_name": model.run_name,
                "run_dir": str(model.run_dir),
            },
            "checkpoint": {"checkpoint_step": 0, "checkpoint_path": "unit://checkpoint"},
            "embedding_key": "finetuned_g4",
            "pooling": "g4",
            "embedding_dim": 3,
            "split_by_manifest_row_index": split_by_row,
        },
        out_dir / "adapted_embeddings.pt",
    )


def _write_config(tmp_path: Path) -> Path:
    root = tmp_path / "diseased-eyes-224"
    for label, name, color in [
        ("CAP", "cap_od.png", "red"),
        ("ptosis", "ptosis_os.png", "green"),
        ("TED", "ted_od.png", "blue"),
    ]:
        (root / label).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), color=color).save(root / label / name)

    manifest = root / "manifest.csv"
    pd.DataFrame(
        [
            {
                "output_path": "CAP/cap_od.png",
                "source_image_path": "CAP/cap.png",
                "folder_label": "CAP",
                "filename": "cap.png",
                "eye": "OD",
                "disease_status": "dis",
                "source_mode": "split_half",
            },
            {
                "output_path": "ptosis/ptosis_os.png",
                "source_image_path": "ptosis/ptosis.png",
                "folder_label": "ptosis",
                "filename": "ptosis.png",
                "eye": "OS",
                "disease_status": "dis",
                "source_mode": "split_half",
            },
            {
                "output_path": "TED/ted_od.png",
                "source_image_path": "TED/ted.png",
                "folder_label": "TED",
                "filename": "ted.png",
                "eye": "OD",
                "disease_status": "dis",
                "source_mode": "split_half",
            },
        ]
    ).to_csv(manifest, index=False)

    run_dir = tmp_path / "runs/unit"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("model:\n  feat_dim: 1\nssl:\n  method: unit\n", encoding="utf-8")
    (run_dir / "checkpoints").mkdir()
    torch.save({"encoder": {}, "objective": {}}, run_dir / "checkpoints/ckpt_step_0000000.pth")

    cfg = {
        "study": {"name": "unit", "output_root": str(tmp_path / "outputs")},
        "dataset": {
            "root": str(root),
            "manifest_csv": str(manifest),
            "image_size": 16,
            "normalize_imagenet": True,
        },
        "pooling": "g4",
        "embedding_key": "backbone/features",
        "models": [
            {
                "model_id": "unit_model",
                "label": "Unit Model",
                "source": "checkpoint",
                "run_name": "unit-run",
                "run_dir": str(run_dir),
                "checkpoint_step": 0,
            }
        ],
        "extraction": {"batch_size": 2, "num_workers": 0, "device": "cpu", "precision": "fp32"},
        "reduction": {"method": "pca", "random_state": 0, "tsne_perplexity": 2},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path
