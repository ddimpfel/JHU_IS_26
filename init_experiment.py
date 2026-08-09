"""Shared, reproducible setup for the GTSRB continual-learning MoE study.

This module is the single source of truth for the training and held-out-test
workflows.  The notebooks deliberately contain only presentation and orchestration
code; they must not redefine data splits, model components, or training losses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal, Sequence, cast
from importlib import import_module
import json
import os
import random
import re
import subprocess
import sys
from urllib.request import urlretrieve

import kagglehub
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from statsmodels.stats.multitest import multipletests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from continual_learning import (
    CILComputerVisionModel,
    build_complete_dataloader,
    bootstrap_performance_diff,
    serialize_save_json,
    serialize_save_model_training,
    train_final_cil_model,
    train_and_evaluate_cil_model,
)
from gre_model_base import (
    GeneralistRouterExperts,
    cost_proxy,
    expected_calibration_error,
    expert_metrics,
    router_entropy,
)
from joint_embedding import JointEmbeddingModule


DEFAULT_GTSRB_CLASSES = (1, 2, 3, 4, 5, 7, 8, 9, 10)
SIGNNAMES_URL = (
    "https://raw.githubusercontent.com/georgesung/traffic_sign_classification_german/"
    "master/signnames.csv"
)


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved paths used by notebooks in either VS Code or Google Colab."""

    project_dir: Path
    results_dir: Path
    in_colab: bool


def prepare_notebook_runtime(
    results_namespace: str,
    *,
    repo_url: str = "https://github.com/ddimpfel/JHU_IS_26.git",
    colab_drive_root: str = "/content/drive/MyDrive/is26_models",
) -> RuntimePaths:
    """Locate this package locally or clone it once when a notebook runs in Colab."""
    try:
        drive: Any = import_module("google.colab.drive")
        in_colab = True
    except ImportError:
        drive = None
        in_colab = False

    if in_colab:
        assert drive is not None
        drive.mount("/content/drive")
        repo_root = Path("/content") / Path(repo_url).stem
        if not repo_root.exists():
            subprocess.run(["git", "clone", repo_url], check=True)
        project_dir = repo_root
        if not (project_dir / "init_experiment.py").exists():
            raise FileNotFoundError(f"Expected the experiment package at {project_dir}.")
        results_dir = Path(colab_drive_root) / results_namespace
    else:
        current = Path.cwd().resolve()
        candidates = (current, current / "Current Work", *current.parents)
        project_dir = next((candidate for candidate in candidates if (candidate / "init_experiment.py").exists()), None)
        if project_dir is None:
            raise FileNotFoundError("Run this notebook from Current Work or a parent workspace directory.")
        results_dir = project_dir / "results" / results_namespace

    project_dir = project_dir.resolve()
    os.chdir(project_dir)
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    results_dir.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(project_dir=project_dir, results_dir=results_dir, in_colab=in_colab)


def default_experiment_config(runtime: RuntimePaths, *, run_seeds: tuple[int, ...] = (7,)) -> "ExperimentConfig":
    """Return identical practical defaults for every notebook execution environment."""
    return ExperimentConfig(
        run_seeds=run_seeds,
        num_workers=4 if runtime.in_colab else 0,
        pin_memory=runtime.in_colab,
        persistent_workers=runtime.in_colab,
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """All study conditions shared by baseline and joint-embedding experiments."""

    seed: int = 7
    run_seeds: tuple[int, ...] = (7,)
    cv_folds: int = 3
    cv_repeats: int = 1
    cv_epochs: int = 4
    final_run_seeds: tuple[int, ...] = (7, 17, 27)
    throughput_warmup_batches: int = 3
    dataset_name: str = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
    class_ids: tuple[int, ...] = DEFAULT_GTSRB_CLASSES
    validation_fraction: float = 0.20
    image_size: int = 224
    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    num_tasks: int = 3
    epochs: int = 8
    exemplar_ratio: float = 0.065
    task_1_lr: float = 0.001
    later_task_lr_factor: float = 0.1
    use_class_masking: bool = True
    kd_temperature: float = 2.0
    lambda_kd: float = 0.5
    num_experts: int = 6
    hidden_expert_size: int = 128
    dropout: float = 0.1
    top_k: int = 2
    lambda_aux: float = 0.05
    transformer_d_model: int = 32
    transformer_nhead: int = 4
    je_embedding_dim: int = 256
    je_projection_dim: int = 256
    je_feature_key: Literal["embeddings", "projections"] = "projections"
    je_temperature: float = 0.07
    je_contrastive_weight: float = 0.1
    pretrained_backbones: bool = True
    device: str | None = None

    def __post_init__(self) -> None:
        if len(self.class_ids) != len(set(self.class_ids)):
            raise ValueError("class_ids must not contain duplicates.")
        if len(self.class_ids) % self.num_tasks != 0:
            raise ValueError("The selected classes must divide evenly into num_tasks.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in the open interval (0, 1).")
        if not 0 < self.exemplar_ratio <= 1:
            raise ValueError("exemplar_ratio must be in the interval (0, 1].")
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")
        if self.transformer_d_model % self.transformer_nhead != 0:
            raise ValueError("transformer_d_model must be divisible by transformer_nhead.")
        if self.je_feature_key not in {"embeddings", "projections"}:
            raise ValueError("je_feature_key must be 'embeddings' or 'projections'.")
        if not self.run_seeds:
            raise ValueError("run_seeds must include at least one seed.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")
        if self.cv_repeats < 1 or self.cv_epochs < 1:
            raise ValueError("cv_repeats and cv_epochs must be positive.")
        if not self.final_run_seeds or len(self.final_run_seeds) != len(set(self.final_run_seeds)):
            raise ValueError("final_run_seeds must include unique seeds.")
        if self.throughput_warmup_batches < 0:
            raise ValueError("throughput_warmup_batches cannot be negative.")

    @property
    def total_classes(self) -> int:
        return len(self.class_ids)

    @property
    def resolved_device(self) -> str:
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def resolved_pin_memory(self) -> bool:
        return bool(self.resolved_device == "cuda") if self.pin_memory is None else self.pin_memory

    @property
    def resolved_persistent_workers(self) -> bool:
        if self.num_workers == 0:
            return False
        return bool(self.resolved_device == "cuda") if self.persistent_workers is None else self.persistent_workers

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as handle:
            values = json.load(handle)
        values["class_ids"] = tuple(values["class_ids"])
        values["run_seeds"] = tuple(values["run_seeds"])
        if "final_run_seeds" in values:
            values["final_run_seeds"] = tuple(values["final_run_seeds"])
        return cls(**values)


def set_reproducibility(seed: int) -> tuple[np.random.Generator, torch.Generator]:
    """Reset all stochastic sources used by a single experimental replicate."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed), torch.Generator(device="cpu").manual_seed(seed)


def environment_summary(config: ExperimentConfig) -> dict[str, str | bool | int]:
    """Return, rather than print, execution details for notebook display and manifests."""
    cuda_available = torch.cuda.is_available()
    summary: dict[str, str | bool | int] = {
        "device": config.resolved_device,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "torch": torch.__version__,
        "torchvision": torchvision_version(),
    }
    if cuda_available:
        summary["cuda_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return summary


def torchvision_version() -> str:
    import torchvision

    return torchvision.__version__


def task_class_split(config: ExperimentConfig) -> tuple[tuple[int, ...], ...]:
    """Create the fixed, seed-governed raw-label split used by every model family."""
    rng = np.random.default_rng(config.seed)
    classes = np.asarray(config.class_ids, dtype=int).copy()
    rng.shuffle(classes)
    task_size = len(classes) // config.num_tasks
    return tuple(tuple(int(value) for value in classes[start:start + task_size]) for start in range(0, len(classes), task_size))


class TrafficSignDataset(Dataset):
    """GTSRB dataset adapter with the CIL engine's `(image, target, path)` contract."""

    def __init__(
        self,
        data_dir: str | Path,
        image_file_paths: Sequence[str],
        targets: Sequence[int],
        transformer,
        class_mapping: dict[int, int],
    ) -> None:
        if len(targets) != len(image_file_paths):
            raise ValueError("image_file_paths and targets must have the same length.")
        self.data_dir = Path(data_dir)
        self.image_file_paths = list(image_file_paths)
        self.targets = [int(target) for target in targets]
        self.transformer = transformer
        self.class_mapping = class_mapping

    def __len__(self) -> int:
        return len(self.image_file_paths)

    def __getitem__(self, index: int):
        image_path = self.data_dir / self.image_file_paths[index]
        image = Image.open(image_path).convert("RGB")
        target = {"label": torch.tensor(self.class_mapping[self.targets[index]], dtype=torch.long)}
        return self.transformer(image), target, str(image_path)


def collate_traffic_signs(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)


@dataclass
class DataBundle:
    """Immutable dataset split plus reproducible loader factories for every replicate."""

    config: ExperimentConfig
    download_dir: Path
    source_train_dataset: TrafficSignDataset
    train_dataset: TrafficSignDataset
    validation_dataset: TrafficSignDataset
    test_dataset: TrafficSignDataset
    task_classes: tuple[tuple[int, ...], ...]
    class_mapping: dict[int, int]
    class_weights: torch.Tensor
    source_train_frame: pd.DataFrame
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame

    def _loader(self, dataset: TrafficSignDataset, indices: Sequence[int], *, shuffle: bool, generator=None) -> DataLoader:
        return DataLoader(
            torch.utils.data.Subset(dataset, list(indices)),
            batch_size=self.config.batch_size,
            collate_fn=collate_traffic_signs,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.resolved_pin_memory,
            persistent_workers=self.config.resolved_persistent_workers,
            generator=generator,
        )

    def _task_indices(self, dataset: TrafficSignDataset, task: Sequence[int]) -> list[int]:
        return [index for index, target in enumerate(dataset.targets) if target in task]

    def _task_loader(self, dataset: TrafficSignDataset, task: Sequence[int], *, shuffle: bool, generator=None) -> DataLoader:
        return self._loader(dataset, self._task_indices(dataset, task), shuffle=shuffle, generator=generator)

    def training_loaders(self, seed: int) -> tuple[list[DataLoader], list[DataLoader]]:
        """Give each model the same task ordering and shuffle sequence for a replicate."""
        train_generator = torch.Generator(device="cpu").manual_seed(seed)
        train_loaders = [self._task_loader(self.train_dataset, task, shuffle=True, generator=train_generator) for task in self.task_classes]
        validation_loaders = [self._task_loader(self.validation_dataset, task, shuffle=False) for task in self.task_classes]
        return train_loaders, validation_loaders

    def final_training_loaders(self, seed: int) -> list[DataLoader]:
        """Create final CIL task loaders using every selected source-training record."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return [
            self._task_loader(self.source_train_dataset, task, shuffle=True, generator=generator)
            for task in self.task_classes
        ]

    def cross_validation_loaders(
        self,
        repeat: int,
        fold: int,
        seed: int,
    ) -> tuple[list[DataLoader], list[DataLoader], CrossEntropyLoss]:
        """Return task-local CIL train/validation loaders for one stratified CV fold.

        Each pseudo-task is partitioned independently. For a three-fold run, fold 0
        trains task `t` on `tf2 + tf3` and validates on `tf1`, and so on.
        """
        if not 0 <= fold < self.config.cv_folds:
            raise ValueError(f"fold must be in [0, {self.config.cv_folds - 1}].")

        train_indices: list[list[int]] = []
        validation_indices: list[list[int]] = []
        for task_index, task in enumerate(self.task_classes):
            task_indices = np.asarray(self._task_indices(self.source_train_dataset, task), dtype=int)
            task_targets = np.asarray([self.source_train_dataset.targets[index] for index in task_indices], dtype=int)
            splitter = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.seed + repeat * self.config.num_tasks + task_index,
            )
            partitions = list(splitter.split(task_indices, task_targets))
            held_out_indices: set[int] = set()
            for partition_train, partition_validation in partitions:
                partition_train_set = set(task_indices[partition_train].tolist())
                partition_validation_set = set(task_indices[partition_validation].tolist())
                if partition_train_set & partition_validation_set:
                    raise RuntimeError("Task-local cross-validation produced overlapping train and validation indices.")
                held_out_indices.update(partition_validation_set)
            if held_out_indices != set(task_indices.tolist()):
                raise RuntimeError("Task-local cross-validation folds do not cover every source-training record exactly once.")
            train_partition, validation_partition = partitions[fold]
            train_indices.append(task_indices[train_partition].tolist())
            validation_indices.append(task_indices[validation_partition].tolist())

        train_targets = np.asarray(
            [self.source_train_dataset.targets[index] for indices in train_indices for index in indices],
            dtype=int,
        )
        weights = compute_class_weight(
            "balanced",
            classes=np.asarray(sorted(self.class_mapping)),
            y=train_targets,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        train_loaders = [
            self._loader(self.source_train_dataset, indices, shuffle=True, generator=generator)
            for indices in train_indices
        ]
        validation_loaders = [
            self._loader(self.source_train_dataset, indices, shuffle=False)
            for indices in validation_indices
        ]
        loss_fn = CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.config.resolved_device)
        )
        return train_loaders, validation_loaders, loss_fn

    def test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_traffic_signs,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.resolved_pin_memory,
            persistent_workers=self.config.resolved_persistent_workers,
        )

    def loss_fn(self, device: str | None = None) -> CrossEntropyLoss:
        return CrossEntropyLoss(weight=self.class_weights.to(device or self.config.resolved_device))

    def source_loss_fn(self, device: str | None = None) -> CrossEntropyLoss:
        weights = compute_class_weight(
            "balanced",
            classes=np.asarray(sorted(self.class_mapping)),
            y=np.asarray(self.source_train_dataset.targets, dtype=int),
        )
        return CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device or self.config.resolved_device)
        )


def build_gtsrb_data(config: ExperimentConfig) -> DataBundle:
    """Download, stratify, label-map, and package GTSRB exactly once for all suites."""
    download_dir = Path(kagglehub.dataset_download(config.dataset_name))
    csv_paths: dict[str, Path | None] = {name: None for name in ("train.csv", "test.csv", "meta.csv")}
    for path in download_dir.rglob("*.csv"):
        normalized_name = path.name.lower()
        if normalized_name in csv_paths:
            csv_paths[normalized_name] = path
    missing = [name for name, path in csv_paths.items() if path is None]
    if missing:
        raise FileNotFoundError(f"The downloaded dataset does not contain: {', '.join(missing)}")

    train_csv_path = csv_paths["train.csv"]
    test_csv_path = csv_paths["test.csv"]
    assert train_csv_path is not None and test_csv_path is not None
    train_csv = pd.read_csv(train_csv_path)
    selected = list(config.class_ids)
    source_train_frame = train_csv[train_csv.ClassId.isin(selected)].reset_index(drop=True)
    train_frame, validation_frame = train_test_split(
        train_csv,
        test_size=config.validation_fraction,
        shuffle=True,
        stratify=train_csv.ClassId,
        random_state=config.seed,
    )
    train_frame = train_frame[train_frame.ClassId.isin(selected)].reset_index(drop=True)
    validation_frame = validation_frame[validation_frame.ClassId.isin(selected)].reset_index(drop=True)
    test_frame = pd.read_csv(test_csv_path)
    test_frame = test_frame[test_frame.ClassId.isin(selected)].reset_index(drop=True)
    if set(train_frame.ClassId.unique()) != set(selected):
        raise ValueError("Every selected class must be represented in the training split.")

    class_mapping = {raw_id: index for index, raw_id in enumerate(sorted(selected))}
    weights = compute_class_weight("balanced", classes=np.asarray(sorted(selected)), y=train_frame.ClassId.to_numpy())
    transformer = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    build_dataset = lambda frame: TrafficSignDataset(download_dir, frame.Path.tolist(), frame.ClassId.tolist(), transformer, class_mapping)
    return DataBundle(
        config=config,
        download_dir=download_dir,
        source_train_dataset=build_dataset(source_train_frame),
        train_dataset=build_dataset(train_frame),
        validation_dataset=build_dataset(validation_frame),
        test_dataset=build_dataset(test_frame),
        task_classes=task_class_split(config),
        class_mapping=class_mapping,
        class_weights=torch.tensor(weights, dtype=torch.float32),
        source_train_frame=source_train_frame,
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
    )


def load_sign_names(destination: str | Path = "signnames.csv") -> pd.DataFrame:
    """Download sign metadata on demand for notebook-only visualisation."""
    destination = Path(destination)
    if not destination.exists():
        urlretrieve(SIGNNAMES_URL, destination)
    return pd.read_csv(destination)


class MobileNetGeneralist(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_large(weights=weights)
        self.backbone = base_model.features
        self.pool = base_model.avgpool
        self.out_channels = cast(nn.Linear, base_model.classifier[0]).in_features
        self.classifier = nn.Sequential(nn.Linear(self.out_channels, num_classes), nn.LayerNorm(num_classes))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.pool(self.backbone(x)).flatten(1)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


class ConvNeXtTinyGeneralist(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.1) -> None:
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        base_model = models.convnext_tiny(weights=weights)
        self.backbone = base_model.features
        self.pool = base_model.avgpool
        self.out_channels = cast(nn.Linear, base_model.classifier[2]).in_features
        self.classifier = nn.Sequential(nn.LayerNorm(self.out_channels), nn.Dropout(dropout), nn.Linear(self.out_channels, num_classes))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.pool(self.backbone(x)).flatten(1)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


class RegressionRouter(nn.Module):
    def __init__(self, input_size: int, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.gating = nn.Linear(input_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gating(x)
        return logits + torch.randn_like(logits) / self.num_experts if self.training else logits


class MlpRouter(nn.Module):
    def __init__(self, input_size: int, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ff1 = nn.Linear(input_size, input_size * 2)
        self.ff2 = nn.Linear(input_size * 2, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.ff2(self.ff1(x))
        return logits + torch.randn_like(logits) / self.num_experts if self.training else logits


class MlpExpert(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerExpert(nn.Module):
    def __init__(self, in_features: int, d_model: int, nhead: int, output_size: int) -> None:
        super().__init__()
        if in_features % d_model != 0:
            raise ValueError("d_model must perfectly divide in_features.")
        self.num_patches = in_features // d_model
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True), num_layers=1)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        tokens = x.view(batch_size, self.num_patches, self.d_model)
        tokens = torch.cat((self.cls_token.expand(batch_size, -1, -1), tokens), dim=1) + self.pos_embed
        return self.classifier(self.transformer(tokens)[:, 0, :])


class ResidualGatedMlpExpert(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(nn.LayerNorm(input_size), nn.Linear(input_size, hidden_size), nn.GELU(), nn.Dropout(dropout))
        self.block1 = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size * 2), nn.GLU(dim=1), nn.Dropout(dropout))
        self.block2 = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size * 2), nn.GLU(dim=1), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x)
        hidden = hidden + self.block1(hidden)
        hidden = hidden + self.block2(hidden)
        return self.classifier(hidden)


class JointEmbeddingRouterExperts(GeneralistRouterExperts):
    """GRE variant that routes a selected JE latent space and retains it for SupCon."""

    def __init__(self, *args, feature_key: Literal["embeddings", "projections"], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feature_key = feature_key
        self._last_joint_embedding_outputs: dict[str, torch.Tensor] | None = None

    def reset_routing_state(self) -> None:
        super().reset_routing_state()
        self._last_joint_embedding_outputs = None

    def get_joint_embedding_loss(self, labels: torch.Tensor) -> torch.Tensor:
        if self._last_joint_embedding_outputs is None:
            parameter = next(self.parameters())
            return torch.zeros((), device=parameter.device, dtype=parameter.dtype)
        generalist = cast(JointEmbeddingModule, self.generalist)
        return generalist.compute_contrastive_loss(
            labels=labels,
            **{self.feature_key: self._last_joint_embedding_outputs[self.feature_key]},
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        generalist = cast(JointEmbeddingModule, self.generalist)
        encoded = generalist.backbone.encode(x)
        generalist_logits = encoded["logits"]
        features = encoded[self.feature_key]
        if features.ndim != 2:
            features = torch.flatten(features, 1)
        router_probs = F.softmax(self.router(features), dim=1)
        topk_probs, topk_indices = torch.topk(router_probs, k=self.k, dim=1)
        topk_weights = topk_probs / topk_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        expert_logits = torch.zeros_like(generalist_logits, device=features.device, dtype=features.dtype)
        for expert_index, expert in enumerate(self.experts):
            selected_rows, selected_slots = torch.where(topk_indices == expert_index)
            if selected_rows.numel():
                weights = topk_weights[selected_rows, selected_slots].unsqueeze(1)
                expert_logits[selected_rows] += expert(features[selected_rows]) * weights
        self._last_router_probs = router_probs.detach()
        self._last_topk_indices = topk_indices.detach()
        self._last_aux_loss = self._compute_auxiliary_loss(router_probs, topk_indices)
        self._last_joint_embedding_outputs = {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}
        return (generalist_logits + expert_logits) / self.temperature


class JointEmbeddingCILComputerVisionModel(CILComputerVisionModel):
    """Adds the JE-only contrastive objective without changing baseline loss behavior."""

    def __init__(self, *args, lambda_contrastive: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_contrastive = lambda_contrastive

    def _joint_embedding_loss(self, target: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "get_joint_embedding_loss"):
            return cast(JointEmbeddingRouterExperts, self.model).get_joint_embedding_loss(target)
        parameter = next(self.model.parameters())
        return torch.zeros((), device=self.device, dtype=parameter.dtype)

    def loss(self, loss_fn, images: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        new_predictions = self.model(images)
        prediction_loss = loss_fn(self._mask_logits(new_predictions), target)
        auxiliary_loss = self._get_model_aux_loss()
        contrastive_loss = self._joint_embedding_loss(target)
        if self.prev_model is None:
            return prediction_loss + auxiliary_loss + self.lambda_contrastive * contrastive_loss
        with torch.no_grad():
            old_predictions = self.prev_model(images)
        indices = self.prev_seen_classes_tensor
        distillation_loss = KLDivLoss(reduction="batchmean")(
            F.log_softmax(new_predictions[:, indices] / self.kd_temperature, dim=1),
            F.softmax(old_predictions[:, indices] / self.kd_temperature, dim=1),
        ) * self.kd_temperature**2
        return (1 - self.lambda_kd) * prediction_loss + self.lambda_kd * distillation_loss + auxiliary_loss + self.lambda_contrastive * contrastive_loss


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: Literal["Baseline", "Joint Embedding"]
    backbone: str
    router: str
    expert: str
    feature_space: str
    uses_joint_embedding: bool


def _expert_factory(expert_name: str, feature_dim: int, config: ExperimentConfig):
    def factory() -> nn.Module:
        if expert_name == "MLP Experts":
            return MlpExpert(feature_dim, config.hidden_expert_size, config.total_classes, config.dropout)
        if expert_name == "Transformer Experts":
            return TransformerExpert(feature_dim, config.transformer_d_model, config.transformer_nhead, config.total_classes)
        if expert_name == "ResMLP Experts":
            return ResidualGatedMlpExpert(feature_dim, config.hidden_expert_size, config.total_classes, config.dropout)
        raise KeyError(f"Unknown expert family: {expert_name}")

    return factory


def _router(router_name: str, feature_dim: int, config: ExperimentConfig) -> nn.Module:
    router_type = {"Regression Router": RegressionRouter, "MLP Router": MlpRouter}.get(router_name)
    if router_type is None:
        raise KeyError(f"Unknown router family: {router_name}")
    return router_type(feature_dim, config.num_experts)


def build_model(spec: ModelSpec, config: ExperimentConfig) -> GeneralistRouterExperts:
    """Instantiate a model only from a registered condition, never notebook-local code."""
    if spec.uses_joint_embedding:
        if spec.backbone == "JE MobileNet Large":
            backbone_name: Literal["mobilenet_v3_large", "convnext_tiny"] = "mobilenet_v3_large"
        elif spec.backbone == "JE ConvNeXt Tiny":
            backbone_name = "convnext_tiny"
        else:
            raise KeyError(f"Unsupported JE backbone: {spec.backbone}")
        generalist = JointEmbeddingModule(
            num_classes=config.total_classes,
            backbone=backbone_name,
            embedding_dim=config.je_embedding_dim,
            projection_dim=config.je_projection_dim,
            pretrained=config.pretrained_backbones,
            temperature=config.je_temperature,
        )
        feature_dim = config.je_embedding_dim if spec.feature_space == "embeddings" else config.je_projection_dim
        return JointEmbeddingRouterExperts(
            generalist=generalist,
            router=_router(spec.router, feature_dim, config),
            expert_factory=_expert_factory(spec.expert, feature_dim, config),
            num_experts=config.num_experts,
            num_classes=config.total_classes,
            k=config.top_k,
            lambda_aux=config.lambda_aux,
            feature_key=cast(Literal["embeddings", "projections"], spec.feature_space),
        )

    if spec.backbone == "MobileNet Large":
        generalist = MobileNetGeneralist(config.total_classes, config.pretrained_backbones)
    elif spec.backbone == "ConvNeXt Tiny":
        generalist = ConvNeXtTinyGeneralist(config.total_classes, config.pretrained_backbones, config.dropout)
    else:
        raise KeyError(f"Unsupported baseline backbone: {spec.backbone}")
    feature_dim = generalist.out_channels
    return GeneralistRouterExperts(
        generalist=generalist,
        router=_router(spec.router, feature_dim, config),
        expert_factory=_expert_factory(spec.expert, feature_dim, config),
        num_experts=config.num_experts,
        num_classes=config.total_classes,
        k=config.top_k,
        lambda_aux=config.lambda_aux,
    )


def build_model_specs(config: ExperimentConfig, family: Literal["baseline", "joint_embedding", "all"] = "all") -> dict[str, ModelSpec]:
    """Build the complete factorial grid, with identical router/expert conditions."""
    specs: dict[str, ModelSpec] = {}
    routers = ("Regression Router", "MLP Router")
    experts = ("MLP Experts", "Transformer Experts", "ResMLP Experts")
    if family in {"baseline", "all"}:
        for backbone, router, expert in product(("MobileNet Large", "ConvNeXt Tiny"), routers, experts):
            name = f"{backbone} + {router} + {expert}"
            specs[name] = ModelSpec(name, "Baseline", backbone, router, expert, "backbone_features", False)
    if family in {"joint_embedding", "all"}:
        for backbone, router, expert in product(("JE MobileNet Large", "JE ConvNeXt Tiny"), routers, experts):
            name = f"{backbone} + {router} + {expert}"
            specs[name] = ModelSpec(name, "Joint Embedding", backbone, router, expert, config.je_feature_key, True)
    return specs


def strip_checkpoint_suffix(value: str) -> str:
    return re.sub(r"(?:__seed-\d+|_\d{4}-\d{2}-\d{2})$", "", value)


def build_model_from_name(model_name: str, config: ExperimentConfig) -> GeneralistRouterExperts:
    parts = strip_checkpoint_suffix(Path(model_name).stem).split(" + ")
    if len(parts) != 3:
        raise ValueError("Model names must have the form 'Generalist + Router + Expert'.")
    generalist, router, expert = parts
    all_specs = build_model_specs(config)
    candidate = f"{generalist} + {router} + {expert}"
    if candidate not in all_specs:
        raise KeyError(f"No registered model condition matches: {model_name}")
    return build_model(all_specs[candidate], config)


def build_cil_model(spec: ModelSpec, config: ExperimentConfig, rng: np.random.Generator) -> CILComputerVisionModel:
    kwargs = {
        "model": build_model(spec, config),
        "exemplar_ratio": config.exemplar_ratio,
        "rng": rng,
        "device": config.resolved_device,
        "use_class_masking": config.use_class_masking,
        "kd_temperature": config.kd_temperature,
        "lambda_kd": config.lambda_kd,
    }
    if spec.uses_joint_embedding:
        return JointEmbeddingCILComputerVisionModel(**kwargs, lambda_contrastive=config.je_contrastive_weight)
    return CILComputerVisionModel(**kwargs)


def collect_gre_metrics(cil_model: CILComputerVisionModel, eval_dataloader: DataLoader, config: ExperimentConfig, prefix: str) -> dict:
    model = cil_model.model
    if not isinstance(model, GeneralistRouterExperts):
        return {}
    entropy = router_entropy(model, eval_dataloader, device=config.resolved_device)
    expert_summary = expert_metrics(model, eval_dataloader, device=config.resolved_device)
    ece = expected_calibration_error(model, config.total_classes, eval_dataloader, device=config.resolved_device)
    expected_calls = expert_summary["Expected Expert Calls"]
    return {
        f"{prefix} Router Entropy": entropy["Mean Router Entropy"],
        f"{prefix} Normalized Router Entropy": entropy["Normalized Router Entropy"],
        f"{prefix} Router Entropy Std": entropy["Std Router Entropy"],
        f"{prefix} Avg Router Prob": entropy["Avg Router Prob"],
        f"{prefix} ECE": ece,
        f"{prefix} Expected Expert Calls": expected_calls,
        f"{prefix} Expert Utilization Ratio": expert_summary["Expert Selection Ratio"],
        f"{prefix} Cost Proxy": cost_proxy(expected_calls, model.get_model_parameters_by_component()),
    }


def _run_directory(results_dir: str | Path, run_name: str | None, resume: bool) -> Path:
    name = run_name or f"experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    directory = Path(results_dir) / name
    if directory.exists() and not resume:
        raise FileExistsError(f"Experiment run already exists: {directory}. Set resume=True to continue it.")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_compatible(payload), handle, indent=2)
    temporary_path.replace(path)


@dataclass
class ExperimentRun:
    comparison_df: pd.DataFrame
    loss_histories: dict[str, list[list[float]]]
    checkpoint_paths: dict[str, Path]
    run_dir: Path
    stage: str


class ExperimentRunner:
    """Runs resumable CIL cross-validation and final-training work units."""

    def __init__(
        self,
        config: ExperimentConfig,
        data: DataBundle,
        results_dir: str | Path,
        *,
        run_name: str | None = None,
        resume: bool = False,
    ) -> None:
        self.config = config
        self.data = data
        self.run_dir = _run_directory(results_dir, run_name, resume)
        self.models_dir = self.run_dir / "models"
        self.results_dir = self.run_dir / "results"
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.progress_path = self.run_dir / "progress.json"
        existing_config = self.run_dir / "experiment_config.json"
        if existing_config.exists():
            if ExperimentConfig.load(existing_config).to_dict() != config.to_dict():
                raise ValueError("The supplied configuration does not match the existing resumable run.")
        else:
            config.save(existing_config)
        if not self.progress_path.exists():
            _atomic_json_dump({"units": {}}, self.progress_path)
        manifest_path = self.run_dir / "manifest.json"
        if not manifest_path.exists():
            _atomic_json_dump(
                {
                    "environment": environment_summary(config),
                    "task_classes": data.task_classes,
                    "class_mapping": data.class_mapping,
                    "protocol": {
                        "cv": "task-local stratified CIL folds",
                        "final": "full selected train.csv CIL replicas",
                        "test_partition_used_for_selection": False,
                    },
                },
                manifest_path,
            )

    def _progress(self) -> dict[str, Any]:
        with self.progress_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _set_unit_status(self, unit_id: str, **status: Any) -> None:
        progress = self._progress()
        progress.setdefault("units", {})[unit_id] = status
        _atomic_json_dump(progress, self.progress_path)

    def _record_specs(self, specs: Sequence[ModelSpec]) -> None:
        manifest_path = self.run_dir / "manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        registered_specs = {spec["name"]: spec for spec in manifest.get("registered_specs", [])}
        registered_specs.update({spec.name: asdict(spec) for spec in specs})
        manifest["registered_specs"] = list(registered_specs.values())
        _atomic_json_dump(manifest, manifest_path)

    def _unit_is_complete(self, unit_id: str, result_path: Path, checkpoint_path: Path | None = None) -> bool:
        unit = self._progress().get("units", {}).get(unit_id, {})
        return unit.get("status") == "complete" and result_path.is_file() and (
            checkpoint_path is None or checkpoint_path.is_file()
        )

    def _comparison(self, stage: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result_path in sorted(self.results_dir.glob(f"{stage}__*.json")):
            with result_path.open(encoding="utf-8") as handle:
                rows.append(json.load(handle))
        comparison_df = pd.DataFrame(rows)
        comparison_df.to_csv(self.run_dir / f"{stage}_comparison.csv", index=False)
        return comparison_df

    def _run_cv_unit(self, spec: ModelSpec, repeat: int, fold: int, verbose: bool) -> tuple[dict[str, Any], list[list[float]]]:
        unit_seed = self.config.seed + repeat * self.config.cv_folds + fold
        np_rng, _ = set_reproducibility(unit_seed)
        train_loaders, validation_loaders, loss_fn = self.data.cross_validation_loaders(repeat, fold, unit_seed)
        cil_model = build_cil_model(spec, self.config, np_rng)
        results, losses = train_and_evaluate_cil_model(
            cil_model=cil_model,
            optimizer_cls=AdamW,
            loss_fn=loss_fn,
            train_dataloaders=train_loaders,
            val_dataloaders=validation_loaders,
            epochs=self.config.cv_epochs,
            num_tasks=self.config.num_tasks,
            base_lr=self.config.task_1_lr,
            later_task_lr_factor=self.config.later_task_lr_factor,
            device=self.config.resolved_device,
            verbose=verbose,
            throughput_warmup_batches=self.config.throughput_warmup_batches,
        )
        results.update(collect_gre_metrics(cil_model, build_complete_dataloader(validation_loaders), self.config, "Validation"))
        row = {
            "Stage": "cv",
            "Model": spec.name,
            "Family": spec.family,
            "Backbone": spec.backbone,
            "Router": spec.router,
            "Expert": spec.expert,
            "Feature Space": spec.feature_space,
            "CV Repeat": repeat + 1,
            "CV Fold": fold + 1,
            "Run Seed": unit_seed,
            **results,
        }
        del cil_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return row, losses

    def run_cross_validation_specs(self, specs: Sequence[ModelSpec], verbose: bool = False) -> ExperimentRun:
        self._record_specs(specs)
        loss_histories: dict[str, list[list[float]]] = {}
        for repeat in range(self.config.cv_repeats):
            for fold in range(self.config.cv_folds):
                for spec in specs:
                    unit_id = f"cv::{spec.name}::repeat-{repeat + 1}::fold-{fold + 1}"
                    result_path = self.results_dir / f"cv__{spec.name}__repeat-{repeat + 1}__fold-{fold + 1}.json"
                    if self._unit_is_complete(unit_id, result_path):
                        continue
                    self._set_unit_status(unit_id, status="running")
                    try:
                        print(f"\n=== CV {spec.name} (repeat={repeat + 1}, fold={fold + 1}) ===")
                        row, losses = self._run_cv_unit(spec, repeat, fold, verbose)
                        _atomic_json_dump(row, result_path)
                        self._set_unit_status(unit_id, status="complete", result=str(result_path.relative_to(self.run_dir)))
                        self._comparison("cv")
                        loss_histories[unit_id] = losses
                    except BaseException as exc:
                        self._set_unit_status(unit_id, status="failed", error=repr(exc))
                        raise
        return ExperimentRun(self._comparison("cv"), loss_histories, {}, self.run_dir, "cv")

    def _run_final_unit(self, spec: ModelSpec, final_seed: int, verbose: bool) -> tuple[dict[str, Any], list[list[float]], Path]:
        np_rng, _ = set_reproducibility(final_seed)
        cil_model = build_cil_model(spec, self.config, np_rng)
        train_loaders = self.data.final_training_loaders(final_seed)
        results, losses = train_final_cil_model(
            cil_model,
            AdamW,
            self.data.source_loss_fn(),
            train_loaders,
            epochs=self.config.epochs,
            base_lr=self.config.task_1_lr,
            later_task_lr_factor=self.config.later_task_lr_factor,
            verbose=verbose,
            throughput_warmup_batches=self.config.throughput_warmup_batches,
        )
        results.update(collect_gre_metrics(cil_model, build_complete_dataloader(train_loaders), self.config, "Training"))
        checkpoint_path = self.models_dir / f"{spec.name}__seed-{final_seed}.pth"
        cil_model.save(str(checkpoint_path))
        row = {
            "Stage": "final",
            "Model": spec.name,
            "Family": spec.family,
            "Backbone": spec.backbone,
            "Router": spec.router,
            "Expert": spec.expert,
            "Feature Space": spec.feature_space,
            "Final Seed": final_seed,
            **results,
        }
        del cil_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return row, losses, checkpoint_path

    def run_final_specs(
        self,
        specs: Sequence[ModelSpec],
        seeds: Sequence[int] | None = None,
        verbose: bool = False,
    ) -> ExperimentRun:
        self._record_specs(specs)
        loss_histories: dict[str, list[list[float]]] = {}
        checkpoint_paths: dict[str, Path] = {}
        for final_seed in tuple(seeds or self.config.final_run_seeds):
            for spec in specs:
                unit_id = f"final::{spec.name}::seed-{final_seed}"
                result_path = self.results_dir / f"final__{spec.name}__seed-{final_seed}.json"
                checkpoint_path = self.models_dir / f"{spec.name}__seed-{final_seed}.pth"
                if self._unit_is_complete(unit_id, result_path, checkpoint_path):
                    checkpoint_paths[unit_id] = checkpoint_path
                    continue
                self._set_unit_status(unit_id, status="running")
                try:
                    print(f"\n=== Final {spec.name} (seed={final_seed}) ===")
                    row, losses, checkpoint_path = self._run_final_unit(spec, final_seed, verbose)
                    _atomic_json_dump(row, result_path)
                    self._set_unit_status(unit_id, status="complete", result=str(result_path.relative_to(self.run_dir)), checkpoint=str(checkpoint_path.relative_to(self.run_dir)))
                    self._comparison("final")
                    loss_histories[unit_id] = losses
                    checkpoint_paths[unit_id] = checkpoint_path
                except BaseException as exc:
                    self._set_unit_status(unit_id, status="failed", error=repr(exc))
                    raise
        return ExperimentRun(self._comparison("final"), loss_histories, checkpoint_paths, self.run_dir, "final")

    def run_baseline_cross_validation(self, verbose: bool = False) -> ExperimentRun:
        return self.run_cross_validation_specs(list(build_model_specs(self.config, "baseline").values()), verbose)

    def run_joint_embedding_cross_validation(self, verbose: bool = False) -> ExperimentRun:
        return self.run_cross_validation_specs(list(build_model_specs(self.config, "joint_embedding").values()), verbose)

    def run_baseline_final(self, seeds: Sequence[int] | None = None, verbose: bool = False) -> ExperimentRun:
        return self.run_final_specs(list(build_model_specs(self.config, "baseline").values()), seeds, verbose)

    def run_joint_embedding_final(self, seeds: Sequence[int] | None = None, verbose: bool = False) -> ExperimentRun:
        return self.run_final_specs(list(build_model_specs(self.config, "joint_embedding").values()), seeds, verbose)


def initialize_experiment(config: ExperimentConfig | None = None, results_dir: str | Path = "results") -> ExperimentRunner:
    """Create a runner after building the shared data split once."""
    config = config or ExperimentConfig()
    set_reproducibility(config.seed)
    return ExperimentRunner(config, build_gtsrb_data(config), results_dir)


def add_model_component_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Model" not in df:
        raise KeyError("Expected a 'Model' column.")
    parts = df["Model"].astype(str).str.split(r"\s+\+\s+", expand=True)
    if parts.shape[1] < 3:
        raise ValueError("Expected model names in the form 'Generalist + Router + Expert'.")
    result = df.copy()
    result[["Generalist", "Router", "Expert"]] = parts.iloc[:, :3]
    return result


def normalize_backbone_label(generalist_name: str) -> str | float:
    name = str(generalist_name).strip().lower()
    if name.startswith("je mobilenet"):
        return "JE MobileNet Large"
    if name.startswith("je convnext"):
        return "JE ConvNeXt Tiny"
    if name.startswith("mobilenet"):
        return "MobileNet Large"
    if name.startswith("convnext"):
        return "ConvNeXt Tiny"
    return np.nan


def summarize_component_performance(df: pd.DataFrame, component: str, metrics: Sequence[str]) -> pd.DataFrame:
    metrics = [metric for metric in metrics if metric in df]
    if not metrics:
        return pd.DataFrame()
    result = df.groupby(component, observed=True)[metrics].mean().reset_index()
    return result.sort_values(metrics[0], ascending=False).reset_index(drop=True)


def build_component_delta_table(df: pd.DataFrame, component: str, baseline: str, comparisons: Sequence[str], fixed_columns: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    metrics = [metric for metric in metrics if metric in df]
    rows = []
    for comparison in comparisons:
        subset = df[df[component].isin((baseline, comparison))]
        identity_columns = [*fixed_columns, component]
        if subset.duplicated(identity_columns).any():
            raise ValueError(f"Duplicate paired identities for {component}: include fold, repeat, or seed in fixed_columns.")
        pivot = subset.pivot(index=list(fixed_columns), columns=component, values=metrics)
        for metric in metrics:
            baseline_key, comparison_key = (metric, baseline), (metric, comparison)
            if baseline_key not in pivot or comparison_key not in pivot:
                continue
            paired = pivot[[baseline_key, comparison_key]].dropna()
            result = paired.index.to_frame(index=False)
            result[component] = comparison
            result[f"{metric} [{baseline}]"] = paired[baseline_key].to_numpy()
            result[f"{metric} [{comparison}]"] = paired[comparison_key].to_numpy()
            result[f"{metric} Delta ({comparison} - {baseline})"] = result[f"{metric} [{comparison}]"] - result[f"{metric} [{baseline}]"]
            rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_paired_component_tests(df: pd.DataFrame, component: str, baseline: str, comparisons: Sequence[str], fixed_columns: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    metrics = [metric for metric in metrics if metric in df]
    rows: list[dict] = []
    for comparison in comparisons:
        subset = df[df[component].isin((baseline, comparison))]
        identity_columns = [*fixed_columns, component]
        if subset.duplicated(identity_columns).any():
            raise ValueError(f"Duplicate paired identities for {component}: include fold, repeat, or seed in fixed_columns.")
        pivot = subset.pivot(index=list(fixed_columns), columns=component, values=metrics)
        for metric in metrics:
            baseline_key, comparison_key = (metric, baseline), (metric, comparison)
            if baseline_key not in pivot or comparison_key not in pivot:
                continue
            paired = pivot[[baseline_key, comparison_key]].dropna()
            deltas = paired[comparison_key].to_numpy(dtype=float) - paired[baseline_key].to_numpy(dtype=float)
            nonzero = deltas[~np.isclose(deltas, 0.0)]
            if len(nonzero) == 0:
                p_value, test = 1.0, "All deltas are zero"
            elif len(nonzero) < 2:
                p_value, test = np.nan, "Insufficient non-zero pairs"
            else:
                try:
                    _, p_value = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
                    test = "Wilcoxon signed-rank"
                except ValueError:
                    p_value, test = np.nan, "Wilcoxon unavailable"
            rows.append({component: comparison, "Metric": metric, "Paired Configurations": len(deltas), f"Mean {baseline}": paired[baseline_key].mean(), f"Mean {comparison}": paired[comparison_key].mean(), "Mean Delta": deltas.mean(), "Median Delta": np.median(deltas), "Std Delta": deltas.std(ddof=0), "Raw P Value": p_value, "Test": test})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["FDR BH P Value"] = np.nan
    valid = result["Raw P Value"].notna()
    if valid.any():
        _, corrected, _, _ = multipletests(result.loc[valid, "Raw P Value"], method="fdr_bh")
        result.loc[valid, "FDR BH P Value"] = corrected
    result["Significant @ 0.05"] = result["FDR BH P Value"] < 0.05
    return result.sort_values([component, "Metric"]).reset_index(drop=True)


def evaluate_checkpoint(checkpoint_path: str | Path, data: DataBundle, output_dir: str | Path | None = None) -> tuple[dict, CILComputerVisionModel]:
    """Reconstruct and evaluate either new or archived checkpoints on the shared test set."""
    checkpoint_path = Path(checkpoint_path)
    model_name = strip_checkpoint_suffix(checkpoint_path.stem)
    seed_match = re.search(r"__seed-(\d+)$", checkpoint_path.stem)
    model = build_model_from_name(model_name, data.config)
    wrapper = CILComputerVisionModel.load(model, filename=str(checkpoint_path), map_location=data.config.resolved_device, device=data.config.resolved_device)
    test_loss, test_metrics = wrapper.evaluate(data.test_loader(), data.loss_fn(), apply_class_masking=data.config.use_class_masking)
    spec = build_model_specs(data.config)[model_name]
    result = {
        "Model": model_name,
        "Source Checkpoint": str(checkpoint_path),
        "Family": spec.family,
        "Backbone": spec.backbone,
        "Router": spec.router,
        "Expert": spec.expert,
        "Test Loss": test_loss,
        "Test Macro F1": test_metrics["Macro F1"],
        "Test Micro F1": test_metrics["Micro F1"],
        "Test Weighted F1": test_metrics["Weighted F1"],
        "Num Parameters": sum(parameter.numel() for parameter in wrapper.model.parameters()),
    }
    if seed_match:
        result["Final Seed"] = int(seed_match.group(1))
    result.update(collect_gre_metrics(wrapper, data.test_loader(), data.config, "Test"))
    if output_dir is not None:
        output_path = Path(output_dir) / f"test_{checkpoint_path.stem}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialize_save_json(result, str(output_path))
    return result, wrapper


def evaluate_checkpoints(checkpoint_paths: Sequence[str | Path], data: DataBundle, output_dir: str | Path) -> tuple[pd.DataFrame, dict[str, nn.Module]]:
    rows: list[dict] = []
    loaded_models: dict[str, nn.Module] = {}
    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating {Path(checkpoint_path).name}...")
        result, wrapper = evaluate_checkpoint(checkpoint_path, data, output_dir)
        rows.append(result)
        loaded_models[Path(checkpoint_path).stem] = wrapper.model
    return pd.DataFrame(rows), loaded_models


__all__ = [
    "ExperimentConfig", "ExperimentRunner", "ExperimentRun", "DataBundle", "TrafficSignDataset", "RuntimePaths",
    "initialize_experiment", "build_gtsrb_data", "build_model_specs", "build_model", "build_model_from_name", "build_cil_model",
    "set_reproducibility", "environment_summary", "task_class_split", "load_sign_names", "evaluate_checkpoints",
    "prepare_notebook_runtime", "default_experiment_config",
    "add_model_component_columns", "normalize_backbone_label", "summarize_component_performance",
    "build_component_delta_table", "run_paired_component_tests", "bootstrap_performance_diff",
]

