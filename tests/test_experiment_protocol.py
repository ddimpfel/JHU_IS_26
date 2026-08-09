import sys
import unittest
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

CURRENT_WORK = Path(__file__).resolve().parents[1]
if str(CURRENT_WORK) not in sys.path:
    sys.path.insert(0, str(CURRENT_WORK))

from continual_learning import TrainingThroughput
from init_experiment import DataBundle, ExperimentConfig, TrafficSignDataset, build_component_delta_table


class ToyDataset(Dataset):
    def __init__(self, targets: list[int]) -> None:
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return torch.zeros(3, 4, 4), {"label": torch.tensor(0)}, str(index)


class CrossValidationProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            class_ids=(1, 2, 3, 4, 5, 6),
            num_tasks=3,
            cv_folds=3,
            batch_size=4,
            pretrained_backbones=False,
        )
        targets = [class_id for class_id in self.config.class_ids for _ in range(6)]
        dataset = cast(TrafficSignDataset, ToyDataset(targets))
        frame = pd.DataFrame({"ClassId": targets})
        self.bundle = DataBundle(
            config=self.config,
            download_dir=Path("."),
            source_train_dataset=dataset,
            train_dataset=dataset,
            validation_dataset=dataset,
            test_dataset=dataset,
            task_classes=((1, 2), (3, 4), (5, 6)),
            class_mapping={class_id: index for index, class_id in enumerate(self.config.class_ids)},
            class_weights=torch.ones(len(self.config.class_ids)),
            source_train_frame=frame,
            train_frame=frame,
            validation_frame=frame,
            test_frame=frame,
        )

    def test_task_local_cv_folds_are_disjoint_and_cover_each_task(self) -> None:
        validation_coverage = [set() for _ in self.bundle.task_classes]
        for fold in range(self.config.cv_folds):
            train_loaders, validation_loaders, _ = self.bundle.cross_validation_loaders(0, fold, 100 + fold)
            for task_index, (train_loader, validation_loader) in enumerate(zip(train_loaders, validation_loaders)):
                train_indices = set(cast(Subset, train_loader.dataset).indices)
                validation_indices = set(cast(Subset, validation_loader.dataset).indices)
                expected_indices = set(self.bundle._task_indices(self.bundle.source_train_dataset, self.bundle.task_classes[task_index]))
                self.assertFalse(train_indices & validation_indices)
                self.assertEqual(train_indices | validation_indices, expected_indices)
                validation_coverage[task_index].update(validation_indices)
        for task_index, covered_indices in enumerate(validation_coverage):
            self.assertEqual(
                covered_indices,
                set(self.bundle._task_indices(self.bundle.source_train_dataset, self.bundle.task_classes[task_index])),
            )

    def test_final_loaders_use_each_selected_source_record_once(self) -> None:
        loaders = self.bundle.final_training_loaders(seed=7)
        indices = [index for loader in loaders for index in cast(Subset, loader.dataset).indices]
        self.assertEqual(sorted(indices), list(range(len(self.bundle.source_train_dataset))))


class ThroughputProtocolTests(unittest.TestCase):
    def test_warmup_batches_are_not_counted_as_timed_work(self) -> None:
        measurement = TrainingThroughput()
        measurement.record(batch_size=4, elapsed_seconds=None)
        measurement.record(batch_size=4, elapsed_seconds=None)
        measurement.record(batch_size=4, elapsed_seconds=None)
        measurement.record(batch_size=4, elapsed_seconds=0.5)
        measurement.record(batch_size=2, elapsed_seconds=0.5)

        result = measurement.to_dict()
        self.assertEqual(result["Training Samples"], 18)
        self.assertEqual(result["Timed Training Samples"], 6)
        self.assertEqual(result["Training Optimizer Steps"], 5)
        self.assertEqual(result["Timed Optimizer Steps"], 2)
        self.assertEqual(result["Training Throughput (samples/s)"], 6.0)


class PairingProtocolTests(unittest.TestCase):
    def test_duplicate_pair_identity_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "Backbone": ["A", "A", "B"],
                "Router": ["R", "R", "R"],
                "Fold": [1, 1, 1],
                "Metric": [0.1, 0.2, 0.3],
            }
        )
        with self.assertRaises(ValueError):
            build_component_delta_table(frame, "Backbone", "A", ["B"], ["Router", "Fold"], ["Metric"])


if __name__ == "__main__":
    unittest.main()
