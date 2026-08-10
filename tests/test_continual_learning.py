import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

CURRENT_WORK = Path(__file__).resolve().parents[1]
if str(CURRENT_WORK) not in sys.path:
    sys.path.insert(0, str(CURRENT_WORK))

from continual_learning import (
    CILComputerVisionModel,
    InferenceThroughput,
    TrainingThroughput,
    average_accuracy,
    average_forgetting,
    backward_transfer,
    backward_transfer_per_task,
    bootstrap_performance_diff,
    forward_transfer,
    forward_transfer_per_task,
    measure_inference_throughput,
    serialize_save_json,
    serialize_save_model_training,
    summarize_continual_metric_results,
    task_forgetting,
)


class DummyDataset(Dataset):
    def __init__(self, targets: list[int]) -> None:
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return torch.zeros(3, 4, 4), {"label": torch.tensor(self.targets[index] % 6)}, str(index)


class DummyModel(torch.nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.flatten(1)
        return self.linear(x)


class ContinualLearningMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        # 3 tasks evaluation matrix across 3 training stages
        self.eval_matrix = np.array([
            [0.90, 0.85, 0.80],  # Task 0 performance
            [0.00, 0.88, 0.82],  # Task 1 performance
            [0.00, 0.00, 0.86],  # Task 2 performance
        ])
        self.forward_eval_matrix = np.array([
            [np.nan, np.nan, np.nan],
            [0.30, np.nan, np.nan],  # Task 1 score before training task 1
            [np.nan, 0.40, np.nan],  # Task 2 score before training task 2
        ])
        self.baseline_scores = np.array([0.10, 0.10, 0.10])

    def test_average_accuracy(self) -> None:
        expected = np.mean([0.80, 0.82, 0.86])
        self.assertAlmostEqual(average_accuracy(self.eval_matrix), expected)

    def test_task_forgetting_and_average(self) -> None:
        forgetting = task_forgetting(self.eval_matrix)
        # Task 0 max before final stage: max(0.90, 0.85)=0.90, final=0.80 -> 0.10
        # Task 1 max before final stage: max(0.88)=0.88, final=0.82 -> 0.06
        np.testing.assert_allclose(forgetting, [0.10, 0.06], atol=1e-5)
        self.assertAlmostEqual(average_forgetting(forgetting), 0.08)

    def test_backward_transfer(self) -> None:
        bwt_per_task = backward_transfer_per_task(self.eval_matrix)
        # Task 0: 0.80 - 0.90 = -0.10
        # Task 1: 0.82 - 0.88 = -0.06
        np.testing.assert_allclose(bwt_per_task, [-0.10, -0.06], atol=1e-5)
        self.assertAlmostEqual(backward_transfer(self.eval_matrix), -0.08)

    def test_forward_transfer(self) -> None:
        fwt_per_task = forward_transfer_per_task(self.forward_eval_matrix, self.baseline_scores)
        # Task 1: 0.30 - 0.10 = 0.20
        # Task 2: 0.40 - 0.10 = 0.30
        np.testing.assert_allclose(fwt_per_task, [0.20, 0.30], atol=1e-5)
        self.assertAlmostEqual(forward_transfer(self.forward_eval_matrix, self.baseline_scores), 0.25)

    def test_summarize_continual_metric_results(self) -> None:
        summary = summarize_continual_metric_results(
            eval_matrix=self.eval_matrix,
            forward_eval_matrix=self.forward_eval_matrix,
            baseline_forward_scores=self.baseline_scores,
            suffix="TestMetric",
        )
        self.assertIn("AvgAcc TestMetric", summary)
        self.assertIn("Backward Transfer TestMetric", summary)
        self.assertIn("Forward Transfer TestMetric", summary)
        self.assertIn("Average Forgetting TestMetric", summary)
        self.assertAlmostEqual(summary["AvgAcc TestMetric"], np.mean([0.80, 0.82, 0.86]))


class ClassIncrementalEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_model = DummyModel(num_classes=6)
        self.cil_model = CILComputerVisionModel(
            model=self.raw_model,
            exemplar_ratio=0.1,
            use_class_masking=True,
            device="cpu",
        )

    def test_class_masking_zeroes_unseen_classes(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        self.cil_model.seen_classes_tensor = torch.tensor([0, 1], dtype=torch.long)
        masked_logits = self.cil_model._mask_logits(logits)
        
        # Classes 0 and 1 should remain untouched
        self.assertEqual(masked_logits[0, 0].item(), 1.0)
        self.assertEqual(masked_logits[0, 1].item(), 2.0)
        
        # Classes 2..5 should be set to torch.finfo.min
        min_val = torch.finfo(logits.dtype).min
        for c in range(2, 6):
            self.assertEqual(masked_logits[0, c].item(), min_val)

    def test_save_and_load_checkpoint(self) -> None:
        self.cil_model.seen_classes = {0, 1, 2}
        self.cil_model.seen_classes_tensor = torch.tensor([0, 1, 2], dtype=torch.long)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "cil_model.pth"
            self.cil_model.save(str(ckpt_path))
            self.assertTrue(ckpt_path.exists())

            loaded_cil = CILComputerVisionModel.load(
                model_instance=DummyModel(num_classes=6),
                filename=str(ckpt_path),
                device="cpu",
            )
            self.assertEqual(loaded_cil.seen_classes, {0, 1, 2})
            self.assertEqual(loaded_cil.exemplar_ratio, 0.1)
            self.assertTrue(loaded_cil.use_class_masking)


class InferenceThroughputTests(unittest.TestCase):
    def test_inference_throughput_dataclass(self) -> None:
        measurement = InferenceThroughput()
        measurement.record(batch_size=8, elapsed_seconds=None)  # Warmup batch
        measurement.record(batch_size=8, elapsed_seconds=0.016)
        measurement.record(batch_size=8, elapsed_seconds=0.016)

        results = measurement.to_dict()
        self.assertEqual(results["Inference Samples"], 24)
        self.assertEqual(results["Timed Inference Samples"], 16)
        self.assertEqual(results["Timed Inference Batches"], 2)
        self.assertAlmostEqual(results["Measured Inference Time (s)"], 0.032)
        self.assertAlmostEqual(results["Inference Throughput (samples/s)"], 500.0)

    def test_measure_inference_throughput_function(self) -> None:
        model = DummyModel(num_classes=6)
        dataset = DummyDataset(targets=[0, 1, 2, 3, 4, 5, 0, 1])
        dataloader = DataLoader(
            dataset, batch_size=4, collate_fn=lambda b: (list(zip(*b))[0], list(zip(*b))[1], list(zip(*b))[2])
        )
        results = measure_inference_throughput(model, dataloader, device="cpu", warmup_batches=1)
        self.assertIn("Inference Throughput (samples/s)", results)
        self.assertIn("Inference Latency (ms/sample)", results)


class JSONSerializationTests(unittest.TestCase):
    def test_serialize_save_json_numpy_types(self) -> None:
        data = {
            "ndarray": np.array([1, 2, 3]),
            "float": np.float64(3.14),
            "int": np.int64(42),
            "nested": {"val": np.array([0.1, 0.2])},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "results.json"
            serialize_save_json(data, str(file_path))
            self.assertTrue(file_path.exists())


class BootstrapProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.all_targets = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0] * 10)

    def test_bootstrap_performance_diff_with_1d_predictions(self) -> None:
        model_preds = {
            "Model_A": self.all_targets.copy(),
            "Model_B": np.array([(t + 1) % 3 for t in self.all_targets]),
        }
        pairwise_results, model_performance = bootstrap_performance_diff(
            all_targets=self.all_targets,
            model_preds=model_preds,
            rng=self.rng,
            resamples=100,
            average="macro",
            stratified=True,
        )
        self.assertEqual(len(model_performance), 2)
        self.assertEqual(len(pairwise_results), 1)
        row = pairwise_results[0]
        self.assertIn("Observed Delta (A-B)", row)
        self.assertIn("FDR-BH Adjusted Tail Probability", row)
        self.assertGreater(row["Observed Delta (A-B)"], 0.0)
        self.assertTrue(row["95% CI Lower"] <= row["95% CI Upper"])

    def test_bootstrap_performance_diff_with_2d_logits(self) -> None:
        logits_A = np.zeros((len(self.all_targets), 3))
        for i, t in enumerate(self.all_targets):
            logits_A[i, t] = 5.0

        logits_B = np.zeros((len(self.all_targets), 3))
        for i, t in enumerate(self.all_targets):
            wrong_class = (t + 1) % 3
            logits_B[i, wrong_class] = 5.0

        model_preds = {
            "Model_A_Logits": logits_A,
            "Model_B_Logits": logits_B,
        }
        pairwise_results, model_performance = bootstrap_performance_diff(
            all_targets=self.all_targets,
            model_preds=model_preds,
            rng=self.rng,
            resamples=50,
            average="macro",
            stratified=True,
        )
        self.assertEqual(len(model_performance), 2)
        self.assertEqual(len(pairwise_results), 1)
        self.assertAlmostEqual(model_performance[0]["Observed F1"], 1.0)
        self.assertAlmostEqual(model_performance[1]["Observed F1"], 0.0)

    def test_bootstrap_tie_splitting_tail_probability(self) -> None:
        model_preds = {
            "Model_A": self.all_targets.copy(),
            "Model_B": self.all_targets.copy(),
        }
        pairwise_results, _ = bootstrap_performance_diff(
            all_targets=self.all_targets,
            model_preds=model_preds,
            rng=self.rng,
            resamples=100,
            average="macro",
            stratified=True,
        )
        row = pairwise_results[0]
        self.assertEqual(row["Observed Delta (A-B)"], 0.0)
        self.assertAlmostEqual(row["Bootstrap Tail Probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
