import sys
import unittest
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

CURRENT_WORK = Path(__file__).resolve().parents[1]
if str(CURRENT_WORK) not in sys.path:
    sys.path.insert(0, str(CURRENT_WORK))

from gre_model_base import (
    GeneralistRouterExperts,
    calibrate_temperature,
    cost_proxy,
    expected_calibration_error,
    expert_metrics,
    router_entropy,
)


class DummyGeneralist(nn.Module):
    def __init__(self, in_features: int = 16, feature_dim: int = 8, num_classes: int = 4) -> None:
        super().__init__()
        self.fc_feat = nn.Linear(in_features, feature_dim)
        self.fc_out = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = torch.relu(self.fc_feat(x))
        logits = self.fc_out(features)
        return (logits, features) if return_features else logits


class DummyRouter(nn.Module):
    def __init__(self, feature_dim: int = 8, num_experts: int = 3) -> None:
        super().__init__()
        self.gating = nn.Linear(feature_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gating(x)


class DummyExpert(nn.Module):
    def __init__(self, feature_dim: int = 8, num_classes: int = 4) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class DummyDataset(Dataset):
    def __init__(self, num_samples: int = 12, in_features: int = 16, num_classes: int = 4) -> None:
        self.data = torch.randn(num_samples, in_features)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], {"label": self.labels[index]}, f"path_{index}"


class GeneralistRouterExpertsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.in_features = 16
        self.feature_dim = 8
        self.num_classes = 4
        self.num_experts = 3
        self.k = 2

        self.generalist = DummyGeneralist(self.in_features, self.feature_dim, self.num_classes)
        self.router = DummyRouter(self.feature_dim, self.num_experts)
        expert_factory = lambda: DummyExpert(self.feature_dim, self.num_classes)

        self.model = GeneralistRouterExperts(
            generalist=self.generalist,
            router=self.router,
            expert_factory=expert_factory,
            num_experts=self.num_experts,
            num_classes=self.num_classes,
            k=self.k,
            lambda_aux=0.05,
        )

    def test_forward_pass_shape_and_routing_state(self) -> None:
        batch_size = 6
        x = torch.randn(batch_size, self.in_features)
        output = self.model(x)

        self.assertEqual(output.shape, (batch_size, self.num_classes))
        self.assertIsNotNone(self.model._last_router_probs)
        self.assertIsNotNone(self.model._last_topk_indices)
        self.assertEqual(cast(torch.Tensor, self.model._last_router_probs).shape, (batch_size, self.num_experts))
        self.assertEqual(cast(torch.Tensor, self.model._last_topk_indices).shape, (batch_size, self.k))

    def test_auxiliary_loss(self) -> None:
        x = torch.randn(4, self.in_features)
        _ = self.model(x)
        aux_loss = self.model.get_auxiliary_loss()

        self.assertIsInstance(aux_loss, torch.Tensor)
        self.assertEqual(aux_loss.ndim, 0)
        self.assertGreater(aux_loss.item(), 0.0)

    def test_reset_routing_state(self) -> None:
        x = torch.randn(4, self.in_features)
        _ = self.model(x)
        self.model.reset_routing_state()

        self.assertIsNone(self.model._last_router_probs)
        self.assertIsNone(self.model._last_topk_indices)
        self.assertIsNone(self.model._last_aux_loss)
        
        # get_auxiliary_loss should return a zero tensor when state is reset
        zero_aux = self.model.get_auxiliary_loss()
        self.assertEqual(zero_aux.item(), 0.0)

    def test_routing_summary(self) -> None:
        self.assertIsNone(self.model.routing_summary())
        x = torch.randn(4, self.in_features)
        _ = self.model(x)
        summary = self.model.routing_summary()

        if summary is not None:
            self.assertIsNotNone(summary)
            self.assertIn("avg_router_prob", summary)
            self.assertIn("selection_rate", summary)
            self.assertEqual(len(summary["avg_router_prob"]), self.num_experts)
            self.assertAlmostEqual(summary["selection_rate"].sum().item(), 1.0, places=5)
        else:
            self.fail("summary returned as None.")

    def test_parameter_counting_utilities(self) -> None:
        comp_params = self.model.get_model_parameters_by_component()
        self.assertIn("generalist", comp_params)
        self.assertIn("router", comp_params)
        self.assertIn("expert", comp_params)

        trainable, total, ratio = self.model.get_trainable_parameters()
        self.assertGreater(total, 0)
        self.assertEqual(trainable, total)
        self.assertEqual(ratio, 1.0)

        active, active_total, active_ratio = self.model.get_active_parameters()
        self.assertLess(active, active_total)
        self.assertGreater(active_ratio, 0.0)
        self.assertLess(active_ratio, 1.0)


class GREMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.in_features = 16
        self.feature_dim = 8
        self.num_classes = 4
        self.num_experts = 3
        self.k = 2

        self.generalist = DummyGeneralist(self.in_features, self.feature_dim, self.num_classes)
        self.router = DummyRouter(self.feature_dim, self.num_experts)
        expert_factory = lambda: DummyExpert(self.feature_dim, self.num_classes)

        self.model = GeneralistRouterExperts(
            generalist=self.generalist,
            router=self.router,
            expert_factory=expert_factory,
            num_experts=self.num_experts,
            num_classes=self.num_classes,
            k=self.k,
            lambda_aux=0.05,
        )
        self.dataset = DummyDataset(num_samples=12, in_features=self.in_features, num_classes=self.num_classes)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=4,
            collate_fn=lambda batch: (
                [b[0] for b in batch],
                [b[1] for b in batch],
                [b[2] for b in batch],
            ),
        )

    def test_expert_metrics(self) -> None:
        results = expert_metrics(self.model, self.dataloader, device="cpu")
        self.assertEqual(results["Model k"], self.k)
        self.assertAlmostEqual(results["Expected Expert Calls"], float(self.k))
        self.assertEqual(len(results["Expert Selection Ratio"]), self.num_experts)
        self.assertAlmostEqual(results["Expert Selection Ratio"].sum(), 1.0, places=5)

    def test_cost_proxy(self) -> None:
        comp_params = self.model.get_model_parameters_by_component()
        cost = cost_proxy(float(self.k), comp_params) # type: ignore
        expected = self.k * comp_params["expert"] + comp_params["generalist"] + comp_params["router"]
        self.assertEqual(cost, expected)

    def test_router_entropy(self) -> None:
        results = router_entropy(self.model, self.dataloader, device="cpu")
        self.assertIn("Mean Router Entropy", results)
        self.assertIn("Normalized Router Entropy", results)
        self.assertIn("Std Router Entropy", results)
        self.assertIn("Avg Router Prob", results)
        self.assertGreaterEqual(results["Mean Router Entropy"], 0.0)
        self.assertTrue(0.0 <= results["Normalized Router Entropy"] <= 1.0)

    def test_expected_calibration_error(self) -> None:
        ece_value = expected_calibration_error(self.model, self.num_classes, self.dataloader, device="cpu")
        self.assertIsInstance(ece_value, float)
        self.assertGreaterEqual(ece_value, 0.0)

    def test_calibrate_temperature(self) -> None:
        val_logits = torch.randn(20, self.num_classes)
        val_labels = torch.randint(0, self.num_classes, (20,))
        loss_fn = nn.CrossEntropyLoss()

        calibrate_temperature(self.model, val_logits, val_labels, loss_fn, device="cpu")
        self.assertGreater(self.model.temperature.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
