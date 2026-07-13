import unittest

import torch

from joint_embedding import (
    JointEmbeddingBackbone,
    JointEmbeddingModule,
    SupervisedContrastiveLoss,
)


class JointEmbeddingBackboneTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.images = torch.randn(2, 3, 224, 224)

    def test_forward_returns_logits_and_embedding_features_by_default(self) -> None:
        model = JointEmbeddingBackbone(
            num_classes=4,
            backbone="mobilenet_v3_small",
            embedding_dim=32,
            projection_dim=16,
            pretrained=False,
        )

        logits, features = model(self.images, return_features=True)
        encoded = model.encode(self.images)

        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(features.shape, (2, 32))
        self.assertTrue(torch.allclose(features, encoded["embeddings"]))

    def test_forward_returns_only_logits_when_requested(self) -> None:
        model = JointEmbeddingBackbone(
            num_classes=3,
            backbone="mobilenet_v3_small",
            embedding_dim=24,
            projection_dim=12,
            pretrained=False,
        )

        logits = model(self.images, return_features=False)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (2, 3))

        logits, features = model(self.images, return_features=True)
        encoded = model.encode(self.images)

        self.assertEqual(logits.shape, (2, 5))
        self.assertEqual(features.shape[1], model.backbone_feature_dim)
        self.assertTrue(torch.allclose(features, encoded["backbone_features"]))


class ContrastiveLossTests(unittest.TestCase):
    def test_supervised_contrastive_loss_returns_zero_without_positive_pairs(self) -> None:
        loss_fn = SupervisedContrastiveLoss()
        embeddings = torch.randn(3, 12)
        labels = torch.tensor([0, 1, 2])

        loss = loss_fn(embeddings, labels)

        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.item(), 0.0)

    def test_supervised_contrastive_loss_returns_finite_scalar_with_positive_pairs(self) -> None:
        loss_fn = SupervisedContrastiveLoss()
        embeddings = torch.randn(4, 12)
        labels = torch.tensor([0, 0, 1, 1])

        loss = loss_fn(embeddings, labels)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))


class JointEmbeddingModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.images = torch.randn(2, 3, 224, 224)

    def test_module_forward_matches_backbone_public_contract(self) -> None:
        module = JointEmbeddingModule(
            num_classes=4,
            backbone="mobilenet_v3_small",
            embedding_dim=32,
            projection_dim=16,
            pretrained=False,
        )

        logits, features = module(self.images, return_features=True)

        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(features.shape, (2, 32))

    def test_module_supcon_dispatch_accepts_embeddings(self) -> None:
        module = JointEmbeddingModule(
            num_classes=4,
            backbone="mobilenet_v3_small",
            embedding_dim=32,
            projection_dim=16,
            pretrained=False,
        )
        encoded = module.backbone.encode(self.images)
        labels = torch.tensor([0, 0])

        loss = module.compute_contrastive_loss(
            labels=labels,
            embeddings=encoded["embeddings"],
        )

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()