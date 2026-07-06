import unittest

import torch

from joint_embedding import (
    JointEmbeddingBackbone,
    JointEmbeddingModule,
    NTXentLoss,
    SupervisedContrastiveLoss,
    TripletLoss,
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

    def test_backbone_feature_source_is_returned_when_configured(self) -> None:
        model = JointEmbeddingBackbone(
            num_classes=5,
            backbone="mobilenet_v3_small",
            embedding_dim=20,
            projection_dim=10,
            pretrained=False,
            expert_feature_source="backbone",
        )

        logits, features = model(self.images, return_features=True)
        encoded = model.encode(self.images)

        self.assertEqual(logits.shape, (2, 5))
        self.assertEqual(features.shape[1], model.backbone_feature_dim)
        self.assertTrue(torch.allclose(features, encoded["backbone_features"]))

    def test_router_feature_source_controls_classifier_input_space(self) -> None:
        embedding_model = JointEmbeddingBackbone(
            num_classes=6,
            backbone="mobilenet_v3_small",
            embedding_dim=28,
            projection_dim=14,
            pretrained=False,
            router_feature_source="embedding",
        )
        backbone_model = JointEmbeddingBackbone(
            num_classes=6,
            backbone="mobilenet_v3_small",
            embedding_dim=28,
            projection_dim=14,
            pretrained=False,
            router_feature_source="backbone",
        )

        embedding_outputs = embedding_model.encode(self.images)
        backbone_outputs = backbone_model.encode(self.images)

        self.assertEqual(
            embedding_model.classification_head.in_features,
            embedding_model.embedding_dim,
        )
        self.assertEqual(
            backbone_model.classification_head.in_features,
            backbone_model.backbone_feature_dim,
        )
        self.assertEqual(
            embedding_outputs["router_features"].shape,
            embedding_outputs["embeddings"].shape,
        )
        self.assertEqual(
            backbone_outputs["router_features"].shape,
            backbone_outputs["backbone_features"].shape,
        )

    def test_supported_backbones_produce_consistent_public_outputs(self) -> None:
        for backbone_name in ["resnet18", "mobilenet_v3_small", "convnext_tiny"]:
            with self.subTest(backbone=backbone_name):
                model = JointEmbeddingBackbone(
                    num_classes=4,
                    backbone=backbone_name,
                    embedding_dim=16,
                    projection_dim=8,
                    pretrained=False,
                )

                logits, features = model(self.images, return_features=True)
                encoded = model.encode(self.images)

                self.assertEqual(logits.shape, (2, 4))
                self.assertEqual(features.shape, (2, 16))
                self.assertEqual(encoded["projections"].shape, (2, 8))
                self.assertEqual(encoded["backbone_features"].shape[1], model.backbone_feature_dim)


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

    def test_ntxent_loss_requires_matching_shapes(self) -> None:
        loss_fn = NTXentLoss()
        z_i = torch.randn(2, 8)
        z_j = torch.randn(3, 8)

        with self.assertRaises(ValueError):
            loss_fn(z_i, z_j)

    def test_triplet_loss_returns_finite_scalar(self) -> None:
        loss_fn = TripletLoss(margin=0.5)
        anchor = torch.randn(3, 10)
        positive = anchor + 0.05 * torch.randn(3, 10)
        negative = torch.randn(3, 10) + 5.0

        loss = loss_fn(anchor, positive, negative)

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
            loss_type="supcon",
        )
        encoded = module.backbone.encode(self.images)
        labels = torch.tensor([0, 0])

        loss = module.compute_contrastive_loss(
            labels=labels,
            embeddings=encoded["embeddings"],
        )

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_module_ntxent_dispatch_accepts_paired_projections(self) -> None:
        module = JointEmbeddingModule(
            num_classes=4,
            backbone="mobilenet_v3_small",
            embedding_dim=32,
            projection_dim=16,
            pretrained=False,
            loss_type="ntxent",
        )
        z_i = torch.randn(2, 16)
        z_j = torch.randn(2, 16)

        loss = module.compute_contrastive_loss(paired_projections=(z_i, z_j))

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()