from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


BackboneName = Literal[
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "convnext_tiny",
    "convnext_small",
]


# =========================================================================================
# Joint Embedding Backbone
# =========================================================================================


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointEmbeddingBackbone(nn.Module):
    """
    Backbone for GRE that separates three concerns:

    1. Raw visual feature extraction.
    2. Projection of raw features into a latent embedding space.
    3. Production of classification logits from either space.

    The GRE generalist contract stays unchanged:
    `forward(..., return_features=True)` returns `(logits, features)`.

    The `features` tensor is configurable at initialization so experts can be
    trained on either:
    - latent embeddings, which is the default joint-embedding study condition, or
    - raw backbone features, which enables an ablation against the latent space.

    Extra tensors such as projections and raw backbone features are available
    through `encode(...)` rather than additional `forward(...)` return modes.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: BackboneName = "mobilenet_v3_small",
        embedding_dim: int = 128,
        projection_dim: int = 128,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim

        self.feature_encoder, self.backbone_feature_dim = self._build_feature_encoder(
            backbone=backbone,
            pretrained=pretrained,
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(self.backbone_feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
        )

        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=projection_dim,
        )

        self.classification_head = nn.Linear(self.embedding_dim, num_classes)

    def _build_feature_encoder(
        self,
        backbone: BackboneName,
        pretrained: bool,
    ) -> tuple[nn.Module, int]:
        if backbone == "mobilenet_v3_small":
            model = self._load_torchvision_model(
                builder=models.mobilenet_v3_small,
                weights_enum_name="MobileNet_V3_Small_Weights",
                pretrained=pretrained,
            )
            encoder = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1))
            feature_dim = model.classifier[0].in_features
            return encoder, feature_dim

        if backbone == "mobilenet_v3_large":
            model = self._load_torchvision_model(
                builder=models.mobilenet_v3_large,
                weights_enum_name="MobileNet_V3_Large_Weights",
                pretrained=pretrained,
            )
            encoder = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1))
            feature_dim = model.classifier[0].in_features
            return encoder, feature_dim

        if backbone == "convnext_tiny":
            model = self._load_torchvision_model(
                builder=models.convnext_tiny,
                weights_enum_name="ConvNeXt_Tiny_Weights",
                pretrained=pretrained,
            )
            encoder = nn.Sequential(model.features, model.avgpool)
            feature_dim = model.classifier[2].in_features
            return encoder, feature_dim

        if backbone == "convnext_small":
            model = self._load_torchvision_model(
                builder=models.convnext_small,
                weights_enum_name="ConvNeXt_Small_Weights",
                pretrained=pretrained,
            )
            encoder = nn.Sequential(model.features, model.avgpool)
            feature_dim = model.classifier[2].in_features
            return encoder, feature_dim

        raise ValueError(f"Unsupported backbone: {backbone}")

    def _load_torchvision_model(
        self,
        builder,
        weights_enum_name: str,
        pretrained: bool,
    ) -> nn.Module:
        if pretrained:
            weights_enum = getattr(models, weights_enum_name, None)
            if weights_enum is not None:
                return builder(weights=weights_enum.DEFAULT)
            return builder(pretrained=True)

        try:
            return builder(weights=None)
        except TypeError:
            return builder(pretrained=False)

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_features = self.feature_encoder(x)
        backbone_features = torch.flatten(backbone_features, 1)
        embeddings = self.embedding_head(backbone_features)
        projections = self.projection_head(embeddings)
        logits = self.classification_head(embeddings)

        return {
            "logits": logits,
            "embeddings": embeddings,
            "backbone_features": backbone_features,
            "projections": projections,
        }

    def forward(self, x: torch.Tensor, return_features: bool = True):
        outputs = self.encode(x)

        if not return_features:
            return outputs["logits"]

        return outputs["logits"], outputs["projections"]


# =========================================================================================
# Contrastive Objectives
# =========================================================================================


class SupervisedContrastiveLoss(nn.Module):
    """
    Vectorized supervised contrastive loss.

    This is the objective that best matches the current continual-learning study:
    it uses class labels to compact same-class samples and separate different classes,
    which is directly aligned with reducing interference across tasks.
    
    This is similar to soft nearest-neighbor loss by comparing multiple positive and
    negative samples together.
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-12) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (batch_size, feature_dim)")
        if labels.ndim != 1:
            labels = labels.view(-1)
        if embeddings.size(0) != labels.size(0):
            raise ValueError("embeddings and labels must have the same batch size")

        features = F.normalize(embeddings, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).to(dtype=logits.dtype)
        logits_mask = torch.ones_like(positive_mask) - torch.eye(
            embeddings.size(0),
            device=embeddings.device,
            dtype=logits.dtype,
        )

        positive_mask = positive_mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps))

        positive_count = positive_mask.sum(dim=1)
        valid_rows = positive_count > 0
        if not torch.any(valid_rows):
            print("no valid rows found")
            return embeddings.new_zeros(())

        mean_log_prob_pos = (
            (positive_mask[valid_rows] * log_prob[valid_rows]).sum(dim=1)
            / positive_count[valid_rows]
        )
        return -mean_log_prob_pos.mean()


# =========================================================================================
# Joint Embedding Training Wrapper
# =========================================================================================


class JointEmbeddingModule(nn.Module):
    """
    Thin wrapper around the backbone that exposes a contrastive criterion.

    This stays separate from `CILComputerVisionModel` because the continual-learning
    wrapper expects a standard classifier interface during optimization. The joint
    embedding objective can therefore be added explicitly in the training loop when
    the study requires it, instead of silently changing the GRE loss contract.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: BackboneName = "mobilenet_v3_small",
        embedding_dim: int = 128,
        projection_dim: int = 128,
        pretrained: bool = True,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()

        self.backbone = JointEmbeddingBackbone(
            num_classes=num_classes,
            backbone=backbone,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            pretrained=pretrained,
        )

        self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)

    def forward(self, x: torch.Tensor, return_features: bool = True):
        return self.backbone(x, return_features=return_features)

    def compute_contrastive_loss(
        self,
        *,
        labels: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
        projections: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature_tensor = embeddings if embeddings is not None else projections
        if feature_tensor is None or labels is None:
            raise ValueError("supcon loss requires labels and embeddings or projections")
        return self.contrastive_loss(feature_tensor, labels)
