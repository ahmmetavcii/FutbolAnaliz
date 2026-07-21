"""Compact temporal jersey-number model."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

from .schemas import NUM_CLASSES
from .temporal_pooling import QualityWeightedTemporalPooling


class TemporalJerseyRecognizer(nn.Module):
    """MobileNetV3-small frame encoder plus learned temporal quality pooling."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        *,
        dropout: float = 0.2,
        width_mult: float = 1.0,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = "DEFAULT" if pretrained and abs(width_mult - 1.0) < 1e-6 else None
        backbone = mobilenet_v3_small(weights=weights, width_mult=width_mult)
        feature_dim = int(backbone.classifier[0].in_features)
        self.encoder = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten(1))
        self.pool = QualityWeightedTemporalPooling(feature_dim, hidden_dim=96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.num_classes = num_classes
        self.pretrained = bool(weights)

    def forward(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        frame_quality: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [batch, time, channels, height, width]")
        batch, time, channels, height, width = frames.shape
        encoded = self.encoder(frames.reshape(batch * time, channels, height, width))
        encoded = encoded.reshape(batch, time, -1)
        pooled, weights = self.pool(encoded, frame_mask, frame_quality)
        return {
            "logits": self.classifier(pooled),
            "frame_weights": weights,
            "frame_features": encoded,
        }


def build_model(config: dict[str, object] | None = None) -> TemporalJerseyRecognizer:
    config = config or {}
    return TemporalJerseyRecognizer(
        num_classes=int(config.get("num_classes", NUM_CLASSES)),
        dropout=float(config.get("dropout", 0.2)),
        width_mult=float(config.get("width_mult", 1.0)),
        pretrained=bool(config.get("pretrained", False)),
    )
