"""Learned quality-aware pooling over tracklet frames."""

from __future__ import annotations

import torch
from torch import nn


class QualityWeightedTemporalPooling(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        frame_quality: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, frames, channels]")
        batch, frames, _ = features.shape
        mask = (
            torch.ones((batch, frames), dtype=torch.bool, device=features.device)
            if frame_mask is None
            else frame_mask.to(device=features.device, dtype=torch.bool)
        )
        scores = self.quality_head(features).squeeze(-1)
        if frame_quality is not None:
            quality = frame_quality.to(device=features.device, dtype=features.dtype).clamp_min(1e-4)
            scores = scores + quality.log()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * mask.to(features.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return pooled, weights
