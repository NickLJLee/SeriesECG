from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """DINO/iBOT-style projection head."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 8192,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, output_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


def build_prediction_head(d_model: int, output_dim: int, head_type: str = "mlp", dropout: float = 0.1) -> nn.Module:
    if head_type == "linear":
        return nn.Linear(d_model, output_dim)
    if head_type != "mlp":
        raise ValueError(f"Unsupported head_type={head_type!r}")
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, output_dim),
    )

