from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Compatibility wrapper for PyTorch versions before native SDPA."""
    native = getattr(F, "scaled_dot_product_attention", None)
    if native is not None:
        return native(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask.to(dtype=scores.dtype, device=scores.device)
    weights = torch.softmax(scores, dim=-1)
    if dropout_p:
        weights = F.dropout(weights, p=dropout_p, training=True)
    return torch.matmul(weights, v)
