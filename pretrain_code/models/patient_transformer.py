from __future__ import annotations

import torch
import torch.nn as nn

from .attention import scaled_dot_product_attention


class PatientTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, token_count, d_model = x.shape
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).reshape(batch_size, token_count, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(batch_size, token_count, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(batch_size, token_count, self.n_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = torch.where(attn_mask, float("-inf"), 0.0).to(dtype=q.dtype)
        attn = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(batch_size, token_count, d_model)
        x = x + self.out_proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class PatientAggregator(nn.Module):
    """Aggregate one or more ECG record embeddings into a patient/case embedding."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 3,
        n_heads: int = 6,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        num_registers: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_registers = max(0, int(num_registers))
        dim_feedforward = dim_feedforward or 4 * d_model
        self.case_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_registers, d_model)) if self.num_registers else None
        self.blocks = nn.ModuleList(
            [
                PatientTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.case_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=0.02)

    def forward(self, record_tokens: torch.Tensor, record_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if record_tokens.dim() == 2:
            record_tokens = record_tokens.unsqueeze(0)
        batch_size = record_tokens.shape[0]
        tokens = [self.case_token.expand(batch_size, -1, -1)]
        if self.register_tokens is not None:
            tokens.append(self.register_tokens.expand(batch_size, -1, -1))
        tokens.append(record_tokens)
        x = torch.cat(tokens, dim=1)

        key_padding_mask = None
        if record_padding_mask is not None:
            prefix = torch.zeros(
                record_padding_mask.shape[0],
                1 + self.num_registers,
                dtype=torch.bool,
                device=record_padding_mask.device,
            )
            key_padding_mask = torch.cat([prefix, record_padding_mask], dim=1)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.norm(x)[:, 0]
