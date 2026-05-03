from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_fn


def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    def slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if n_heads & (n_heads - 1) == 0:
        slopes = slopes_power_of_2(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = slopes_power_of_2(closest)
        slopes += slopes_power_of_2(2 * closest)[0::2][: n_heads - closest]
    return -torch.tensor(slopes, dtype=torch.float32).view(1, n_heads, 1, 1)


class ECGALiBi(nn.Module):
    """Time-aware attention bias for ECG tokens."""

    def __init__(self, n_heads: int, lead_penalty: float = 0.15, learnable: bool = False) -> None:
        super().__init__()
        slopes = _get_alibi_slopes(n_heads)
        if learnable:
            self.slopes = nn.Parameter(slopes)
        else:
            self.register_buffer("slopes", slopes)
        self.lead_penalty = float(lead_penalty)

    def forward(self, time_index: torch.Tensor, lead_index: torch.Tensor, prefix_tokens: int) -> torch.Tensor:
        device = time_index.device
        token_count = time_index.numel()
        total = prefix_tokens + token_count
        bias = torch.zeros(1, self.slopes.shape[1], total, total, device=device)
        dt = torch.abs(time_index[:, None] - time_index[None, :]).float()
        dl = (lead_index[:, None] != lead_index[None, :]).float() * self.lead_penalty
        distance = dt + dl
        bias[:, :, prefix_tokens:, prefix_tokens:] = self.slopes.to(device) * distance.view(1, 1, token_count, token_count)
        return bias


class ECGTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        qk_norm: bool = False,
    ) -> None:
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
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_dropout = attn_dropout
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, token_count, d_model = x.shape
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).reshape(batch_size, token_count, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(batch_size, token_count, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(batch_size, token_count, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias.to(dtype=q.dtype) if attn_bias is not None else None,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(batch_size, token_count, d_model)
        x = x + self.out_proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class ECGTokenEncoder(nn.Module):
    """Encode one ECG recording into token and record-level embeddings."""

    def __init__(
        self,
        lead_num: int = 12,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        dim_feedforward: int = 1536,
        patch_samples: int = 250,
        patch_stride: int | None = None,
        num_registers: int = 2,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        qk_norm: bool = False,
        lead_penalty: float = 0.15,
    ) -> None:
        super().__init__()
        self.lead_num = lead_num
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.patch_samples = patch_samples
        self.patch_stride = patch_stride or patch_samples
        self.num_registers = max(0, int(num_registers))
        self.activation_checkpointing = False

        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch_samples, stride=self.patch_stride)
        self.lead_embedding = nn.Parameter(torch.zeros(lead_num, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_registers, d_model)) if self.num_registers else None
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_bias = ECGALiBi(n_heads=n_heads, lead_penalty=lead_penalty)
        self.blocks = nn.ModuleList(
            [
                ECGTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    qk_norm=qk_norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.lead_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)

    def set_activation_checkpointing(self, enabled: bool = True) -> None:
        self.activation_checkpointing = bool(enabled)

    def token_count(self, sample_count: int) -> int:
        return (sample_count - self.patch_samples) // self.patch_stride + 1

    def forward(
        self,
        ecg: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ecg.dim() != 3:
            raise ValueError(f"Expected ECG tensor [B, C, T], got {tuple(ecg.shape)}")
        batch_size, channels, _ = ecg.shape
        if channels > self.lead_num:
            ecg = ecg[:, : self.lead_num]
            channels = self.lead_num
        elif channels < self.lead_num:
            pad = ecg.new_zeros(batch_size, self.lead_num - channels, ecg.shape[-1])
            ecg = torch.cat([ecg, pad], dim=1)
            channels = self.lead_num

        x = ecg.reshape(batch_size * channels, 1, ecg.shape[-1])
        x = self.patch_embed(x).transpose(1, 2)
        token_count = x.shape[1]
        x = x.reshape(batch_size, channels, token_count, self.d_model)
        x = x + self.lead_embedding[:channels].view(1, channels, 1, self.d_model)
        x = x.reshape(batch_size, channels * token_count, self.d_model)

        if token_mask is None:
            mask_flat = torch.zeros(batch_size, channels * token_count, dtype=torch.bool, device=ecg.device)
        else:
            mask_flat = token_mask.to(device=ecg.device, dtype=torch.bool).reshape(batch_size, channels * token_count)
            x = torch.where(mask_flat.unsqueeze(-1), self.mask_token.expand_as(x), x)

        prefix = [self.cls_token.expand(batch_size, -1, -1)]
        if self.register_tokens is not None:
            prefix.append(self.register_tokens.expand(batch_size, -1, -1))
        x = torch.cat(prefix + [x], dim=1)

        lead_index = torch.arange(channels, device=ecg.device).repeat_interleave(token_count)
        time_index = torch.arange(token_count, device=ecg.device).repeat(channels)
        attn_bias = self.pos_bias(time_index, lead_index, prefix_tokens=1 + self.num_registers)

        use_checkpoint = self.activation_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_checkpoint:
                x = checkpoint_fn(lambda z, b=block: b(z, attn_bias), x, use_reentrant=False)
            else:
                x = block(x, attn_bias)
        x = self.norm(x)
        cls = x[:, 0]
        tokens = x[:, 1 + self.num_registers :]
        return cls, tokens, mask_flat
