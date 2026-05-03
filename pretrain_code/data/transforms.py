from __future__ import annotations

import math

import torch
import torch.nn as nn


class ECGAugmenter(nn.Module):
    """GPU-friendly augmentations for ECG self-supervision."""

    def __init__(
        self,
        lead_dropout_prob: float = 0.15,
        time_mask_prob: float = 0.50,
        max_time_mask_ratio: float = 0.10,
        amplitude_min: float = 0.80,
        amplitude_max: float = 1.20,
        noise_std: float = 0.01,
        baseline_wander_prob: float = 0.30,
        baseline_wander_amplitude: float = 0.05,
    ) -> None:
        super().__init__()
        self.lead_dropout_prob = lead_dropout_prob
        self.time_mask_prob = time_mask_prob
        self.max_time_mask_ratio = max_time_mask_ratio
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.noise_std = noise_std
        self.baseline_wander_prob = baseline_wander_prob
        self.baseline_wander_amplitude = baseline_wander_amplitude

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return ecg
        out = ecg.clone()
        batch_size, channels, length = out.shape

        if self.amplitude_max != self.amplitude_min:
            scale = torch.empty(batch_size, 1, 1, device=out.device, dtype=out.dtype)
            scale.uniform_(self.amplitude_min, self.amplitude_max)
            out = out * scale

        if channels > 1 and self.lead_dropout_prob > 0:
            keep = torch.rand(batch_size, channels, 1, device=out.device) > self.lead_dropout_prob
            keep = keep | (~keep.any(dim=1, keepdim=True))
            out = out * keep.to(dtype=out.dtype)

        if self.time_mask_prob > 0 and self.max_time_mask_ratio > 0:
            max_width = max(1, int(length * self.max_time_mask_ratio))
            apply_mask = torch.rand(batch_size, device=out.device) < self.time_mask_prob
            for idx in torch.where(apply_mask)[0].tolist():
                width = int(torch.randint(1, max_width + 1, (1,), device=out.device).item())
                start = int(torch.randint(0, max(1, length - width + 1), (1,), device=out.device).item())
                out[idx, :, start : start + width] = 0

        if self.baseline_wander_prob > 0 and self.baseline_wander_amplitude > 0:
            apply_wander = torch.rand(batch_size, 1, 1, device=out.device) < self.baseline_wander_prob
            time = torch.linspace(0, 1, length, device=out.device, dtype=out.dtype).view(1, 1, length)
            freq = torch.empty(batch_size, 1, 1, device=out.device, dtype=out.dtype).uniform_(0.15, 0.50)
            phase = torch.empty(batch_size, 1, 1, device=out.device, dtype=out.dtype).uniform_(0, 2 * math.pi)
            amp = torch.empty(batch_size, 1, 1, device=out.device, dtype=out.dtype).uniform_(
                -self.baseline_wander_amplitude,
                self.baseline_wander_amplitude,
            )
            out = out + apply_wander.to(dtype=out.dtype) * amp * torch.sin(2 * math.pi * freq * time + phase)

        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std
        return out


def make_contiguous_token_mask(
    batch_size: int,
    lead_num: int,
    token_count: int,
    *,
    mask_ratio_min: float = 0.10,
    mask_ratio_max: float = 0.50,
    mask_leads_together: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create contiguous time-token masks, returned as `[B, lead_num, token_count]`."""
    mask = torch.zeros(batch_size, lead_num, token_count, dtype=torch.bool, device=device)
    for batch_idx in range(batch_size):
        ratio = float(torch.empty(1, device=device).uniform_(mask_ratio_min, mask_ratio_max).item())
        width = max(1, int(round(token_count * ratio)))
        start = int(torch.randint(0, max(1, token_count - width + 1), (1,), device=device).item())
        if mask_leads_together:
            mask[batch_idx, :, start : start + width] = True
        else:
            lead_mask = torch.rand(lead_num, device=device) < 0.5
            if not lead_mask.any():
                lead_mask[torch.randint(0, lead_num, (1,), device=device)] = True
            mask[batch_idx, lead_mask, start : start + width] = True
    return mask

