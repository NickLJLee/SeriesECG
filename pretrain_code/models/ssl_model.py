from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrain_code.data.transforms import ECGAugmenter, make_contiguous_token_mask

from .ecg_encoder import ECGTokenEncoder
from .heads import ProjectionHead


def _soft_ce(student_logits: torch.Tensor, teacher_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    return -(teacher_probs * F.log_softmax(student_logits / temperature, dim=-1)).sum(dim=-1).mean()


class ECGSSLModel(nn.Module):
    """Masked self-distillation for ECG record pretraining."""

    def __init__(
        self,
        encoder: ECGTokenEncoder,
        *,
        output_dim: int = 8192,
        proj_hidden_dim: int = 2048,
        proj_bottleneck_dim: int = 256,
        student_temp: float = 0.10,
        teacher_temp: float = 0.07,
        teacher_patch_temp: float = 0.07,
        center_momentum: float = 0.90,
        mask_ratio_min: float = 0.10,
        mask_ratio_max: float = 0.50,
    ) -> None:
        super().__init__()
        self.student_encoder = encoder
        self.teacher_encoder = copy.deepcopy(encoder)
        self.student_head = ProjectionHead(encoder.d_model, proj_hidden_dim, proj_bottleneck_dim, output_dim)
        self.teacher_head = ProjectionHead(encoder.d_model, proj_hidden_dim, proj_bottleneck_dim, output_dim)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        self.augmenter = ECGAugmenter()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.teacher_patch_temp = teacher_patch_temp
        self.center_momentum = center_momentum
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.register_buffer("center_cls", torch.zeros(1, output_dim))
        self.register_buffer("center_patch", torch.zeros(1, output_dim))

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_head.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for student, teacher in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            teacher.data.mul_(momentum).add_(student.data, alpha=1.0 - momentum)
        for student, teacher in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            teacher.data.mul_(momentum).add_(student.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def _update_centers(self, cls_logits: torch.Tensor, patch_logits: torch.Tensor) -> None:
        self.center_cls.mul_(self.center_momentum).add_(cls_logits.mean(dim=0, keepdim=True), alpha=1.0 - self.center_momentum)
        self.center_patch.mul_(self.center_momentum).add_(
            patch_logits.reshape(-1, patch_logits.shape[-1]).mean(dim=0, keepdim=True),
            alpha=1.0 - self.center_momentum,
        )

    def forward(self, ecg: torch.Tensor) -> dict[str, torch.Tensor]:
        view_student = self.augmenter(ecg)
        view_teacher = self.augmenter(ecg)
        token_count = self.student_encoder.token_count(ecg.shape[-1])
        token_mask = make_contiguous_token_mask(
            ecg.shape[0],
            self.student_encoder.lead_num,
            token_count,
            mask_ratio_min=self.mask_ratio_min,
            mask_ratio_max=self.mask_ratio_max,
            device=ecg.device,
        )

        student_cls, student_tokens, mask_flat = self.student_encoder(view_student, token_mask=token_mask)
        student_cls_logits = self.student_head(student_cls)
        student_patch_logits = self.student_head(student_tokens)

        with torch.no_grad():
            teacher_cls, teacher_tokens, _ = self.teacher_encoder(view_teacher)
            teacher_cls_logits_raw = self.teacher_head(teacher_cls)
            teacher_patch_logits_raw = self.teacher_head(teacher_tokens)
            teacher_cls_probs = F.softmax((teacher_cls_logits_raw - self.center_cls) / self.teacher_temp, dim=-1)
            teacher_patch_probs = F.softmax(
                (teacher_patch_logits_raw - self.center_patch.view(1, 1, -1)) / self.teacher_patch_temp,
                dim=-1,
            )

        loss_cls = _soft_ce(student_cls_logits, teacher_cls_probs, self.student_temp)
        if mask_flat.any():
            loss_mim = _soft_ce(student_patch_logits[mask_flat], teacher_patch_probs[mask_flat], self.student_temp)
        else:
            loss_mim = student_cls_logits.new_zeros(())
        loss = loss_cls + loss_mim

        if self.training:
            self._update_centers(teacher_cls_logits_raw.detach(), teacher_patch_logits_raw.detach())

        return {
            "loss": loss,
            "loss_cls": loss_cls.detach(),
            "loss_mim": loss_mim.detach(),
            "mask_ratio": mask_flat.float().mean().detach(),
        }


def cosine_teacher_momentum(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
    if total_steps <= 1:
        return end
    progress = min(1.0, max(0.0, step / float(total_steps - 1)))
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0

