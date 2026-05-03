import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_like(features: torch.Tensor) -> torch.Tensor:
    return features.new_zeros(())


def _get_vector(
    metadata: Dict[str, torch.Tensor],
    key: str,
    size: int,
    device: torch.device,
    dtype: torch.dtype,
    default: float,
) -> torch.Tensor:
    value = metadata.get(key)
    if value is None:
        return torch.full((size,), default, device=device, dtype=dtype)
    return value.to(device=device, dtype=dtype).view(-1)


def _get_long_vector(
    metadata: Dict[str, torch.Tensor],
    key: str,
    size: int,
    device: torch.device,
    default: int,
) -> torch.Tensor:
    value = metadata.get(key)
    if value is None:
        return torch.full((size,), default, device=device, dtype=torch.long)
    return value.to(device=device, dtype=torch.long).view(-1)


def _get_label_matrix(
    metadata: Dict[str, torch.Tensor],
    size: int,
    num_diseases: int,
    device: torch.device,
) -> torch.Tensor:
    value = metadata.get("disease_labels")
    if value is None or num_diseases == 0:
        return torch.zeros((size, num_diseases), device=device)
    value = value.to(device=device, dtype=torch.float32)
    if value.ndim == 1:
        value = value.view(size, -1)
    return value[:, :num_diseases]


class ECGAugmenter(nn.Module):
    """GPU-friendly ECG augmentations for contrastive SSL."""

    def __init__(
        self,
        lead_dropout_prob: float = 0.15,
        time_mask_prob: float = 0.5,
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

        if self.amplitude_max > 0 and self.amplitude_max != self.amplitude_min:
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


class SymmetricContrastiveLoss(nn.Module):
    def forward(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits_left = logit_scale * left_features @ right_features.t()
        logits_right = logits_left.t()
        labels = torch.arange(left_features.shape[0], device=left_features.device)
        loss_left = F.cross_entropy(logits_left, labels)
        loss_right = F.cross_entropy(logits_right, labels)
        loss = 0.5 * (loss_left + loss_right)
        return loss, {
            "loss_ecg_to_text": loss_left.detach(),
            "loss_text_to_ecg": loss_right.detach(),
            "logit_scale": logit_scale.detach(),
        }


class SymmetricViewLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor) -> torch.Tensor:
        logit_scale = view_a.new_tensor(1.0 / self.temperature)
        loss, _ = SymmetricContrastiveLoss()(view_a, view_b, logit_scale)
        return loss


class NormalAwareMultiPositiveContrastiveLoss(nn.Module):
    """Normal-Aware Multi-Positive Contrastive Loss (NA-MPCL)."""

    def __init__(
        self,
        num_diseases: int,
        temperature: float = 0.07,
        beta: float = 2.0,
        gamma: float = 0.1,
        sigma_age: float = 10.0,
        sigma_hr: float = 15.0,
        lambda_aug: float = 1.0,
        lambda_patient: float = 1.0,
        lambda_disease: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_diseases = num_diseases
        self.temperature = temperature
        self.beta = beta
        self.gamma = gamma
        self.sigma_age = sigma_age
        self.sigma_hr = sigma_hr
        self.lambda_aug = lambda_aug
        self.lambda_patient = lambda_patient
        self.lambda_disease = lambda_disease
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
        positive_view_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.shape[0] == 0:
            return _zero_like(features)

        features = F.normalize(features, dim=-1)
        batch_size = features.shape[0]
        device = features.device

        disease_labels = _get_label_matrix(metadata, batch_size, self.num_diseases, device)
        normal_state = _get_long_vector(metadata, "normal_state", batch_size, device, -1)
        nsr = _get_long_vector(metadata, "nsr", batch_size, device, 0)
        patient_id = _get_long_vector(metadata, "patient_id", batch_size, device, -1)
        sample_id = _get_long_vector(metadata, "sample_id", batch_size, device, -1)
        age = _get_vector(metadata, "age", batch_size, device, torch.float32, -1.0)
        heart_rate = _get_vector(metadata, "heart_rate", batch_size, device, torch.float32, -1.0)
        sex = _get_long_vector(metadata, "sex", batch_size, device, -1)
        site = _get_long_vector(metadata, "site", batch_size, device, -1)

        if positive_view_features is not None:
            features = torch.cat([features, F.normalize(positive_view_features, dim=-1)], dim=0)
            disease_labels = torch.cat([disease_labels, disease_labels], dim=0)
            normal_state = torch.cat([normal_state, normal_state], dim=0)
            nsr = torch.cat([nsr, nsr], dim=0)
            patient_id = torch.cat([patient_id, patient_id], dim=0)
            sample_id = torch.cat([sample_id, sample_id], dim=0)
            age = torch.cat([age, age], dim=0)
            heart_rate = torch.cat([heart_rate, heart_rate], dim=0)
            sex = torch.cat([sex, sex], dim=0)
            site = torch.cat([site, site], dim=0)

        n_items = features.shape[0]
        eye = torch.eye(n_items, device=device, dtype=torch.bool)
        similarity = features @ features.t()

        positive_weights = features.new_zeros((n_items, n_items))
        same_sample = sample_id[:, None] == sample_id[None, :]
        known_sample = (sample_id[:, None] >= 0) & (sample_id[None, :] >= 0)
        positive_weights = positive_weights + self.lambda_aug * (same_sample & known_sample & ~eye).float()

        same_patient = patient_id[:, None] == patient_id[None, :]
        known_patient = (patient_id[:, None] >= 0) & (patient_id[None, :] >= 0)
        same_state = (normal_state[:, None] == normal_state[None, :]) & (normal_state[:, None] >= 0)
        patient_positive = same_patient & known_patient & same_state & ~same_sample & ~eye
        positive_weights = positive_weights + self.lambda_patient * patient_positive.float()

        if self.num_diseases > 0:
            intersection = disease_labels @ disease_labels.t()
            label_count = disease_labels.sum(dim=1, keepdim=True)
            union = label_count + label_count.t() - intersection
            jaccard = torch.where(union > 0, intersection / union.clamp_min(self.eps), torch.zeros_like(union))
            different_patient = ~same_patient | ~known_patient
            disease_positive = (
                (nsr[:, None] > 0)
                & (nsr[None, :] > 0)
                & different_patient
                & ~eye
            )
            positive_weights = positive_weights + self.lambda_disease * jaccard * disease_positive.float()

        positive_mask = positive_weights > 0
        negative_mask = ~eye & ~positive_mask

        healthy = normal_state == 0
        disease_positive_state = normal_state > 0
        hard_negative = (
            (healthy[:, None] & disease_positive_state[None, :])
            | (disease_positive_state[:, None] & healthy[None, :])
        )

        matching = self._clinical_matching(age, heart_rate, sex, site)
        omega = 1.0 + self.beta * matching * hard_negative.float()
        negative_logits = (similarity + self.gamma * hard_negative.float()) / self.temperature
        negative_exp = (omega * negative_logits.exp() * negative_mask.float()).sum(dim=1).clamp_min(self.eps)

        positive_exp = (similarity / self.temperature).exp()
        log_prob = torch.log(positive_exp / (positive_exp + negative_exp[:, None] + self.eps) + self.eps)
        weight_sum = positive_weights.sum(dim=1)
        valid_anchor = weight_sum > self.eps

        if not valid_anchor.any():
            return _zero_like(features)

        per_anchor = -(positive_weights * log_prob).sum(dim=1) / weight_sum.clamp_min(self.eps)
        return per_anchor[valid_anchor].mean()

    def _clinical_matching(
        self,
        age: torch.Tensor,
        heart_rate: torch.Tensor,
        sex: torch.Tensor,
        site: torch.Tensor,
    ) -> torch.Tensor:
        age_known = (age[:, None] >= 0) & (age[None, :] >= 0)
        hr_known = (heart_rate[:, None] >= 0) & (heart_rate[None, :] >= 0)

        age_term = torch.ones_like(age[:, None] + age[None, :], dtype=torch.float32)
        hr_term = torch.ones_like(age_term)
        age_term = torch.where(
            age_known,
            torch.exp(-torch.abs(age[:, None] - age[None, :]) / max(self.sigma_age, self.eps)),
            age_term,
        )
        hr_term = torch.where(
            hr_known,
            torch.exp(-torch.abs(heart_rate[:, None] - heart_rate[None, :]) / max(self.sigma_hr, self.eps)),
            hr_term,
        )

        sex_known = (sex[:, None] >= 0) & (sex[None, :] >= 0)
        site_known = (site[:, None] >= 0) & (site[None, :] >= 0)
        sex_term = torch.where(sex_known, (sex[:, None] == sex[None, :]).float(), torch.ones_like(age_term))
        site_term = torch.where(site_known, (site[:, None] == site[None, :]).float(), torch.ones_like(age_term))
        return age_term * hr_term * sex_term * site_term


class PrototypeMarginLoss(nn.Module):
    def __init__(
        self,
        num_diseases: int,
        feature_dim: int = 1024,
        margin: float = 0.2,
        ema_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.num_diseases = num_diseases
        self.margin = margin
        self.ema_momentum = ema_momentum
        self.register_buffer("prototypes", torch.zeros(num_diseases + 1, feature_dim))
        self.register_buffer("prototype_counts", torch.zeros(num_diseases + 1))

    def forward(self, features: torch.Tensor, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.num_diseases == 0 or features.shape[0] == 0:
            return _zero_like(features)

        features = F.normalize(features, dim=-1)
        if self.prototypes.shape[1] != features.shape[1]:
            self.prototypes = features.new_zeros(self.num_diseases + 1, features.shape[1])
            self.prototype_counts = features.new_zeros(self.num_diseases + 1)

        batch_size = features.shape[0]
        device = features.device
        disease_labels = _get_label_matrix(metadata, batch_size, self.num_diseases, device)
        normal_state = _get_long_vector(metadata, "normal_state", batch_size, device, -1)

        self._update_prototypes(features.detach(), disease_labels, normal_state)

        healthy_ready = self.prototype_counts[0] > 0
        disease_ready = self.prototype_counts[1:] > 0
        disease_mask = (normal_state[:, None] > 0) & (disease_labels > 0) & disease_ready[None, :]

        if not healthy_ready or not disease_mask.any():
            return _zero_like(features)

        healthy_similarity = features @ self.prototypes[0].detach()
        disease_similarity = features @ self.prototypes[1:].detach().t()
        margins = F.relu(self.margin + healthy_similarity[:, None] - disease_similarity)
        return margins[disease_mask].mean()

    @torch.no_grad()
    def _update_prototypes(
        self,
        features: torch.Tensor,
        disease_labels: torch.Tensor,
        normal_state: torch.Tensor,
    ) -> None:
        healthy_mask = normal_state == 0
        self._ema_update(0, features, healthy_mask)

        disease_nsr = normal_state > 0
        for disease_idx in range(self.num_diseases):
            mask = disease_nsr & (disease_labels[:, disease_idx] > 0)
            self._ema_update(disease_idx + 1, features, mask)

    @torch.no_grad()
    def _ema_update(self, proto_idx: int, features: torch.Tensor, mask: torch.Tensor) -> None:
        if not mask.any():
            return
        mean_feature = F.normalize(features[mask].mean(dim=0), dim=0)
        if self.prototype_counts[proto_idx] == 0:
            self.prototypes[proto_idx].copy_(mean_feature)
        else:
            updated = self.ema_momentum * self.prototypes[proto_idx] + (1 - self.ema_momentum) * mean_feature
            self.prototypes[proto_idx].copy_(F.normalize(updated, dim=0))
        self.prototype_counts[proto_idx] += mask.sum().to(dtype=self.prototype_counts.dtype)


class MultiLabelFocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logits is None or targets.numel() == 0:
            return logits.new_zeros(()) if logits is not None else targets.new_zeros(())

        targets = targets.to(device=logits.device, dtype=logits.dtype)
        if valid is not None:
            valid = valid.to(device=logits.device, dtype=torch.bool).view(-1)
            if not valid.any():
                return logits.new_zeros(())
            logits = logits[valid]
            targets = targets[valid]

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)
        loss = bce * (1 - pt).pow(self.gamma)
        if self.alpha is not None:
            alpha = logits.new_tensor(self.alpha)
            loss = loss * (targets * alpha + (1 - targets))
        return loss.mean()


class AlphaECGPretrainingLoss(nn.Module):
    """Full multimodal + SSL objective for AlphaECG pretraining."""

    def __init__(
        self,
        num_diseases: int,
        lambda_multimodal: float = 1.0,
        lambda_view: float = 1.0,
        lambda_na: float = 1.0,
        lambda_proto: float = 0.2,
        lambda_cls: float = 0.1,
        temperature: float = 0.07,
        view_temperature: float = 0.07,
        beta: float = 2.0,
        gamma: float = 0.1,
        proto_margin: float = 0.2,
        sigma_age: float = 10.0,
        sigma_hr: float = 15.0,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.lambda_multimodal = lambda_multimodal
        self.lambda_view = lambda_view
        self.lambda_na = lambda_na
        self.lambda_proto = lambda_proto
        self.lambda_cls = lambda_cls
        self.augmenter = ECGAugmenter()
        self.multimodal_loss = SymmetricContrastiveLoss()
        self.view_loss = SymmetricViewLoss(view_temperature)
        self.na_loss = NormalAwareMultiPositiveContrastiveLoss(
            num_diseases=num_diseases,
            temperature=temperature,
            beta=beta,
            gamma=gamma,
            sigma_age=sigma_age,
            sigma_hr=sigma_hr,
        )
        self.proto_loss = PrototypeMarginLoss(
            num_diseases=num_diseases,
            margin=proto_margin,
        )
        self.cls_loss = MultiLabelFocalBCELoss(gamma=focal_gamma, alpha=focal_alpha)

    def forward(
        self,
        model: nn.Module,
        ecg: torch.Tensor,
        text: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        metadata = metadata or {}

        view_a = self.augmenter(ecg)
        view_b = self.augmenter(ecg)

        ecg_raw = model.encode_image(view_a)
        ecg_features = F.normalize(ecg_raw, dim=-1)
        text_features = F.normalize(model.encode_text(text), dim=-1)
        view_b_features = F.normalize(model.encode_image(view_b), dim=-1)

        logit_scale = model.logit_scale.exp()
        multimodal, mm_metrics = self.multimodal_loss(ecg_features, text_features, logit_scale)
        view = self.view_loss(ecg_features, view_b_features)
        normal_aware = self.na_loss(ecg_features, metadata, positive_view_features=view_b_features)
        proto = self.proto_loss(ecg_features, metadata)

        disease_labels = _get_label_matrix(
            metadata,
            ecg.shape[0],
            getattr(model, "num_diseases", 0),
            ecg.device,
        )
        has_labels = metadata.get("has_disease_labels")
        disease_logits = model.predict_disease(ecg_raw) if hasattr(model, "predict_disease") else None
        cls = self.cls_loss(disease_logits, disease_labels, has_labels) if disease_logits is not None else _zero_like(ecg_features)

        total = (
            self.lambda_multimodal * multimodal
            + self.lambda_view * view
            + self.lambda_na * normal_aware
            + self.lambda_proto * proto
            + self.lambda_cls * cls
        )

        metrics = {
            "loss_total": total.detach(),
            "loss_multimodal": multimodal.detach(),
            "loss_view": view.detach(),
            "loss_na": normal_aware.detach(),
            "loss_proto": proto.detach(),
            "loss_cls": cls.detach(),
        }
        metrics.update(mm_metrics)
        return total, metrics
