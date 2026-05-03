from __future__ import annotations

import torch
import torch.nn as nn

from .ecg_encoder import ECGTokenEncoder
from .heads import build_prediction_head
from .patient_transformer import PatientAggregator


class ECGPatientModel(nn.Module):
    """Record encoder + patient aggregator + supervised prediction head."""

    def __init__(
        self,
        record_encoder: ECGTokenEncoder,
        output_dim: int,
        *,
        head_type: str = "mlp",
        head_dropout: float = 0.1,
        patient_layers: int = 3,
        patient_heads: int = 6,
        patient_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.record_encoder = record_encoder
        self.patient_aggregator = PatientAggregator(
            d_model=record_encoder.d_model,
            n_layers=patient_layers,
            n_heads=patient_heads,
            dropout=patient_dropout,
        )
        self.head = build_prediction_head(
            d_model=record_encoder.d_model,
            output_dim=output_dim,
            head_type=head_type,
            dropout=head_dropout,
        )

    def encode_records(self, records: torch.Tensor) -> torch.Tensor:
        if records.dim() != 4:
            raise ValueError(f"Expected records [B, R, C, T], got {tuple(records.shape)}")
        batch_size, record_count, channels, length = records.shape
        flat_records = records.reshape(batch_size * record_count, channels, length)
        flat_cls, _, _ = self.record_encoder(flat_records)
        return flat_cls.reshape(batch_size, record_count, -1)

    def forward(self, records: torch.Tensor, record_padding_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        record_tokens = self.encode_records(records)
        if record_padding_mask is not None:
            record_tokens = record_tokens.masked_fill(record_padding_mask.unsqueeze(-1), 0.0)
        patient_embedding = self.patient_aggregator(record_tokens, record_padding_mask=record_padding_mask)
        logits = self.head(patient_embedding)
        return {
            "embedding": patient_embedding,
            "record_tokens": record_tokens,
            "logits": logits,
        }

