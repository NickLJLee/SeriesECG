from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .ecg_io import preprocess_record


def _split_columns(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _resolve_path(raw_path: object, root: str | os.PathLike[str] | None = None) -> str:
    path = Path(str(raw_path))
    if not path.is_absolute() and root:
        path = Path(root) / path
    return str(path)


def _first_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


class ECGManifestDataset(Dataset):
    """Record-level ECG dataset driven by a CSV manifest."""

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        *,
        ecg_root: str | os.PathLike[str] | None = None,
        path_column: str | None = None,
        record_id_column: str | None = None,
        patient_id_column: str | None = None,
        text_column: str | None = None,
        label_columns: str | Sequence[str] | None = None,
        sample_rate_column: str | None = None,
        signal_key: str | None = None,
        target_fs: float = 500.0,
        lead_num: int = 12,
        window_size: int = 5000,
        crop: str = "center",
        normalize: str = "per_lead",
        apply_filter: bool = True,
        drop_missing: bool = True,
    ) -> None:
        self.manifest_path = str(manifest_path)
        self.ecg_root = str(ecg_root) if ecg_root else None
        self.data = pd.read_csv(manifest_path)
        self.path_column = path_column or _first_existing_column(
            self.data.columns,
            ("path", "ecg_path", "record_path", "filename", "HashFileName"),
        )
        if self.path_column is None:
            raise ValueError("Manifest must contain a path/ecg_path/record_path/filename column.")
        self.record_id_column = record_id_column or _first_existing_column(
            self.data.columns,
            ("record_id", "sample_id", "HashFileName", "filename"),
        )
        self.patient_id_column = patient_id_column or _first_existing_column(
            self.data.columns,
            ("patient_id", "PatientID", "BDSPPatientID", "case_id"),
        )
        self.text_column = text_column or _first_existing_column(
            self.data.columns,
            ("text", "txt", "diagnosis", "deid_t_diagnosis_original", "deid_t_diagnosis", "ICD_text"),
        )
        self.sample_rate_column = sample_rate_column or _first_existing_column(
            self.data.columns,
            ("sample_rate", "fs", "sampling_rate"),
        )
        self.label_columns = _split_columns(label_columns)
        self.signal_key = signal_key
        self.target_fs = target_fs
        self.lead_num = lead_num
        self.window_size = window_size
        self.crop = crop
        self.normalize = normalize
        self.apply_filter = apply_filter

        if drop_missing:
            keep = []
            for _, row in self.data.iterrows():
                path = _resolve_path(row[self.path_column], self.ecg_root)
                if Path(path).exists():
                    keep.append(True)
                    continue
                if Path(path).with_suffix(".mat").exists():
                    keep.append(True)
                    continue
                keep.append(False)
            self.data = self.data.loc[keep].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def _row_path(self, row: pd.Series) -> str:
        path = _resolve_path(row[self.path_column], self.ecg_root)
        if Path(path).exists():
            return path
        mat_path = str(Path(path).with_suffix(".mat"))
        return mat_path if Path(mat_path).exists() else path

    def _labels(self, row: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.label_columns:
            return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.bool)
        values = []
        mask = []
        for column in self.label_columns:
            value = row[column] if column in row.index else np.nan
            valid = not pd.isna(value)
            values.append(float(value) if valid else 0.0)
            mask.append(valid)
        return torch.tensor(values, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.data.iloc[idx]
        path = self._row_path(row)
        sample_rate = None
        if self.sample_rate_column and self.sample_rate_column in row.index and not pd.isna(row[self.sample_rate_column]):
            sample_rate = float(row[self.sample_rate_column])
        ecg = preprocess_record(
            path,
            signal_key=self.signal_key,
            sample_rate=sample_rate,
            target_fs=self.target_fs,
            lead_num=self.lead_num,
            window_size=self.window_size,
            crop=self.crop,
            normalize=self.normalize,
            apply_filter=self.apply_filter,
        )
        labels, label_mask = self._labels(row)
        record_id = str(row[self.record_id_column]) if self.record_id_column else Path(path).stem
        patient_id = str(row[self.patient_id_column]) if self.patient_id_column else record_id
        text = str(row[self.text_column]) if self.text_column and not pd.isna(row[self.text_column]) else ""
        return {
            "ecg": torch.from_numpy(ecg),
            "labels": labels,
            "label_mask": label_mask,
            "text": text,
            "path": path,
            "record_id": record_id,
            "patient_id": patient_id,
        }


class PatientECGDataset(Dataset):
    """Patient/case-level dataset: one item contains one or more ECG records."""

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        *,
        ecg_root: str | os.PathLike[str] | None = None,
        case_id_column: str | None = None,
        max_records_per_case: int = 8,
        record_sort_column: str | None = None,
        **record_dataset_kwargs,
    ) -> None:
        self.record_dataset = ECGManifestDataset(
            manifest_path,
            ecg_root=ecg_root,
            drop_missing=record_dataset_kwargs.pop("drop_missing", True),
            **record_dataset_kwargs,
        )
        data = self.record_dataset.data
        self.case_id_column = case_id_column or _first_existing_column(
            data.columns,
            ("case_id", "patient_id", "PatientID", "BDSPPatientID"),
        )
        if self.case_id_column is None:
            self.case_id_column = self.record_dataset.record_id_column or self.record_dataset.path_column
        self.max_records_per_case = max(1, int(max_records_per_case))
        self.record_sort_column = record_sort_column or _first_existing_column(
            data.columns,
            ("record_time", "acquisition_time", "date", "timestamp"),
        )

        groups: dict[str, list[int]] = defaultdict(list)
        for idx, row in data.iterrows():
            groups[str(row[self.case_id_column])].append(idx)
        self.case_ids = sorted(groups.keys())
        self.groups = []
        for case_id in self.case_ids:
            indices = groups[case_id]
            if self.record_sort_column:
                indices = sorted(indices, key=lambda i: str(data.iloc[i][self.record_sort_column]))
            self.groups.append(indices[: self.max_records_per_case])

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, object]:
        indices = self.groups[idx]
        samples = [self.record_dataset[i] for i in indices]
        records = torch.stack([sample["ecg"] for sample in samples], dim=0)
        labels = samples[0]["labels"].clone()
        label_mask = samples[0]["label_mask"].clone()
        for sample in samples[1:]:
            sample_mask = sample["label_mask"]
            fill = ~label_mask & sample_mask
            if fill.any():
                labels[fill] = sample["labels"][fill]
                label_mask[fill] = True
        return {
            "records": records,
            "labels": labels,
            "label_mask": label_mask,
            "case_id": self.case_ids[idx],
            "record_ids": [str(sample["record_id"]) for sample in samples],
            "paths": [str(sample["path"]) for sample in samples],
        }


def collate_record_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "ecg": torch.stack([item["ecg"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "label_mask": torch.stack([item["label_mask"] for item in batch], dim=0),
        "texts": [str(item["text"]) for item in batch],
        "record_ids": [str(item["record_id"]) for item in batch],
        "patient_ids": [str(item["patient_id"]) for item in batch],
        "paths": [str(item["path"]) for item in batch],
    }


def collate_patient_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    max_records = max(int(item["records"].shape[0]) for item in batch)
    channels = int(batch[0]["records"].shape[1])
    length = int(batch[0]["records"].shape[2])
    records = torch.zeros(len(batch), max_records, channels, length, dtype=torch.float32)
    record_padding_mask = torch.ones(len(batch), max_records, dtype=torch.bool)
    for batch_idx, item in enumerate(batch):
        count = int(item["records"].shape[0])
        records[batch_idx, :count] = item["records"]
        record_padding_mask[batch_idx, :count] = False
    return {
        "records": records,
        "record_padding_mask": record_padding_mask,
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "label_mask": torch.stack([item["label_mask"] for item in batch], dim=0),
        "case_ids": [str(item["case_id"]) for item in batch],
        "record_ids": [item["record_ids"] for item in batch],
        "paths": [item["paths"] for item in batch],
    }
