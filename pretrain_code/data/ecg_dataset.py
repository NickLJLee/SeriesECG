from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .ecg_io import preprocess_record

_ICD_I_RE = re.compile(r"^I[0-9A-Z]{2}$")
_SORT_COLUMN_CANDIDATES = ("record_time", "acquisition_time", "date", "timestamp")


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


def _read_manifest_columns(manifest_path: str | os.PathLike[str]) -> list[str]:
    return list(pd.read_csv(manifest_path, nrows=0).columns)


def _expand_label_columns(value: str | Sequence[str] | None, columns: Sequence[str]) -> list[str]:
    requested = _split_columns(value)
    if len(requested) == 1 and requested[0].lower() == "all_icd_i":
        return [column for column in columns if _ICD_I_RE.match(str(column))]
    return requested


def _record_key(raw_path: object) -> str:
    text = str(raw_path)
    if not text:
        return text
    return Path(text).stem


def _has_records_file(root: str | os.PathLike[str] | None) -> bool:
    if not root:
        return False
    root_path = Path(root)
    if not root_path.exists():
        return False
    try:
        next(root_path.rglob("RECORDS"))
        return True
    except StopIteration:
        return False


def _normalize_ecg_layout(layout: str, ecg_root: str | os.PathLike[str] | None) -> str:
    normalized = layout.lower()
    if normalized not in {"auto", "heedb_wfdb", "flat"}:
        raise ValueError("--ecg_layout must be one of: auto, heedb_wfdb, flat.")
    if normalized == "auto":
        return "heedb_wfdb" if _has_records_file(ecg_root) else "flat"
    return normalized


def _load_path_index(path: str | os.PathLike[str]) -> dict[str, str]:
    index_path = Path(path)
    if not index_path.exists():
        return {}
    frame = pd.read_csv(index_path, usecols=["record_id", "path"])
    return dict(zip(frame["record_id"].astype(str), frame["path"].astype(str)))


def _save_path_index(path: str | os.PathLike[str], index: dict[str, str]) -> None:
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"record_id": list(index.keys()), "path": list(index.values())})
    frame.to_csv(index_path, index=False)


def _build_heedb_wfdb_index(
    ecg_root: str | os.PathLike[str],
    wanted_records: set[str] | None = None,
) -> dict[str, str]:
    root = Path(ecg_root)
    index: dict[str, str] = {}
    saw_records = False
    for records_path in root.rglob("RECORDS"):
        saw_records = True
        base_dir = records_path.parent
        try:
            with records_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    record = line.strip().split()
                    if not record:
                        continue
                    record_name = record[0]
                    record_id = Path(record_name).stem
                    if wanted_records is not None and record_id not in wanted_records:
                        continue
                    signal_path = base_dir / record_name
                    if signal_path.suffix.lower() not in {".mat", ".hea"}:
                        signal_path = signal_path.with_suffix(".mat")
                    elif signal_path.suffix.lower() == ".hea":
                        signal_path = signal_path.with_suffix(".mat")
                    index[record_id] = str(signal_path)
        except OSError:
            continue
    if saw_records:
        return index

    for hea_path in root.rglob("*.hea"):
        record_id = hea_path.stem
        if wanted_records is not None and record_id not in wanted_records:
            continue
        index[record_id] = str(hea_path.with_suffix(".mat"))
    return index


def _resolve_path_index(
    *,
    ecg_root: str | os.PathLike[str] | None,
    path_index: str | os.PathLike[str] | None,
    wanted_records: set[str] | None,
) -> dict[str, str]:
    index = _load_path_index(path_index) if path_index else {}
    missing_records = None
    if wanted_records is not None:
        missing_records = wanted_records.difference(index)
    if ecg_root and (not index or (missing_records is not None and missing_records)):
        built = _build_heedb_wfdb_index(ecg_root, wanted_records=missing_records or wanted_records)
        index.update(built)
        if path_index:
            _save_path_index(path_index, index)
    return index


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
        ecg_layout: str = "auto",
        path_index: str | os.PathLike[str] | None = None,
        extra_columns: Sequence[str] | None = None,
        include_text: bool = True,
    ) -> None:
        self.manifest_path = str(manifest_path)
        self.ecg_root = str(ecg_root) if ecg_root else None
        self.manifest_columns = _read_manifest_columns(manifest_path)
        self.path_column = path_column or _first_existing_column(
            self.manifest_columns,
            ("path", "ecg_path", "record_path", "filename", "HashFileName"),
        )
        if self.path_column is None:
            raise ValueError("Manifest must contain a path/ecg_path/record_path/filename column.")
        self.record_id_column = record_id_column or _first_existing_column(
            self.manifest_columns,
            ("record_id", "sample_id", "HashFileName", "filename"),
        )
        self.patient_id_column = patient_id_column or _first_existing_column(
            self.manifest_columns,
            ("patient_id", "PatientID", "BDSPPatientID", "case_id"),
        )
        self.text_column = None
        if include_text:
            self.text_column = text_column or _first_existing_column(
                self.manifest_columns,
                ("text", "txt", "diagnosis", "deid_t_diagnosis_original", "deid_t_diagnosis", "ICD_text"),
            )
        self.sample_rate_column = sample_rate_column or _first_existing_column(
            self.manifest_columns,
            ("sample_rate", "fs", "sampling_rate"),
        )
        self.label_columns = _expand_label_columns(label_columns, self.manifest_columns)
        self.signal_key = signal_key
        self.target_fs = target_fs
        self.lead_num = lead_num
        self.window_size = window_size
        self.crop = crop
        self.normalize = normalize
        self.apply_filter = apply_filter
        self.ecg_layout = _normalize_ecg_layout(ecg_layout, self.ecg_root)
        self.path_index_path = str(path_index) if path_index else None
        self.resolved_path_column = "__resolved_ecg_path"

        usecols = {
            column
            for column in (
                self.path_column,
                self.record_id_column,
                self.patient_id_column,
                self.text_column,
                self.sample_rate_column,
                *self.label_columns,
                *_SORT_COLUMN_CANDIDATES,
                *(extra_columns or ()),
            )
            if column and column in self.manifest_columns
        }
        self.data = pd.read_csv(manifest_path, usecols=list(usecols))

        if drop_missing:
            self._drop_missing_records()
        elif self.ecg_layout == "heedb_wfdb":
            self._attach_heedb_paths(drop_missing=False)

    def __len__(self) -> int:
        return len(self.data)

    def _attach_heedb_paths(self, *, drop_missing: bool) -> None:
        record_ids = self.data[self.path_column].map(_record_key).astype(str)
        wanted_records = set(record_ids.unique())
        index = _resolve_path_index(
            ecg_root=self.ecg_root,
            path_index=self.path_index_path,
            wanted_records=wanted_records,
        )
        resolved = record_ids.map(index)
        self.data[self.resolved_path_column] = resolved
        if drop_missing:
            self.data = self.data.loc[resolved.notna()].reset_index(drop=True)

    def _drop_missing_records(self) -> None:
        if self.ecg_layout == "heedb_wfdb":
            self._attach_heedb_paths(drop_missing=True)
            return

        keep = []
        resolved = []
        for _, row in self.data.iterrows():
            path = _resolve_path(row[self.path_column], self.ecg_root)
            if Path(path).exists():
                keep.append(True)
                resolved.append(path)
                continue
            mat_path = str(Path(path).with_suffix(".mat"))
            if Path(mat_path).exists():
                keep.append(True)
                resolved.append(mat_path)
                continue
            keep.append(False)
            resolved.append(path)
        self.data[self.resolved_path_column] = resolved
        self.data = self.data.loc[keep].reset_index(drop=True)

    def _row_path(self, row: pd.Series) -> str:
        if self.resolved_path_column in row.index and not pd.isna(row[self.resolved_path_column]):
            return str(row[self.resolved_path_column])
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
        manifest_columns = _read_manifest_columns(manifest_path)
        inferred_case_column = case_id_column or _first_existing_column(
            manifest_columns,
            ("case_id", "patient_id", "PatientID", "BDSPPatientID"),
        )
        inferred_sort_column = record_sort_column or _first_existing_column(
            manifest_columns,
            _SORT_COLUMN_CANDIDATES,
        )
        self.record_dataset = ECGManifestDataset(
            manifest_path,
            ecg_root=ecg_root,
            drop_missing=record_dataset_kwargs.pop("drop_missing", True),
            extra_columns=[column for column in (inferred_case_column, inferred_sort_column) if column],
            **record_dataset_kwargs,
        )
        data = self.record_dataset.data
        self.case_id_column = inferred_case_column
        if self.case_id_column is None:
            self.case_id_column = self.record_dataset.record_id_column or self.record_dataset.path_column
        self.max_records_per_case = max(1, int(max_records_per_case))
        self.record_sort_column = inferred_sort_column if inferred_sort_column in data.columns else None

        if self.record_sort_column:
            ordered = data.sort_values([self.case_id_column, self.record_sort_column], kind="mergesort")
        else:
            ordered = data
        self.case_ids = []
        self.groups = []
        for case_id, frame in ordered.groupby(self.case_id_column, sort=True):
            indices = list(frame.index)
            selected = indices[-self.max_records_per_case :]
            self.case_ids.append(str(case_id))
            self.groups.append(selected)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, object]:
        indices = self.groups[idx]
        samples = [self.record_dataset[i] for i in indices]
        records = torch.stack([sample["ecg"] for sample in samples], dim=0)
        labels = torch.stack([sample["labels"] for sample in samples], dim=0)
        label_masks = torch.stack([sample["label_mask"] for sample in samples], dim=0)
        label_mask = label_masks.any(dim=0)
        if labels.numel():
            labels = torch.where(label_masks, labels, torch.zeros_like(labels)).amax(dim=0)
        else:
            labels = torch.empty(0, dtype=torch.float32)
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
