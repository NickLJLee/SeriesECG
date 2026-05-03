import hashlib
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from mne.filter import filter_data, notch_filter
from scipy.interpolate import interp1d
from scipy.io import loadmat
from torch.utils import data
from torch.utils.data import Dataset

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer


DEFAULT_ECG_PATH = "/data2/2shared/AAA_public_data/HEEDB/HEEDB/ECG/WFDB/"
DEFAULT_FALLBACK_RECORD = (
    "/data2/2shared/AAA_public_data/HEEDB/HEEDB/ECG/WFDB/"
    "S0001/2018/12/de_112344838_20190330104655_20190512102946"
)

DEFAULT_DISEASE_DEFINITIONS: Sequence[Tuple[str, Sequence[str]]] = (
    ("heart_failure", ("heart failure", "congestive heart failure", "hfpef", "hfref")),
    ("atrial_fibrillation", ("atrial fibrillation", "atrial flutter", "afib")),
    ("myocardial_infarction", ("myocardial infarction", "acute myocardial infarction", "old myocardial infarction")),
    ("ischemic_heart_disease", ("chronic ischemic heart disease", "coronary artery disease", "angina pectoris")),
    ("aortic_valve_disease", ("aortic valve", "aortic stenosis", "nonrheumatic aortic valve")),
    ("mitral_valve_disease", ("mitral valve", "nonrheumatic mitral valve")),
    ("cardiomyopathy", ("cardiomyopathy",)),
    ("cvd_death", ("cardiovascular death", "cvd death")),
)

STRICT_ABNORMAL_ECG_TERMS: Sequence[str] = (
    "abnormal ecg",
    "borderline ecg",
    "infarct",
    "ischemia",
    "hypertrophy",
    "bundle branch block",
    "axis deviation",
    "atrial fibrillation",
    "atrial flutter",
    "pacemaker",
    "premature",
    "ectopic",
    "tachycardia",
    "bradycardia",
    "low voltage",
    "st abnormality",
    "nonspecific st",
    "t wave",
    "qrs",
    "qt",
)


def _stable_int(value: object, default: int = -1) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:15], 16)


def _safe_float(value: object, default: float = -1.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _encode_sex(value: object) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -1
    text = str(value).strip().lower()
    if text in {"m", "male", "1"}:
        return 1
    if text in {"f", "female", "0"}:
        return 0
    return -1


class ECGDataset(Dataset):
    def __init__(
        self,
        txt_path: str,
        ecg_path: str = DEFAULT_ECG_PATH,
        lead_num: int = 12,
        disease_definitions: Sequence[Tuple[str, Sequence[str]]] = DEFAULT_DISEASE_DEFINITIONS,
        window_size: int = 5000,
        target_fs: float = 500.0,
    ) -> None:
        self.window_size = window_size
        self.fs = target_fs
        self.ecg_path = ecg_path
        self.lead_num = lead_num
        self.disease_definitions = tuple(disease_definitions)
        self.disease_names = [name for name, _ in self.disease_definitions]

        self.data = pd.read_csv(txt_path)
        required = ["HashFileName", "deid_t_diagnosis_original"]
        self.data = self.data.dropna(subset=[col for col in required if col in self.data.columns]).reset_index(drop=True)
        self.columns = {column.lower(): column for column in self.data.columns}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.data.iloc[idx]
        hash_file_name = str(self._value(row, ["HashFileName"]))
        diag = str(self._value(row, ["deid_t_diagnosis_original", "deid_t_diagnosis"], ""))
        diag_structured = str(self._value(row, ["deid_t_diagnosis"], ""))
        icd_text = str(self._value(row, ["ICD_text", "icd_text"], ""))

        txt = self._build_text(diag, icd_text)
        ecg, site = self._load_ecg(hash_file_name)
        disease_labels = self._disease_labels(icd_text)
        nsr = int(self._is_nsr(f"{diag_structured} {diag}"))
        normal_state = self._normal_state(diag, diag_structured, disease_labels, nsr)

        return {
            "ecg": ecg,
            "txt": txt,
            "disease_labels": torch.tensor(disease_labels, dtype=torch.float32),
            "normal_state": torch.tensor(normal_state, dtype=torch.long),
            "nsr": torch.tensor(nsr, dtype=torch.long),
            "patient_id": torch.tensor(
                _stable_int(self._value(row, ["BDSPPatientID", "patient_id", "PatientID", "bdsp_id"], idx)),
                dtype=torch.long,
            ),
            "sample_id": torch.tensor(_stable_int(hash_file_name, idx), dtype=torch.long),
            "age": torch.tensor(_safe_float(self._value(row, ["age", "Age"], -1.0)), dtype=torch.float32),
            "sex": torch.tensor(_encode_sex(self._value(row, ["sex", "Sex", "gender", "Gender"], None)), dtype=torch.long),
            "heart_rate": torch.tensor(
                _safe_float(self._value(row, ["heart_rate", "HR", "hr", "ventricular_rate"], -1.0)),
                dtype=torch.float32,
            ),
            "site": torch.tensor(
                _stable_int(self._value(row, ["site", "Site", "center", "Center"], site), site),
                dtype=torch.long,
            ),
            "has_disease_labels": torch.tensor(int("icd_text" in self.columns), dtype=torch.long),
        }

    def _value(self, row: pd.Series, candidates: Iterable[str], default: object = None) -> object:
        for candidate in candidates:
            column = candidate if candidate in row.index else self.columns.get(candidate.lower())
            if column is None:
                continue
            value = row[column]
            if pd.isna(value):
                continue
            return value
        return default

    def _build_text(self, diag: str, icd_text: str) -> str:
        if icd_text.lower() == "nan" or not icd_text.strip():
            return diag
        return f"{diag}; {icd_text}"

    def _disease_labels(self, icd_text: str) -> np.ndarray:
        text = "" if icd_text.lower() == "nan" else icd_text.lower()
        return np.asarray(
            [float(any(keyword in text for keyword in keywords)) for _, keywords in self.disease_definitions],
            dtype=np.float32,
        )

    def _is_nsr(self, diagnosis_text: str) -> bool:
        text = diagnosis_text.lower()
        return any(
            phrase in text
            for phrase in (
                "normal sinus rhythm",
                "sinus rhythm",
                "sinus bradycardia",
                "sinus tachycardia",
            )
        )

    def _normal_state(self, diag: str, diag_structured: str, disease_labels: np.ndarray, nsr: int) -> int:
        if not nsr:
            return -1
        if disease_labels.sum() > 0:
            return int(np.flatnonzero(disease_labels)[0]) + 1
        text = f"{diag_structured} {diag}".lower()
        has_normal_phrase = "normal ecg" in text or "otherwise normal ecg" in text
        has_abnormal_phrase = any(term in text for term in STRICT_ABNORMAL_ECG_TERMS)
        return 0 if has_normal_phrase and not has_abnormal_phrase else -1

    def _load_ecg(self, hash_file_name: str) -> Tuple[torch.Tensor, int]:
        base_path, site = self._find_record(hash_file_name)
        if base_path is None:
            base_path, site = DEFAULT_FALLBACK_RECORD, 1
            print(f"No ECG file found for {hash_file_name}; using fallback record.")

        try:
            ecg_data, sample_rate = self._read_record(base_path)
        except Exception as exc:
            print(f"Error reading ECG file {base_path}: {exc}; using fallback record.")
            ecg_data, sample_rate = self._read_record(DEFAULT_FALLBACK_RECORD)
            site = 1

        ecg_data = np.asarray(ecg_data, dtype=float)
        ecg_data = self._ensure_channel_first(ecg_data)
        ecg_data = self._select_leads(ecg_data)
        ecg_data = self._normalize(ecg_data)
        ecg_data = self._preprocess(ecg_data, sample_rate)
        return torch.tensor(ecg_data, dtype=torch.float32), site

    def _find_record(self, hash_file_name: str) -> Tuple[Optional[str], int]:
        for site_idx in range(1, 5):
            site_dir = f"S{site_idx:04d}"
            for year in range(1987, 2027):
                for month in range(1, 13):
                    base_path = os.path.join(self.ecg_path, site_dir, str(year), f"{month:02d}", hash_file_name)
                    if os.path.exists(f"{base_path}.mat"):
                        return base_path, site_idx
        return None, -1

    def _read_record(self, base_path: str) -> Tuple[np.ndarray, int]:
        mat_data = loadmat(f"{base_path}.mat")
        sample_rate = 500
        hea_path = f"{base_path}.hea"
        if os.path.exists(hea_path):
            with open(hea_path, "r") as hea_file:
                elements = hea_file.readline().strip().split()
                if len(elements) > 2:
                    sample_rate = int(float(elements[2]))
        return mat_data["val"], sample_rate

    def _ensure_channel_first(self, ecg_data: np.ndarray) -> np.ndarray:
        if ecg_data.ndim != 2:
            raise ValueError(f"Expected a 2D ECG array, got shape {ecg_data.shape}.")
        if ecg_data.shape[0] > ecg_data.shape[1] and ecg_data.shape[1] <= 12:
            ecg_data = ecg_data.T
        return ecg_data

    def _select_leads(self, ecg_data: np.ndarray) -> np.ndarray:
        if self.lead_num == 1:
            original_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            candidate_leads = ["I", "aVL", "-aVR", "II", "-III", "aVF", "-aVF"]
            selected = np.random.choice(candidate_leads)
            sign = -1 if selected.startswith("-") else 1
            lead = selected[1:] if selected.startswith("-") else selected
            lead_idx = original_leads.index(lead)
            return sign * ecg_data[lead_idx : lead_idx + 1]

        if ecg_data.shape[0] >= self.lead_num:
            return ecg_data[: self.lead_num]

        pad = np.zeros((self.lead_num - ecg_data.shape[0], ecg_data.shape[1]))
        return np.concatenate([ecg_data, pad], axis=0)

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def _resample_unequal(self, values: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
        if fs_in == 0 or fs_in == fs_out:
            return values
        if 2 * fs_out == fs_in:
            return values[::2]
        duration = len(values) / fs_in
        x_old = np.linspace(0, 1, num=len(values), endpoint=True)
        x_new = np.linspace(0, 1, num=int(duration * fs_out), endpoint=True)
        return interp1d(x_old, values, kind="linear")(x_new)

    def _preprocess(self, ecg_data: np.ndarray, sample_rate: int) -> np.ndarray:
        processed = []
        for lead in ecg_data:
            if sample_rate != self.fs:
                lead = self._resample_unequal(lead, int(sample_rate), int(self.fs))
            lead = notch_filter(lead, self.fs, 60, verbose="ERROR")
            lead = filter_data(lead, self.fs, 0.5, 50, verbose="ERROR")
            processed.append(lead)

        out = np.asarray(processed)
        length = out.shape[1]
        if length > self.window_size:
            start = (length - self.window_size) // 2
            out = out[:, start : start + self.window_size]
        elif length < self.window_size:
            pad = np.zeros((out.shape[0], self.window_size - length))
            out = np.concatenate([out, pad], axis=1)
        return out


def load_data(
    txt_path: str,
    ecg_path: str = DEFAULT_ECG_PATH,
    batch_size: int = 128,
    lead_num: int = 12,
    num_workers: Optional[int] = None,
    shuffle: bool = True,
) -> data.DataLoader:
    dataset = ECGDataset(txt_path=txt_path, ecg_path=ecg_path, lead_num=lead_num)
    workers = min(8, os.cpu_count() or 1) if num_workers is None else num_workers
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_clip(
    model_path: Optional[str] = None,
    pretrained: bool = False,
    context_length: int = 77,
    lead_num: int = 12,
    num_diseases: int = 0,
    device: Optional[torch.device] = None,
) -> CLIP:
    params = {
        "context_length": context_length,
        "vocab_size": 49408,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
        "lead_num": lead_num,
        "num_diseases": num_diseases,
    }

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pretrained:
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        print("Loaded pretrained CLIP weights.")
    else:
        model = CLIP(**params)
        print("Loaded AlphaECG CLIP model.")

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    return model


def preprocess_text(texts: List[str], model: CLIP) -> torch.LongTensor:
    tokenizer = SimpleTokenizer()
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(str(text)) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), model.context_length, dtype=torch.long)

    for idx, tokens in enumerate(all_tokens):
        if len(tokens) > model.context_length:
            tokens = tokens[: model.context_length]
            tokens[model.context_length - 1] = eot_token
        result[idx, : len(tokens)] = torch.tensor(tokens)
    return result
