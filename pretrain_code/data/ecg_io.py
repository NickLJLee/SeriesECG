from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, resample_poly


DEFAULT_LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
DEFAULT_SIGNAL_KEYS = ("val", "ecg", "signal", "signals", "data", "x")


def read_header_sample_rate(path: str | os.PathLike[str], default_fs: float = 500.0) -> float:
    """Read WFDB-like `.hea` sampling rate when available."""
    base = Path(path)
    hea_path = base.with_suffix(".hea")
    if not hea_path.exists() and base.suffix:
        hea_path = Path(str(base) + ".hea")
    if not hea_path.exists():
        return float(default_fs)

    try:
        with hea_path.open("r", encoding="utf-8", errors="ignore") as handle:
            first = handle.readline().strip().split()
        if len(first) > 2:
            return float(first[2])
    except OSError:
        pass
    return float(default_fs)


def _first_numeric_array(values: Iterable[object]) -> np.ndarray | None:
    for value in values:
        arr = np.asarray(value)
        if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number):
            return arr
    return None


def load_ecg_array(path: str | os.PathLike[str], signal_key: str | None = None) -> np.ndarray:
    """Load ECG samples from `.mat`, `.npy`, `.npz`, `.h5`, or plain CSV/TXT."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".mat":
        mat = loadmat(path)
        if signal_key and signal_key in mat:
            return np.asarray(mat[signal_key])
        for key in DEFAULT_SIGNAL_KEYS:
            if key in mat:
                return np.asarray(mat[key])
        arr = _first_numeric_array(v for k, v in mat.items() if not k.startswith("__"))
        if arr is None:
            raise ValueError(f"No numeric ECG array found in {path}.")
        return arr

    if suffix == ".npy":
        return np.load(path)

    if suffix == ".npz":
        archive = np.load(path)
        if signal_key and signal_key in archive:
            return archive[signal_key]
        for key in DEFAULT_SIGNAL_KEYS:
            if key in archive:
                return archive[key]
        if not archive.files:
            raise ValueError(f"No arrays found in {path}.")
        return archive[archive.files[0]]

    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as handle:
            if signal_key and signal_key in handle:
                return handle[signal_key][()]
            for key in DEFAULT_SIGNAL_KEYS:
                if key in handle:
                    return handle[key][()]
            for key in handle.keys():
                value = handle[key]
                if hasattr(value, "shape") and len(value.shape) >= 2:
                    return value[()]
        raise ValueError(f"No 2D ECG dataset found in {path}.")

    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        return np.loadtxt(path, delimiter=delimiter)

    raise ValueError(f"Unsupported ECG file type: {path}")


def ensure_channel_first(ecg: np.ndarray, max_leads: int = 16) -> np.ndarray:
    """Return ECG in `[lead, time]` shape."""
    ecg = np.asarray(ecg, dtype=np.float32)
    ecg = np.squeeze(ecg)
    if ecg.ndim != 2:
        raise ValueError(f"Expected a 2D ECG array, got shape {ecg.shape}.")
    if ecg.shape[0] > ecg.shape[1] and ecg.shape[1] <= max_leads:
        ecg = ecg.T
    elif ecg.shape[0] > max_leads and ecg.shape[1] <= max_leads:
        ecg = ecg.T
    return np.ascontiguousarray(ecg, dtype=np.float32)


def select_or_pad_leads(ecg: np.ndarray, lead_num: int = 12) -> np.ndarray:
    """Select the first `lead_num` leads or zero-pad missing leads."""
    if ecg.shape[0] >= lead_num:
        return ecg[:lead_num]
    pad = np.zeros((lead_num - ecg.shape[0], ecg.shape[1]), dtype=ecg.dtype)
    return np.concatenate([ecg, pad], axis=0)


def resample_ecg(ecg: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if not fs_in or abs(float(fs_in) - float(fs_out)) < 1e-6:
        return ecg
    fs_in_i = int(round(float(fs_in)))
    fs_out_i = int(round(float(fs_out)))
    divisor = math.gcd(fs_in_i, fs_out_i)
    up = fs_out_i // divisor
    down = fs_in_i // divisor
    return resample_poly(ecg, up=up, down=down, axis=1).astype(np.float32)


def filter_ecg(
    ecg: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 50.0,
    notch_hz: float = 60.0,
) -> np.ndarray:
    """Apply conservative ECG filtering. Set cutoffs to <=0 to disable."""
    out = ecg.astype(np.float32, copy=True)
    nyq = float(fs) / 2.0

    if notch_hz and 0.0 < notch_hz < nyq:
        b_notch, a_notch = iirnotch(w0=notch_hz / nyq, Q=30.0)
        out = filtfilt(b_notch, a_notch, out, axis=1).astype(np.float32)

    if lowcut and highcut and 0.0 < lowcut < highcut < nyq:
        b_band, a_band = butter(3, [lowcut / nyq, highcut / nyq], btype="bandpass")
        out = filtfilt(b_band, a_band, out, axis=1).astype(np.float32)
    return out


def crop_or_pad(ecg: np.ndarray, window_size: int, crop: str = "center") -> np.ndarray:
    length = ecg.shape[1]
    if length == window_size:
        return ecg
    if length > window_size:
        if crop == "random":
            start = np.random.randint(0, length - window_size + 1)
        else:
            start = (length - window_size) // 2
        return ecg[:, start : start + window_size]
    pad = np.zeros((ecg.shape[0], window_size - length), dtype=ecg.dtype)
    return np.concatenate([ecg, pad], axis=1)


def normalize_ecg(ecg: np.ndarray, mode: str = "per_lead") -> np.ndarray:
    if mode == "none":
        return ecg.astype(np.float32)
    if mode == "global":
        return ((ecg - ecg.mean()) / (ecg.std() + 1e-8)).astype(np.float32)
    mean = ecg.mean(axis=1, keepdims=True)
    std = ecg.std(axis=1, keepdims=True)
    return ((ecg - mean) / (std + 1e-8)).astype(np.float32)


def preprocess_record(
    path: str | os.PathLike[str],
    *,
    signal_key: str | None = None,
    sample_rate: float | None = None,
    target_fs: float = 500.0,
    lead_num: int = 12,
    window_size: int = 5000,
    crop: str = "center",
    normalize: str = "per_lead",
    apply_filter: bool = True,
    default_fs: float = 500.0,
) -> np.ndarray:
    """Load and preprocess one ECG record as `[lead_num, window_size]` float32."""
    ecg = load_ecg_array(path, signal_key=signal_key)
    ecg = ensure_channel_first(ecg)
    ecg = select_or_pad_leads(ecg, lead_num=lead_num)
    fs = float(sample_rate) if sample_rate is not None and sample_rate > 0 else read_header_sample_rate(path, default_fs)
    ecg = resample_ecg(ecg, fs_in=fs, fs_out=target_fs)
    if apply_filter:
        ecg = filter_ecg(ecg, fs=target_fs)
    ecg = crop_or_pad(ecg, window_size=window_size, crop=crop)
    return normalize_ecg(ecg, mode=normalize)

