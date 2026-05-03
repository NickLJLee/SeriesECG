from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, resample_poly


DEFAULT_LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
DEFAULT_SIGNAL_KEYS = ("val", "ecg", "signal", "signals", "data", "x")
_GAIN_RE = re.compile(
    r"^(?P<gain>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)?"
    r"(?:\((?P<baseline>[-+]?\d+(?:\.\d*)?)\))?"
    r"(?:/(?P<units>\S+))?$"
)


@dataclass(frozen=True)
class WFDBSignal:
    lead_name: str
    gain: float | None = None
    baseline: float | None = None
    adc_zero: float | None = None
    units: str = ""


@dataclass(frozen=True)
class WFDBHeader:
    sample_rate: float
    signal_count: int | None
    signal_length: int | None
    signals: tuple[WFDBSignal, ...]


def _as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _as_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def paired_header_path(path: str | os.PathLike[str]) -> Path:
    base = Path(path)
    if base.suffix.lower() == ".hea":
        return base
    if base.suffix:
        return base.with_suffix(".hea")
    return Path(str(base) + ".hea")


def paired_signal_path(path: str | os.PathLike[str], suffix: str = ".mat") -> Path:
    base = Path(path)
    if base.suffix.lower() == ".hea":
        return base.with_suffix(suffix)
    if base.suffix:
        return base
    return Path(str(base) + suffix)


def read_wfdb_header(path: str | os.PathLike[str], default_fs: float = 500.0) -> WFDBHeader | None:
    """Parse the WFDB `.hea` sidecar when it is available."""
    hea_path = paired_header_path(path)
    if not hea_path.exists():
        return None

    try:
        lines = [
            line.strip()
            for line in hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except OSError:
        return None
    if not lines:
        return None

    first = lines[0].split()
    signal_count = _as_int(first[1]) if len(first) > 1 else None
    sample_rate = _as_float(first[2].split("/")[0]) if len(first) > 2 else None
    signal_length = _as_int(first[3]) if len(first) > 3 else None

    signals: list[WFDBSignal] = []
    for line in lines[1 : 1 + (signal_count or len(lines) - 1)]:
        parts = line.split()
        if len(parts) < 3:
            continue
        gain = None
        baseline = None
        units = ""
        match = _GAIN_RE.match(parts[2])
        if match:
            gain = _as_float(match.group("gain"))
            baseline = _as_float(match.group("baseline"))
            units = match.group("units") or ""
        adc_zero = _as_float(parts[4]) if len(parts) > 4 else None
        lead_name = " ".join(parts[8:]) if len(parts) > 8 else f"lead_{len(signals)}"
        signals.append(WFDBSignal(lead_name=lead_name, gain=gain, baseline=baseline, adc_zero=adc_zero, units=units))

    return WFDBHeader(
        sample_rate=float(sample_rate if sample_rate is not None else default_fs),
        signal_count=signal_count,
        signal_length=signal_length,
        signals=tuple(signals),
    )


def read_header_sample_rate(path: str | os.PathLike[str], default_fs: float = 500.0) -> float:
    """Read WFDB-like `.hea` sampling rate when available."""
    header = read_wfdb_header(path, default_fs=default_fs)
    return float(header.sample_rate) if header is not None else float(default_fs)


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

    if suffix == ".hea":
        path = paired_signal_path(path, suffix=".mat")
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


def scale_with_header(ecg: np.ndarray, header: WFDBHeader | None) -> np.ndarray:
    """Convert WFDB digital samples to physical units when gains are known."""
    if header is None or not header.signals:
        return ecg.astype(np.float32, copy=False)
    out = ecg.astype(np.float32, copy=True)
    for lead_idx, signal in enumerate(header.signals[: out.shape[0]]):
        if signal.gain is None or signal.gain == 0:
            continue
        baseline = signal.baseline
        if baseline is None:
            baseline = signal.adc_zero if signal.adc_zero is not None else 0.0
        out[lead_idx] = (out[lead_idx] - float(baseline)) / float(signal.gain)
    return out


def select_or_pad_leads(
    ecg: np.ndarray,
    lead_num: int = 12,
    source_leads: Sequence[str] | None = None,
    target_leads: Sequence[str] = DEFAULT_LEAD_ORDER,
) -> np.ndarray:
    """Select/reorder leads into the requested lead layout and zero-pad missing leads."""
    if source_leads:
        source_map = {str(lead): idx for idx, lead in enumerate(source_leads)}
        selected = []
        for lead in target_leads[:lead_num]:
            source_idx = source_map.get(str(lead))
            if source_idx is None or source_idx >= ecg.shape[0]:
                selected.append(np.zeros(ecg.shape[1], dtype=ecg.dtype))
            else:
                selected.append(ecg[source_idx])
        if len(selected) < lead_num:
            selected.extend(np.zeros(ecg.shape[1], dtype=ecg.dtype) for _ in range(lead_num - len(selected)))
        return np.stack(selected, axis=0).astype(ecg.dtype, copy=False)
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
    header = read_wfdb_header(path, default_fs=default_fs)
    ecg = load_ecg_array(path, signal_key=signal_key)
    ecg = ensure_channel_first(ecg)
    ecg = scale_with_header(ecg, header)
    source_leads = [signal.lead_name for signal in header.signals] if header is not None else None
    ecg = select_or_pad_leads(ecg, lead_num=lead_num, source_leads=source_leads)
    fs = float(sample_rate) if sample_rate is not None and sample_rate > 0 else (header.sample_rate if header else default_fs)
    ecg = resample_ecg(ecg, fs_in=fs, fs_out=target_fs)
    if apply_filter:
        ecg = filter_ecg(ecg, fs=target_fs)
    ecg = crop_or_pad(ecg, window_size=window_size, crop=crop)
    return normalize_ecg(ecg, mode=normalize)
