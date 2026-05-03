from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pretrain_code.data.ecg_io import preprocess_record
from pretrain_code.models import ECGPatientModel, ECGTokenEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode one patient case into an ECG-MOOZY embedding.")
    parser.add_argument("ecg_paths", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help=".npy output path.")
    parser.add_argument("--lead_num", type=int, default=12)
    parser.add_argument("--window_size", type=int, default=5000)
    parser.add_argument("--target_fs", type=float, default=500.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_filter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint.get("args", {})
    label_columns = [col.strip() for col in str(model_args.get("label_columns", "")).split(",") if col.strip()]
    output_dim = max(1, len(label_columns))
    encoder = ECGTokenEncoder(
        lead_num=int(model_args.get("lead_num", args.lead_num)),
        d_model=int(model_args.get("d_model", 384)),
        n_heads=int(model_args.get("n_heads", 6)),
        n_layers=int(model_args.get("n_layers", 6)),
        patch_samples=int(model_args.get("patch_samples", 250)),
    )
    model = ECGPatientModel(
        encoder,
        output_dim=output_dim,
        patient_layers=int(model_args.get("patient_layers", 3)),
        patient_heads=int(model_args.get("n_heads", 6)),
    )
    if "record_encoder" in checkpoint:
        model.record_encoder.load_state_dict(checkpoint["record_encoder"], strict=False)
    elif "teacher_encoder" in checkpoint:
        model.record_encoder.load_state_dict(checkpoint["teacher_encoder"], strict=False)
    if "patient_aggregator" in checkpoint:
        model.patient_aggregator.load_state_dict(checkpoint["patient_aggregator"], strict=False)
    model.to(args.device).eval()

    records = []
    for path in args.ecg_paths:
        ecg = preprocess_record(
            path,
            lead_num=args.lead_num,
            window_size=args.window_size,
            target_fs=args.target_fs,
            apply_filter=not args.no_filter,
        )
        records.append(torch.from_numpy(ecg))
    batch = torch.stack(records, dim=0).unsqueeze(0).to(args.device)
    padding_mask = torch.zeros(1, len(records), dtype=torch.bool, device=args.device)
    with torch.no_grad():
        embedding = model(batch, record_padding_mask=padding_mask)["embedding"].squeeze(0).cpu().numpy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embedding.astype(np.float32))


if __name__ == "__main__":
    main()
