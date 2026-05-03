from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from pretrain_code.data import PatientECGDataset, collate_patient_batch
from pretrain_code.distributed import cleanup_distributed, is_main_process, setup_distributed, unwrap_model
from pretrain_code.models import ECGPatientModel, ECGTokenEncoder


def encoder_config(encoder: ECGTokenEncoder) -> dict[str, int]:
    return {
        "lead_num": encoder.lead_num,
        "d_model": encoder.d_model,
        "n_heads": encoder.n_heads,
        "n_layers": encoder.n_layers,
        "patch_samples": encoder.patch_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 patient-aware ECG supervised alignment.")
    parser.add_argument("--manifest", required=True, help="CSV with ECG paths, patient IDs, and label columns.")
    parser.add_argument("--label_columns", required=True, help="Comma-separated multi-label targets, or all_icd_i.")
    parser.add_argument("--ecg_root", default="")
    parser.add_argument("--ecg_layout", default="auto", choices=("auto", "heedb_wfdb", "flat"))
    parser.add_argument("--path_index", default="", help="Optional CSV cache mapping record IDs to ECG file paths.")
    parser.add_argument("--teacher_checkpoint", default="", help="Stage 1 checkpoint containing teacher_encoder.")
    parser.add_argument("--output_dir", default="outputs/ecg_stage2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_records_per_case", type=int, default=8)
    parser.add_argument("--lead_num", type=int, default=12)
    parser.add_argument("--window_size", type=int, default=5000)
    parser.add_argument("--target_fs", type=float, default=500.0)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--patch_samples", type=int, default=250)
    parser.add_argument("--patient_layers", type=int, default=3)
    parser.add_argument("--freeze_record_encoder", action="store_true")
    parser.add_argument("--save_every_epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_filter", action="store_true")
    return parser.parse_args()


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    label_mask = label_mask.to(device=logits.device, dtype=torch.bool)
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    if not label_mask.any():
        return logits.sum() * 0.0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return loss[label_mask].mean()


def build_encoder(args: argparse.Namespace) -> ECGTokenEncoder:
    checkpoint = None
    checkpoint_args = {}
    if args.teacher_checkpoint:
        checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu")
        checkpoint_args = checkpoint.get("args", {})
    encoder = ECGTokenEncoder(
        lead_num=int(checkpoint_args.get("lead_num", args.lead_num)),
        d_model=int(checkpoint_args.get("d_model", args.d_model)),
        n_heads=int(checkpoint_args.get("n_heads", args.n_heads)),
        n_layers=int(checkpoint_args.get("n_layers", args.n_layers)),
        patch_samples=int(checkpoint_args.get("patch_samples", args.patch_samples)),
    )
    if checkpoint is not None:
        state = checkpoint.get("teacher_encoder", checkpoint.get("state_dict", checkpoint))
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing encoder keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"Warning: unexpected encoder keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return encoder


def save_checkpoint(model: ECGPatientModel, optimizer: torch.optim.Optimizer, path: str, epoch: int, args: argparse.Namespace) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    saved_args = vars(args).copy()
    saved_args.update(encoder_config(model.record_encoder))
    saved_args["output_dim"] = model.head[-1].out_features if isinstance(model.head, torch.nn.Sequential) else model.head.out_features
    torch.save(
        {
            "epoch": epoch,
            "args": saved_args,
            "record_encoder": model.record_encoder.state_dict(),
            "patient_aggregator": model.patient_aggregator.state_dict(),
            "head": model.head.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    dist_ctx = setup_distributed(args.device)
    args.device = str(dist_ctx.device)
    if is_main_process(dist_ctx):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dataset = PatientECGDataset(
        args.manifest,
        ecg_root=args.ecg_root or None,
        label_columns=args.label_columns,
        max_records_per_case=args.max_records_per_case,
        lead_num=args.lead_num,
        window_size=args.window_size,
        target_fs=args.target_fs,
        apply_filter=not args.no_filter,
        ecg_layout=args.ecg_layout,
        path_index=args.path_index or None,
        include_text=False,
    )
    label_columns = dataset.record_dataset.label_columns
    if not label_columns:
        raise ValueError("--label_columns did not resolve to any manifest columns.")
    args.label_columns = ",".join(label_columns)
    sampler = DistributedSampler(dataset, shuffle=True) if dist_ctx.enabled else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=dist_ctx.device.type == "cuda",
        collate_fn=collate_patient_batch,
    )
    encoder = build_encoder(args)
    model = ECGPatientModel(
        encoder,
        output_dim=len(label_columns),
        patient_layers=args.patient_layers,
        patient_heads=encoder.n_heads,
    ).to(dist_ctx.device)
    if args.freeze_record_encoder:
        for param in model.record_encoder.parameters():
            param.requires_grad = False
    if dist_ctx.enabled:
        model = DistributedDataParallel(
            model,
            device_ids=[dist_ctx.local_rank] if dist_ctx.device.type == "cuda" else None,
            output_device=dist_ctx.local_rank if dist_ctx.device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        loop = tqdm(loader, desc=f"stage2 epoch {epoch}", disable=not is_main_process(dist_ctx))
        loss_sum = 0.0
        steps = 0
        for batch in loop:
            records = batch["records"].to(dist_ctx.device, non_blocking=True)
            record_padding_mask = batch["record_padding_mask"].to(dist_ctx.device, non_blocking=True)
            labels = batch["labels"].to(dist_ctx.device, non_blocking=True)
            label_mask = batch["label_mask"].to(dist_ctx.device, non_blocking=True)
            out = model(records, record_padding_mask=record_padding_mask)
            loss = masked_bce_loss(out["logits"], labels, label_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            optimizer.step()
            loss_sum += float(loss.item())
            steps += 1
            if is_main_process(dist_ctx):
                loop.set_postfix(loss=f"{loss.item():.4f}")
        if is_main_process(dist_ctx):
            print(f"epoch={epoch} mean_loss={loss_sum / max(1, steps):.6f}")
        if is_main_process(dist_ctx) and args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
            save_checkpoint(unwrap_model(model), optimizer, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt"), epoch, args)

    if is_main_process(dist_ctx):
        save_checkpoint(unwrap_model(model), optimizer, os.path.join(args.output_dir, "checkpoint_final.pt"), args.epochs, args)
    cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
