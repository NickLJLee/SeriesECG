from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from pretrain_code.data import ECGManifestDataset, collate_record_batch
from pretrain_code.distributed import cleanup_distributed, is_main_process, setup_distributed, unwrap_model
from pretrain_code.models import ECGSSLModel, ECGTokenEncoder
from pretrain_code.models.ssl_model import cosine_teacher_momentum


def encoder_config(encoder: ECGTokenEncoder) -> dict[str, int]:
    return {
        "lead_num": encoder.lead_num,
        "d_model": encoder.d_model,
        "n_heads": encoder.n_heads,
        "n_layers": encoder.n_layers,
        "patch_samples": encoder.patch_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 ECG masked self-distillation pretraining.")
    parser.add_argument("--manifest", required=True, help="CSV with at least a path/ecg_path column.")
    parser.add_argument("--ecg_root", default="", help="Optional root directory for relative ECG paths.")
    parser.add_argument("--ecg_layout", default="auto", choices=("auto", "heedb_wfdb", "flat"))
    parser.add_argument("--path_index", default="", help="Optional CSV cache mapping record IDs to ECG file paths.")
    parser.add_argument("--output_dir", default="outputs/ecg_stage1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.04)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lead_num", type=int, default=12)
    parser.add_argument("--window_size", type=int, default=5000)
    parser.add_argument("--target_fs", type=float, default=500.0)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--patch_samples", type=int, default=250)
    parser.add_argument("--output_dim", type=int, default=8192)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_filter", action="store_true")
    return parser.parse_args()


def save_checkpoint(model: ECGSSLModel, optimizer: torch.optim.Optimizer, path: str, step: int, args: argparse.Namespace) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    saved_args = vars(args).copy()
    saved_args.update(encoder_config(model.student_encoder))
    torch.save(
        {
            "step": step,
            "args": saved_args,
            "student_encoder": model.student_encoder.state_dict(),
            "teacher_encoder": model.teacher_encoder.state_dict(),
            "student_head": model.student_head.state_dict(),
            "teacher_head": model.teacher_head.state_dict(),
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
    dataset = ECGManifestDataset(
        args.manifest,
        ecg_root=args.ecg_root or None,
        lead_num=args.lead_num,
        window_size=args.window_size,
        target_fs=args.target_fs,
        crop="random",
        apply_filter=not args.no_filter,
        ecg_layout=args.ecg_layout,
        path_index=args.path_index or None,
        include_text=False,
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if dist_ctx.enabled else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=dist_ctx.device.type == "cuda",
        collate_fn=collate_record_batch,
        drop_last=True,
    )
    encoder = ECGTokenEncoder(
        lead_num=args.lead_num,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        patch_samples=args.patch_samples,
    )
    model = ECGSSLModel(encoder, output_dim=args.output_dim).to(dist_ctx.device)
    if dist_ctx.enabled:
        model = DistributedDataParallel(
            model,
            device_ids=[dist_ctx.local_rank] if dist_ctx.device.type == "cuda" else None,
            output_device=dist_ctx.local_rank if dist_ctx.device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(loader) * args.epochs)
    global_step = 0

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        loop = tqdm(loader, desc=f"stage1 epoch {epoch}", disable=not is_main_process(dist_ctx))
        for batch in loop:
            ecg = batch["ecg"].to(dist_ctx.device, non_blocking=True)
            out = model(ecg)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            optimizer.step()
            unwrap_model(model).update_teacher(cosine_teacher_momentum(global_step, total_steps))
            if is_main_process(dist_ctx):
                loop.set_postfix(loss=f"{out['loss'].item():.4f}", mim=f"{out['loss_mim'].item():.4f}")
            global_step += 1
            if is_main_process(dist_ctx) and args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(
                    unwrap_model(model),
                    optimizer,
                    os.path.join(args.output_dir, f"checkpoint_step_{global_step}.pt"),
                    global_step,
                    args,
                )

    if is_main_process(dist_ctx):
        save_checkpoint(unwrap_model(model), optimizer, os.path.join(args.output_dir, "checkpoint_final.pt"), global_step, args)
    cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
