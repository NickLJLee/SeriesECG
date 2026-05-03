# ECG-MOOZY Rewrite

This folder keeps the original ECG-CLIP baseline and adds a patient-first ECG
foundation-model path inspired by MOOZY.

## Data Manifest

Use a CSV with at least:

```text
ecg_path,patient_id,record_id,atrial_fibrillation,bundle_branch_block,myocardial_infarction
sample_001.mat,P001,R001,1,0,0
sample_002.mat,P001,R002,1,0,0
```

Supported ECG files: `.mat`, `.npy`, `.npz`, `.h5`, `.csv`, `.txt`.

## Stage 1: Masked Self-Distillation

```bash
python -m pretrain_code.train_stage1 \
  --manifest /path/to/manifest.csv \
  --ecg_root /path/to/ecg/files \
  --output_dir outputs/ecg_stage1
```

The model uses a lead-aware time-token transformer. It masks contiguous ECG
time tokens, uses an EMA teacher, and trains with CLS distillation plus masked
token prediction.

## HEEDB Multi-GPU Pretraining

Use the repository-level launcher for the full HEEDB Stage 1 -> Stage 2
pipeline:

```bash
setsid env CONDA_ENV=ecg bash /data1/1shared/lijun/ecg/SeriesECG/run_train.sh \
  > /data1/1shared/lijun/ecg/SeriesECG/outputs/heedb_pretrain/launcher.out 2>&1 < /dev/null &
```

Defaults in `run_train.sh`:

```bash
MULTI_GPU=1
GPU_IDS=1,2,3,4,5,6
STAGE1_BATCH_SIZE=128
STAGE2_BATCH_SIZE=32
```

`STAGE1_BATCH_SIZE` and `STAGE2_BATCH_SIZE` are per-GPU batch sizes under
DDP. With six GPUs, the effective global batch sizes are:

```text
Stage 1: 128 x 6 = 768
Stage 2:  32 x 6 = 192
```

The current default avoids GPU 0 because `nvidia-smi` showed another Python
process there, and uses six RTX 4090 cards. The defaults are 4x the previous
per-GPU batch sizes (`32 -> 128` for Stage 1 and `8 -> 32` for Stage 2). To
choose different cards:

```bash
GPU_IDS=2,3,4,5,6,7 bash run_train.sh
```

To increase or reduce per-GPU batch size:

```bash
STAGE1_BATCH_SIZE=64 STAGE2_BATCH_SIZE=16 bash run_train.sh
```

For single-GPU pretraining:

```bash
MULTI_GPU=0 GPU_IDS=1 bash run_train.sh
```

For CPU/debug runs:

```bash
DEVICE=cpu MULTI_GPU=0 bash run_train.sh
```

## Stage 2: Patient-Aware Supervised Alignment

```bash
python -m pretrain_code.train_stage2 \
  --manifest /path/to/manifest.csv \
  --ecg_root /path/to/ecg/files \
  --label_columns atrial_fibrillation,bundle_branch_block,myocardial_infarction \
  --teacher_checkpoint outputs/ecg_stage1/checkpoint_final.pt \
  --output_dir outputs/ecg_stage2
```

Each patient/case contains one or more ECG records. The record encoder produces
one token per ECG, then a patient transformer with a learnable `[CASE]` token
produces the final patient embedding.

## Encode

```bash
python -m pretrain_code.encode \
  record_1.mat record_2.mat \
  --checkpoint outputs/ecg_stage2/checkpoint_final.pt \
  --output case_embedding.npy
```
