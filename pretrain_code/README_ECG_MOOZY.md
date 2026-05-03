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

