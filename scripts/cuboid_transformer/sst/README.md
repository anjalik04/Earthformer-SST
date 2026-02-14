# SST Forecasting with Earthformer

Sea surface temperature (SST) forecasting using the Cuboid Transformer and optional teacher-student distillation.

## Data

- **Source**: NOAA OISST (e.g. weekly mean `sst.week.mean.nc` from `https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.week.mean.nc`).
- Place the file under a data directory, e.g. `data/sst.week.mean.nc`, and set `dataset.data_root` in config to that directory (e.g. `"/content/data"` or `"data"`).

## 1. Train the teacher (single patch)

Train an Earthformer teacher on one fixed SST patch (default: lat 15.625–20.625, lon 65.625–72.375).

```bash
# From repo root with PYTHONPATH including Earthformer (or from Earthformer with PYTHONPATH=.).
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"
python scripts/cuboid_transformer/sst/train_cuboid_sst.py \
  --cfg scripts/cuboid_transformer/sst/sst.yaml \
  --save sst_colab_run_1 \
  --gpus 1
```

- Checkpoints and config are written to `scripts/cuboid_transformer/sst/experiments/<save>/`.
- Use `--ckpt_name model-epoch=019.ckpt` to resume from a specific checkpoint.

## 2. Earthformer–Earthformer distillation (teacher fixed, student on striding patches)

After the teacher is trained, train a student Earthformer that learns from the teacher’s **intermediate features** (encoder and decoder) while seeing **striding patches** over the globe. The student does not see the teacher’s patch distribution directly; it learns how the representation evolves across patches (distribution shift).

- **Offline**: teacher is loaded from checkpoint and frozen.
- **Feature distillation**: last two encoder blocks + decoder pre-projection (configurable).
- **Patch settings**: teacher patch is fixed; student patch grid uses the same patch size (default 5° lat × 6.75° lon) and stride (default same as patch size for non-overlapping tiles).

### Config

- **`sst_distill_earthformer.yaml`**: teacher path, dataset (patch datamodule), distillation weights, optimizer, trainer.

Important fields:

- `teacher_ckpt_path`: path to teacher checkpoint (e.g. `sst_colab_run_1/checkpoints/last.ckpt` under `experiments/`).
- `teacher_cfg_path`: path to the YAML used to train the teacher (e.g. `sst.yaml`); must be next to the distillation config or given as an absolute path.
- `dataset`: `SSTPatchDataModule`; set `data_root`, `in_len`, `out_len`, `batch_size`, `train_end_year`, `val_end_year`. Optional: `patch_lat_deg`, `patch_lon_deg`, `stride_lat_deg`, `stride_lon_deg`, and `student_lat_range` / `student_lon_range` to restrict the student grid.
- `distill`: `enc_block_indices` (e.g. `[-2, -1]`), `loss_weight_enc`, `loss_weight_dec`, `loss_weight_output`.

### Run distillation

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"
cd scripts/cuboid_transformer/sst
python train_cuboid_sst_distill.py \
  --cfg sst_distill_earthformer.yaml \
  --save sst_distill_run \
  --gpus 1
```

- Student checkpoints are saved under `experiments/<save>/checkpoints/` (e.g. `student-epoch=XXX.ckpt`, `last.ckpt`).
- To resume: `--ckpt_name last.ckpt` (or another filename in that folder).

### Patch and stride defaults

- **Teacher patch**: lat 15.625–20.625, lon 65.625–72.375 (single patch used for normalization and teacher input).
- **Student patches**: same size in degrees (`patch_lat_deg: 5.0`, `patch_lon_deg: 6.75`), stride equal to patch size by default (`stride_lat_deg: 5.0`, `stride_lon_deg: 6.75`) for a non-overlapping grid. Adjust in `sst_distill_earthformer.yaml` if you want overlapping or a different region (`student_lat_range` / `student_lon_range`).

### South-stride mode (10 patches, 30% latitude stride)

With `student_stride_south: true`, `student_stride_lat_fraction: 0.30`, and `student_num_stride_patches: 10`, the student is trained on 10 patches striding **south** from the teacher patch by 30% of the patch latitude range (1.5°) each time. Patches share the same longitude as the teacher (65.625–72.375°). Training uses the same `train_cuboid_sst_distill.py` and `sst_distill_earthformer.yaml`.

### Inference on an unseen patch (south)

After training, run inference on a **completely unseen** patch one stride further south (lat 0.625–5.625°, lon 65.625–72.375°) and save a prediction vs actual plot:

```bash
cd scripts/cuboid_transformer/sst
python infer_distill_unseen_south.py \
  --ckpt experiments/sst_distill_run/checkpoints/last.ckpt \
  --data_root /path/to/data \
  --output unseen_south_patch_prediction.png
```

- Uses test period from `--test_start_year` (default 2021). Override `--cfg` if your config path differs.

## Other scripts

- **ConvLSTM student (different setup)**: `train_student.py`, `train_student_multi_patch.py`, and `sst_distill.yaml` are for a ConvLSTM student, not the Earthformer–Earthformer setup above.
- **Evaluation / plotting**: `evaluate_student.py`, `evaluate_generalization.py`, `evaluate_multi_generalized_student.py`, `plot_predictions.py`, `plot_north_atlantic.py`, `compare_forecast_models.py`, `debug_data_consistency.py`.
