"""
Inference on an unseen patch (south of the 10 student training patches) for the
Earthformer-Earthformer distilled model. Student was trained with 10 patches
striding south by 30% of latitude range (1.5 deg). Unseen patch = one more
stride south (lat 0.625-5.625, lon 65.625-72.375). Loads student from distillation
checkpoint and plots prediction vs actual.
"""
import os
import sys
import argparse
import numpy as np
import torch
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from sklearn.metrics import mean_squared_error

_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)
_exps_dir = os.path.join(_curr_dir, "experiments")

from train_cuboid_sst_distill import CuboidDistillPLModule, build_cuboid_from_oc
from train_cuboid_sst import CuboidSSTPLModule


# Teacher patch and stride (must match sst_distill_earthformer.yaml)
TEACHER_LAT = (15.625, 20.625)
TEACHER_LON = (65.625, 72.375)
PATCH_LAT_DEG = 5.0
PATCH_LON_DEG = 6.75
STRIDE_LAT_FRACTION = 0.30
NUM_STRIDE_PATCHES = 10
STRIDE_DEG = STRIDE_LAT_FRACTION * PATCH_LAT_DEG  # 1.5

# Unseen patch: one stride south of the last (10th) training patch
# Last training patch: 15.625 - 9*1.5 = 2.125  ->  2.125 to 7.125
# Unseen: 2.125 - 1.5 = 0.625  ->  0.625 to 5.625
UNSEEN_LAT = (TEACHER_LAT[0] - NUM_STRIDE_PATCHES * STRIDE_DEG, TEACHER_LAT[1] - NUM_STRIDE_PATCHES * STRIDE_DEG)
UNSEEN_LON = TEACHER_LON


def _resize_patch(data: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize (T, H, W) to (T, target_h, target_w)."""
    t, h, w = data.shape
    if h == target_h and w == target_w:
        return data
    out = np.zeros((t, target_h, target_w), dtype=data.dtype)
    for i in range(t):
        out[i] = _resize_2d(data[i], target_h, target_w)
    return out


def _resize_2d(x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = x.shape
    scale_h, scale_w = h / target_h, w / target_w
    out = np.zeros((target_h, target_w), dtype=x.dtype)
    for i in range(target_h):
        for j in range(target_w):
            i0 = int(i * scale_h)
            i1 = min(int((i + 1) * scale_h), h)
            j0 = int(j * scale_w)
            j1 = min(int((j + 1) * scale_w), w)
            out[i, j] = np.nanmean(x[i0:i1, j0:j1])
    return out


def get_parser():
    p = argparse.ArgumentParser(description="Inference on unseen south patch for distilled student")
    p.add_argument("--cfg", type=str, default=os.path.join(_curr_dir, "sst_distill_earthformer.yaml"), help="Distillation config YAML")
    p.add_argument("--ckpt", type=str, required=True, help="Path to distillation checkpoint (e.g. experiments/sst_distill_run/checkpoints/last.ckpt)")
    p.add_argument("--data_root", type=str, default=None, help="Override data root (default: from config)")
    p.add_argument("--filename", type=str, default="sst.week.mean.nc")
    p.add_argument("--output", type=str, default="unseen_south_patch_prediction.png", help="Output plot path")
    p.add_argument("--test_start_year", type=int, default=2021, help="Start year for test period")
    return p


def main():
    args = get_parser().parse_args()
    oc = OmegaConf.load(open(args.cfg, "r"))
    cfg_dir = os.path.dirname(os.path.abspath(args.cfg))
    teacher_cfg_path = oc.teacher_cfg_path
    if not os.path.isabs(teacher_cfg_path):
        teacher_cfg_path = os.path.join(cfg_dir, teacher_cfg_path)
    teacher_ckpt = oc.teacher_ckpt_path
    if not os.path.isabs(teacher_ckpt):
        teacher_ckpt = os.path.join(_exps_dir, teacher_ckpt)

    data_root = args.data_root or oc.dataset.get("data_root", "/content/data")
    in_len = int(oc.dataset.get("in_len", 12))
    out_len = int(oc.dataset.get("out_len", 12))
    train_end_year = int(oc.dataset.get("train_end_year", 2015))
    val_end_year = int(oc.dataset.get("val_end_year", 2020))
    file_path = os.path.join(data_root, args.filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data not found: {file_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading distillation checkpoint (teacher + student)...")
    pl_module = CuboidDistillPLModule.load_from_checkpoint(
        args.ckpt,
        teacher_ckpt_path=teacher_ckpt,
        oc_file=teacher_cfg_path,
        save_dir=os.path.dirname(os.path.dirname(args.ckpt)),
    )
    student = pl_module.student.to(device).eval()

    print("Loading SST and computing normalization from teacher patch...")
    ds = xr.open_dataset(file_path)
    train_slice = slice(None, str(train_end_year))
    ds_teacher = ds.sel(lat=slice(*TEACHER_LAT), lon=slice(*TEACHER_LON))
    train_teacher = ds_teacher["sst"].sel(time=train_slice).values.astype(np.float32)
    mean = float(np.nanmean(train_teacher))
    std = float(np.nanstd(train_teacher))
    if std < 1e-8:
        std = 1.0

    print(f"Unseen patch: lat {UNSEEN_LAT}, lon {UNSEEN_LON}")
    ds_unseen = ds.sel(lat=slice(UNSEEN_LAT[0], UNSEEN_LAT[1]), lon=slice(UNSEEN_LON[0], UNSEEN_LON[1]))
    test_slice = slice(str(args.test_start_year), None)
    unseen_raw = ds_unseen["sst"].sel(time=test_slice).values.astype(np.float32)
    unseen_raw = np.nan_to_num(unseen_raw, nan=mean)
    time_index = ds_unseen["sst"].sel(time=test_slice).get_index("time")
    ds.close()

    # Normalize and add channel; resize to 21x28 to match model
    unseen_norm = (unseen_raw - mean) / std
    target_h, target_w = 21, 28
    if unseen_norm.shape[1] != target_h or unseen_norm.shape[2] != target_w:
        unseen_norm = _resize_patch(unseen_norm, target_h, target_w)
    unseen_norm = unseen_norm[:, np.newaxis, :, :]  # (T, 1, H, W)
    seq_len = in_len + out_len
    if len(unseen_norm) < seq_len:
        raise ValueError(f"Test period has {len(unseen_norm)} steps; need at least {seq_len}")

    # Build sequences for test (use last available chunk for a single plot, or average over several)
    num_test_seqs = len(unseen_norm) - seq_len + 1
    all_preds = []
    all_actuals = []
    with torch.no_grad():
        for i in range(num_test_seqs):
            chunk = unseen_norm[i : i + seq_len]
            x = torch.from_numpy(chunk[:in_len]).float().unsqueeze(0).to(device)  # (1, T_in, 1, H, W)
            y_true = chunk[in_len:seq_len]
            x = x.permute(0, 1, 3, 4, 2)  # (1, T_in, H, W, 1)
            pred = student(x)  # (1, T_out, H, W, 1)
            pred = pred.cpu().numpy().squeeze(0).squeeze(-1)  # (T_out, H, W)
            all_preds.append(pred)
            all_actuals.append(y_true.squeeze(1))  # (T_out, H, W)
    all_preds = np.concatenate(all_preds, axis=0)  # (N*T_out, H, W) then we take first step or mean
    all_actuals = np.concatenate(all_actuals, axis=0)

    # Spatially average for time series plot (first output timestep only for clarity)
    pred_first = all_preds[0::out_len].mean(axis=(1, 2))  # first predicted step per sequence
    actual_first = all_actuals[0::out_len].mean(axis=(1, 2))
    # Align lengths
    n = min(len(pred_first), len(actual_first))
    pred_first = pred_first[:n]
    actual_first = actual_first[:n]
    times = time_index[in_len : in_len + n]

    pred_deg = pred_first * std + mean
    actual_deg = actual_first * std + mean
    mse = mean_squared_error(actual_deg, pred_deg)
    print(f"Unseen patch MSE (spatially averaged, first output step): {mse:.4f}")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times, actual_deg, label="Actual", color="blue", linewidth=1.5)
    ax.plot(times, pred_deg, label="Student predicted", color="red", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.set_title(f"Unseen patch (lat {UNSEEN_LAT[0]:.2f}-{UNSEEN_LAT[1]:.2f}, lon {UNSEEN_LON[0]:.2f}-{UNSEEN_LON[1]:.2f}°)\n"
                 f"Spatially averaged SST – first output step | MSE = {mse:.4f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("SST (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = args.output
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
