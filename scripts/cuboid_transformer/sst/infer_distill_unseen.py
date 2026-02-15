"""
Inference on the westward (left) patch of the base teacher patch.
Supports both Earthformer and ConvLSTM student architectures.
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

# ... (Previous imports for your custom modules remain same) ...
from train_cuboid_sst_distill import CuboidDistillPLModule, build_cuboid_from_oc
from train_cuboid_sst import CuboidSSTPLModule

# Teacher patch configuration
TEACHER_LAT = (15.625, 20.625)
TEACHER_LON = (65.625, 72.375)
PATCH_LAT_DEG = 5.0
PATCH_LON_DEG = 6.75

# --- CHANGE 1: Define the Westward (Left) Patch ---
# We shift the longitude to the left (subtract)
WEST_LAT = TEACHER_LAT
WEST_LON = (TEACHER_LON[0] - PATCH_LON_DEG, TEACHER_LON[1] - PATCH_LON_DEG)

def _resize_patch(data: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    t, h, w = data.shape
    if h == target_h and w == target_w: return data
    out = np.zeros((t, target_h, target_w), dtype=data.dtype)
    for i in range(t):
        out[i] = _resize_2d(data[i], target_h, target_w)
    return out

def _resize_2d(x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Robust 2D resize with boundary safety."""
    h, w = x.shape
    scale_h, scale_w = h / target_h, w / target_w
    out = np.zeros((target_h, target_w), dtype=x.dtype)
    for i in range(target_h):
        for j in range(target_w):
            i0, i1 = int(i * scale_h), min(int((i + 1) * scale_h), h)
            j0, j1 = int(j * scale_w), min(int((j + 1) * scale_w), w)
            # Ensure window is not empty (Math major: non-empty support)
            i1 = max(i1, i0 + 1)
            j1 = max(j1, j0 + 1)
            out[i, j] = np.nanmean(x[i0:i1, j0:j1])
    return out

def get_parser():
    p = argparse.ArgumentParser(description="Inference on westward patch")
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, default="/content/data")
    p.add_argument("--filename", type=str, default="sst.week.mean.nc")
    p.add_argument("--output", type=str, default="west_patch_prediction.png")
    p.add_argument("--test_start_year", type=int, default=2021)
    return p

def main():
    args = get_parser().parse_args()
    oc = OmegaConf.load(open(args.cfg, "r"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {args.ckpt}")
    
    # Load module
    pl_module = CuboidDistillPLModule.load_from_checkpoint(
        args.ckpt,
        teacher_ckpt_path=os.path.join(_exps_dir, oc.teacher_ckpt_path),
        oc_file=os.path.join(os.path.dirname(args.cfg), oc.teacher_cfg_path),
    )
    student = pl_module.student.to(device).eval()

    # Determine Student Type
    is_convlstm = "ConvLSTM" in str(type(student))
    print(f"Detected Student Architecture: {'ConvLSTM' if is_convlstm else 'Earthformer'}")

    print("Computing normalization stats from base teacher region...")
    ds = xr.open_dataset(os.path.join(args.data_root, args.filename))
    train_teacher = ds.sel(lat=slice(*TEACHER_LAT), lon=slice(*TEACHER_LON))["sst"].sel(time=slice(None, "2015")).values
    mean, std = np.nanmean(train_teacher), np.nanstd(train_teacher)

    print(f"Extracting Westward Patch: Lat {WEST_LAT}, Lon {WEST_LON}")
    ds_west = ds.sel(lat=slice(*WEST_LAT), lon=slice(*WEST_LON))
    test_slice = slice(str(args.test_start_year), None)
    west_raw = np.nan_to_num(ds_west["sst"].sel(time=test_slice).values.astype(np.float32), nan=mean)
    time_index = ds_west["sst"].sel(time=test_slice).get_index("time")
    
    # Normalize and Resize
    west_norm = (west_raw - mean) / std
    west_norm = _resize_patch(west_norm, 21, 28)
    
    # Inference loop
    in_len, out_len = 12, 12
    num_test_seqs = len(west_norm) - (in_len + out_len) + 1
    all_preds = []
    all_actuals = []

    

    with torch.no_grad():
        for i in range(num_test_seqs):
            chunk = west_norm[i : i + in_len + out_len]
            x = torch.from_numpy(chunk[:in_len]).float().unsqueeze(0).to(device) # (1, T_in, H, W)
            
            # --- CHANGE 2: Handle Architecture Specific Dimensions ---
            if is_convlstm:
                # ConvLSTM expects (B, T, C, H, W)
                x = x.unsqueeze(2) 
            else:
                # Earthformer expects (B, T, H, W, C)
                x = x.unsqueeze(-1) 
            
            pred = student(x) 
            
            # Standardize output to (T_out, H, W)
            if is_convlstm:
                pred = pred.cpu().numpy().squeeze(0).squeeze(1)
            else:
                pred = pred.cpu().numpy().squeeze(0).squeeze(-1)
                
            all_preds.append(pred)
            all_actuals.append(chunk[in_len:])

    # Spatially average the first predicted timestep of each sequence
    pred_ts = np.concatenate(all_preds, axis=0)[0::out_len].mean(axis=(1, 2))
    actual_ts = np.concatenate(all_actuals, axis=0)[0::out_len].mean(axis=(1, 2))
    
    # Denormalize
    pred_deg = pred_ts * std + mean
    actual_deg = actual_ts * std + mean
    times = time_index[in_len : in_len + len(pred_deg)]
    mse = mean_squared_error(actual_deg, pred_deg)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(times, actual_deg, label="Actual (West Patch)", color="blue")
    plt.plot(times, pred_deg, label="Student Prediction", color="red", linestyle="--")
    plt.title(f"West Patch Generalization | MSE: {mse:.4f}")
    plt.ylabel("SST (°C)")
    plt.legend()
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
