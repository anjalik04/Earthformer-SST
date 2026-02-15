"""
Comprehensive inference testing on multiple unseen patches for the
Earthformer-Earthformer distilled model.

Tests multiple scenarios:
1. Southward patches with varying overlap (0%, 30%, 60%, 90%, 100%) with last training patch
2. Westward patches with varying overlap (0%, 30%, 60%, 90%, 100%) with last training patch
3. Random patches: one near training region, one far from training region
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
from typing import Tuple, List, Dict

_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)
_exps_dir = os.path.join(_curr_dir, "experiments")

from train_cuboid_sst_distill import CuboidDistillPLModule, build_cuboid_from_oc
from train_cuboid_sst import CuboidSSTPLModule


# Teacher patch and stride configuration (must match training config)
TEACHER_LAT = (15.625, 20.625)
TEACHER_LON = (65.625, 72.375)
PATCH_LAT_DEG = 5.0
PATCH_LON_DEG = 6.75
STRIDE_LAT_FRACTION = 0.30
NUM_STRIDE_PATCHES = 10
STRIDE_DEG = STRIDE_LAT_FRACTION * PATCH_LAT_DEG  # 1.5 degrees

# Last training patch (10th patch, 9 strides south of teacher)
LAST_PATCH_LAT = (TEACHER_LAT[0] - 9 * STRIDE_DEG, TEACHER_LAT[1] - 9 * STRIDE_DEG)
LAST_PATCH_LON = TEACHER_LON

# Global lat/lon bounds for safety checks
MIN_LAT, MAX_LAT = -90.0, 90.0
MIN_LON, MAX_LON = 0.0, 360.0


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
    """Simple 2D resize using area averaging."""
    h, w = x.shape
    scale_h, scale_w = h / target_h, w / target_w
    out = np.zeros((target_h, target_w), dtype=x.dtype)
    for i in range(target_h):
        for j in range(target_w):
            i0 = int(i * scale_h)
            i1 = min(int((i + 1) * scale_h), h)
            j0 = int(j * scale_w)
            j1 = min(int((j + 1) * scale_w), w)
            i1 = max(i1, i0 + 1)
            j1 = max(j1, j0 + 1)
            window = x[min(i0, h-1):min(i1, h), min(j0, w-1):min(j1, w)]
            if window.size == 0:
                out[i, j] = x[min(i0, h-1), min(j0, w-1)]
            else:
                out[i, j] = np.nanmean(window)
    return out


def compute_southward_patches() -> List[Tuple[str, Tuple[float, float], Tuple[float, float]]]:
    """
    Compute southward test patches with varying overlap percentages.
    Returns list of (name, lat_range, lon_range).
    """
    patches = []
    overlaps = [0, 30, 60, 90, 100]
    
    for overlap_pct in overlaps:
        # Calculate stride based on overlap
        # overlap_pct=0 means full stride (1.5 deg), overlap_pct=100 means 0 stride
        effective_stride = STRIDE_DEG * (1 - overlap_pct / 100.0)
        
        new_lat_start = LAST_PATCH_LAT[0] - effective_stride
        new_lat_end = LAST_PATCH_LAT[1] - effective_stride
        
        # Safety check
        if new_lat_start < MIN_LAT or new_lat_end < MIN_LAT:
            print(f"Warning: Southward {overlap_pct}% overlap patch goes below minimum latitude, skipping")
            continue
        
        patches.append((
            f"south_{overlap_pct}pct_overlap",
            (new_lat_start, new_lat_end),
            LAST_PATCH_LON
        ))
    
    return patches


def compute_westward_patches() -> List[Tuple[str, Tuple[float, float], Tuple[float, float]]]:
    """
    Compute westward test patches with varying overlap percentages.
    Returns list of (name, lat_range, lon_range).
    """
    patches = []
    overlaps = [0, 35, 60, 90, 100]
    
    for overlap_pct in overlaps:
        # Calculate stride based on overlap
        effective_stride = PATCH_LON_DEG * (1 - overlap_pct / 100.0)
        
        # Move west (decrease longitude)
        new_lon_start = LAST_PATCH_LON[0] - effective_stride
        new_lon_end = LAST_PATCH_LON[1] - effective_stride
        
        # Safety check
        if new_lon_start < MIN_LON:
            print(f"Warning: Westward {overlap_pct}% overlap patch goes below minimum longitude, skipping")
            continue
        
        patches.append((
            f"west_{overlap_pct}pct_overlap",
            LAST_PATCH_LAT,
            (new_lon_start, new_lon_end)
        ))
    
    return patches


def compute_random_patches(seed: int = 42) -> List[Tuple[str, Tuple[float, float], Tuple[float, float]]]:
    """
    Compute two random test patches:
    1. Near training region (within ±10 degrees of training patches)
    2. Far from training region (>20 degrees away)
    
    Returns list of (name, lat_range, lon_range).
    """
    np.random.seed(seed)
    patches = []
    
    # Training region center
    train_lat_center = (TEACHER_LAT[0] + LAST_PATCH_LAT[0]) / 2 + PATCH_LAT_DEG / 2
    train_lon_center = (TEACHER_LON[0] + TEACHER_LON[1]) / 2
    
    # Random patch near training region
    near_lat_offset = np.random.uniform(-10, 10)
    near_lon_offset = np.random.uniform(-10, 10)
    near_lat_start = train_lat_center + near_lat_offset
    near_lat_end = near_lat_start + PATCH_LAT_DEG
    near_lon_start = train_lon_center + near_lon_offset
    near_lon_end = near_lon_start + PATCH_LON_DEG
    
    # Ensure within bounds
    near_lat_start = max(MIN_LAT, min(MAX_LAT - PATCH_LAT_DEG, near_lat_start))
    near_lat_end = near_lat_start + PATCH_LAT_DEG
    near_lon_start = max(MIN_LON, min(MAX_LON - PATCH_LON_DEG, near_lon_start))
    near_lon_end = near_lon_start + PATCH_LON_DEG
    
    patches.append((
        "random_near",
        (near_lat_start, near_lat_end),
        (near_lon_start, near_lon_end)
    ))
    
    # Random patch far from training region
    # Pick a random quadrant far away
    far_directions = [
        (1, 1),   # NE
        (1, -1),  # NW
        (-1, 1),  # SE
        (-1, -1)  # SW
    ]
    dir_lat, dir_lon = far_directions[np.random.randint(0, 4)]
    
    far_lat_offset = np.random.uniform(25, 40) * dir_lat
    far_lon_offset = np.random.uniform(30, 50) * dir_lon
    far_lat_start = train_lat_center + far_lat_offset
    far_lat_end = far_lat_start + PATCH_LAT_DEG
    far_lon_start = train_lon_center + far_lon_offset
    far_lon_end = far_lon_start + PATCH_LON_DEG
    
    # Ensure within bounds
    far_lat_start = max(MIN_LAT, min(MAX_LAT - PATCH_LAT_DEG, far_lat_start))
    far_lat_end = far_lat_start + PATCH_LAT_DEG
    far_lon_start = max(MIN_LON, min(MAX_LON - PATCH_LON_DEG, far_lon_start))
    far_lon_end = far_lon_start + PATCH_LON_DEG
    
    patches.append((
        "random_far",
        (far_lat_start, far_lat_end),
        (far_lon_start, far_lon_end)
    ))
    
    return patches


def run_inference_on_patch(
    student: torch.nn.Module,
    ds: xr.Dataset,
    patch_lat: Tuple[float, float],
    patch_lon: Tuple[float, float],
    mean: float,
    std: float,
    in_len: int,
    out_len: int,
    test_start_year: int,
    device: torch.device,
    target_h: int = 21,
    target_w: int = 28,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run inference on a specific patch.
    
    Returns:
        pred_deg: Predicted values (denormalized)
        actual_deg: Actual values (denormalized)
        times: Time indices
        mse: Mean squared error
    """
    # Extract patch data
    ds_patch = ds.sel(
        lat=slice(patch_lat[0], patch_lat[1]),
        lon=slice(patch_lon[0], patch_lon[1])
    )
    test_slice = slice(str(test_start_year), None)
    patch_raw = ds_patch["sst"].sel(time=test_slice).values.astype(np.float32)
    patch_raw = np.nan_to_num(patch_raw, nan=mean)
    time_index = ds_patch["sst"].sel(time=test_slice).get_index("time")
    
    # Normalize and resize
    patch_norm = (patch_raw - mean) / std
    if patch_norm.shape[1] != target_h or patch_norm.shape[2] != target_w:
        patch_norm = _resize_patch(patch_norm, target_h, target_w)
    print(f"  [Post-Resize Check] Grid shape: {patch_norm.shape}")
    print(f"  [Post-Resize Check] NaNs in grid: {np.isnan(patch_norm).sum()}")
    patch_norm = patch_norm[:, np.newaxis, :, :]  # (T, 1, H, W)
    
    seq_len = in_len + out_len
    if len(patch_norm) < seq_len:
        print(f"Warning: Patch has {len(patch_norm)} timesteps, need at least {seq_len}. Using available data.")
        return None, None, None, float('inf')
    
    # Run inference
    num_test_seqs = len(patch_norm) - seq_len + 1
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for i in range(num_test_seqs):
            chunk = patch_norm[i : i + seq_len]
            x = torch.from_numpy(chunk[:in_len]).float().unsqueeze(0).to(device)
            y_true = chunk[in_len:seq_len]
            x = x.permute(0, 1, 3, 4, 2)  # (1, T_in, H, W, 1)
            pred = student(x)  # (1, T_out, H, W, 1)
            pred = pred.cpu().numpy().squeeze(0).squeeze(-1)
            all_preds.append(pred)
            all_actuals.append(y_true.squeeze(1))
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)
    
    # Spatially average (first output timestep)
    pred_first = all_preds[0::out_len].mean(axis=(1, 2))
    actual_first = all_actuals[0::out_len].mean(axis=(1, 2))
    
    # Align lengths
    n = min(len(pred_first), len(actual_first))
    pred_first = pred_first[:n]
    actual_first = actual_first[:n]
    times = time_index[in_len : in_len + n]
    
    # Denormalize
    pred_deg = pred_first * std + mean
    actual_deg = actual_first * std + mean
    mse = mean_squared_error(actual_deg, pred_deg)
    
    return pred_deg, actual_deg, times, mse


def get_parser():
    p = argparse.ArgumentParser(description="Multi-patch inference testing for distilled student")
    p.add_argument("--cfg", type=str, default=os.path.join(_curr_dir, "sst_distill_earthformer.yaml"), 
                   help="Distillation config YAML")
    p.add_argument("--ckpt", type=str, required=True, 
                   help="Path to distillation checkpoint")
    p.add_argument("--data_root", type=str, default=None, 
                   help="Override data root")
    p.add_argument("--filename", type=str, default="sst.week.mean.nc")
    p.add_argument("--output_dir", type=str, default="inference_results", 
                   help="Output directory for plots and results")
    p.add_argument("--test_start_year", type=int, default=2021, 
                   help="Start year for test period")
    p.add_argument("--test_scenarios", type=str, default="all",
                   choices=["all", "south", "west", "random"],
                   help="Which test scenarios to run")
    p.add_argument("--random_seed", type=int, default=42,
                   help="Random seed for random patch generation")
    return p


def main():
    args = get_parser().parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
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
    file_path = os.path.join(data_root, args.filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data not found: {file_path}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading distillation checkpoint (teacher + student)...")
    pl_module = CuboidDistillPLModule.load_from_checkpoint(
        args.ckpt,
        teacher_ckpt_path=teacher_ckpt,
        oc_file=teacher_cfg_path,
        save_dir=os.path.dirname(os.path.dirname(args.ckpt)),
    )
    student = pl_module.student.to(device).eval()
    
    # Load data and compute normalization
    print("Loading SST dataset...")
    ds = xr.open_dataset(file_path)
    train_slice = slice(None, str(train_end_year))
    ds_teacher = ds.sel(lat=slice(*TEACHER_LAT), lon=slice(*TEACHER_LON))
    train_teacher = ds_teacher["sst"].sel(time=train_slice).values.astype(np.float32)
    mean = float(np.nanmean(train_teacher))
    std = float(np.nanstd(train_teacher))
    if std < 1e-8:
        std = 1.0
    print(f"Normalization: mean={mean:.4f}, std={std:.4f}")
    
    # Generate test patches
    all_patches = []
    if args.test_scenarios in ["all", "south"]:
        all_patches.extend(compute_southward_patches())
    if args.test_scenarios in ["all", "west"]:
        all_patches.extend(compute_westward_patches())
    if args.test_scenarios in ["all", "random"]:
        all_patches.extend(compute_random_patches(args.random_seed))
    
    print(f"\nTesting {len(all_patches)} patches:")
    print(f"Last training patch: lat {LAST_PATCH_LAT}, lon {LAST_PATCH_LON}")
    
    # Store results
    results = []
    
    # Run inference on each patch
    for patch_name, patch_lat, patch_lon in all_patches:
        print(f"\n{'='*60}")
        print(f"Testing patch: {patch_name}")
        print(f"  Latitude:  {patch_lat[0]:.3f} to {patch_lat[1]:.3f}")
        print(f"  Longitude: {patch_lon[0]:.3f} to {patch_lon[1]:.3f}")
        
        pred_deg, actual_deg, times, mse = run_inference_on_patch(
            student=student,
            ds=ds,
            patch_lat=patch_lat,
            patch_lon=patch_lon,
            mean=mean,
            std=std,
            in_len=in_len,
            out_len=out_len,
            test_start_year=args.test_start_year,
            device=device,
        )
        
        if pred_deg is None:
            print(f"  Skipping {patch_name} due to insufficient data")
            continue
        
        print(f"  MSE: {mse:.4f}")
        results.append({
            "name": patch_name,
            "lat": patch_lat,
            "lon": patch_lon,
            "mse": mse,
            "pred": pred_deg,
            "actual": actual_deg,
            "times": times,
        })
        
        # Plot individual patch
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(times, actual_deg, label="Actual", color="blue", linewidth=1.5)
        ax.plot(times, pred_deg, label="Student predicted", color="red", 
                linestyle="--", linewidth=1.2, alpha=0.9)
        ax.set_title(f"{patch_name}\nLat {patch_lat[0]:.2f}-{patch_lat[1]:.2f}°, "
                     f"Lon {patch_lon[0]:.2f}-{patch_lon[1]:.2f}° | MSE = {mse:.4f}")
        ax.set_xlabel("Time")
        ax.set_ylabel("SST (°C)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(args.output_dir, f"{patch_name}_prediction.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {plot_path}")
    
    ds.close()
    
    # Create summary plot
    if results:
        print(f"\n{'='*60}")
        print("Creating summary comparison plot...")
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot 1: MSE comparison
        ax = axes[0]
        patch_names = [r["name"] for r in results]
        mses = [r["mse"] for r in results]
        colors = []
        for name in patch_names:
            if "south" in name:
                colors.append("steelblue")
            elif "west" in name:
                colors.append("coral")
            else:
                colors.append("forestgreen")
        
        bars = ax.bar(range(len(patch_names)), mses, color=colors, alpha=0.7, edgecolor="black")
        ax.set_xticks(range(len(patch_names)))
        ax.set_xticklabels(patch_names, rotation=45, ha="right")
        ax.set_ylabel("MSE")
        ax.set_title("Model Performance Across Different Test Patches")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Add value labels on bars
        for bar, mse in zip(bars, mses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mse:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Sample time series comparison (first 3 patches)
        ax = axes[1]
        n_samples = min(3, len(results))
        for i in range(n_samples):
            r = results[i]
            times = r["times"][:50]  # First 50 timesteps for clarity
            actual = r["actual"][:50]
            pred = r["pred"][:50]
            ax.plot(times, actual, label=f'{r["name"]} (actual)', 
                   linewidth=1.5, alpha=0.7)
            ax.plot(times, pred, label=f'{r["name"]} (pred)', 
                   linestyle="--", linewidth=1.2, alpha=0.7)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("SST (°C)")
        ax.set_title("Sample Time Series Predictions (first 50 timesteps)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        summary_path = os.path.join(args.output_dir, "summary_comparison.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Summary plot saved: {summary_path}")
        
        # Save results to text file
        results_txt = os.path.join(args.output_dir, "results_summary.txt")
        with open(results_txt, "w") as f:
            f.write("Multi-Patch Inference Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"Training configuration:\n")
            f.write(f"  Teacher patch: lat {TEACHER_LAT}, lon {TEACHER_LON}\n")
            f.write(f"  Last training patch: lat {LAST_PATCH_LAT}, lon {LAST_PATCH_LON}\n")
            f.write(f"  Patch size: {PATCH_LAT_DEG}° x {PATCH_LON_DEG}°\n")
            f.write(f"  Training patches: {NUM_STRIDE_PATCHES}\n\n")
            f.write(f"Results:\n")
            f.write("-"*60 + "\n")
            for r in results:
                f.write(f"\nPatch: {r['name']}\n")
                f.write(f"  Latitude:  {r['lat'][0]:.3f} to {r['lat'][1]:.3f}\n")
                f.write(f"  Longitude: {r['lon'][0]:.3f} to {r['lon'][1]:.3f}\n")
                f.write(f"  MSE: {r['mse']:.4f}\n")
        print(f"Results summary saved: {results_txt}")
    
    print(f"\n{'='*60}")
    print(f"All done! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
