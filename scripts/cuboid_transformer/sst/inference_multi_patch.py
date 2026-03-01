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

# Accuracy threshold: prediction is "correct" if within this many °C of actual
ACCURACY_THRESHOLD_DEG = 0.5


def compute_accuracy(actual: np.ndarray, pred: np.ndarray, threshold: float = ACCURACY_THRESHOLD_DEG) -> float:
    """
    Accuracy % = percentage of predictions within `threshold` °C of actual value.
    """
    correct = np.abs(actual - pred) <= threshold
    return float(correct.mean()) * 100.0


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
    Compute southward test patches with varying overlap percentages relative to
    the last training patch.

    overlap=0%    full stride south (no overlap with last training patch)
    overlap=100%  zero stride (identical position to last training patch)

    Returns list of (name, lat_range, lon_range).
    """
    patches = []
    overlaps = [0, 30, 60, 90, 100]
    for overlap_pct in overlaps:
        # Calculate stride based on overlap:
        # overlap_pct=0  full stride (1.5 deg south), no spatial overlap
        # overlap_pct=100  zero stride, identical to last training patch
        effective_stride = STRIDE_DEG * (1 - overlap_pct / 100.0)

        new_lat_start = LAST_PATCH_LAT[0] - effective_stride
        new_lat_end = LAST_PATCH_LAT[1] - effective_stride

        # Safety check: don't go below the southern pole
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
    Compute westward test patches with varying overlap percentages relative to
    the last training patch's longitude band.

    overlap=0%   full patch-width stride west (no overlap)
    overlap=100% zero stride (same longitude as last training patch)

    Returns list of (name, lat_range, lon_range).
    """
    patches = []
    overlaps = [0, 30, 60, 90, 100]
    for overlap_pct in overlaps:
        # Calculate stride based on overlap percentage along longitude axis
        effective_stride = PATCH_LON_DEG * (1 - overlap_pct / 100.0)
        # Move west (decrease longitude)
        new_lon_start = LAST_PATCH_LON[0] - effective_stride
        new_lon_end = LAST_PATCH_LON[1] - effective_stride
        # Safety check: don't go below 0° longitude
        if new_lon_start < MIN_LON:
            print(f"Warning: Westward {overlap_pct}% overlap patch goes below minimum longitude, skipping")
            continue
        patches.append((
            f"west_{overlap_pct}pct_overlap",
            LAST_PATCH_LAT,
            (new_lon_start, new_lon_end)
        ))
    return patches


def compute_random_patches(ds: xr.Dataset = None, seed: int = 42) -> List[Tuple[str, Tuple[float, float], Tuple[float, float]]]:
    """
    Returns the two fixed test patches with explicit bounds.
    1. Random Near: Centered at 10.87, 70.38 (Lat: 8.37 to 13.37, Lon: 67.005 to 73.755)
       — close to the training region to test near-distribution generalisation.
    2. Random Far: Fixed bounds Lat: -25.32 to -20.32, Lon: 110.94 to 117.69
       — Australian/Indian Ocean region to test out-of-distribution generalisation.
    """
    patches = []
    p1_lat = (0.37, 5.37)
    p1_lon = (77.005, 83.755)
    patches.append(("random_near", p1_lat, p1_lon))
    p2_lat = (-25.32, -20.32)
    p2_lon = (100.94, 112.69)
    patches.append(("random_far", p2_lat, p2_lon))
    print(f"  [Random Patches] 'near' bounds: Lat {p1_lat}, Lon {p1_lon}")
    print(f"  [Random Patches] 'far'  bounds: Lat {p2_lat}, Lon {p2_lon}")
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Run inference on a specific patch.

    Returns:
        pred_deg:   Predicted values (denormalized)
        actual_deg: Actual values (denormalized)
        times:      Time indices
        mse:        Mean squared error
        accuracy:   Accuracy % (predictions within ACCURACY_THRESHOLD_DEG of actual)
    """
    # Extract patch data for the test period
    ds_patch = ds.sel(
        lat=slice(patch_lat[0], patch_lat[1]),
        lon=slice(patch_lon[0], patch_lon[1])
    )
    test_slice = slice(str(test_start_year), None)
    patch_raw = ds_patch["sst"].sel(time=test_slice).values.astype(np.float32)
    patch_raw = np.nan_to_num(patch_raw, nan=mean)  # replace NaN (land/missing) with mean
    time_index = ds_patch["sst"].sel(time=test_slice).get_index("time")

    # Normalize using training statistics and resize to model's expected spatial dims
    patch_norm = (patch_raw - mean) / std
    if patch_norm.shape[1] != target_h or patch_norm.shape[2] != target_w:
        patch_norm = _resize_patch(patch_norm, target_h, target_w)
    print(f"  [Post-Resize Check] Grid shape: {patch_norm.shape}")
    print(f"  [Post-Resize Check] NaNs in grid: {np.isnan(patch_norm).sum()}")
    patch_norm = patch_norm[:, np.newaxis, :, :]  # (T, 1, H, W) — add channel dim

    seq_len = in_len + out_len
    if len(patch_norm) < seq_len:
        print(f"Warning: Patch has {len(patch_norm)} timesteps, need at least {seq_len}. Using available data.")
        return None, None, None, float('inf'), 0.0

    # Sliding window inference over all available test sequences
    num_test_seqs = len(patch_norm) - seq_len + 1
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for i in range(num_test_seqs):
            chunk = patch_norm[i : i + seq_len]
            x = torch.from_numpy(chunk[:in_len]).float().unsqueeze(0).to(device)
            y_true = chunk[in_len:seq_len]
            x = x.permute(0, 1, 3, 4, 2)  # (1, T_in, H, W, 1) — model expects channel last
            pred = student(x)              # (1, T_out, H, W, 1)
            pred = pred.cpu().numpy().squeeze(0).squeeze(-1)
            all_preds.append(pred)
            all_actuals.append(y_true.squeeze(1))

    all_preds = np.concatenate(all_preds, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)

    # Take first output timestep of each sequence and spatially average for a scalar time series
    pred_first = all_preds[0::out_len].mean(axis=(1, 2))
    actual_first = all_actuals[0::out_len].mean(axis=(1, 2))

    # Align lengths and build time axis
    n = min(len(pred_first), len(actual_first))
    pred_first = pred_first[:n]
    actual_first = actual_first[:n]
    times = time_index[in_len : in_len + n]

    # Denormalize back to °C
    pred_deg = pred_first * std + mean
    actual_deg = actual_first * std + mean

    mse = mean_squared_error(actual_deg, pred_deg)
    accuracy = compute_accuracy(actual_deg, pred_deg)

    return pred_deg, actual_deg, times, mse, accuracy


def get_parser():
    p = argparse.ArgumentParser(
        description="Multi-patch inference testing for distilled student model. "
                    "Evaluates generalisation across southward, westward, and random patches "
                    "that were unseen during training."
    )
    p.add_argument(
        "--cfg", type=str,
        default=os.path.join(_curr_dir, "sst_distill_earthformer.yaml"),
        help="Path to the distillation config YAML used during training."
    )
    p.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to the distillation checkpoint (.ckpt) to load the student from."
    )
    p.add_argument(
        "--data_root", type=str, default=None,
        help="Override the data root directory from the config. "
             "If not set, uses the value from the config YAML."
    )
    p.add_argument(
        "--filename", type=str, default="sst.week.mean.nc",
        help="NetCDF filename inside data_root. Default: sst.week.mean.nc"
    )
    p.add_argument(
        "--output_dir", type=str, default="inference_results",
        help="Directory where plots and result summaries will be saved."
    )
    p.add_argument(
        "--test_start_year", type=int, default=2021,
        help="Start year for the test period (data from this year onwards is used for inference)."
    )
    p.add_argument(
        "--test_scenarios", type=str, default="all",
        choices=["all", "south", "west", "random"],
        help="Which test scenarios to run: "
             "'south' = southward patches with varying overlap, "
             "'west' = westward patches with varying overlap, "
             "'random' = one near and one far random patch, "
             "'all' = run all three scenarios."
    )
    p.add_argument(
        "--random_seed", type=int, default=42,
        help="Random seed for reproducibility of random patch generation."
    )
    return p


def main():
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading distillation checkpoint (teacher + student)...")
    pl_module = CuboidDistillPLModule.load_from_checkpoint(
        args.ckpt,
        teacher_ckpt_path=teacher_ckpt,
        oc_file=teacher_cfg_path,
        save_dir=os.path.dirname(os.path.dirname(args.ckpt)),
        strice=False
    )
    student = pl_module.student.to(device).eval()

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
    print(f"Accuracy threshold: ±{ACCURACY_THRESHOLD_DEG}°C")

    all_patches = []
    if args.test_scenarios in ["all", "south"]:
        all_patches.extend(compute_southward_patches())
    if args.test_scenarios in ["all", "west"]:
        all_patches.extend(compute_westward_patches())
    if args.test_scenarios in ["all", "random"]:
        all_patches.extend(compute_random_patches(ds, args.random_seed))

    print(f"\nTesting {len(all_patches)} patches:")
    print(f"Last training patch: lat {LAST_PATCH_LAT}, lon {LAST_PATCH_LON}")

    results = []

    for patch_name, patch_lat, patch_lon in all_patches:
        print(f"\n{'='*60}")
        print(f"Testing patch: {patch_name}")
        print(f"  Latitude:  {patch_lat[0]:.3f} to {patch_lat[1]:.3f}")
        print(f"  Longitude: {patch_lon[0]:.3f} to {patch_lon[1]:.3f}")

        pred_deg, actual_deg, times, mse, accuracy = run_inference_on_patch(
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

        print(f"  MSE:      {mse:.4f}")
        print(f"  Accuracy: {accuracy:.1f}% (within ±{ACCURACY_THRESHOLD_DEG}°C)")
        results.append({
            "name": patch_name,
            "lat": patch_lat,
            "lon": patch_lon,
            "mse": mse,
            "accuracy": accuracy,
            "pred": pred_deg,
            "actual": actual_deg,
            "times": times,
        })

        # Individual patch plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        ax.plot(times, actual_deg, label="Actual", color="blue", linewidth=1.5)
        ax.plot(times, pred_deg, label="Student predicted", color="red",
                linestyle="--", linewidth=1.2, alpha=0.9)
        ax.set_title(
            f"{patch_name}\n"
            f"Lat {patch_lat[0]:.2f}–{patch_lat[1]:.2f}°, "
            f"Lon {patch_lon[0]:.2f}–{patch_lon[1]:.2f}° | "
            f"MSE = {mse:.4f} | Accuracy = {accuracy:.1f}%"
        )
        ax.set_ylabel("SST (°C)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Error subplot
        ax2 = axes[1]
        error = np.abs(pred_deg - actual_deg)
        ax2.fill_between(times, error, alpha=0.4, color="orange", label="|Error|")
        ax2.axhline(ACCURACY_THRESHOLD_DEG, color="red", linestyle="--",
                    linewidth=1, label=f"±{ACCURACY_THRESHOLD_DEG}°C threshold")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("|Error| (°C)")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = os.path.join(args.output_dir, f"{patch_name}_prediction.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {plot_path}")

    ds.close()

    # Summary plot
    if results:
        print(f"\n{'='*60}")
        print("Creating summary comparison plot...")

        fig, axes = plt.subplots(3, 1, figsize=(16, 14))

        patch_names = [r["name"] for r in results]
        mses = [r["mse"] for r in results]
        accuracies = [r["accuracy"] for r in results]

        bar_colors = []
        for name in patch_names:
            if "south" in name:
                bar_colors.append("steelblue")
            elif "west" in name:
                bar_colors.append("coral")
            else:
                bar_colors.append("forestgreen")

        x = range(len(patch_names))

        # Plot 1: MSE
        ax = axes[0]
        bars = ax.bar(x, mses, color=bar_colors, alpha=0.7, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(patch_names, rotation=45, ha="right")
        ax.set_ylabel("MSE")
        ax.set_title("MSE Across Test Patches")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, mses):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 2: Accuracy %
        ax = axes[1]
        bars = ax.bar(x, accuracies, color=bar_colors, alpha=0.7, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(patch_names, rotation=45, ha="right")
        ax.set_ylabel(f"Accuracy % (±{ACCURACY_THRESHOLD_DEG}°C)")
        ax.set_title("Prediction Accuracy Across Test Patches")
        ax.set_ylim(0, 110)
        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

        # Plot 3: Sample time series
        ax = axes[2]
        n_samples = min(3, len(results))
        for i in range(n_samples):
            r = results[i]
            t = r["times"][:50]
            ax.plot(t, r["actual"][:50], label=f'{r["name"]} (actual)', linewidth=1.5, alpha=0.7)
            ax.plot(t, r["pred"][:50], label=f'{r["name"]} (pred)', linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel("Time")
        ax.set_ylabel("SST (°C)")
        ax.set_title("Sample Time Series (first 50 timesteps)")
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
            f.write("=" * 60 + "\n\n")
            f.write("Training configuration:\n")
            f.write(f"  Teacher patch: lat {TEACHER_LAT}, lon {TEACHER_LON}\n")
            f.write(f"  Last training patch: lat {LAST_PATCH_LAT}, lon {LAST_PATCH_LON}\n")
            f.write(f"  Patch size: {PATCH_LAT_DEG}° x {PATCH_LON_DEG}°\n")
            f.write(f"  Training patches: {NUM_STRIDE_PATCHES}\n\n")
            f.write(f"Accuracy definition: % of predictions within ±{ACCURACY_THRESHOLD_DEG}°C of actual\n\n")
            f.write("Results:\n")
            f.write("-" * 60 + "\n")
            for r in results:
                f.write(f"\nPatch: {r['name']}\n")
                f.write(f"  Latitude:  {r['lat'][0]:.3f} to {r['lat'][1]:.3f}\n")
                f.write(f"  Longitude: {r['lon'][0]:.3f} to {r['lon'][1]:.3f}\n")
                f.write(f"  MSE:       {r['mse']:.4f}\n")
                f.write(f"  Accuracy:  {r['accuracy']:.1f}%\n")
        print(f"Results summary saved: {results_txt}")

    print(f"\n{'='*60}")
    print(f"All done! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
