# ==============================================================================
# SCRIPT FOR EVALUATING THE *GENERALIZED* CONVLSTM MODEL (EARTHFORMER LOGIC)
# ==============================================================================

import matplotlib
matplotlib.use('Agg')
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import xarray as xr
import sys
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# --- Constants from Earthformer Inference (CRITICAL FOR PARITY) ---
TEACHER_LAT = (15.625, 20.625)
TEACHER_LON = (65.625, 72.375)
PATCH_LAT_DEG = 5.0
PATCH_LON_DEG = 6.75
STRIDE_LAT_FRACTION = 0.30
STRIDE_DEG = STRIDE_LAT_FRACTION * PATCH_LAT_DEG  # 1.5 degrees
# Last training patch (9 strides south of teacher)
LAST_PATCH_LAT = (TEACHER_LAT[0] - 9 * STRIDE_DEG, TEACHER_LAT[1] - 9 * STRIDE_DEG)
LAST_PATCH_LON = TEACHER_LON

# --- Add Repository Root to Python Path ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from scripts.cuboid_transformer.sst.train_cuboid_sst import CuboidSSTPLModule
from scripts.student_model import ConvLSTMStudent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

# ==============================================================================
# PATCH COMPUTATION LOGIC (COPIED FROM EARTHFORMER INFERENCE)
# ==============================================================================

def slice_type(x):
    """Helper to convert string 'start:end' into a python slice object."""
    try:
        start, end = map(float, x.split(':'))
        return slice(start, end)
    except ValueError:
        raise argparse.ArgumentTypeError("Slice must be in the format 'start:end' (e.g., 15.625:20.625)")

def compute_southward_patches():
    patches = []
    overlaps = [0, 30, 60, 90, 100]
    for overlap_pct in overlaps:
        effective_stride = STRIDE_DEG * (1 - overlap_pct / 100.0)
        new_lat_start = LAST_PATCH_LAT[0] - effective_stride
        new_lat_end = LAST_PATCH_LAT[1] - effective_stride
        patches.append((f"South_Olp_{overlap_pct}", (new_lat_start, new_lat_end), LAST_PATCH_LON))
    return patches

def compute_westward_patches():
    patches = []
    overlaps = [0, 30, 60, 90, 100]
    for overlap_pct in overlaps:
        effective_stride = PATCH_LON_DEG * (1 - overlap_pct / 100.0)
        new_lon_start = LAST_PATCH_LON[0] - effective_stride
        new_lon_end = LAST_PATCH_LON[1] - effective_stride
        patches.append((f"West_Olp_{overlap_pct}", LAST_PATCH_LAT, (new_lon_start, new_lon_end)))
    return patches

def compute_random_patches(ds, seed=42):
    """
    Randomly selects patches across the global ocean.
    Uses a 'Pure Ocean' check to ensure the patches don't hit land.
    """
    np.random.seed(seed)
    patches = []
    
    # Reference training center for Indian Ocean
    train_lat_center, train_lon_center = 18.125, 69.0 
    lat_dim, lon_dim = 5.0, 6.75

    def is_pure_ocean(lat_rng, lon_rng):
        try:
            # Check the first timestep for NaNs (land pixels)
            sample = ds["sst"].isel(time=0).sel(lat=slice(*lat_rng), lon=slice(*lon_rng)).values
            # Ensure the slice isn't empty and has no NaNs
            return sample.size > 0 and not np.any(np.isnan(sample))
        except Exception:
            return False

    # Modes: Near (±10 deg) and Far (±40 deg)
    for mode in ["random_near", "random_far"]:
        found = False
        attempts = 0
        while not found and attempts < 200: # Increased attempts for 'far' patches
            offset = 10 if "near" in mode else 40
            lat_start = train_lat_center + np.random.uniform(-offset, offset)
            lon_start = train_lon_center + np.random.uniform(-offset, offset)
            
            # Constraints to keep it within global SST bounds
            lat_start = np.clip(lat_start, -60, 60) 
            lon_start = lon_start % 360 # Handle longitude wrap-around
            
            lats = (lat_start, lat_start + lat_dim)
            lons = (lon_start, lon_start + lon_dim)
            
            if is_pure_ocean(lats, lons):
                patches.append((mode, lats, lons))
                found = True
            attempts += 1
            
        if not found:
            logging.warning(f"Could not find a pure ocean patch for {mode} after 200 attempts.")
            
    return patches
# ==============================================================================
# RESIZING AND UTILS
# ==============================================================================

def _resize_2d(x, target_h, target_w):
    h, w = x.shape
    scale_h, scale_w = h / target_h, w / target_w
    out = np.zeros((target_h, target_w), dtype=x.dtype)
    for i in range(target_h):
        for j in range(target_w):
            i0, i1 = int(i * scale_h), min(int((i + 1) * scale_h), h)
            j0, j1 = int(j * scale_w), min(int((j + 1) * scale_w), w)
            window = x[i0:max(i1, i0+1), j0:max(j1, j0+1)]
            out[i, j] = np.nanmean(window) if window.size > 0 else x[min(i0, h-1), min(j0, w-1)]
    return out

def _resize_patch(data, target_h, target_w):
    t, h, w = data.shape
    out = np.zeros((t, target_h, target_w), dtype=data.dtype)
    for i in range(t):
        out[i] = _resize_2d(data[i], target_h, target_w)
    return out

def create_sequences(data, in_len, out_len):
    sequences = []
    total_seq_len = in_len + out_len
    for i in range(len(data) - total_seq_len + 1):
        sequences.append(data[i:i + total_seq_len])
    if not sequences: return None, None
    sequences = np.array(sequences, dtype=np.float32)
    return torch.from_numpy(sequences[:, :in_len]), torch.from_numpy(sequences[:, in_len:])

# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_comparison(actuals_denorm, preds_denorm, time_axis, title, save_path, y_ticks, lat_range, lon_range):
    final_mse = mean_squared_error(actuals_denorm, preds_denorm)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 8))
    
    ax.plot(time_axis, actuals_denorm, label='Actual Spatially-Averaged SST', color='blue', linewidth=2)
    ax.plot(time_axis, preds_denorm, label='Predicted SST', linestyle='--', color='red', alpha=0.9, linewidth=1.5)
    
    range_str = f"Range: Lat {lat_range[0]:.3f} to {lat_range[1]:.3f} | Lon {lon_range[0]:.3f} to {lon_range[1]:.3f}"
    ax.text(0.5, 1.05, range_str, transform=ax.transAxes, ha='center', fontsize=13, fontweight='bold')

    ax.set_title(f"{title}\nFinal MSE: {final_mse:.4f}", fontsize=16, pad=30)
    ax.set_yticks(y_ticks)
    ax.set_xlabel('Date'); ax.set_ylabel('SST (°C)')
    ax.legend(loc='upper left'); ax.grid(True)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close(fig)

# ==============================================================================
# EVALUATION CORE
# ==============================================================================

def run_evaluation_for_patch(teacher_model, student_model, ds_full, hparams, mean_orig, std_orig, lat_range, lon_range, scenario_name, plot_save_dir, device):
    logging.info(f"Evaluating: {scenario_name}")
    
    ds_patch = ds_full.sel(lat=slice(*lat_range), lon=slice(*lon_range))
    ds_test = ds_patch.sel(time=slice('2021', '2026'))
    
    raw_data = ds_test['sst'].values.astype(np.float32)
    filled_data = np.nan_to_num(raw_data, nan=mean_orig)
    resized_data = _resize_patch(filled_data, 21, 28)
    
    norm_data = (resized_data - mean_orig) / std_orig
    norm_data = norm_data[:, np.newaxis, :, :] # (T, 1, H, W)

    in_len, out_len = hparams.dataset.in_len, hparams.dataset.out_len
    input_seqs, target_seqs = create_sequences(norm_data, in_len, out_len)
    if input_seqs is None: return

    loader = DataLoader(TensorDataset(input_seqs, target_seqs), batch_size=16)
    all_student_preds, all_actuals = [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            # Match Distillation Logic: Teacher predicts -> Student consumes first step
            teacher_in = x_batch.permute(0, 1, 3, 4, 2)
            teacher_out = teacher_model(teacher_in)
            student_in = teacher_out.permute(0, 1, 4, 2, 3)[:, 0:1]
            student_pred = student_model(student_in)

            all_student_preds.extend(student_pred.mean(dim=(2,3,4)).cpu().numpy())
            all_actuals.extend(y_batch[:, 0:1].mean(dim=(2,3,4)).cpu().numpy())

    actuals_denorm = (np.array(all_actuals).flatten() * std_orig) + mean_orig
    student_denorm = (np.array(all_student_preds).flatten() * std_orig) + mean_orig
    
    time_axis = ds_test.get_index("time")[in_len : len(actuals_denorm) + in_len]
    y_ticks = np.arange(np.floor(actuals_denorm.min()), np.ceil(actuals_denorm.max()) + 1, 1.0)

    plot_comparison(actuals_denorm, student_denorm, time_axis, f"ConvLSTM: {scenario_name}", 
                    os.path.join(plot_save_dir, f"student_{scenario_name}.png"), 
                    y_ticks, lat_range, lon_range)

def get_args_parser():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Generalized Student Evaluation Script')
    parser.add_argument('--teacher_ckpt_path', type=str, required=True,
                        help='Path to the pre-trained Earthformer (Teacher) .ckpt file.')
    parser.add_argument('--student_ckpt_path', type=str, required=True,
                        help='Path to the *generalized* (multi-patch) Student .pth file.')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to the .yaml config file.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the single .nc file.')
    parser.add_argument('--plot_save_dir', type=str, default='evaluation_plots_generalized_convlstm',
                        help='Directory to save the new comparison plots.')
    parser.add_argument('--train_end_year', type=int, default=2015)
    
    # Base patch args for normalization stats
    parser.add_argument('--base_lat_slice', type=slice_type, default="15.625:20.625")
    parser.add_argument('--base_lon_slice', type=slice_type, default="65.625:72.375")
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation.')
    return parser

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.plot_save_dir, exist_ok=True)

    # 1. Load Teacher & Student (Existing Logic)
    pl_module = CuboidSSTPLModule.load_from_checkpoint(args.teacher_ckpt_path, oc_file=args.cfg, save_dir=args.plot_save_dir)
    teacher_model = pl_module.torch_nn_module.to(device).eval()
    hparams = pl_module.hparams

    if args.student_ckpt_path.endswith('.pth'):
        student_model = ConvLSTMStudent(input_dim=1, hidden_dim=12, kernel_size=(3, 3), num_layers=2).to(device)
        student_model.load_state_dict(torch.load(args.student_ckpt_path, map_location=device))
    else:
        student_pl = CuboidSSTPLModule.load_from_checkpoint(args.student_ckpt_path, oc_file=args.cfg)
        student_model = student_pl.torch_nn_module.to(device)
    student_model.eval()

    # 2. Load Dataset & Stats (Existing Logic)
    ds_full = xr.open_dataset(args.data_path)
    train_stats = ds_full['sst'].sel(time=slice(None, str(args.train_end_year)), 
                                     lat=args.base_lat_slice, 
                                     lon=args.base_lon_slice).values
    mean_original = np.nanmean(train_stats)
    std_original = np.nanstd(train_stats)

    # --- THE FIX: CALL THE FUNCTIONS TO DEFINE THE LISTS ---
    scenarios = []
    
    # Call the functions defined earlier in your script
    south_patches = compute_southward_patches() 
    west_patches = compute_westward_patches()
    random_patches = compute_random_patches(ds_full, seed=42)

    # Add them to the master scenarios list
    for name, lat_rng, lon_rng in south_patches:
        scenarios.append({"name": name, "lat": lat_rng, "lon": lon_rng})

    for name, lat_rng, lon_rng in west_patches:
        scenarios.append({"name": name, "lat": lat_rng, "lon": lon_rng})

    for name, lat_rng, lon_rng in random_patches:
        scenarios.append({"name": name, "lat": lat_rng, "lon": lon_rng})

    # 3. Execution Loop with Printing
    print(f"\n{'='*60}")
    print(f"STARTING GENERALIZED CONVLSTM INFERENCE")
    print(f"{'='*60}")

    for s in scenarios:
        # Print ranges to console as requested
        print(f"\nTesting patch: {s['name']}")
        print(f"  Latitude:  {s['lat'][0]:.3f} to {s['lat'][1]:.3f}")
        print(f"  Longitude: {s['lon'][0]:.3f} to {s['lon'][1]:.3f}")
        
        run_evaluation_for_patch(
            teacher_model=teacher_model, 
            student_model=student_model, 
            ds_full=ds_full, 
            hparams=hparams, 
            mean_orig=mean_original, 
            std_orig=std_original, 
            lat_range=s['lat'], 
            lon_range=s['lon'], 
            scenario_name=s['name'], 
            plot_save_dir=args.plot_save_dir, 
            device=device
        )

    ds_full.close()
    logging.info("Evaluation complete.")

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    # Add your argparse and initialization here
    main()
