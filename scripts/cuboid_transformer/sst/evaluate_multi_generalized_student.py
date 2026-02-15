# ==============================================================================
# SCRIPT FOR EVALUATING THE *GENERALIZED* (TEACHER + STUDENT) MODEL
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
from omegaconf import OmegaConf

# --- Add Repository Root to Python Path ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

try:
    from scripts.cuboid_transformer.sst.train_cuboid_sst import CuboidSSTPLModule
    from scripts.student_model import ConvLSTMStudent
except ImportError as e:
    print(f"Error: Could not import modules. {e}")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

def slice_type(x):
    start, end = map(float, x.split(':'))
    return slice(start, end)

def create_sequences(data, in_len, out_len):
    sequences = []
    total_seq_len = in_len + out_len
    for i in range(len(data) - total_seq_len + 1):
        sequences.append(data[i:i + total_seq_len])
    if not sequences: return None, None
    sequences = np.array(sequences, dtype=np.float32)
    x = torch.from_numpy(sequences[:, :in_len])
    y = torch.from_numpy(sequences[:, in_len:])
    return x, y

def plot_comparison(actuals_denorm, preds_denorm, time_axis, title, save_path, y_ticks, label="Predicted SST"):
    try:
        logging.info(f"Saving plot to {save_path}...")
        final_mse = mean_squared_error(actuals_denorm, preds_denorm)
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.plot(time_axis, actuals_denorm, label='Actual Spatially-Averaged SST', color='blue', linewidth=2)
        ax.plot(time_axis, preds_denorm, label=label, linestyle='--', color='red', alpha=0.9, linewidth=1.5)
        ax.set_title(f"{title}\nFinal MSE: {final_mse:.4f}", fontsize=16)
        ax.set_yticks(y_ticks)
        ax.set_xlabel('Date', fontsize=14); ax.set_ylabel('SST (°C)', fontsize=14)
        ax.legend(loc='upper left'); ax.grid(True)
        fig.tight_layout()
        plt.savefig(save_path, dpi=300); plt.close(fig)
    except Exception as e:
        logging.error(f"Error during plotting: {e}")

# --- CHANGE 1: Full Implementation of Pure Ocean Random Patches ---
def compute_random_patches(ds, seed=42):
    np.random.seed(seed)
    patches = []
    # Reference training center for Indian Ocean
    train_lat_center, train_lon_center = 18.125, 69.0 
    lat_dim, lon_dim = 5.0, 6.75

    def is_pure_ocean(lat_rng, lon_rng):
        try:
            sample = ds["sst"].isel(time=0).sel(lat=slice(*lat_rng), lon=slice(*lon_rng)).values
            return sample.size > 0 and not np.any(np.isnan(sample))
        except: return False

    for mode in ["random_near", "random_far"]:
        found = False
        attempts = 0
        while not found and attempts < 100:
            offset = 10 if "near" in mode else 40
            lat_start = train_lat_center + np.random.uniform(-offset, offset)
            lon_start = train_lon_center + np.random.uniform(-offset, offset)
            lats, lons = (lat_start, lat_start + lat_dim), (lon_start, lon_start + lon_dim)
            if is_pure_ocean(lats, lons):
                patches.append((mode, lats, lons))
                found = True
            attempts += 1
    return patches

def run_evaluation_for_patch(
    teacher_model, student_type, student_model, ds_full, hparams, 
    mean_original, std_original, center_lat, center_lon, 
    scenario_name, plot_save_dir, device
):
    logging.info(f"\n===== Scenario: {scenario_name} =====")
    patch_height, patch_width = 21, 28
    
    # --- CHANGE 2: Robust Boundary Indexing ---
    center_lat_idx = np.abs(ds_full.lat.values - center_lat).argmin()
    center_lon_idx = np.abs(ds_full.lon.values - center_lon).argmin()
    start_lat_idx = int(np.clip(center_lat_idx - patch_height // 2, 0, len(ds_full.lat) - patch_height))
    start_lon_idx = int(np.clip(center_lon_idx - patch_width // 2, 0, len(ds_full.lon) - patch_width))
    
    ds_new_patch = ds_full.isel(lat=slice(start_lat_idx, start_lat_idx + patch_height), 
                                lon=slice(start_lon_idx, start_lon_idx + patch_width))
    
    new_patch_data_raw = ds_new_patch['sst'].values.astype(np.float32)
    new_patch_data_filled = np.nan_to_num(new_patch_data_raw, nan=mean_original)
    new_patch_normalized = (new_patch_data_filled - mean_original) / std_original
    new_patch_normalized = new_patch_normalized[:, np.newaxis, :, :] # (T, 1, H, W)

    in_len, out_len = hparams.dataset.in_len, hparams.dataset.out_len
    input_seqs, target_seqs = create_sequences(new_patch_normalized, in_len, out_len)
    if input_seqs is None: return

    full_loader = DataLoader(TensorDataset(input_seqs, target_seqs), batch_size=16, shuffle=False)

    # --- CHANGE 3: Correct Student Inference Logic ---
    all_teacher_preds, all_student_preds, all_actuals = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(full_loader, desc=f"Inference"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Teacher: [B, T, H, W, C]
            teacher_in = x_batch.permute(0, 1, 3, 4, 2)
            teacher_out_raw = teacher_model(teacher_in) 
            teacher_pred_norm = teacher_out_raw.permute(0, 1, 4, 2, 3) # [B, T, C, H, W]

            # Student: Consumes Teacher's first output step
            # Note: We take the teacher's prediction for the first future step
            student_in = teacher_pred_norm[:, 0:1, :, :, :] 
            student_pred_norm = student_model(student_in)

            all_teacher_preds.extend(teacher_pred_norm[:, 0:1].mean(dim=(2,3,4)).cpu().numpy())
            all_student_preds.extend(student_pred_norm.mean(dim=(2,3,4)).cpu().numpy())
            all_actuals.extend(y_batch[:, 0:1].mean(dim=(2,3,4)).cpu().numpy())

    actuals_denorm = (np.array(all_actuals).flatten() * std_original) + mean_original
    teacher_denorm = (np.array(all_teacher_preds).flatten() * std_original) + mean_original
    student_denorm = (np.array(all_student_preds).flatten() * std_original) + mean_original
    
    time_axis = ds_new_patch.get_index("time")[in_len : len(actuals_denorm) + in_len]
    y_ticks = np.arange(np.floor(actuals_denorm.min()), np.ceil(actuals_denorm.max()) + 1, 1.0)

    plot_comparison(actuals_denorm, teacher_denorm, time_axis, f"Teacher: {scenario_name}", 
                    os.path.join(plot_save_dir, f"teacher_{scenario_name}.png"), y_ticks)
    plot_comparison(actuals_denorm, student_denorm, time_axis, f"Student: {scenario_name}", 
                    os.path.join(plot_save_dir, f"student_{scenario_name}.png"), y_ticks)

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.plot_save_dir, exist_ok=True)

    # Load Teacher
    pl_module = CuboidSSTPLModule.load_from_checkpoint(args.teacher_ckpt_path, oc_file=args.cfg, save_dir=args.plot_save_dir)
    teacher_model = pl_module.torch_nn_module.to(device).eval()
    hparams = pl_module.hparams

    # Load Student
    if args.student_ckpt_path.endswith('.pth'):
        student_model = ConvLSTMStudent(input_dim=1, hidden_dim=12, kernel_size=(3, 3), num_layers=2).to(device)
        student_model.load_state_dict(torch.load(args.student_ckpt_path, map_location=device))
        student_type = "convlstm"
    else:
        student_pl = CuboidSSTPLModule.load_from_checkpoint(args.student_ckpt_path, oc_file=args.cfg, save_dir=args.plot_save_dir)
        student_model = student_pl.torch_nn_module.to(device)
        student_type = "earthformer"
    student_model.eval()

    ds_full = xr.open_dataset(args.data_path)
    train_stats = ds_full['sst'].sel(time=slice(None, str(args.train_end_year)), lat=args.base_lat_slice, lon=args.base_lon_slice).values
    mean_original, std_original = np.nanmean(train_stats), np.nanstd(train_stats)
    logging.info(f"Stats: Mean {mean_original:.4f}, Std {std_original:.4f}")

    # --- CHANGE 4: Define variables and call random patches correctly ---
    overlaps = [0, 30, 60, 90]
    LAT_DIM, LON_DIM = 5.0, 6.75
    last_center_lat, last_center_lon = 18.125, 85.875
    scenarios = []

    for p in overlaps:
        stride = LAT_DIM * (1 - p / 100.0)
        scenarios.append({"name": f"South_Olp_{p}", "center_lat": last_center_lat - stride, "center_lon": last_center_lon})
        
    for p in overlaps:
        stride = LON_DIM * (1 - p / 100.0)
        scenarios.append({"name": f"West_Olp_{p}", "center_lat": last_center_lat, "center_lon": last_center_lon - stride})

    # Correct call to the random function
    random_results = compute_random_patches(ds_full, seed=42)
    for name, lats, lons in random_results:
        scenarios.append({"name": name, "center_lat": (lats[0]+lats[1])/2, "center_lon": (lons[0]+lons[1])/2})

    for scenario in scenarios:
        run_evaluation_for_patch(teacher_model, student_type, student_model, ds_full, hparams, 
                                 mean_original, std_original, scenario["center_lat"], scenario["center_lon"],
                                 scenario["name"], args.plot_save_dir, device)

    ds_full.close()
    logging.info("Evaluation complete.")

if __name__ == '__main__':
    main()
