# ==============================================================================
# SCRIPT FOR EVALUATING THE *GENERALIZED* (TEACHER + STUDENT) MODEL
#
# (Version 12 - Extended to evaluate P4, P5, P6, and P7)
#
# ==============================================================================

# --- ROBUST FIX for Matplotlib Backend Error ---
import matplotlib
matplotlib.use('Agg')
# --- END OF FIX ---

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

# --- Imports from your Earthformer repository ---
try:
    from scripts.cuboid_transformer.sst.train_cuboid_sst import CuboidSSTPLModule
    from scripts.student_model import ConvLSTMStudent
except ImportError as e:
    print(f"Error: Could not import modules. {e}")
    exit(1)

# --- Setup Logging & Warnings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", "This DataLoader will create .* worker processes .*")
warnings.filterwarnings("ignore", "torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

# --- Helper function for parsing slice arguments ---
def slice_type(x):
    """Parses a 'start:end' string into a slice object."""
    try:
        start, end = map(float, x.split(':'))
        return slice(start, end)
    except ValueError:
        raise argparse.ArgumentTypeError("Slice must be in 'start:end' format (e.g., '15.5:20.5')")

def create_sequences(data, in_len, out_len):
    """Creates overlapping sequences with a stride of 1."""
    sequences = []
    total_seq_len = in_len + out_len
    for i in range(len(data) - total_seq_len + 1):
        sequences.append(data[i:i + total_seq_len])
    if not sequences:
        return None, None
    
    sequences = np.array(sequences, dtype=np.float32)
    x = torch.from_numpy(sequences[:, :in_len])
    y = torch.from_numpy(sequences[:, in_len:])
    return x, y

def plot_comparison(actuals_denorm, preds_denorm, time_axis, title, save_path, y_ticks, label="Predicted SST"):
    """
    Helper function to generate and save a single plot.
    """
    try:
        logging.info(f"Saving plot to {save_path}...")
        final_mse = mean_squared_error(actuals_denorm, preds_denorm)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(20, 8))

        ax.plot(time_axis, actuals_denorm, label='Actual Spatially-Averaged SST', color='blue', linewidth=2)
        ax.plot(time_axis, preds_denorm, label=label, 
                linestyle='--', color='red', alpha=0.9, linewidth=1.5)

        full_title = (f"{title}\n"
                      f"Final MSE (Full Period): {final_mse:.4f}")
        ax.set_title(full_title, fontsize=16)
        
        ax.set_yticks(y_ticks) # Use the provided 1-degree ticks
        
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Spatially-Averaged SST (°C)', fontsize=14)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True)
        fig.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close(fig) # Close the figure to save memory
        logging.info(f"Plot saved successfully to {save_path}")
    except Exception as e:
        logging.error(f"Error during plotting {save_path}: {e}")

def run_evaluation_for_patch(
    teacher_model, student_model, ds_full, hparams, 
    mean_original, std_original, center_lat, center_lon, 
    scenario_name, plot_save_dir, device
):
    """
    This function processes one patch and generates TWO plots
    (Teacher-Only and Teacher+Student).
    """
    logging.info(f"\n===== Generating Plots for Scenario: {scenario_name} =====")
    
    patch_height, patch_width = 21, 28
    DEVICE = device 

    # --- 1. PROCESS THE NEW PATCH (using your working .isel logic) ---
    logging.info(f"Step 1: Extracting a {patch_height}x{patch_width} patch centered near ({center_lat}°N, {center_lon}°E)...")
    
    center_lat_idx = np.abs(ds_full.lat.values - center_lat).argmin()
    center_lon_idx = np.abs(ds_full.lon.values - center_lon).argmin()

    start_lat_idx = max(0, min(center_lat_idx - patch_height // 2, len(ds_full.lat) - patch_height))
    end_lat_idx = start_lat_idx + patch_height
    start_lon_idx = max(0, min(center_lon_idx - patch_width // 2, len(ds_full.lon) - patch_width))
    end_lon_idx = start_lon_idx + patch_width
    
    if start_lat_idx < 0 or end_lat_idx > len(ds_full.lat) or start_lon_idx < 0 or end_lon_idx > len(ds_full.lon):
        logging.warning(f"Skipping scenario '{scenario_name}': patch is too close to the dataset edge.")
        return

    ds_new_patch = ds_full.isel(lat=slice(start_lat_idx, end_lat_idx), lon=slice(start_lon_idx, end_lon_idx))
    
    new_patch_data_raw = ds_new_patch['sst'].values.astype(np.float32)
    
    # --- Check for valid patch size ---
    if new_patch_data_raw.shape[1:] != (patch_height, patch_width):
         logging.warning(f"Skipping scenario '{scenario_name}': extracted patch has wrong shape {new_patch_data_raw.shape[1:]}. Expected {(patch_height, patch_width)}")
         return
         
    new_patch_time_index = ds_new_patch.get_index("time")

    new_patch_data_filled = np.nan_to_num(new_patch_data_raw, nan=mean_original)
    new_patch_normalized = (new_patch_data_filled - mean_original) / std_original
    new_patch_normalized = new_patch_normalized[:, np.newaxis, :, :] # (T, C, H, W)

    in_len = hparams.dataset.in_len
    out_len = hparams.dataset.out_len
    input_seqs, target_seqs = create_sequences(new_patch_normalized, in_len, out_len)
    
    if input_seqs is None:
        logging.warning(f"Could not create sequences for '{scenario_name}'. Time series may be too short.")
        return
        
    logging.info(f"Created {len(input_seqs)} sequences for the new patch.")
    
    full_dataset = TensorDataset(input_seqs, target_seqs)
    batch_size = hparams.dataset.get('batch_size', 16) 
    num_workers = hparams.dataset.get('num_workers', 4)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- 2. RUN INFERENCE (Teacher and Student) ---
    logging.info("Step 2: Running model inference for Teacher and Student...")
    
    all_teacher_preds_norm = []
    all_student_preds_norm = []
    all_actuals_norm = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(full_loader, desc=f"Evaluating {scenario_name}"):
            x_batch = x_batch.to(DEVICE) # (B, 12, 1, H, W)
            y_batch = y_batch.to(DEVICE) # (B, 12, 1, H, W)

            # --- Teacher Pass ---
            input_seq_teacher = x_batch.permute(0, 1, 3, 4, 2) # (B, T, H, W, C)
            teacher_output_permuted = teacher_model(input_seq_teacher) # (B, 12, H, W, 1)
            teacher_output = teacher_output_permuted.permute(0, 1, 4, 2, 3) # (B, 12, 1, H, W)

            if student_type == "convlstm":
                student_input = teacher_output_permuted.permute(0, 1, 4, 2, 3) 
                student_pred_norm = student_model(student_input[:, 0:1, ...])
            else:
                student_input = teacher_output_permuted[:, 0:1, ...] 
                student_pred_norm_permuted = student_model(student_input)
                # Permute back to (B, T, C, H, W) for the mean calculation logic below
                student_pred_norm = student_pred_norm_permuted.permute(0, 1, 4, 2, 3)
            
            teacher_pred_step1 = teacher_output[:, 0:1, :, :, :]
            
            # --- Student Pass ---
            student_pred_step1 = student_model(teacher_pred_step1)
            
            target_step1 = y_batch[:, 0:1, :, :, :]

            # --- Spatially Average ---
            teacher_avg = teacher_pred_step1.mean(dim=(2, 3, 4))
            student_avg = student_pred_step1.mean(dim=(2, 3, 4))
            actual_avg = target_step1.mean(dim=(2, 3, 4))
            
            all_teacher_preds_norm.extend(teacher_avg.cpu().numpy())
            all_student_preds_norm.extend(student_avg.cpu().numpy())
            all_actuals_norm.extend(actual_avg.cpu().numpy())

    logging.info("Inference complete.")

    # --- 3. DE-NORMALIZE & PREPARE FOR PLOTTING ---
    logging.info("Step 3: De-normalizing data and preparing plots...")
    
    actuals_denorm = (np.array(all_actuals_norm) * std_original) + mean_original
    teacher_preds_denorm = (np.array(all_teacher_preds_norm) * std_original) + mean_original
    student_preds_denorm = (np.array(all_student_preds_norm) * std_original) + mean_original
    
    time_axis = new_patch_time_index[in_len : len(actuals_denorm) + in_len]
    
    # --- 4. Create Y-Ticks ---
    all_data = np.concatenate([actuals_denorm, teacher_preds_denorm, student_preds_denorm])
    y_min = np.floor(np.nanmin(all_data))
    y_max = np.ceil(np.nanmax(all_data))
    y_ticks = np.arange(y_min, y_max + 1, 1.0)
    
    # --- 5. Plot and Save Both Graphs ---
    
    # Plot 1: Teacher-Only
    plot_comparison(
        actuals_denorm, 
        teacher_preds_denorm, 
        time_axis,
        title=f"Teacher-Only Forecast vs. Actual\nPatch: {scenario_name}",
        save_path=os.path.join(plot_save_dir, f"teacher_only_{scenario_name}.png"),
        y_ticks=y_ticks,
        label="Predicted SST (Teacher-Only)"
    )
    
    # Plot 2: Teacher+Student
    plot_comparison(
        actuals_denorm, 
        student_preds_denorm, 
        time_axis,
        title=f"Generalized Student Forecast vs. Actual\nPatch: {scenario_name}",
        save_path=os.path.join(plot_save_dir, f"generalized_student_corrected_{scenario_name}.png"),
        y_ticks=y_ticks,
        label="Predicted SST (Generalized Student)"
    )

def get_args_parser():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Generalized Student Evaluation Script')
    parser.add_argument('--teacher_ckpt_path', type=str, required=True,
                        help='Path to the pre-trained Earthformer (Teacher) .ckpt file.')
    parser.add_argument('--student_ckpt_path', type=str, required=True,
                        help='Path to the *generalized* (multi-patch) Student .pth file.')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to the .yaml config file (e.g., sst.yaml).')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the single .nc file (e.g., sst.mon.mean.nc).')
    parser.add_argument('--plot_save_dir', type=str, default='evaluation_plots_generalized',
                        help='Directory to save the new comparison plots.')
    
    # Base patch args (for stats)
    parser.add_argument('--base_lat_slice', type=slice_type, default="15.625:20.625")
    parser.add_argument('--base_lon_slice', type=slice_type, default="65.625:72.375")
    parser.add_argument('--train_end_year', type=int, default=2015)
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation (cuda or cpu).')
    return parser

def compute_random_patches(ds, seed=42):
    np.random.seed(seed)
    patches = []
    def is_pure_ocean(lat_rng, lon_rng):
        sample = ds["sst"].isel(time=0).sel(lat=slice(*lat_rng), lon=slice(*lon_rng)).values
        return sample.size > 0 and not np.any(np.isnan(sample))
    return patches
    
def main():
    """
    Main function to load models once and loop through scenarios.
    """
    parser = get_args_parser()
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.plot_save_dir, exist_ok=True)

    # --- 1. LOAD TRAINED TEACHER MODEL ---
    logging.info(f"Loading Teacher model from checkpoint: {args.teacher_ckpt_path}")
    try:
        pl_module = CuboidSSTPLModule.load_from_checkpoint(
            args.teacher_ckpt_path,
            oc_file=args.cfg,
            save_dir=os.path.dirname(args.student_ckpt_path) # Use student dir as dummy
        )
        teacher_model = pl_module.torch_nn_module
        hparams = pl_module.hparams
        teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        logging.info("Teacher model loaded and frozen.")
    except Exception as e:
        logging.error(f"Error loading teacher checkpoint: {e}")
        exit(1)

    # --- 2. LOAD TRAINED GENERALIZED STUDENT MODEL ---
    logging.info(f"Loading Student model from checkpoint: {args.student_ckpt_path}")
    try:
        if args.student_ckpt_path.endswith('.pth'):
            # It's your manual ConvLSTM
            student_model = ConvLSTMStudent(
                input_dim=1, hidden_dim=12, kernel_size=(3, 3), num_layers=2
            ).to(device)
            student_model.load_state_dict(torch.load(args.student_ckpt_path, map_location=device))
            student_type = "convlstm"
        else:
            # It's an Earthformer Student (Lightning Checkpoint)
            student_pl = CuboidSSTPLModule.load_from_checkpoint(
                args.student_ckpt_path,
                oc_file=args.cfg,
                save_dir=os.path.dirname(args.student_ckpt_path)
            )
            student_model = student_pl.torch_nn_module.to(device)
            student_type = "earthformer"
        # student_model = ConvLSTMStudent(
        #     input_dim=1, hidden_dim=12, kernel_size=(3, 3), num_layers=2
        # ).to(device)
        # student_model.load_state_dict(torch.load(args.student_ckpt_path, map_location=device))
        student_model.eval()
        for param in student_model.parameters():
            param.requires_grad = False
        logging.info("Generalized ConvLSTM Student model loaded and frozen.")
    except Exception as e:
        logging.error(f"Error loading student checkpoint: {e}")
        exit(1)

    # --- 3. LOAD DATA & GET ORIGINAL STATS ---
    logging.info(f"Loading full dataset and calculating original normalization stats from: {args.data_path}")
    ds_full = xr.open_dataset(args.data_path)

    base_train_slice = slice(None, str(args.train_end_year))
    train_data_for_stats = ds_full['sst'].sel(
        time=base_train_slice, lat=args.base_lat_slice, lon=args.base_lon_slice
    ).values.astype(np.float32)
    
    mean_original = np.nanmean(train_data_for_stats)
    std_original = np.nanstd(train_data_for_stats)
    logging.info(f"Original Training Mean: {mean_original:.4f}, Original Training Std: {std_original:.4f}")

    # --- 4. Define 4 New *Test* Scenarios (P4, P5, P6, P7) ---
    # P3 (last training patch) was lon=82.5:89.25 (Center: 85.875)
    # 50% shift = 3.375 degrees. 100% shift = 6.75 degrees.
    scenarios = [
        # {
        #     "name": "P4_Test_50_Overlap",
        #     "center_lat": 18.125,
        #     "center_lon": 89.25 # 85.875 + 3.375
        # },
        # {
        #     "name": "P5_Test_0_Overlap",
        #     "center_lat": 18.125,
        #     "center_lon": 96.0 # 89.25 + 6.75
        # },
        # # --- NEW PATCHES ADDED HERE ---
        # {
        #     "name": "P6_Test_50_Overlap",
        #     "center_lat": 18.125,
        #     "center_lon": 99.375 # 96.0 + 3.375
        # },
        # {
        #     "name": "P7_Test_0_Overlap",
        #     "center_lat": 18.125,
        #     "center_lon": 106.125 # 99.375 + 6.75
        # },
    ]
    overlaps = [0, 30, 60, 90]
    LAT_DIM = 5.0
    LON_DIM = 6.75
    last_center_lat = 18.125 
    last_center_lon = 85.875
    for p in overlaps:
        stride = LAT_DIM * (1 - p / 100.0)
        scenarios.append({
            "name": f"South_Overlap_{p}pct",
            "center_lat": last_center_lat - stride,
            "center_lon": last_center_lon
        })
        
    for p in overlaps:
        stride = LON_DIM * (1 - p / 100.0)
        scenarios.append({
            "name": f"West_Overlap_{p}pct",
            "center_lat": last_center_lat,
            "center_lon": last_center_lon - stride
        })

    for name, lats, lons in random_results:
        scenarios.append({
            "name": name,
            "center_lat": (lats[0] + lats[1]) / 2,
            "center_lon": (lons[0] + lons[1]) / 2
        })
    
    # --- 5. Loop through scenarios and generate 8 plots ---
    for scenario in scenarios:
        run_evaluation_for_patch(
            teacher_model=teacher_model,
            student_type = student_type,
            student_model=student_model,
            ds_full=ds_full,
            hparams=hparams,
            mean_original=mean_original,
            std_original=std_original,
            center_lat=scenario["center_lat"],
            center_lon=scenario["center_lon"],
            scenario_name=scenario["name"],
            plot_save_dir=args.plot_save_dir,
            device=device
        )
    
    ds_full.close()
    logging.info("--- All generalization plots saved. ---")

if __name__ == '__main__':
    main()
