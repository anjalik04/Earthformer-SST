# ==============================================================================
# SCRIPT FOR TRAINING A *GENERALIZED* STUDENT (ConvLSTM) MODEL
#
# (Version 2 - Fixes RuntimeError by using index-based slicing)
#
# ==============================================================================

import sys
from unittest.mock import MagicMock
sys.modules["pkg_resources"] = MagicMock()

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import xarray as xr
import warnings

# --- Add Repository Root to Python Path ---
# This finds the 'src' folder by going up 3 levels
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

def get_args_parser():
    """Parses command line arguments for student training."""
    parser = argparse.ArgumentParser(description='Multi-Patch Teacher-Student Training')

    # --- Model Paths ---
    parser.add_argument('--teacher_ckpt_path', type=str, required=True,
                        help='Path to the pre-trained Earthformer (Teacher) .ckpt file.')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to the .yaml config file (e.g., sst.yaml).')
    parser.add_argument('--student_save_dir', type=str, default='checkpoints/student_generalized',
                        help='Directory to save the trained generalized ConvLSTM (Student) model.')
    
    # === MODIFICATION 1: ADD RESUME ARGUMENT ===
    parser.add_argument('--student_resume_ckpt', type=str, default=None,
                        help='Optional path to a previously trained Student model .pth file to resume training.')
    # ==========================================

    # --- Data Path (Single .nc file) ---
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the single .nc file (e.g., sst.mon.mean.nc).')
    
    # --- Base Patch Coordinate Arguments (for stats ONLY) ---
    parser.add_argument('--base_lat_slice', type=slice_type, default="15.625:20.625",
                        help='Latitude slice for Base Patch (for stats).')
    parser.add_argument('--base_lon_slice', type=slice_type, default="65.625:72.375",
                        help='Longitude slice for Base Patch (for stats).')

    # --- Training Parameters ---
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train the student model.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for student training.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the student model optimizer.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu).')
    
    # DataModule split args
    parser.add_argument('--train_end_year', type=int, default=2015, help='Last year of training data')
    parser.add_argument('--val_end_year', type=int, default=2020, help='Last year of validation data')

    return parser

def load_and_prep_multi_patch_data(args, hparams):
    in_len = hparams.dataset.in_len
    out_len = hparams.dataset.out_len
    if args.data_path.endswith('.pt'):
        logging.info(f"Detected .pt cache. Loading pre-processed data from: {args.data_path}")
        
        cache = torch.load(args.data_path, map_location="cpu", weights_only=False)
        
        data_tensor = cache['all_student_data']
        time_index = cache['time_index']
        train_mask = (time_index.year <= args.train_end_year)
        val_mask = (time_index.year == (args.train_end_year + 1))
        test_mask = (time_index.year > (args.train_end_year + 1))

        def slice_and_flatten(mask):
            if not any(mask): return None
            sliced = data_tensor[:, mask]
            return sliced.reshape(-1, *sliced.shape[2:])

        train_raw = slice_and_flatten(train_mask)
        val_raw = slice_and_flatten(val_mask)
        test_raw = slice_and_flatten(test_mask)
    
        # 3. Create Sequences
        all_x_train, all_y_train = create_sequences(train_raw.numpy(), in_len, out_len)
        all_x_val, all_y_val = create_sequences(val_raw.numpy(), in_len, out_len)
        
        # Handle the Test set carefully in case it's empty
        if test_raw is not None:
            all_x_test, all_y_test = create_sequences(test_raw.numpy(), in_len, out_len)
        else:
            all_x_test, all_y_test = None, None
        
        train_loader = DataLoader(TensorDataset(all_x_train, all_y_train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(TensorDataset(all_x_val, all_y_val), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = None
        if all_x_test is not None:
            test_loader = DataLoader(TensorDataset(all_x_test, all_y_test), 
                                     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            logging.info(f"Test Loader created with {len(test_loader)} batches.")
        else:
            logging.info("Test set is empty for the current date range.")
    
        return train_loader, val_loader, test_loader
    elif args.data_path.endswith('.nc'):
        logging.info(f"Detected .nc file. Slicing patches on the fly from: {args.data_path}")
        """
        Loads and prepares data from ALL specified patches and combines them
        into a single training and validation set.
        """
        logging.info(f"Loading full dataset from: {args.data_path}")
        ds_full = xr.open_dataset(args.data_path)
    
        # --- 1. Get Base Patch Stats ---
        logging.info("Calculating normalization stats from Base Patch...")
        base_train_slice = slice(None, str(args.train_end_year))
        train_data_for_stats = ds_full['sst'].sel(
            time=base_train_slice, lat=args.base_lat_slice, lon=args.base_lon_slice
        ).values.astype(np.float32)
        
        mean_base = np.nanmean(train_data_for_stats)
        std_base = np.nanstd(train_data_for_stats)
        logging.info(f"Base Patch Stats: Mean={mean_base:.4f}, Std={std_base:.4f}")
    
        # --- 2. Define ALL Patches for Training ---
        # (Base, P0, P1, P2, P3)
        # --- FIX: Using center_lat/lon to slice by index ---
        scenarios = [
            {"name": "Base_Patch", "center_lat": 18.125, "center_lon": 69.0},     # 65.625:72.375
            {"name": "P0_Shift",   "center_lat": 18.125, "center_lon": 75.75},    # 72.375:79.125
            {"name": "P1_Shift",   "center_lat": 18.125, "center_lon": 79.125},   # 75.75:82.5
            {"name": "P2_Shift",   "center_lat": 18.125, "center_lon": 82.5},     # 79.125:85.875
            {"name": "P3_Shift",   "center_lat": 18.125, "center_lon": 85.875}    # 82.5:89.25
        ]
    
        all_train_x, all_train_y = [], []
        all_val_x, all_val_y = [], []
    
        in_len = hparams.dataset.in_len
        out_len = hparams.dataset.out_len
        val_slice = slice(str(args.train_end_year + 1), str(args.val_end_year))
        
        patch_height, patch_width = 21, 28
    
        for scenario in scenarios:
            logging.info(f"Processing patch: {scenario['name']}...")
            
            # --- 3. Slice and Normalize Data (Robust Index Slicing) ---
            center_lat = scenario['center_lat']
            center_lon = scenario['center_lon']
            
            center_lat_idx = np.abs(ds_full.lat.values - center_lat).argmin()
            center_lon_idx = np.abs(ds_full.lon.values - center_lon).argmin()
    
            start_lat_idx = center_lat_idx - patch_height // 2
            end_lat_idx = start_lat_idx + patch_height
            start_lon_idx = center_lon_idx - patch_width // 2
            end_lon_idx = start_lon_idx + patch_width
            
            if start_lat_idx < 0 or end_lat_idx > len(ds_full.lat) or start_lon_idx < 0 or end_lon_idx > len(ds_full.lon):
                logging.warning(f"Skipping scenario '{scenario['name']}': patch is too close to the dataset edge.")
                continue
                
            ds_patch = ds_full.isel(lat=slice(start_lat_idx, end_lat_idx), 
                                     lon=slice(start_lon_idx, end_lon_idx))
            # --- End of Fix ---
            
            patch_data_raw = ds_patch['sst'].values.astype(np.float32)
            patch_time_index = ds_patch.get_index("time")
            
            # Verify shape
            if patch_data_raw.shape[1:] != (patch_height, patch_width):
                 logging.warning(f"Skipping scenario '{scenario['name']}': extracted patch has wrong shape {patch_data_raw.shape[1:]}. Expected {(patch_height, patch_width)}")
                 continue
    
            patch_data_filled = np.nan_to_num(patch_data_raw, nan=mean_base)
            patch_data_norm = (patch_data_filled - mean_base) / std_base
            patch_data_norm = patch_data_norm[:, np.newaxis, :, :] # (T, C, H, W)
    
            # --- 4. Create Train/Val Splits for this patch ---
            train_indices = patch_time_index.slice_indexer(base_train_slice.start, base_train_slice.stop)
            val_indices = patch_time_index.slice_indexer(val_slice.start, val_slice.stop)
            
            train_array = patch_data_norm[train_indices]
            val_array = patch_data_norm[val_indices]
    
            # --- 5. Create sequences and append to master lists ---
            train_x, train_y = create_sequences(train_array, in_len, out_len)
            val_x, val_y = create_sequences(val_array, in_len, out_len)
            
            if train_x is not None and val_x is not None:
                all_train_x.append(train_x)
                all_train_y.append(train_y)
                all_val_x.append(val_x)
                all_val_y.append(val_y)
                logging.info(f"  ...added {len(train_x)} train seqs, {len(val_x)} val seqs.")
            else:
                logging.warning(f"  ...could not create sequences for {scenario['name']}.")
    
        # --- 6. Concatenate all sequences into giant datasets ---
        logging.info("Concatenating all patch data...")
        combined_train_x = torch.cat(all_train_x, dim=0)
        combined_train_y = torch.cat(all_train_y, dim=0)
        combined_val_x = torch.cat(all_val_x, dim=0)
        combined_val_y = torch.cat(all_val_y, dim=0)
    
        logging.info(f"Total Training Sequences (from {len(all_train_x)} patches): {len(combined_train_x)}")
        logging.info(f"Total Validation Sequences (from {len(all_val_x)} patches): {len(combined_val_x)}")
        
        train_dataset = TensorDataset(combined_train_x, combined_train_y)
        val_dataset = TensorDataset(combined_val_x, combined_val_y)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
                                
        ds_full.close()
        return train_loader, val_loader
    else:
        raise ValueError(f"Unsupported file format: {args.data_path}. Must be .nc or .pt")


# --- train_one_epoch and validate_one_epoch are UNCHANGED ---

def train_one_epoch(teacher_model, student_model, dataloader, optimizer, criterion, device):
    """Runs a single training epoch based on your specified logic."""
    student_model.train()
    teacher_model.eval()
    
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training Epoch', leave=False)

    for input_seq, target_seq in progress_bar:
        # input_seq: (B, 12, 1, H, W)
        # target_seq: (B, 12, 1, H, W)
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        
        # --- 1. Permute for Teacher ---
        input_seq_teacher = input_seq.permute(0, 1, 3, 4, 2)

        # --- 2. Teacher Forward Pass (Frozen) ---
        with torch.no_grad():
            teacher_output_permuted = teacher_model(input_seq_teacher) # (B, 12, H, W, 1)
        
        # --- 3. Permute Back for Student ---
        teacher_output = teacher_output_permuted.permute(0, 1, 4, 2, 3) # (B, 12, 1, H, W)
        
        # --- 4. Get 1st Step (Student Input) ---
        student_input = teacher_output[:, 0:1, :, :, :] # (B, 1, 1, H, W)

        # --- 5. Get 1st Step (Student Target) ---
        student_target = target_seq[:, 0:1, :, :, :] # (B, 1, 1, H, W)

        # --- 6. Student Forward Pass (Trainable) ---
        optimizer.zero_grad()
        student_prediction = student_model(student_input) # (B, 1, 1, H, W)
        
        # --- 7. Loss and Backprop ---
        loss = criterion(student_prediction, student_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_loss

def validate_one_epoch(teacher_model, student_model, dataloader, criterion, device):
    """Runs a single validation epoch."""
    student_model.eval()
    teacher_model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            input_seq_teacher = input_seq.permute(0, 1, 3, 4, 2)
            teacher_output_permuted = teacher_model(input_seq_teacher)
            teacher_output = teacher_output_permuted.permute(0, 1, 4, 2, 3)
            
            student_input = teacher_output[:, 0:1, :, :, :]
            student_target = target_seq[:, 0:1, :, :, :]
            
            student_prediction = student_model(student_input)
            
            loss = criterion(student_prediction, student_target)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_loss

def main(args):
    """Main function to run the multi-patch student training."""
    
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # --- 1. Load & Freeze Teacher Model (Earthformer) ---
    logging.info(f"Loading Teacher model from checkpoint: {args.teacher_ckpt_path}")
    try:
        pl_module = CuboidSSTPLModule.load_from_checkpoint(
            args.teacher_ckpt_path,
            oc_file=args.cfg,
            save_dir=os.path.dirname(args.teacher_ckpt_path) # Dummy save_dir
        )
        teacher_model = pl_module.torch_nn_module
        hparams = pl_module.hparams
        teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        logging.info("Teacher model (CuboidTransformerModel) loaded and frozen.")
    except Exception as e:
        logging.error(f"Error loading teacher checkpoint: {e}")
        exit(1)
        
    # --- 2. Setup Data ---
    train_loader, val_loader = load_and_prep_multi_patch_data(args, hparams)
    
    logging.info(f"Combined DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    if len(train_loader) == 0:
        logging.error("Training DataLoader is empty. Check data paths and split years.")
        exit(1)

    total_samples = len(train_loader.dataset) + len(val_loader.dataset)
    # Samples per patch = Total Timesteps (2317) - Window Size (24) + 1 = 2294
    samples_per_patch = 2294 
    num_patches = round(total_samples / samples_per_patch)
    
    # --- 3. Initialize Student Model (ConvLSTM) ---
    student_model = ConvLSTMStudent(
        input_dim=1,
        hidden_dim=12,
        kernel_size=(3, 3),
        num_layers=2
    ).to(device)
    logging.info("ConvLSTM Student model initialized.")
    
    # === MODIFICATION 2: IMPLEMENT CHECKPOINT LOADING LOGIC ===
    if args.student_resume_ckpt:
        try:
            logging.info(f"Attempting to resume Student training from: {args.student_resume_ckpt}")
            student_model.load_state_dict(torch.load(args.student_resume_ckpt, map_location=device))
            logging.info("Student model state loaded successfully for resume.")
        except Exception as e:
            logging.error(f"Error loading student checkpoint for resume: {e}")
            sys.exit(1)
    # ==========================================================

    # --- 4. Setup Optimizer and Criterion ---
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # --- 5. Training Loop ---
    best_val_loss = float('inf')
    os.makedirs(args.student_save_dir, exist_ok=True)
    
    logging.info(f"--- Starting GENERALIZED Student Training (on {num_patches} patches) ---")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(teacher_model, student_model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(teacher_model, student_model, val_loader, criterion, device)
        
        logging.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the student's state dict
            save_path = os.path.join(args.student_save_dir, 'best_generalized_student.pth')
            torch.save(student_model.state_dict(), save_path)
            logging.info(f"New best model saved to {save_path}")

    logging.info("--- Generalized Student Training Finished ---")
    logging.info(f"Best validation loss: {best_val_loss:.6f}")
    logging.info(f"Trained generalized student model saved to: {os.path.join(args.student_save_dir, 'best_generalized_student.pth')}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
