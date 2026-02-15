
# ==============================================================================
# SCRIPT FOR DEBUGGING DATA PIPELINE CONSISTENCY (CORRECTED & VERIFIED)
#
# This script loads the data processing pipeline from two different model
# checkpoints, extracts the "actual" (ground truth) normalized data from each,
# and compares them to ensure they are identical.
#
# It helps diagnose non-deterministic behavior in the data loading process,
# such as differences caused by dataloader shuffling.
#
# INSTRUCTIONS:
# 1. Save this file using the `%%writefile` command in a Colab cell.
# 2. Update the two CHECKPOINT_PATH variables in the main function below.
# 3. Run from a new Colab cell using the command:
#    !python debug_data_consistency.py
# ==============================================================================

import torch
import numpy as np
import os
from omegaconf import OmegaConf
import warnings

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", "This DataLoader will create .* worker processes .*")
warnings.filterwarnings("ignore", "torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

# Import custom modules from your project
from src.earthformer.datasets.sst.sst_datamodule import SSTDataModule
from train_cuboid_sst import CuboidSSTPLModule

def get_actual_data_for_checkpoint(checkpoint_path: str, cfg_path: str, save_dir: str):
    """
    Loads a model checkpoint, sets up its corresponding data pipeline,
    and extracts the actual normalized data from all data splits.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}.")

    print(f"\n--- Processing Checkpoint: {os.path.basename(checkpoint_path)} ---")

    # Step 1: Load Model to get its saved hyperparameters. This is the key to
    #         reproducing the exact data configuration used during training.
    print("  Loading model to access hparams...")
    model = CuboidSSTPLModule.load_from_checkpoint(
        checkpoint_path,
        oc_file=cfg_path,
        save_dir=save_dir,
        weights_only = False,
        strict = False
    )
    model.eval()

    # Step 2: Prepare the DataModule using the checkpoint's specific hparams.
    print("  Setting up datamodule from checkpoint configuration...")
    dataset_cfg = OmegaConf.to_object(model.hparams.dataset)
    # Clean up config for DataModule instantiation
    safe_keys = [
        "data_path", "img_size", "in_len", "out_len", "stride", 
        "batch_size", "num_workers", "pin_memory"
    ]
    
    # Construct the clean config: dataset_cfg = {Intersection of full_cfg and safe_keys}
    dataset_cfg = {k: full_dataset_cfg[k] for k in safe_keys if k in full_dataset_cfg}
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    print("  Data is ready.")

    # Step 3: Helper function to extract and process actual values from a dataloader.
    def extract_actuals(loader, split_name):
        print(f"    Extracting actuals from {split_name}...")
        all_actuals_onestep = []
        for i, batch in enumerate(loader):
            _, y = batch  # We only need the actual data 'y'
            # Spatially average the data to get a single time-series value
            avg_actuals = y.mean(dim=[3, 4])
            # Select the first time step and convert to NumPy array
            all_actuals_onestep.append(avg_actuals[:, 0].cpu().numpy())
        return np.concatenate(all_actuals_onestep)

    # Step 4: Extract data from all dataloaders.
    train_actuals = extract_actuals(train_loader, "train_loader")
    val_actuals = extract_actuals(val_loader, "val_loader")
    test_actuals = extract_actuals(test_loader, "test_loader")
    
    print(f"--- Finished processing {os.path.basename(checkpoint_path)} ---")
    return train_actuals, val_actuals, test_actuals

def compare_data_pipelines():
    """
    Main function to define two checkpoints and compare their data pipelines.
    """
    # --- 1. CONFIGURATION ---
    print("Step 1: Configuring paths...")
    
    CFG_PATH_1= "/kaggle/working/Earthformer-SST/scripts/cuboid_transformer/sst/sst.yaml"
    CFG_PATH_2= "/kaggle/working/Earthformer-SST/scripts/cuboid_transformer/sst/sst_distill_earthformer.yaml"
    
    # !!! UPDATE THESE PATHS !!!
    # Define the two checkpoints you want to compare
    CHECKPOINT_PATH_1 = "/kaggle/working/Earthformer-SST/scripts/cuboid_transformer/sst/experiments/convlstm_southward_run/best_generalized_student.ckpt"
    CHECKPOINT_PATH_2 = "/kaggle/working/Earthformer-SST/scripts/cuboid_transformer/sst/experiments/sst_southward_run/checkpoints/student-epoch=043.ckpt" # Example: a different epoch
    
    # This usually stays the same if checkpoints are from the same run
    save_directory_name = "consistency_check"

    # --- 2. GET DATA FROM BOTH CHECKPOINTS ---
    print("\nStep 2: Getting data from the first checkpoint pipeline...")
    train1, val1, test1 = get_actual_data_for_checkpoint(
        checkpoint_path=CHECKPOINT_PATH_1,
        cfg_path=CFG_PATH_1,
        save_dir=save_directory_name
    )

    print("\nStep 3: Getting data from the second checkpoint pipeline...")
    train2, val2, test2 = get_actual_data_for_checkpoint(
        checkpoint_path=CHECKPOINT_PATH_2,
        cfg_path=CFG_PATH_2,
        save_dir=save_directory_name
    )

    # --- 4. COMPARE THE DATASETS ---
    print("\nStep 4: Comparing the extracted actual data arrays...")
    
    # Using np.isclose is the correct way to compare floating point numbers,
    # as it accounts for tiny precision differences. '!=' is not reliable.
    # We count the number of elements that are NOT close.
    diff_train = np.sum(~np.isclose(train1, train2))
    diff_val = np.sum(~np.isclose(val1, val2))
    diff_test = np.sum(~np.isclose(test1, test2))
    
    print("\n==================== COMPARISON RESULTS ====================")
    print(f"Number of differing data points in TRAIN set:      {diff_train}")
    print(f"Number of differing data points in VALIDATION set:  {diff_val}")
    print(f"Number of differing data points in TEST set:        {diff_test}")
    print("==========================================================")
    
    if diff_train == 0 and diff_val == 0 and diff_test == 0:
        print("\nConclusion: The data pipelines are consistent. The actual data is identical.")
    else:
        print("\nConclusion: The data pipelines are NOT consistent.")
        if diff_train > 0:
            print("INFO: The difference in the TRAIN set is expected if your train_dataloader uses 'shuffle=True'.")
        if diff_val > 0 or diff_test > 0:
            print("WARNING: The difference in the VALIDATION or TEST sets is UNEXPECTED and points to a data consistency issue. These dataloaders should have 'shuffle=False'.")

if __name__ == '__main__':
    compare_data_pipelines()
