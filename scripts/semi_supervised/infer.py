import os
import argparse
import torch
import sys
import logging
import pandas as pd
import re
import numpy as np
import h5py  # Make sure to import h5py

from networks.my_net import LeFeD_Net
from test_util import test_all_case_all
from collections import OrderedDict
from networks.unet_3d import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default=r"F:\\hcp_ours_cn",
    help="Name of Experiment",
) # This is for fully supervised testing using 3D U-Net.
# parser.add_argument(
#     "--root_path",
#     type=str,
#     default=r"F:\\Alou_disk_1\\SOTA\\data_split\\HCP",
#     help="Name of Experiment",
# ) # This is for semi-supervised stage
parser.add_argument("--dataset_name", type=str, default="HCP", help="dataset_name")
parser.add_argument("--exp", type=str, default="Unified_SSL_sup", help="dataset_name")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument(
    "--detail", type=int, default=1, help="print metrics for every samples?"
)
parser.add_argument(
    "--method",
    type=str,
    default="unet_3D",
    choices=["unet_3D", "lefed"],
    help="SSL method",
)
parser.add_argument("--nms", type=int, default=1, help="apply NMS post-procssing?")
parser.add_argument("--labelnum", type=int, default=131, help="number of labeled data")
parser.add_argument(
    "--n_folds", type=int, default=5, help="Number of folds for cross-validation"
)
parser.add_argument(
    "--voxel_spacing", 
    type=float, 
    nargs=3, 
    default=[1.25, 1.25, 1.25],
    help="Default voxel spacing (x y z) if not found in HDF5 files. Example: --voxel_spacing 1.25 1.25 1.25"
)

FLAGS = parser.parse_args()
args = parser.parse_args()

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

# Fix the snapshot path - use os.path.join for proper path handling
snapshot_path = os.path.join(".", "model", f"{FLAGS.exp}_{FLAGS.labelnum}", FLAGS.method)
# Also create a Windows-friendly version with backslashes for display
snapshot_path_windows = snapshot_path.replace('/', '\\')

num_classes = 2
is_dt = FLAGS.method == "ugmcl"
is_urpc = FLAGS.method == "urpc"
is_lfd = FLAGS.method == "lefed"

# Fix the image list construction
test_list_path = os.path.join(FLAGS.root_path, "test.list")
print(f"Reading test list from: {test_list_path}")

with open(test_list_path, "r") as f:
    image_list = f.readlines()

# Clean up the image paths properly
image_list = []
with open(test_list_path, "r") as f:
    for line in f:
        # Remove whitespace and newlines
        line = line.strip()
        if line:
            # Extract the base filename without extension
            base_name = line.split('.')[0] if '.' in line else line
            # Construct the full path properly using os.path.join
            full_path = os.path.join(FLAGS.root_path, "data", f"{base_name}.h5")
            image_list.append(full_path)

print(f"Found {len(image_list)} test files")
print(f"First few test files:")
for i, path in enumerate(image_list[:3]):
    print(f"  {i}: {path}")

# Create snapshot directory if it doesn't exist
os.makedirs(snapshot_path, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(snapshot_path, 'test_log.txt')),
        logging.StreamHandler()
    ]
)

logging.info(f"Snapshot path: {snapshot_path}")
logging.info(f"Number of test files: {len(image_list)}")


def check_voxel_spacing_in_h5(image_list):
    """
    Check voxel spacing in HDF5 files and report statistics
    """
    spacing_values = []
    missing_spacing = []
    invalid_files = []
    
    for image_path in image_list:
        try:
            # Check if file exists first
            if not os.path.exists(image_path):
                logging.warning(f"File does not exist: {image_path}")
                invalid_files.append(image_path)
                continue
                
            with h5py.File(image_path, "r") as h5f:
                if "voxel_spacing" in h5f.attrs:
                    spacing = h5f.attrs["voxel_spacing"]
                    spacing_values.append(spacing)
                else:
                    missing_spacing.append(image_path)
        except Exception as e:
            logging.warning(f"Could not read {image_path}: {e}")
            invalid_files.append(image_path)
    
    if spacing_values:
        spacing_array = np.array(spacing_values)
        logging.info(f"Voxel spacing statistics:")
        logging.info(f"  Mean: {np.mean(spacing_array, axis=0)}")
        logging.info(f"  Std: {np.std(spacing_array, axis=0)}")
        logging.info(f"  Min: {np.min(spacing_array, axis=0)}")
        logging.info(f"  Max: {np.max(spacing_array, axis=0)}")
    
    if missing_spacing:
        logging.warning(f"Missing voxel spacing in {len(missing_spacing)} files")
        if FLAGS.voxel_spacing:
            logging.info(f"Using default voxel spacing: {FLAGS.voxel_spacing}")
        else:
            logging.warning("No default voxel_spacing provided, using (1.0, 1.0, 1.0)")
    
    if invalid_files:
        logging.error(f"Found {len(invalid_files)} invalid/unreadable files")
    
    return len(missing_spacing), len(spacing_values), len(invalid_files)


def test_calculate_metric():
    """
    Test model across all folds with voxel spacing support
    """
    # Check voxel spacing in HDF5 files
    missing_count, present_count, invalid_count = check_voxel_spacing_in_h5(image_list)
    logging.info(f"Files with voxel spacing: {present_count}, without: {missing_count}, invalid: {invalid_count}")
    
    if invalid_count > 0:
        logging.error(f"Found {invalid_count} invalid files. Please check the paths.")
        return []
    
    # List to store metrics for all folds
    all_metrics = []

    for fold in range(FLAGS.n_folds):
        logging.info(f"Processing fold {fold + 1}/{FLAGS.n_folds}")

        # Use os.path.join for proper path construction
        test_save_path = os.path.join(snapshot_path, f"test_bis_{fold}")
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
        logging.info(f"Test save path: {test_save_path}")

        # Initialize model based on method
        if is_urpc:
            net = unet_3D_dv_semi(n_classes=num_classes, in_channels=2).cuda()
        elif is_dt:
            net = unet_3D_dt(n_classes=num_classes, in_channels=2).cuda()
        elif is_lfd:
            net = LeFeD_Net(
                n_channels=2,
                n_classes=num_classes,
                normalization="batchnorm",
                has_dropout=False,
            ).cuda()
        else:
            net = unet_3D(n_classes=num_classes, in_channels=2).cuda()

        # Load model weights - use os.path.join
        save_mode_path = os.path.join(snapshot_path, f"fold_{fold}_best.pth")
        
        if not os.path.exists(save_mode_path):
            logging.error(f"Checkpoint not found: {save_mode_path}")
            continue

        logging.info(f"Loading weights from: {save_mode_path}")
        state_dict = torch.load(save_mode_path)
        
        # Handle DataParallel wrapping if needed
        if isinstance(state_dict, dict) and all(k.startswith('module.') for k in state_dict.keys()):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.'
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
            logging.info("Loaded DataParallel checkpoint without module prefix")
        else:
            net.load_state_dict(state_dict)
        
        logging.info(f"Initialized weights from {save_mode_path}")
        net.eval()

        # Run inference
        try:
            avg_metric, std_var = test_all_case_all(
                net,
                image_list,
                num_classes=num_classes,
                patch_size=(112, 112, 80),
                stride_xy=18,
                stride_z=4,
                is_dt=is_dt,
                is_urpc=is_urpc,
                is_lfd=is_lfd,
                save_result=True,
                test_save_path=test_save_path + os.sep,  # Add separator for the path
                metric_detail=FLAGS.detail,
                voxel_spacing=FLAGS.voxel_spacing
            )

            # Validate metric shapes
            if len(avg_metric) != 4 or len(std_var) != 4:
                logging.error(f"Unexpected metric shape for fold {fold}: mean={avg_metric}, std={std_var}")
                continue

            # Store metrics
            metrics_dict = {
                "fold": fold + 1,
                "dice_mean":   avg_metric[0],
                "jc_mean":     avg_metric[1],
                "hd95_mean":   avg_metric[2],
                "asd_mean":    avg_metric[3],
                "dice_std":    std_var[0],
                "jc_std":      std_var[1],
                "hd95_std":    std_var[2],
                "asd_std":     std_var[3],
            }

            all_metrics.append(metrics_dict)

            # Log fold results
            logging.info(f"Fold {fold + 1} metrics:")
            for k, v in metrics_dict.items():
                if isinstance(v, float):
                    logging.info(f"  {k:10}: {v:.4f}")
                else:
                    logging.info(f"  {k:10}: {v}")

        except Exception as e:
            logging.error(f"Error processing fold {fold}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Process and save results
    if not all_metrics:
        logging.error("No metrics collected. Check checkpoints and inference.")
        return []

    # Create DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Reorder columns for clarity
    columns_order = [
        "fold",
        "dice_mean", "dice_std",
        "jc_mean",   "jc_std",
        "hd95_mean", "hd95_std",
        "asd_mean",  "asd_std"
    ]
    metrics_df = metrics_df[columns_order]

    # Save to CSV
    csv_path = os.path.join(snapshot_path, "all_folds_metrics.csv")
    metrics_df.to_csv(csv_path, index=False, float_format="%.4f")
    logging.info(f"\nAll folds metrics saved to: {csv_path}")

    # # Calculate and save summary statistics
    # summary_stats = metrics_df.iloc[:, 1:].agg(['mean', 'std']).round(4)
    # summary_path = os.path.join(snapshot_path, "summary_statistics.csv")
    # summary_stats.to_csv(summary_path)
    # logging.info(f"Summary statistics saved to: {summary_path}")

    # Print average across folds
    avg_across_folds = metrics_df.iloc[:, 1:].mean()
    logging.info("\n" + "="*50)
    logging.info("Average across all folds:")
    for metric, value in avg_across_folds.items():
        logging.info(f"  {metric:15}: {value:.4f}")
    logging.info("="*50)

    return metrics_df


def compare_with_default_spacing():
    """
    Compare metrics with default spacing vs actual spacing
    Useful for debugging the impact of voxel spacing
    """
    logging.info("\n" + "="*50)
    logging.info("Voxel Spacing Impact Analysis")
    logging.info("="*50)
    
    # Check if all files have spacing
    all_have_spacing = True
    spacings = []
    
    for image_path in image_list:
        try:
            with h5py.File(image_path, "r") as h5f:
                if "voxel_spacing" in h5f.attrs:
                    spacings.append(h5f.attrs["voxel_spacing"])
                else:
                    all_have_spacing = False
        except:
            all_have_spacing = False
    
    if all_have_spacing and spacings:
        spacing_array = np.array(spacings)
        unique_spacings = np.unique(spacing_array, axis=0)
        
        if len(unique_spacings) == 1:
            logging.info(f"All files have consistent voxel spacing: {unique_spacings[0]}")
        else:
            logging.info(f"Found {len(unique_spacings)} different voxel spacings:")
            for i, spacing in enumerate(unique_spacings):
                count = np.sum(np.all(spacing_array == spacing, axis=1))
                logging.info(f"  Spacing {spacing}: {count} files")
    else:
        logging.warning("Some files are missing voxel spacing information or files are not accessible")


if __name__ == "__main__":
    logging.info("Starting test with voxel spacing support")
    logging.info(f"Arguments: {FLAGS}")
    
    # Verify that test files exist
    existing_files = [f for f in image_list if os.path.exists(f)]
    missing_files = [f for f in image_list if not os.path.exists(f)]
    
    logging.info(f"Total test files: {len(image_list)}")
    logging.info(f"Existing files: {len(existing_files)}")
    logging.info(f"Missing files: {len(missing_files)}")
    
    if missing_files:
        logging.warning(f"First few missing files: {missing_files[:5]}")
    
    if len(existing_files) == 0:
        logging.error("No test files found! Please check the paths.")
        sys.exit(1)
    
    # Perform voxel spacing analysis
    compare_with_default_spacing()
    
    # Run main test
    metrics = test_calculate_metric()
    
    if metrics is not None and not metrics.empty:
        logging.info(f"\nTest completed successfully. Results saved in {snapshot_path}")
        logging.info(f"Metrics shape: {metrics.shape}")
    else:
        logging.error("Test failed to produce metrics")