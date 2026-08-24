import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
from networks.my_net import LeFeD_Net
from dataloaders import utils
from utils import ramps, losses, metrics, test_3d_patch
from dataloaders.la_heart import (
    LAHeart,
    RandomCrop,
    CenterCrop,
    RandomRotFlip,
    ToTensor,
    TwoStreamBatchSampler,
    HCP,
    MDM,
)
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="./",
    help="Name of Experiment",
)
parser.add_argument("--exp", type=str, default="HCP", help="dataset_name")
parser.add_argument("--model", type=str, default="lefed_net", help="model_name")
parser.add_argument(
    "--max_iterations", type=int, default=15000, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
parser.add_argument(
    "--labeled_bs", type=int, default=2, help="labeled_batch_size per gpu"
)
parser.add_argument("--base_lr", type=float, default=0.01, help="learning rate")
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument(
    "--max_samples", type=int, default=12, help="maximum samples to train"
)
parser.add_argument("--labelnum", type=int, default=2, help="number of labeled data")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default="6", help="GPU to use")
parser.add_argument(
    "--temperature", type=float, default=0.05, help="temperature of sharpening"
)
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
parser.add_argument(
    "--consistency_rampup", type=float, default=40.0, help="consistency_rampup"
)
parser.add_argument(
    "--n_splits", type=int, default=5, help="Number of folds for cross-validation"
)

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model/" + args.exp + "_{}labels/".format(args.labelnum)

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(","))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs



def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P**T / (P**T + (1 - P) ** T)
    return P_sharpen




import torch.nn.functional as F


def gaussian_kernel_3d(size=5, sigma=1.0, device='cuda'):
    """Create 3D Gaussian kernel (differentiable)"""
    size = size // 2
    x, y, z = torch.meshgrid(
        torch.arange(-size, size+1, dtype=torch.float32, device=device),
        torch.arange(-size, size+1, dtype=torch.float32, device=device),
        torch.arange(-size, size+1, dtype=torch.float32, device=device),
        indexing='ij'
    )
    kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size*2+1, size*2+1, size*2+1)  # [1,1,D,H,W]

def compute_boundary_gt(label, sigma=1.0, thresh=0.3, kernel_size=5):
    """
    Compute approximate boundary map from label (foreground=1)
    - Uses 3D Gaussian convolution + gradient magnitude
    - Fully differentiable, runs on GPU
    """
    device = label.device
    
    # Ensure label is long and binary (0/1)
    label = (label > 0).long()  # foreground = 1
    
    # One-hot → take foreground channel [B,1,D,H,W]
    label_onehot = F.one_hot(label, num_classes=2).permute(0,4,1,2,3).float()[:,1:2]  # [B,1,D,H,W]
    
    # Create Gaussian kernel
    kernel = gaussian_kernel_3d(size=kernel_size, sigma=sigma, device=device)
    
    # Apply 3D Gaussian blur (padding to preserve size)
    padding = kernel_size // 2
    blurred = F.conv3d(
        label_onehot,
        kernel,
        padding=padding,
        groups=1
    )
    
    # Compute gradient magnitude (simple central difference)
    grad_x = torch.abs(blurred[:, :, 2:, :, :] - blurred[:, :, :-2, :, :])
    grad_y = torch.abs(blurred[:, :, :, 2:, :] - blurred[:, :, :, :-2, :])
    grad_z = torch.abs(blurred[:, :, :, :, 2:] - blurred[:, :, :, :, :-2])
    
    # Pad gradients back to original size
    grad_x = F.pad(grad_x, (0,0,0,0,1,1), mode='constant', value=0)
    grad_y = F.pad(grad_y, (0,0,1,1,0,0), mode='constant', value=0)
    grad_z = F.pad(grad_z, (1,1,0,0,0,0), mode='constant', value=0)
    
    grad = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
    
    # Threshold to get boundary map
    boundary = (grad > thresh).float()
    
    return boundary  # shape [B,1,D,H,W]


def lefed(model, volume_batch, label_batch, iter_num, losses):
    """
    Updated LeFeD training step with boundary-enhanced discrepancy feedback.
    Assumes model returns 7 items: outputs1, outputs2, masks, deep_seg1, deep_seg2, bound_maps1, bound_maps2
    """
    en_resized = None  # will be computed after first pass

    for num in range(3):
        # ──────────────────────────────────────────────────────────────
        # Forward pass
        # ──────────────────────────────────────────────────────────────
        if num == 0:
            outputs1, outputs2, masks, deep_seg1, deep_seg2, bound_maps1, bound_maps2 = model(volume_batch, [])
        else:
            outputs1, outputs2, masks, deep_seg1, deep_seg2, bound_maps1, bound_maps2 = model(volume_batch, en_resized)

        consistency_weight = get_current_consistency_weight(iter_num // 150)

        # ──────────────────────────────────────────────────────────────
        # Boundary-enhanced discrepancy using FULL-RES side outputs
        # ──────────────────────────────────────────────────────────────
        en = []  # full-resolution enhanced diffs (will be resized later)
        bound_weight = 1.5  # tune: 1.0 ~ 3.0 for boundary emphasis

        for idx in range(5):
            # Full-resolution segmentation logits from both decoders
            logit1 = deep_seg1[idx].detach()  # [B, n_classes, D, H, W]
            logit2 = deep_seg2[idx].detach()

            # Foreground probability difference (stable for boundaries)
            p1 = F.softmax(logit1, dim=1)[:, 1:2]   # [B, 1, D, H, W]
            p2 = F.softmax(logit2, dim=1)[:, 1:2]
            diff = p1 - p2

            # Boundary disagreement between decoders
            bound1 = bound_maps1[idx]  # [B, 1, D, H, W]
            bound2 = bound_maps2[idx]
            bound_diff = torch.abs(bound1 - bound2)

            # Enhanced discrepancy: diff boosted where boundaries disagree
            enhanced_diff = diff * (1.0 + bound_weight * bound_diff)

            en.append(1e-3 * enhanced_diff)  # small global scaling

        # ──────────────────────────────────────────────────────────────
        # NEW: Resize en to match encoder feature map sizes (critical!)
        # ──────────────────────────────────────────────────────────────
        en_resized = []
        current_size = list(volume_batch.shape[2:])  # [D, H, W] full res

        for i in range(5):
            # Process from shallowest (en[4]) to deepest (en[0])
            feat_full = en[4 - i]

            resized = F.interpolate(
                feat_full,
                size=current_size,
                mode='trilinear',
                align_corners=True
            )
            en_resized.append(resized)

            # Halve for next deeper encoder level
            current_size = [max(1, s // 2) for s in current_size]

        # Reverse to match encoder addition order: en_resized[0] = deepest (x5)
        en_resized = en_resized[::-1]

        # ──────────────────────────────────────────────────────────────
        # Boundary auxiliary loss (only on strong decoder = decoder1)
        # ──────────────────────────────────────────────────────────────
        boundary_gt = compute_boundary_gt(label_batch, sigma=1.0, thresh=0.5)

        boundary_loss = 0.0
        for b_pred in bound_maps1:  # multi-scale
            gt_resized = F.interpolate(
                boundary_gt,
                size=b_pred.shape[2:],
                mode='trilinear',
                align_corners=True
            )
            boundary_loss += F.binary_cross_entropy(b_pred, gt_resized)

        boundary_loss = boundary_loss / len(bound_maps1)

        # ──────────────────────────────────────────────────────────────
        # Existing losses — updated to use deep_seg1 for deep supervision
        # ──────────────────────────────────────────────────────────────
        # deep_seg1[0] = deepest side output, deep_seg1[4] = shallowest
        out5_soft = F.softmax(deep_seg1[0], dim=1)  # deepest
        out4_soft = F.softmax(deep_seg1[1], dim=1)
        out3_soft = F.softmax(deep_seg1[2], dim=1)
        out2_soft = F.softmax(deep_seg1[3], dim=1)
        out1_soft = F.softmax(deep_seg1[4], dim=1)  # shallowest

        outputs_soft1 = F.softmax(outputs1, dim=1)
        outputs_soft2 = F.softmax(outputs2, dim=1)

        # Supervised loss (main outputs)
        loss_sup1 = losses.dice_loss(
            outputs_soft1[:labeled_bs, 1, :, :, :],
            label_batch[:labeled_bs] == 1
        )
        loss_sup2 = losses.dice_loss(
            outputs_soft2[:labeled_bs, 1, :, :, :],
            label_batch[:labeled_bs] == 1
        )
        loss_sup = loss_sup1 + loss_sup2

        # Deep supervision (now from deep_seg1 side outputs)
        los1 = losses.dice_loss(out5_soft[:labeled_bs, 1], label_batch[:labeled_bs] == 1)   # deepest = higher weight
        los2 = losses.dice_loss(out4_soft[:labeled_bs, 1], label_batch[:labeled_bs] == 1)
        los3 = losses.dice_loss(out3_soft[:labeled_bs, 1], label_batch[:labeled_bs] == 1)
        los4 = losses.dice_loss(out2_soft[:labeled_bs, 1], label_batch[:labeled_bs] == 1)
        los5 = losses.dice_loss(out1_soft[:labeled_bs, 1], label_batch[:labeled_bs] == 1)   # shallowest = lower weight
        los = 0.8 * los1 + 0.6 * los2 + 0.4 * los3 + 0.2 * los4 + 0.1 * los5
        loss_ds = los

        loss_cons = losses.mse_loss(outputs_soft1, outputs_soft2)

        # ──────────────────────────────────────────────────────────────
        # Total loss — include boundary term
        # ──────────────────────────────────────────────────────────────
        loss = loss_sup + loss_ds + consistency_weight * loss_cons + 0.3 * boundary_loss  # tune 0.1~0.5

    return loss

