import math
import numpy as np
import torch
import os
import random
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from medpy import metric
from skimage.measure import label
from tqdm import tqdm

# ==================== Inference Functions ====================

class PatchInferenceEngine:
    """Handle patch-based inference for 3D volumes."""
    
    def __init__(self, patch_size: Tuple[int, int, int], stride_xy: int = 18, 
                 stride_z: int = 4, num_classes: int = 2):
        self.patch_size = patch_size
        self.stride_xy = stride_xy
        self.stride_z = stride_z
        self.num_classes = num_classes
    
    def _calculate_padding(self, image_shape: Tuple[int, int, int]) -> Tuple:
        """Calculate padding to make image evenly divisible by patch size."""
        pad = [(0, max(0, s - x)) for x, s in zip(image_shape, self.patch_size)]
        return [(p // 2, p - p // 2) for p in [sum(p) for p in zip(*pad)]]
    
    def _compute_grid(self, padded_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Compute sampling grid dimensions."""
        ww, hh, dd = padded_shape
        sx = math.ceil((ww - self.patch_size[0]) / self.stride_xy) + 1
        sy = math.ceil((hh - self.patch_size[1]) / self.stride_xy) + 1
        sz = math.ceil((dd - self.patch_size[2]) / self.stride_z) + 1
        return sx, sy, sz
    
    def __call__(self, model: nn.Module, image: np.ndarray, image1: np.ndarray,
                 is_dt: bool = False, is_urpc: bool = False, 
                 is_lfd: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform patch-based inference on a single case.
        
        Returns:
            label_map: Segmentation map
            score_map: Probability map
        """
        w, h, d = image.shape
        
        # Padding
        pad = self._calculate_padding(image.shape)
        if any(sum(p) > 0 for p in pad):
            image = np.pad(image, pad, mode="constant")
            image1 = np.pad(image1, pad, mode="constant")
        
        ww, hh, dd = image.shape
        sx, sy, sz = self._compute_grid((ww, hh, dd))
        
        score_map = np.zeros((self.num_classes, ww, hh, dd), dtype=np.float32)
        cnt = np.zeros((ww, hh, dd), dtype=np.float32)
        
        # Slide window
        for x in range(sx):
            xs = min(self.stride_xy * x, ww - self.patch_size[0])
            for y in range(sy):
                ys = min(self.stride_xy * y, hh - self.patch_size[1])
                for z in range(sz):
                    zs = min(self.stride_z * z, dd - self.patch_size[2])
                    
                    patch = self._extract_patch(image, image1, xs, ys, zs)
                    out = self._infer_patch(model, patch, is_dt, is_urpc, is_lfd)
                    
                    score_map[:, xs:xs+self.patch_size[0], 
                              ys:ys+self.patch_size[1], 
                              zs:zs+self.patch_size[2]] += out
                    cnt[xs:xs+self.patch_size[0], 
                        ys:ys+self.patch_size[1], 
                        zs:zs+self.patch_size[2]] += 1
        
        # Normalize and remove padding
        score_map = score_map / (cnt + 1e-8)
        label_map = np.argmax(score_map, axis=0)
        
        if any(sum(p) > 0 for p in pad):
            label_map = label_map[[slice(p[0], -p[1] or None) for p in pad]]
        
        return label_map, score_map
    
    def _extract_patch(self, image: np.ndarray, image1: np.ndarray, 
                       xs: int, ys: int, zs: int) -> torch.Tensor:
        """Extract and prepare a patch for inference."""
        patch = image[xs:xs+self.patch_size[0], 
                      ys:ys+self.patch_size[1], 
                      zs:zs+self.patch_size[2]]
        patch1 = image1[xs:xs+self.patch_size[0], 
                        ys:ys+self.patch_size[1], 
                        zs:zs+self.patch_size[2]]
        
        patch = torch.from_numpy(patch[None, None, ...]).float().cuda()
        patch1 = torch.from_numpy(patch1[None, None, ...]).float().cuda()
        return torch.cat((patch, patch1), dim=1)
    
    def _infer_patch(self, model: nn.Module, patch: torch.Tensor,
                     is_dt: bool, is_urpc: bool, is_lfd: bool) -> np.ndarray:
        """Run inference on a single patch."""
        with torch.no_grad():
            if is_urpc:
                out, _, _, _ = model(patch)
            elif is_dt:
                _, out = model(patch)
            elif is_lfd:
                out, y1, _, _, _, _, _ = model(patch, [])
            else:
                out = model(patch)
            
            out = F.softmax(out, dim=1).cpu().numpy()[0]  # [C, H, W, D]
        return out

def get_largest_cc(segmentation: np.ndarray) -> np.ndarray:
    """Extract largest connected component."""
    labels = label(segmentation)
    if labels.max() == 0:
        return segmentation
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_cc


def evaluate_model(model: nn.Module, dataloader: DataLoader, 
                   patch_size: Tuple[int, int, int], stride_xy: int = 18,
                   stride_z: int = 4, num_classes: int = 2,
                   is_dt: bool = False, is_urpc: bool = False,
                   is_lfd: bool = False) -> float:
    """
    Evaluate model on validation set.
    
    Returns:
        Average Dice score
    """
    model.eval()
    inference_engine = PatchInferenceEngine(patch_size, stride_xy, stride_z, num_classes)
    
    total_dice = 0.0
    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"][0, 0].cpu().numpy()
            image1 = batch["image1"][0, 0].cpu().numpy()
            label = batch["label"][0].cpu().numpy()
            
            pred, _ = inference_engine(model, image, image1, is_dt, is_urpc, is_lfd)
            
            # Post-process prediction
            pred = get_largest_cc(pred == 1)
            label = label == 1
            dice = metric.binary.dc(pred, label) if pred.sum() > 0 else 0.0
            total_dice += dice
    
    return total_dice / len(dataloader)