"""
3D Medical Image Segmentation Training Script with Semi-supervised Learning Methods
Supports: Mean Teacher, Uncertainty-aware Mean Teacher, UGMCL, Entropy Minimization, ICT, URPC, LeFeD
"""

import argparse
import logging
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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import KFold
from medpy import metric
from skimage.measure import label
from tensorboardX import SummaryWriter

# Custom imports - UNCOMMENT THESE IN YOUR PROJECT
from networks.unet_3d import unet_3D, unet_3D_dt, unet_3D_dv_semi
from networks.my_net import LeFeD_Net
from utils.lefed import lefed
from utils import ramps, losses
from evaluate import evaluate_model
from dataloaders.la_heart1 import (
    RandomRotFlip, RandomCrop, ToTensor,
    TwoStreamBatchSampler, MDM,
)


"""
3D Medical Image Segmentation Training Script with Semi-supervised Learning Methods
Supports: Mean Teacher, Uncertainty-aware Mean Teacher, UGMCL, Entropy Minimization, ICT, URPC, LeFeD
"""

import argparse
import logging
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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import KFold
from medpy import metric
from skimage.measure import label
from tensorboardX import SummaryWriter

# Custom imports - UNCOMMENT THESE IN YOUR PROJECT
# from networks.unet_3d import unet_3D, unet_3D_dt, unet_3D_dv_semi
# from networks.my_net import LeFeD_Net
# from lefed import lefed
# from utils import ramps, losses
# from dataloaders.la_heart1 import (
#     RandomRotFlip, RandomCrop, ToTensor,
#     TwoStreamBatchSampler, MDM,
# )


# ==================== Configuration ====================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='3D Medical Image Segmentation with Semi-supervised Learning')
    
    # Data parameters
    parser.add_argument("--root_path", type=str, 
                       default="/mnt/siat140_disk2/alou/research/TMI_code/data_split/HCP",
                       help="Path to dataset")
    parser.add_argument("--exp", type=str, default="Unified_SSL_new", help="Experiment name")
    
    # Method parameters
    parser.add_argument("--method", type=str, default="lefed",
                       choices=["mean_teacher", "unc_mean_teacher", "ugmcl", 
                                "entropy_min", "ict", "urpc", "lefed"],
                       help="Semi-supervised learning method")
    
    # Training parameters
    parser.add_argument("--max_iterations", type=int, default=30000, help="Maximum iterations")
    parser.add_argument("--batch_size", type=int, default=4, help="Total batch size")
    parser.add_argument("--labeled_bs", type=int, default=2, help="Labeled batch size")
    parser.add_argument("--labeled_num", type=int, default=18, help="Number of labeled samples")
    parser.add_argument("--base_lr", type=float, default=0.01, help="Initial learning rate")
    
    # Model parameters
    parser.add_argument("--ema_decay", type=float, default=0.99, help="EMA decay rate")
    parser.add_argument("--consistency", type=float, default=0.1, help="Consistency weight")
    parser.add_argument("--patch_size", type=int, nargs="+", default=[112, 112, 80],
                       help="Input patch size")
    
    # Cross-validation parameters
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--deterministic", type=int, default=1, 
                       help="Whether to use deterministic algorithms")
    
    return parser.parse_args()


# ==================== Loss Functions ====================

class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation."""
    
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = 1e-5

    def _one_hot_encoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Convert label tensor to one-hot encoding."""
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, 
                softmax: bool = True) -> torch.Tensor:
        """
        Args:
            inputs: Network logits (B, C, H, W, D)
            target: Ground truth labels (B, H, W, D)
            softmax: Whether to apply softmax to inputs
        """
        if softmax:
            inputs = F.softmax(inputs, dim=1)
        
        target = self._one_hot_encoder(target)
        loss = 0.0
        
        for i in range(1, self.n_classes):  # Skip background
            intersect = torch.sum(inputs[:, i] * target[:, i])
            z_sum = torch.sum(inputs[:, i] * inputs[:, i])
            y_sum = torch.sum(target[:, i] * target[:, i])
            loss += (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
        
        return 1 - loss / (self.n_classes - 1)


def softmax_mse_loss(input_logits: torch.Tensor, 
                     target_logits: torch.Tensor) -> torch.Tensor:
    """MSE loss between softmax outputs."""
    return F.mse_loss(F.softmax(input_logits, dim=1), 
                      F.softmax(target_logits, dim=1))


def entropy_loss(pred: torch.Tensor) -> torch.Tensor:
    """Entropy minimization loss."""
    p = F.softmax(pred, dim=1)
    return -torch.mean(torch.sum(p * torch.log(p + 1e-6), dim=1))


def sigmoid_rampup(current: float, rampup_length: float) -> float:
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(iter_num: int, max_iterations: int, 
                                   base_weight: float) -> float:
    """Calculate current consistency weight with rampup."""
    return base_weight * sigmoid_rampup(iter_num, max_iterations)


def update_ema_variables(model: nn.Module, ema_model: nn.Module, 
                         alpha: float, global_step: int) -> None:
    """Update exponential moving average of model parameters."""
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


# ==================== Model Factory ====================

class ModelFactory:
    """Factory for creating models based on method."""
    
    @staticmethod
    def create_model(method: str, num_classes: int, in_channels: int = 2, 
                     ema: bool = False) -> nn.Module:
        """Create model instance."""
        is_urpc = method == "urpc"
        is_dt = method == "ugmcl"
        is_lfd = method == "lefed"
        
        # Uncomment and modify according to your actual model imports
        if is_urpc:
            model = unet_3D_dv_semi(n_classes=num_classes, in_channels=in_channels)
        elif is_dt:
            model = unet_3D_dt(n_classes=num_classes, in_channels=in_channels)
        elif is_lfd:
            model = LeFeD_Net(n_channels=in_channels, n_classes=num_classes,
                             normalization="batchnorm", has_dropout=True)
        else:
            model = unet_3D(n_classes=num_classes, in_channels=in_channels)
        
        model = model.cuda()
        
        if ema:
            for p in model.parameters():
                p.requires_grad = False
        
        return model


# ==================== Training Classes ====================

class MethodHandler:
    """Handle method-specific forward passes and loss calculations."""
    
    def __init__(self, method: str, model: nn.Module, ema_model: Optional[nn.Module],
                 num_classes: int, max_iterations: int, base_consistency: float):
        self.method = method
        self.model = model
        self.ema_model = ema_model
        self.num_classes = num_classes
        self.max_iterations = max_iterations
        self.base_consistency = base_consistency
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)
        
        # Method flags
        self.is_urpc = method == "urpc"
        self.is_dt = method == "ugmcl"
        self.is_lfd = method == "lefed"
        
        # Storage for intermediate outputs
        self.outputs = {}
    
    def forward(self, volume_batch: torch.Tensor, iter_num: int, 
                label_batch: Optional[torch.Tensor] = None) -> Dict:
        """Forward pass through model."""
        if self.is_urpc:
            dsv1, dsv2, dsv3, dsv4 = self.model(volume_batch)
            outputs_aux = [dsv1, dsv2, dsv3, dsv4]
            outputs_soft = [F.softmax(o, dim=1) for o in outputs_aux]
            ensemble_pred = torch.mean(torch.stack(outputs_soft), dim=0)
            
            self.outputs.update({
                'dsv1': dsv1,
                'outputs_aux': outputs_aux,
                'outputs_soft': outputs_soft,
                'ensemble_pred': ensemble_pred
            })
            
        elif self.is_dt:
            dist_map, seg_logits = self.model(volume_batch)
            self.outputs.update({
                'dist_map': dist_map,
                'seg_logits': seg_logits,
                'seg_soft': F.softmax(seg_logits, dim=1)
            })
            
        elif self.is_lfd:
            # LeFeD forward pass with empty list for prior storage
            seg_logits, y1, masks, stage_out1, _, _, _ = self.model(volume_batch, [])
            self.outputs.update({
                'seg_logits': seg_logits,
                'seg_soft': F.softmax(seg_logits, dim=1),
                'y1': y1,
                'masks': masks,
                'stage_out1': stage_out1
            })
            
        else:
            seg_logits = self.model(volume_batch)
            self.outputs.update({
                'seg_logits': seg_logits,
                'seg_soft': F.softmax(seg_logits, dim=1)
            })
        
        return self.outputs
    
    def compute_supervised_loss(self, labeled_bs: int, label_batch: torch.Tensor,
                                 fa_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute supervised loss on labeled data."""
        if self.is_urpc:
            loss_ce = self.ce_loss(self.outputs['dsv1'][:labeled_bs], 
                                   label_batch[:labeled_bs])
            loss_dice = self.dice_loss(self.outputs['dsv1'][:labeled_bs], 
                                       label_batch[:labeled_bs])
        else:
            loss_ce = self.ce_loss(self.outputs['seg_logits'][:labeled_bs], 
                                   label_batch[:labeled_bs])
            
            loss_dice = self.dice_loss(self.outputs['seg_logits'][:labeled_bs], 
                                       label_batch[:labeled_bs])
        
        return 0.5 * (loss_ce + loss_dice)
    
    def compute_consistency_loss(self, volume_batch: torch.Tensor, 
                                  labeled_bs: int, iter_num: int,
                                  label_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute consistency loss for unlabeled data."""
        if self.method == "mean_teacher":
            return self._mean_teacher_loss(volume_batch, labeled_bs)
        
        elif self.method == "unc_mean_teacher":
            return self._uncertainty_mean_teacher_loss(volume_batch, labeled_bs, iter_num)
        
        elif self.method == "entropy_min":
            return entropy_loss(self.outputs['seg_logits'][labeled_bs:])
        
        elif self.method == "ict":
            # Implement ICT loss
            return self._ict_loss(volume_batch, labeled_bs)
        
        elif self.method == "urpc":
            return self._urpc_loss(labeled_bs)
        
        elif self.method == "ugmcl":
            return self._ugmcl_loss(volume_batch, labeled_bs, iter_num)
        
        elif self.method == "lefed":
            # For LeFeD, the consistency loss is handled separately through the lefed function
            return torch.tensor(0.0).cuda()
        
        return torch.tensor(0.0).cuda()
    
    def compute_lefed_loss(self, volume_batch: torch.Tensor, label_batch: torch.Tensor,
                           iter_num: int) -> torch.Tensor:
        """
        Compute LeFeD loss using the imported lefed function.
        This is a separate method since lefed returns the total loss directly.
        """
        if self.method == "lefed":
            # Call the imported lefed function with the appropriate arguments
            # Note: The exact signature may need adjustment based on your lefed implementation
            total_loss = lefed(
                self.model,           # model
                volume_batch,         # volume batch
                label_batch,          # label batch
                iter_num,             # current iteration
                losses                 # losses module from utils
            )
            return total_loss
        return torch.tensor(0.0).cuda()
    
    def _mean_teacher_loss(self, volume_batch: torch.Tensor, 
                           labeled_bs: int) -> torch.Tensor:
        """Mean teacher consistency loss."""
        unlabeled_volume = volume_batch[labeled_bs:]
        noise = torch.clamp(torch.randn_like(unlabeled_volume) * 0.1, -0.2, 0.2)
        ema_input = unlabeled_volume + noise
        
        with torch.no_grad():
            ema_out = self.ema_model(ema_input)
        
        return softmax_mse_loss(self.outputs['seg_logits'][labeled_bs:], ema_out)
    
    def _uncertainty_mean_teacher_loss(self, volume_batch: torch.Tensor,
                                       labeled_bs: int, iter_num: int) -> torch.Tensor:
        """Uncertainty-aware mean teacher loss."""
        unlabeled_volume = volume_batch[labeled_bs:]
        
        # Standard mean teacher consistency
        noise = torch.clamp(torch.randn_like(unlabeled_volume) * 0.1, -0.2, 0.2)
        ema_input = unlabeled_volume + noise
        
        with torch.no_grad():
            ema_out = self.ema_model(ema_input)
        
        consistency_dist = softmax_mse_loss(
            self.outputs['seg_logits'][labeled_bs:], ema_out
        )
        
        # MC Dropout uncertainty estimation
        preds = []
        self.model.eval()
        for _ in range(8):
            noisy_input = unlabeled_volume + torch.clamp(
                torch.randn_like(unlabeled_volume) * 0.1, -0.2, 0.2
            )
            preds.append(self.ema_model(noisy_input))
        self.model.train()
        
        preds = torch.stack(preds)
        pred_soft = F.softmax(preds, dim=2).mean(0)
        uncertainty = -torch.sum(pred_soft * torch.log(pred_soft + 1e-6), 
                                 dim=1, keepdim=True) / np.log(2)
        
        # Dynamic threshold
        threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num, self.max_iterations)) * np.log(2)
        mask = (uncertainty < threshold).float()
        
        return torch.sum(mask * consistency_dist) / (torch.sum(mask) + 1e-8)
    
    def _ict_loss(self, volume_batch: torch.Tensor, labeled_bs: int) -> torch.Tensor:
        """Interpolation Consistency Training loss."""
        # TODO: Implement ICT loss
        # This would involve mixing labeled and unlabeled data
        # and enforcing consistency between mixed predictions
        return torch.tensor(0.0).cuda()
    
    def _urpc_loss(self, labeled_bs: int) -> torch.Tensor:
        """URPC consistency loss."""
        kl_div = nn.KLDivLoss(reduction="none")
        consistency_loss = 0.0
        
        unlabeled_pred = self.outputs['ensemble_pred'][labeled_bs:]
        
        for i in range(4):
            log_prob = F.log_softmax(self.outputs['outputs_aux'][i][labeled_bs:], dim=1)
            kl_val = kl_div(log_prob, unlabeled_pred.detach()).sum(dim=1, keepdim=True)
            weight = torch.exp(-kl_val)
            mse = F.mse_loss(self.outputs['outputs_soft'][i][labeled_bs:], 
                            unlabeled_pred, reduction="none")
            consistency_loss += (mse * weight).mean() + kl_val.mean()
        
        return consistency_loss / 4
    
    def _ugmcl_loss(self, volume_batch: torch.Tensor, labeled_bs: int, 
                    iter_num: int) -> torch.Tensor:
        """UGMCL (Uncertainty-Guided Mean Teacher with Distance Transform) loss."""
        unlabeled_volume = volume_batch[labeled_bs:]
        
        # MC Dropout uncertainty from EMA teacher
        self.model.eval()
        mc_preds = []
        with torch.no_grad():
            for _ in range(8):
                noisy_input = unlabeled_volume + torch.clamp(
                    torch.randn_like(unlabeled_volume) * 0.1, -0.2, 0.2
                )
                _, ema_seg = self.ema_model(noisy_input)
                mc_preds.append(F.softmax(ema_seg, dim=1))
        self.model.train()
        
        mc_preds = torch.stack(mc_preds)
        mean_pred = mc_preds.mean(0)
        uncertainty = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), 
                                 dim=1, keepdim=True)
        uncertainty = uncertainty / np.log(2)
        
        # Dynamic threshold
        threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num, self.max_iterations)) * np.log(2)
        mask = (uncertainty < threshold).float()
        
        # Mean teacher consistency
        with torch.no_grad():
            noisy_input = unlabeled_volume + torch.clamp(
                torch.randn_like(unlabeled_volume) * 0.1, -0.2, 0.2
            )
            _, ema_seg = self.ema_model(noisy_input)
            ema_seg_soft = F.softmax(ema_seg, dim=1)
        
        consistency_dist = F.mse_loss(
            self.outputs['seg_soft'][labeled_bs:], ema_seg_soft, reduction="none"
        )
        consistency_loss = torch.sum(mask * consistency_dist.mean(1, keepdim=True)) / (
            torch.sum(mask) + 1e-8
        )
        
        return consistency_loss
    
    def get_consistency_weight(self, iter_num: int) -> float:
        """Get current consistency weight."""
        return get_current_consistency_weight(iter_num, self.max_iterations, 
                                              self.base_consistency)


class Trainer:
    """Main trainer class."""
    
    def __init__(self, args: argparse.Namespace, snapshot_path: str, fold: int):
        self.args = args
        self.snapshot_path = Path(snapshot_path)
        self.fold = fold
        self.num_classes = 2
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(log_dir=str(self.snapshot_path / "log"))
        
        # Initialize method flags
        self.is_urpc = args.method == "urpc"
        self.is_dt = args.method == "ugmcl"
        self.is_lfd = args.method == "lefed"
        
        # Initialize data
        self._init_data()
        
        # Initialize models
        self._init_models()
        
        # Initialize optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=args.base_lr, 
            momentum=0.9, 
            weight_decay=1e-4
        )
        
        # Initialize method handler
        self.method_handler = MethodHandler(
            method=args.method,
            model=self.model,
            ema_model=self.ema_model,
            num_classes=self.num_classes,
            max_iterations=args.max_iterations,
            base_consistency=args.consistency
        )
    
    def _init_data(self):
        """Initialize datasets and dataloaders."""
        # Create dataset
        db_train = MDM(
            base_dir=self.args.root_path,
            split="train",
            transform=transforms.Compose([
                RandomRotFlip(),
                RandomCrop(self.args.patch_size),
                ToTensor()
            ])
        )
        
        # K-fold split
        kf = KFold(n_splits=self.args.n_splits, shuffle=True, 
                   random_state=self.args.seed)
        train_idx, val_idx = list(kf.split(db_train))[self.fold]
        
        self.train_dataset = Subset(db_train, train_idx)
        self.val_dataset = Subset(db_train, val_idx)
        
        # Split into labeled and unlabeled
        all_idx = np.arange(len(self.train_dataset))
        np.random.shuffle(all_idx)
        labeled_idxs = all_idx[:self.args.labeled_num].tolist()
        unlabeled_idxs = all_idx[self.args.labeled_num:].tolist()
        
        # Create batch sampler
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, 
            unlabeled_idxs, 
            self.args.batch_size,
            self.args.batch_size - self.args.labeled_bs
        )
        
        self.trainloader = DataLoader(
            self.train_dataset, 
            batch_sampler=batch_sampler, 
            num_workers=4, 
            pin_memory=True
        )
        
        self.valloader = DataLoader(
            self.val_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=1
        )
        
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")
        self.logger.info(f"Labeled samples: {len(labeled_idxs)}")
        self.logger.info(f"Iterations per epoch: {len(self.trainloader)}")
    
    def _init_models(self):
        """Initialize models."""
        self.model = ModelFactory.create_model(
            self.args.method, self.num_classes, in_channels=2, ema=False
        )
        
        self.ema_model = None
        if self.args.method not in ["entropy_min", "urpc", "lefed"]:  # LeFeD doesn't use EMA
            self.ema_model = ModelFactory.create_model(
                self.args.method, self.num_classes, in_channels=2, ema=True
            )
    
    def train(self) -> float:
        """Main training loop."""
        iter_num = 0
        best_dice = 0.0
        max_epoch = self.args.max_iterations // len(self.trainloader) + 1
        
        for epoch in tqdm(range(max_epoch), desc=f"Fold {self.fold}"):
            self.model.train()
            
            for batch in self.trainloader:
                # Prepare batch
                volume_batch = torch.cat((batch["image"], batch["image1"]), dim=1).cuda()
                fa_map = batch["image1"][:self.args.labeled_bs].cuda()
                label_batch = batch["label"].cuda()
                
                if self.args.method == "lefed":
                    # Special handling for LeFeD - the lefed function returns total loss directly
                    total_loss = self.method_handler.compute_lefed_loss(
                        volume_batch, label_batch, iter_num
                    )
                    
                    # Standard backward pass
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    
                    # No EMA update for LeFeD
                    
                else:
                    # Standard forward pass for other methods
                    self.method_handler.forward(volume_batch, iter_num)
                    
                    # Compute losses
                    sup_loss = self.method_handler.compute_supervised_loss(
                        self.args.labeled_bs, label_batch, fa_map
                    )
                    
                    consistency_loss = self.method_handler.compute_consistency_loss(
                        volume_batch, self.args.labeled_bs, iter_num, label_batch
                    )
                    
                    consistency_weight = self.method_handler.get_consistency_weight(iter_num)
                    
                    # Total loss
                    total_loss = sup_loss + consistency_weight * consistency_loss
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    
                    # Update EMA model
                    if self.ema_model is not None:
                        update_ema_variables(
                            self.model, self.ema_model, 
                            self.args.ema_decay, iter_num
                        )
                
                # Update learning rate
                self._update_learning_rate(iter_num)
                
                # Logging
                if self.args.method == "lefed":
                    self._log_training_lefed(iter_num, total_loss)
                else:
                    self._log_training(iter_num, total_loss, sup_loss, 
                                      consistency_loss, consistency_weight)
                
                # Validation
                if iter_num % 200 == 0 and iter_num > 0:
                    best_dice = self._validate(iter_num, best_dice)
                
                iter_num += 1
                if iter_num >= self.args.max_iterations:
                    break
            
            if iter_num >= self.args.max_iterations:
                break
        
        self.writer.close()
        return best_dice
    
    def _update_learning_rate(self, iter_num: int):
        """Update learning rate with polynomial decay."""
        lr = self.args.base_lr * (1.0 - iter_num / self.args.max_iterations) ** 0.9
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _log_training(self, iter_num: int, total_loss: torch.Tensor,
                      sup_loss: torch.Tensor, cons_loss: torch.Tensor,
                      cons_weight: float):
        """Log training metrics."""
        self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], iter_num)
        self.writer.add_scalar("loss/total", total_loss.item(), iter_num)
        self.writer.add_scalar("loss/sup", sup_loss.item(), iter_num)
        
        cons_val = cons_loss if isinstance(cons_loss, float) else cons_loss.item()
        self.writer.add_scalar("loss/cons", cons_val, iter_num)
        self.writer.add_scalar("cons_weight", cons_weight, iter_num)
    
    def _validate(self, iter_num: int, best_dice: float) -> float:
        """Run validation and save best model."""
        dice = evaluate_model(
            self.model,
            self.valloader,
            self.args.patch_size,
            stride_xy=18,
            stride_z=4,
            num_classes=self.num_classes,
            is_dt=self.is_dt,
            is_urpc=self.is_urpc,
            is_lfd=self.is_lfd
        )
        
        if dice > best_dice:
            best_dice = dice
            torch.save(
                self.model.state_dict(),
                self.snapshot_path / f"fold_{self.fold}_best.pth"
            )
        
        self.writer.add_scalar("val/dice", dice, iter_num)
        self.logger.info(f"Iter {iter_num} | Dice: {dice:.4f} | Best: {best_dice:.4f}")
        
        return best_dice


# ==================== Main ====================

def set_seed(seed: int, deterministic: bool):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def setup_logging(snapshot_path: Path):
    """Setup logging configuration."""
    logging.basicConfig(
        filename=str(snapshot_path / "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set up paths
    snapshot_path = Path(f"./model_unified/{args.exp}_{args.labeled_num}/{args.method}")
    snapshot_path.mkdir(parents=True, exist_ok=True)
    
    # Setup
    set_seed(args.seed, args.deterministic)
    setup_logging(snapshot_path)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info(f"Experiment: {args.exp}")
    logger.info(f"Method: {args.method}")
    logger