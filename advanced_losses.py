"""
Advanced Loss Functions for NeSy-CML
Specialized loss functions optimized for F1 score and accuracy improvement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C) for classification or (N, C) for multilabel
            targets: Targets of shape (N, C) for multilabel or (N,) for classification
        """
        if inputs.dim() == 2 and targets.dim() == 2:
            # Multilabel case
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
            
            if self.alpha is not None:
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                alpha_t = self.alpha.unsqueeze(0).expand_as(targets) * targets + \
                         (1 - self.alpha.unsqueeze(0).expand_as(targets)) * (1 - targets)
                focal_loss = alpha_t * focal_loss
        else:
            # Classification case
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
            p_t = torch.exp(-ce_loss)
            focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for multilabel classification
    Good for imbalanced datasets
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Targets of shape (N, C) for multilabel
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        # Dice loss
        dice_loss = 1 - dice
        
        return dice_loss


class F1Loss(nn.Module):
    """
    F1 Loss - directly optimizes F1 score
    """
    
    def __init__(self, threshold: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Targets of shape (N, C) for multilabel
        """
        # Apply sigmoid
        probs = torch.sigmoid(inputs)
        
        # Convert to binary predictions
        preds = (probs > self.threshold).float()
        
        # Calculate TP, FP, FN
        tp = (preds * targets).sum(dim=0)
        fp = (preds * (1 - targets)).sum(dim=0)
        fn = ((1 - preds) * targets).sum(dim=0)
        
        # Calculate F1 per class
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # F1 loss (1 - F1)
        f1_loss = 1 - f1
        
        if self.reduction == 'mean':
            return f1_loss.mean()
        elif self.reduction == 'sum':
            return f1_loss.sum()
        else:
            return f1_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function: Focal + Dice + F1
    Optimized for multilabel classification with imbalanced classes
    """
    
    def __init__(self, 
                 focal_weight: float = 0.4,
                 dice_weight: float = 0.3,
                 f1_weight: float = 0.3,
                 focal_gamma: float = 2.0,
                 focal_alpha: Optional[torch.Tensor] = None,
                 dice_smooth: float = 1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.f1_weight = f1_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.f1_loss = F1Loss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Targets of shape (N, C) for multilabel
        
        Returns:
            Dictionary with total loss and component losses
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        f1 = self.f1_loss(inputs, targets)
        
        total_loss = (self.focal_weight * focal + 
                     self.dice_weight * dice + 
                     self.f1_weight * f1)
        
        return {
            'total_loss': total_loss,
            'focal_loss': focal,
            'dice_loss': dice,
            'f1_loss': f1
        }


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multilabel classification
    Different penalties for false positives vs false negatives
    """
    
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, 
                 clip: float = 0.05, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Targets of shape (N, C) for multilabel
        """
        # Calculate probabilities
        x_sigmoid = torch.sigmoid(inputs)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        # Asymmetric clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Basic BCE
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        
        # Asymmetric focusing
        pt = xs_pos * targets + xs_neg * (1 - targets)
        pt = pt.clamp(min=self.eps, max=1 - self.eps)
        
        # Apply asymmetric gamma
        asymmetric_w = torch.pow(1 - pt, self.gamma_pos * targets + self.gamma_neg * (1 - targets))
        loss *= asymmetric_w
        
        if self.reduction == 'mean':
            return -loss.mean()
        elif self.reduction == 'sum':
            return -loss.sum()
        else:
            return -loss


def create_chess_loss_function(class_weights: Optional[torch.Tensor] = None) -> CombinedLoss:
    """
    Create optimized loss function for chess multilabel classification
    
    Args:
        class_weights: Optional tensor of class weights for balancing
    
    Returns:
        CombinedLoss optimized for chess tasks
    """
    return CombinedLoss(
        focal_weight=0.4,
        dice_weight=0.3,
        f1_weight=0.3,
        focal_gamma=2.0,
        focal_alpha=class_weights,
        dice_smooth=1.0
    )


def create_imbalanced_loss_function(class_counts: List[int], 
                                   use_asymmetric: bool = False) -> nn.Module:
    """
    Create loss function optimized for imbalanced datasets
    
    Args:
        class_counts: List of sample counts per class
        use_asymmetric: Whether to use asymmetric loss
    
    Returns:
        Loss function module
    """
    # Calculate class weights (inverse frequency)
    total_samples = sum(class_counts)
    class_weights = torch.tensor([
        total_samples / (len(class_counts) * count) if count > 0 else 1.0
        for count in class_counts
    ], dtype=torch.float32)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    if use_asymmetric:
        return AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    else:
        return create_chess_loss_function(class_weights)


# Example usage
if __name__ == "__main__":
    print("🧠 Advanced Loss Functions for NeSy-CML")
    print("=" * 50)
    
    # Test with dummy data
    batch_size = 4
    num_classes = 9
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Test Combined Loss
    combined_loss = create_chess_loss_function()
    loss_dict = combined_loss(logits, targets)
    print(f"✅ Combined Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"   Focal: {loss_dict['focal_loss'].item():.4f}")
    print(f"   Dice: {loss_dict['dice_loss'].item():.4f}")
    print(f"   F1: {loss_dict['f1_loss'].item():.4f}")
    
    # Test Asymmetric Loss
    asym_loss = AsymmetricLoss()
    asym_loss_val = asym_loss(logits, targets)
    print(f"✅ Asymmetric Loss: {asym_loss_val.item():.4f}")
    
    print("\n🎉 Advanced loss functions ready!")

