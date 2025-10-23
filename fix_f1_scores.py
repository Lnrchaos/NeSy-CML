#!/usr/bin/env python3
"""
Practical fixes to actually improve F1 scores
"""

import torch
import torch.nn as nn
from train_chess import ChessDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

class BalancedChessLoss(nn.Module):
    """Balanced loss function to handle class imbalance"""
    
    def __init__(self, pos_weights=None, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.pos_weights = pos_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def forward(self, outputs, targets):
        # Use focal loss to handle imbalance
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs, targets.float(), pos_weight=self.pos_weights, reduction='none'
        )
        
        # Apply focal loss weighting
        pt = torch.exp(-bce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss
        
        return focal_loss.mean()

def create_balanced_dataloader(dataset, batch_size=4):
    """Create a balanced dataloader using weighted sampling"""
    
    # Calculate class weights
    all_labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        labels = sample['labels']
        all_labels.append(labels)
    
    all_labels = torch.stack(all_labels)
    
    # Calculate positive rates for each class
    pos_counts = all_labels.sum(dim=0)
    total_samples = len(dataset)
    
    # Create sample weights (higher weight for rare classes)
    sample_weights = []
    for i in range(len(dataset)):
        labels = all_labels[i]
        # Weight based on rarest positive class in this sample
        if labels.sum() > 0:
            # Find the rarest class in this sample
            active_classes = labels.nonzero().flatten()
            rarest_count = pos_counts[active_classes].min().item()
            weight = total_samples / (rarest_count + 1)  # +1 to avoid division by zero
        else:
            weight = 1.0  # Default weight for samples with no labels
        
        sample_weights.append(weight)
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset) * 3,  # Oversample to get more balanced batches
        replacement=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )

def calculate_pos_weights(dataset):
    """Calculate positive weights for balanced loss"""
    
    all_labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        labels = sample['labels']
        all_labels.append(labels)
    
    all_labels = torch.stack(all_labels)
    
    # Calculate positive and negative counts
    pos_counts = all_labels.sum(dim=0)
    neg_counts = len(dataset) - pos_counts
    
    # Calculate pos_weight (neg_count / pos_count)
    pos_weights = []
    for i in range(10):
        if pos_counts[i] > 0:
            weight = neg_counts[i] / pos_counts[i]
        else:
            weight = 1.0  # Default for classes with no positive samples
        pos_weights.append(weight)
    
    return torch.tensor(pos_weights)

def create_focused_dataset(dataset, focus_classes=None):
    """Create a dataset focused on specific classes to improve F1"""
    
    if focus_classes is None:
        # Focus on classes with reasonable sample counts
        focus_classes = ['opening', 'evaluation', 'pieces', 'checkmate']  # Classes with 3+ samples
    
    class_names = ['tactics', 'strategy', 'opening', 'endgame', 'pieces', 
                   'notation', 'middlegame', 'evaluation', 'checkmate', 'draw']
    
    focus_indices = [class_names.index(cls) for cls in focus_classes if cls in class_names]
    
    print(f"🎯 Focusing on classes: {focus_classes}")
    print(f"📊 Class indices: {focus_indices}")
    
    # Create new dataset with only focused classes
    focused_samples = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        original_labels = sample['labels']
        
        # Create new labels with only focused classes
        new_labels = torch.zeros(len(focus_classes))
        for j, class_idx in enumerate(focus_indices):
            new_labels[j] = original_labels[class_idx]
        
        # Only include samples that have at least one focused label
        if new_labels.sum() > 0:
            new_sample = sample.copy()
            new_sample['labels'] = new_labels
            focused_samples.append(new_sample)
    
    print(f"📚 Focused dataset: {len(focused_samples)} samples (from {len(dataset)})")
    
    return focused_samples, focus_classes

def test_improvements():
    """Test the F1 improvement strategies"""
    print("🚀 Testing F1 Improvement Strategies")
    print("=" * 50)
    
    # Load dataset
    dataset = ChessDataset()
    print(f"📚 Original dataset: {len(dataset)} samples")
    
    # Strategy 1: Calculate proper loss weights
    pos_weights = calculate_pos_weights(dataset)
    print(f"\n⚖️  Calculated positive weights:")
    class_names = ['tactics', 'strategy', 'opening', 'endgame', 'pieces', 
                   'notation', 'middlegame', 'evaluation', 'checkmate', 'draw']
    
    for i, (name, weight) in enumerate(zip(class_names, pos_weights)):
        print(f"  {name:12}: {weight:.2f}")
    
    # Strategy 2: Create focused dataset
    focused_samples, focus_classes = create_focused_dataset(dataset)
    
    # Strategy 3: Create balanced dataloader
    balanced_loader = create_balanced_dataloader(dataset, batch_size=2)
    
    print(f"\n🧪 Testing balanced sampling:")
    sample_count = 0
    class_counts = torch.zeros(10)
    
    for batch in balanced_loader:
        labels = batch['labels']
        class_counts += labels.sum(dim=0)
        sample_count += labels.size(0)
        
        if sample_count >= 20:  # Test first 20 samples
            break
    
    print(f"📊 Balanced sampling results (20 samples):")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        percentage = (count / sample_count) * 100
        print(f"  {name:12}: {count:4.1f} ({percentage:5.1f}%)")
    
    # Strategy 4: Create improved loss function
    balanced_loss = BalancedChessLoss(pos_weights=pos_weights)
    
    print(f"\n💡 Recommended Training Configuration:")
    print(f"  1. Use BalancedChessLoss with calculated pos_weights")
    print(f"  2. Use WeightedRandomSampler for balanced batches")
    print(f"  3. Focus on {len(focus_classes)} classes initially: {focus_classes}")
    print(f"  4. Increase batch size to 8-16 for better gradient estimates")
    print(f"  5. Use lower learning rate (1e-4) for stable training")
    
    return pos_weights, balanced_loss, focus_classes

if __name__ == "__main__":
    test_improvements()