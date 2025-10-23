#!/usr/bin/env python3
"""
Improvements to boost chess model accuracy and F1 scores
"""

import torch
import torch.nn as nn
import numpy as np

class ImprovedChessLoss(nn.Module):
    """Enhanced loss function to handle class imbalance better"""
    
    def __init__(self, class_weights=None, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Class weights based on your data distribution
        if class_weights is None:
            # Inverse frequency weighting based on your results
            self.class_weights = torch.tensor([
                2.0,  # tactics (23.9% -> weight 2.0)
                3.0,  # strategy (15.0% -> weight 3.0) 
                2.0,  # opening (20.3% -> weight 2.0)
                2.5,  # endgame (17.0% -> weight 2.5)
                1.0,  # pieces (54.7% -> weight 1.0)
                0.8,  # notation (73.6% -> weight 0.8)
                1.5,  # middlegame (29.8% -> weight 1.5)
                2.0,  # evaluation (21.3% -> weight 2.0)
                1.8,  # checkmate (26.6% -> weight 1.8)
                4.0   # draw (12.8% -> weight 4.0)
            ])
        else:
            self.class_weights = torch.tensor(class_weights)
    
    def forward(self, outputs, targets):
        # Move class weights to same device
        class_weights = self.class_weights.to(outputs.device)
        
        # Weighted binary cross entropy
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs, targets.float(), reduction='none'
        )
        
        # Apply class weights
        weighted_loss = bce_loss * class_weights.unsqueeze(0)
        
        # Focal loss component
        pt = torch.exp(-weighted_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * weighted_loss
        
        return focal_loss.mean()

def calculate_better_metrics(predictions, labels, class_names):
    """Calculate more informative metrics"""
    
    # Per-class metrics
    class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        if i < labels.size(1):
            class_labels = labels[:, i].bool()
            class_preds = predictions[:, i].bool()
            
            tp = (class_preds & class_labels).sum().item()
            fp = (class_preds & ~class_labels).sum().item()
            fn = (~class_preds & class_labels).sum().item()
            tn = (~class_preds & ~class_labels).sum().item()
            
            # Calculate metrics
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'support': tp + fn  # Number of true positives
            }
    
    # Overall metrics
    macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
    macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
    macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
    
    # Weighted metrics (by support)
    total_support = sum(m['support'] for m in class_metrics.values())
    weighted_f1 = sum(m['f1'] * m['support'] for m in class_metrics.values()) / total_support
    
    return {
        'class_metrics': class_metrics,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'weighted_f1': weighted_f1
    }

def print_detailed_results(metrics):
    """Print comprehensive results analysis"""
    
    print(f"\n📊 DETAILED PERFORMANCE ANALYSIS")
    print(f"=" * 60)
    
    # Sort classes by F1 score
    sorted_classes = sorted(
        metrics['class_metrics'].items(), 
        key=lambda x: x[1]['f1'], 
        reverse=True
    )
    
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Accuracy':<10} {'Support':<8}")
    print(f"-" * 60)
    
    for class_name, class_metrics in sorted_classes:
        print(f"{class_name:<12} "
              f"{class_metrics['precision']:<10.3f} "
              f"{class_metrics['recall']:<8.3f} "
              f"{class_metrics['f1']:<8.3f} "
              f"{class_metrics['accuracy']:<10.3f} "
              f"{class_metrics['support']:<8.0f}")
    
    print(f"\n🎯 OVERALL METRICS:")
    print(f"   Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"   Weighted F1:  {metrics['weighted_f1']:.4f}")
    print(f"   Macro Prec:   {metrics['macro_precision']:.4f}")
    print(f"   Macro Recall: {metrics['macro_recall']:.4f}")
    
    # Performance tiers
    excellent = [name for name, m in sorted_classes if m['f1'] >= 0.7]
    good = [name for name, m in sorted_classes if 0.5 <= m['f1'] < 0.7]
    needs_work = [name for name, m in sorted_classes if m['f1'] < 0.5]
    
    print(f"\n🏆 PERFORMANCE TIERS:")
    if excellent:
        print(f"   Excellent (F1≥0.7): {', '.join(excellent)}")
    if good:
        print(f"   Good (F1≥0.5):      {', '.join(good)}")
    if needs_work:
        print(f"   Needs Work (F1<0.5): {', '.join(needs_work)}")

# Quick improvements you can make:
QUICK_IMPROVEMENTS = """
🚀 QUICK WAYS TO BOOST YOUR ACCURACY:

1. 📊 Use Weighted Loss (biggest impact):
   - Rare classes (draw, strategy) get higher weights
   - Common classes (notation, pieces) get lower weights

2. 🎯 Adjust Classification Threshold:
   - Instead of 0.5, try 0.3 or 0.4 for rare classes
   - Use class-specific thresholds

3. 📚 Data Augmentation:
   - Extract more samples from rare class books
   - Paraphrase existing samples

4. 🔧 Model Architecture:
   - Add class-specific heads
   - Use attention mechanisms for rare classes

5. 📈 Ensemble Methods:
   - Train separate models for rare classes
   - Combine predictions

Your current results are actually quite good for multi-label classification!
F1 scores of 0.6-0.9 for several classes is excellent performance.
"""

if __name__ == "__main__":
    print(QUICK_IMPROVEMENTS)