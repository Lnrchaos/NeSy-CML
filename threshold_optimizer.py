"""
Threshold Optimization for Multilabel Classification
Optimizes per-class thresholds to maximize F1 scores
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import f1_score, precision_recall_curve
from tqdm import tqdm


class ThresholdOptimizer:
    """
    Optimizes classification thresholds per class to maximize F1 scores
    """
    
    def __init__(self, num_classes: int, metric: str = 'f1'):
        """
        Args:
            num_classes: Number of output classes
            metric: Metric to optimize ('f1', 'f1_macro', 'f1_weighted')
        """
        self.num_classes = num_classes
        self.metric = metric
        self.optimal_thresholds = None
        self.threshold_history = []
    
    def optimize_thresholds(self, 
                           predictions: torch.Tensor,
                           targets: torch.Tensor,
                           num_thresholds: int = 100) -> Dict[str, float]:
        """
        Optimize thresholds for each class to maximize F1 score
        
        Args:
            predictions: Predicted probabilities of shape (N, C)
            targets: Ground truth labels of shape (N, C)
            num_thresholds: Number of threshold values to test
        
        Returns:
            Dictionary with optimal thresholds and metrics
        """
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        optimal_thresholds = np.zeros(self.num_classes)
        best_f1_scores = np.zeros(self.num_classes)
        
        # Optimize threshold for each class independently
        for class_idx in tqdm(range(self.num_classes), desc="Optimizing thresholds"):
            class_preds = predictions_np[:, class_idx]
            class_targets = targets_np[:, class_idx]
            
            # Skip if no positive samples
            if class_targets.sum() == 0:
                optimal_thresholds[class_idx] = 0.5
                best_f1_scores[class_idx] = 0.0
                continue
            
            # Test different thresholds
            thresholds = np.linspace(0.01, 0.99, num_thresholds)
            best_f1 = 0.0
            best_threshold = 0.5
            
            for threshold in thresholds:
                binary_preds = (class_preds >= threshold).astype(int)
                f1 = f1_score(class_targets, binary_preds, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            optimal_thresholds[class_idx] = best_threshold
            best_f1_scores[class_idx] = best_f1
        
        self.optimal_thresholds = optimal_thresholds
        
        return {
            'optimal_thresholds': optimal_thresholds,
            'per_class_f1': best_f1_scores,
            'macro_f1': best_f1_scores.mean(),
            'weighted_f1': self._calculate_weighted_f1(targets_np, best_f1_scores)
        }
    
    def _calculate_weighted_f1(self, targets: np.ndarray, f1_scores: np.ndarray) -> float:
        """Calculate weighted F1 score based on class frequencies"""
        class_counts = targets.sum(axis=0)
        total_samples = targets.shape[0]
        weights = class_counts / total_samples
        return np.average(f1_scores, weights=weights)
    
    def apply_thresholds(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Apply optimized thresholds to predictions
        
        Args:
            predictions: Predicted probabilities of shape (N, C)
        
        Returns:
            Binary predictions of shape (N, C)
        """
        if self.optimal_thresholds is None:
            # Default to 0.5 if not optimized
            return (predictions >= 0.5).float()
        
        predictions_np = predictions.detach().cpu().numpy()
        binary_preds = (predictions_np >= self.optimal_thresholds).astype(float)
        return torch.from_numpy(binary_preds).to(predictions.device)
    
    def optimize_using_precision_recall(self,
                                       predictions: torch.Tensor,
                                       targets: torch.Tensor) -> Dict[str, float]:
        """
        Optimize thresholds using precision-recall curve
        
        Args:
            predictions: Predicted probabilities of shape (N, C)
            targets: Ground truth labels of shape (N, C)
        
        Returns:
            Dictionary with optimal thresholds and metrics
        """
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        optimal_thresholds = np.zeros(self.num_classes)
        best_f1_scores = np.zeros(self.num_classes)
        
        for class_idx in tqdm(range(self.num_classes), desc="PR curve optimization"):
            class_preds = predictions_np[:, class_idx]
            class_targets = targets_np[:, class_idx]
            
            if class_targets.sum() == 0:
                optimal_thresholds[class_idx] = 0.5
                best_f1_scores[class_idx] = 0.0
                continue
            
            # Get precision-recall curve
            precision, recall, thresholds = precision_recall_curve(
                class_targets, class_preds
            )
            
            # Calculate F1 for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Find best threshold
            best_idx = np.argmax(f1_scores)
            optimal_thresholds[class_idx] = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_f1_scores[class_idx] = f1_scores[best_idx]
        
        self.optimal_thresholds = optimal_thresholds
        
        return {
            'optimal_thresholds': optimal_thresholds,
            'per_class_f1': best_f1_scores,
            'macro_f1': best_f1_scores.mean(),
            'weighted_f1': self._calculate_weighted_f1(targets_np, best_f1_scores)
        }
    
    def get_thresholds(self) -> np.ndarray:
        """Get current optimal thresholds"""
        if self.optimal_thresholds is None:
            return np.ones(self.num_classes) * 0.5
        return self.optimal_thresholds
    
    def save_thresholds(self, filepath: str):
        """Save thresholds to file"""
        if self.optimal_thresholds is not None:
            np.save(filepath, self.optimal_thresholds)
    
    def load_thresholds(self, filepath: str):
        """Load thresholds from file"""
        self.optimal_thresholds = np.load(filepath)


class AdaptiveThresholdOptimizer(ThresholdOptimizer):
    """
    Adaptive threshold optimizer that updates thresholds during training
    """
    
    def __init__(self, num_classes: int, update_frequency: int = 10, 
                 learning_rate: float = 0.1):
        super().__init__(num_classes)
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        self.step_count = 0
    
    def update_thresholds(self,
                         predictions: torch.Tensor,
                         targets: torch.Tensor) -> Dict[str, float]:
        """
        Update thresholds adaptively during training
        
        Args:
            predictions: Predicted probabilities
            targets: Ground truth labels
        
        Returns:
            Dictionary with updated thresholds and metrics
        """
        self.step_count += 1
        
        if self.step_count % self.update_frequency == 0:
            # Full optimization
            results = self.optimize_thresholds(predictions, targets)
            
            # Smooth update
            if self.optimal_thresholds is not None:
                old_thresholds = self.optimal_thresholds.copy()
                new_thresholds = results['optimal_thresholds']
                
                # Exponential moving average
                self.optimal_thresholds = (
                    (1 - self.learning_rate) * old_thresholds +
                    self.learning_rate * new_thresholds
                )
            else:
                self.optimal_thresholds = results['optimal_thresholds']
            
            return results
        else:
            # Return current thresholds
            return {
                'optimal_thresholds': self.get_thresholds(),
                'per_class_f1': np.zeros(self.num_classes),
                'macro_f1': 0.0,
                'weighted_f1': 0.0
            }


def create_threshold_optimizer(num_classes: int, 
                              adaptive: bool = False,
                              **kwargs) -> ThresholdOptimizer:
    """
    Factory function to create threshold optimizer
    
    Args:
        num_classes: Number of classes
        adaptive: Whether to use adaptive optimizer
        **kwargs: Additional arguments
    
    Returns:
        ThresholdOptimizer instance
    """
    if adaptive:
        return AdaptiveThresholdOptimizer(
            num_classes,
            update_frequency=kwargs.get('update_frequency', 10),
            learning_rate=kwargs.get('learning_rate', 0.1)
        )
    else:
        return ThresholdOptimizer(num_classes, metric=kwargs.get('metric', 'f1'))


# Example usage
if __name__ == "__main__":
    print("🎯 Threshold Optimizer for NeSy-CML")
    print("=" * 50)
    
    # Create dummy data
    batch_size = 100
    num_classes = 9
    
    predictions = torch.rand(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Test standard optimizer
    optimizer = create_threshold_optimizer(num_classes, adaptive=False)
    results = optimizer.optimize_thresholds(predictions, targets)
    
    print(f"✅ Optimized Thresholds:")
    for i, threshold in enumerate(results['optimal_thresholds']):
        print(f"   Class {i}: {threshold:.3f} (F1: {results['per_class_f1'][i]:.3f})")
    print(f"   Macro F1: {results['macro_f1']:.3f}")
    print(f"   Weighted F1: {results['weighted_f1']:.3f}")
    
    # Apply thresholds
    binary_preds = optimizer.apply_thresholds(predictions)
    print(f"\n✅ Applied thresholds: {binary_preds.shape}")
    
    print("\n🎉 Threshold optimizer ready!")

