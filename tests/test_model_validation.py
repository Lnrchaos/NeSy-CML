"""
Comprehensive Model Validation Test
Tests the trained NeuroSym-CML model on all 100 images to get full validation accuracy
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Import our components
from secure_image_dataset import SecureDataModule
from meta_model import HybridModel, ModelSpec

class ModelValidator:
    """Comprehensive model validation on all images"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        # Load model and config from checkpoint
        self.model, self.config = self._load_model()
        
        print(f"🔍 Model loaded from: {checkpoint_path}")
        print(f"💻 Device: {self.device}")
        print(f"🏗️  Architecture: {self.config['neural_architecture']}")
        print(f"📊 Classes: {self.config['num_classes']}")
    
    def _load_model(self):
        """Load model from checkpoint"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # Create model with same config
        spec = ModelSpec(
            neural_architecture=config['neural_architecture'],
            num_classes=config['num_classes'],
            hidden_sizes=config['hidden_sizes'],
            use_symbolic_reasoning=config.get('use_symbolic_reasoning', True),
            memory_size=config.get('memory_size', 1000),
            rule_set_size=config.get('rule_set_size', 100),
            learning_rate=config['learning_rate'],
            device=str(self.device)
        )
        
        model = HybridModel(spec)
        
        # Load state dict with strict=False to handle architecture differences
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError as e:
            print(f"⚠️  Strict loading failed, trying flexible loading: {e}")
            # Try loading with strict=False to ignore missing/extra keys
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                print(f"⚠️  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {unexpected_keys}")
        
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def create_full_dataset(self):
        """Create dataset with all images (no limit)"""
        print("📊 Creating full dataset with all images...")
        
        # Dataset paths
        images_dir = "../dataset/images"
        labels_dir = "../dataset/labels"
        classes_file = "../dataset/classes.txt"
        
        # Create data module with all images
        data_module = SecureDataModule(
            images_dir=images_dir,
            labels_dir=labels_dir,
            classes_file=classes_file,
            batch_size=4,  # Small batch size for memory efficiency
            max_samples=None,  # No limit - use all images
            cleanup_after_epoch=False  # Don't cleanup during validation
        )
        
        # Create dataloader
        train_loader, val_loader, dataset = data_module.create_dataloaders(train_split=0.0)  # Use all as validation
        
        print(f"📈 Total images loaded: {len(dataset)}")
        return val_loader, dataset
    
    def validate_model(self, dataloader: DataLoader) -> dict:
        """Comprehensive validation of the model"""
        print("🔍 Starting comprehensive validation...")
        
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        class_accuracy = {}
        class_counts = {}
        
        # Initialize class tracking
        for i in range(42):  # 42 classes
            class_accuracy[i] = 0
            class_counts[i] = 0
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
                # Move data to device
                images = batch['images'].to(self.device)
                texts = batch['texts'].to(self.device)
                rules = batch['rules'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, texts, rules)
                loss = F.binary_cross_entropy_with_logits(outputs, targets)
                
                # Calculate predictions
                predictions = torch.sigmoid(outputs) > 0.5
                confidences = torch.sigmoid(outputs)
                
                # Store for analysis
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                all_confidences.append(confidences.cpu())
                
                # Calculate accuracy
                batch_correct = (predictions == targets).float()
                batch_total = targets.size(0) * targets.size(1)
                
                total_loss += loss.item()
                correct_predictions += batch_correct.sum().item()
                total_predictions += batch_total
                
                # Per-class accuracy
                for i in range(targets.size(1)):  # For each class
                    class_targets = targets[:, i]
                    class_preds = predictions[:, i]
                    class_mask = class_targets == 1  # Only count positive examples
                    
                    if class_mask.sum() > 0:
                        class_correct = (class_preds[class_mask] == class_targets[class_mask]).sum().item()
                        class_accuracy[i] += class_correct
                        class_counts[i] += class_mask.sum().item()
        
        # Calculate final metrics
        avg_loss = total_loss / len(dataloader)
        overall_accuracy = correct_predictions / total_predictions
        
        # Calculate per-class accuracy
        per_class_accuracy = {}
        for i in range(42):
            if class_counts[i] > 0:
                per_class_accuracy[i] = class_accuracy[i] / class_counts[i]
            else:
                per_class_accuracy[i] = 0.0
        
        # Concatenate all predictions for detailed analysis
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_confidences = torch.cat(all_confidences, dim=0)
        
        return {
            'avg_loss': avg_loss,
            'overall_accuracy': overall_accuracy,
            'per_class_accuracy': per_class_accuracy,
            'class_counts': class_counts,
            'total_samples': len(all_predictions),
            'predictions': all_predictions,
            'targets': all_targets,
            'confidences': all_confidences
        }
    
    def print_detailed_results(self, results: dict, class_names: list):
        """Print detailed validation results"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE VALIDATION RESULTS")
        print("="*80)
        
        print(f"📈 Total Samples: {results['total_samples']}")
        print(f"📉 Average Loss: {results['avg_loss']:.4f}")
        print(f"🎯 Overall Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
        
        print(f"\n📋 Per-Class Accuracy:")
        print("-" * 80)
        print(f"{'Class ID':<8} {'Class Name':<20} {'Accuracy':<10} {'Samples':<10} {'Status'}")
        print("-" * 80)
        
        for i in range(42):
            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
            accuracy = results['per_class_accuracy'][i]
            count = results['class_counts'][i]
            status = "✅ Good" if accuracy > 0.7 else "⚠️  Needs Work" if accuracy > 0.3 else "❌ Poor"
            
            print(f"{i:<8} {class_name:<20} {accuracy:.4f}     {count:<10} {status}")
        
        # Find best and worst performing classes
        valid_classes = [(i, acc) for i, acc in results['per_class_accuracy'].items() if results['class_counts'][i] > 0]
        if valid_classes:
            best_class = max(valid_classes, key=lambda x: x[1])
            worst_class = min(valid_classes, key=lambda x: x[1])
            
            print(f"\n🏆 Best Performing Class:")
            print(f"   Class {best_class[0]} ({class_names[best_class[0]]}): {best_class[1]:.4f} ({best_class[1]*100:.2f}%)")
            
            print(f"\n⚠️  Worst Performing Class:")
            print(f"   Class {worst_class[0]} ({class_names[worst_class[0]]}): {worst_class[1]:.4f} ({worst_class[1]*100:.2f}%)")
        
        # Overall statistics
        valid_accuracies = [acc for acc in results['per_class_accuracy'].values() if acc > 0]
        if valid_accuracies:
            mean_class_accuracy = np.mean(valid_accuracies)
            std_class_accuracy = np.std(valid_accuracies)
            
            print(f"\n📊 Class Accuracy Statistics:")
            print(f"   Mean: {mean_class_accuracy:.4f} ({mean_class_accuracy*100:.2f}%)")
            print(f"   Std:  {std_class_accuracy:.4f} ({std_class_accuracy*100:.2f}%)")
    
    def run_validation(self):
        """Run complete validation test"""
        print("🚀 Starting Comprehensive Model Validation")
        print("="*60)
        
        try:
            # Create dataset with all images
            dataloader, dataset = self.create_full_dataset()
            
            # Load class names
            with open("../dataset/classes.txt", 'r') as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
            
            # Run validation
            results = self.validate_model(dataloader)
            
            # Print results
            self.print_detailed_results(results, class_names)
            
            # Cleanup
            dataset.final_cleanup()
            print(f"\n🧹 Dataset cleanup completed")
            
            return results
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return None

def main():
    """Main validation function"""
    print("🔍 NeuroSym-CML Model Validation Test")
    print("="*50)
    
    # Find the best model checkpoint
    checkpoint_dir = "checkpoints"
    best_model_path = os.path.join(checkpoint_dir, "best_secure_model.pt")
    
    if not os.path.exists(best_model_path):
        print(f"❌ Best model checkpoint not found: {best_model_path}")
        print("Available checkpoints:")
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pt'):
                    print(f"  - {file}")
        return
    
    # Create validator
    validator = ModelValidator(best_model_path)
    
    # Run validation
    results = validator.run_validation()
    
    if results:
        print(f"\n✅ Validation completed successfully!")
        print(f"🎯 Final Overall Accuracy: {results['overall_accuracy']*100:.2f}%")
    else:
        print(f"\n❌ Validation failed!")

if __name__ == "__main__":
    main()
