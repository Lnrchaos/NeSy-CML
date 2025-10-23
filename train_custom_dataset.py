"""
Custom Dataset Training Script for NeuroSym-CML
Optimized for the specific dataset structure with labeled and unlabeled images
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

# Import NeuroSym-CML components
from meta_model import HybridModel, ModelSpec, ModelBuilder
from symbolic_controller import SymbolicController
from replay_buffer import ReplayBuffer
from evaluator import evaluate as evaluate_model
from architecture_selector import ArchitectureSelector

class CustomDataset(Dataset):
    """Custom dataset for labeled and unlabeled images"""
    
    def __init__(self, data_dir: str, label_dir: Optional[str] = None, transform=None, is_train: bool = True):
        """
        Initialize custom dataset
        
        Args:
            data_dir: Directory containing images
            label_dir: Directory containing labels (None for unlabeled data)
            transform: Image transformations
            is_train: Whether this is training data (affects data augmentation)
        """
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.transform = transform
        self.is_train = is_train
        
        # Collect all image files
        self.image_paths = []
        self.label_paths = []
        
        # Add root directory images
        self._add_images_from_dir(self.data_dir, is_labeled=True)
        
        # Add subdirectories (unlabeled)
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir() and subdir.name.lower() not in ['labels', 'label']:
                self._add_images_from_dir(subdir, is_labeled=False)
    
    def _add_images_from_dir(self, directory: Path, is_labeled: bool):
        """Add images from a directory to the dataset"""
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for img_path in directory.glob(ext):
                self.image_paths.append(img_path)
                
                # Find corresponding label if it exists
                label_path = None
                if is_labeled and self.label_dir:
                    label_name = f"{img_path.stem}.txt"
                    label_path = self.label_dir / label_name
                    if not label_path.exists():
                        label_path = None
                
                self.label_paths.append(label_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Load label if it exists
        label = 0  # Default label for unlabeled data
        if label_path and label_path.exists():
            try:
                # Assuming label is a single integer per line
                with open(label_path, 'r') as f:
                    label = int(f.readline().strip())
            except (ValueError, FileNotFoundError):
                pass
        
        return {
            'image': image,
            'label': label,
            'is_labeled': label_path is not None,
            'path': str(img_path)
        }

class CustomTrainer:
    """Custom trainer for the dataset structure"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Use the provided device or fall back to CUDA/CPU
        self.device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Enable TF32 for faster training on Ampere GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.scaler = torch.amp.GradScaler(init_scale=2.**16, enabled=config.get('mixed_precision', True)) if torch.cuda.is_available() else None
        
        # Initialize components
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        self.arch_selector = ArchitectureSelector()
        
        # Training state
        self.epoch = 0
        self.best_metric = float('inf')
        self.training_history = []
        
        print(f"🚀 Training initialized on {self.device}")
        print(f"Model architecture: {config.get('neural_architecture', 'custom_cnn')}")
        print(f"Batch size: {config.get('batch_size', 8)}")
        print(f"Mixed precision: {config.get('mixed_precision', True)}")
        print(f"Symbolic reasoning: {config.get('use_symbolic_reasoning', True)}")
        print(f"Rule set size: {config.get('rule_set_size', 100)}")
    
    def _create_model(self):
        """Create the model with memory optimizations"""
        # Create model specification
        spec = ModelSpec(
            neural_architecture=self.config.get('neural_architecture', 'custom_cnn'),
            num_classes=self.config.get('num_classes', 10),
            hidden_sizes=self.config.get('hidden_sizes', [128, 64]),
            use_symbolic_reasoning=self.config.get('use_symbolic_reasoning', True),
            memory_size=self.config.get('memory_size', 1000),
            rule_set_size=self.config.get('rule_set_size', 100),  # Increased default
            rule_embedding_dim=self.config.get('rule_embedding_dim', 64),
            meta_batch_size=self.config.get('meta_batch_size', 4),
            inner_lr=self.config.get('inner_lr', 0.001),
            outer_lr=self.config.get('outer_lr', 0.0001),
            memory_sampling_strategy=self.config.get('memory_sampling_strategy', 'random'),
            learning_rate=self.config.get('learning_rate', 0.001),
            device=str(self.device)
        )
        
        # Create the model
        model = HybridModel(spec)
        model.to(self.device)
        
        # Initialize symbolic controller for advanced reasoning
        # We'll initialize it with a default size and update it dynamically
        self.symbolic_controller = None
        self.symbolic_controller_initialized = False
        
        # Initialize replay buffer for continual learning
        self.replay_buffer = ReplayBuffer(
            memory_size=self.config.get('memory_size', 1000)
        )
        
        print(f"🧠 Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
        print(f"   - Architecture: {self.config.get('neural_architecture')}")
        print(f"   - Symbolic reasoning: {self.config.get('use_symbolic_reasoning', True)}")
        print(f"   - Rule set size: {self.config.get('rule_set_size', 100)}")
        print(f"   - Memory size: {self.config.get('memory_size', 1000)}")
        
        return model
    
    def _initialize_symbolic_controller(self, input_size: int):
        """Initialize symbolic controller with the correct input size"""
        if not hasattr(self, 'symbolic_controller_initialized') or not self.symbolic_controller_initialized:
            self.symbolic_controller = SymbolicController(
                num_rules=self.config.get('rule_set_size', 100),
                input_size=input_size,
                hidden_size=64,
                use_task_metadata=self.config.get('use_task_metadata', True),
                use_prior_state=self.config.get('use_prior_state', True),
                use_attention=self.config.get('use_attention', True)
            ).to(self.device)
            self.symbolic_controller_initialized = True
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for training"""
        params = list(self.model.parameters()) + list(self.symbolic_controller.parameters())
        return optim.AdamW(
            params,
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
    
    def train_epoch(self, train_loader: DataLoader):
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch + 1}', dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            is_labeled = batch['is_labeled'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.scaler is not None):
                # Get features from the backbone
                features = self.model.backbone(images)
                if isinstance(features, tuple):
                    features = features[0]  # Handle models that return (features, aux_output)
                
                # Flatten features if needed
                if features.dim() > 2:
                    features = features.mean([2, 3])  # Global average pooling
                
                # Initialize symbolic controller with correct input size if not done
                if not hasattr(self, 'symbolic_controller_initialized') or not self.symbolic_controller_initialized:
                    self._initialize_symbolic_controller(features.size(1))
                
                # Get rule indices from symbolic controller
                with torch.no_grad():
                    rule_indices, _ = self.symbolic_controller(features.detach())
                
                # Ensure rule_indices is a 1D tensor of indices
                if rule_indices.dim() > 1:
                    rule_indices = rule_indices.argmax(dim=1)
                rule_indices = rule_indices.to(self.device)
                
                # Forward pass through model
                outputs = self.model(
                    x=images,
                    text_embeddings=features,
                    rule_indices=rule_indices
                )
                
                # Only compute loss on labeled data
                if is_labeled.any():
                    loss = self.criterion(outputs[is_labeled], labels[is_labeled]) / accumulation_steps
                else:
                    # Skip this batch if no labeled data
                    continue
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Update metrics
            if is_labeled.any():
                total_loss += loss.item() * images.size(0) * accumulation_steps
                _, predicted = outputs.max(1)
                total += is_labeled.sum().item()
                correct += predicted[is_labeled].eq(labels[is_labeled]).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / total if total > 0 else 0,
                    'acc': 100. * correct / total if total > 0 else 0
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        self.symbolic_controller.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                is_labeled = batch['is_labeled'].to(self.device)
                
                # Forward pass
                features = self.model.extract_features(images)
                rule_indices = self.symbolic_controller(features)
                outputs = self.model(images, text_embeddings=features, rule_indices=rule_indices)
                
                # Only evaluate on labeled data
                if is_labeled.any():
                    loss = self.criterion(outputs[is_labeled], labels[is_labeled])
                    total_loss += loss.item() * is_labeled.sum().item()
                    _, predicted = outputs.max(1)
                    total += is_labeled.sum().item()
                    correct += predicted[is_labeled].eq(labels[is_labeled]).sum().item()
        
        # Calculate validation metrics
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'symbolic_controller_state_dict': self.symbolic_controller.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best if applicable
        if is_best:
            best_path = os.path.join('checkpoints', 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"\nSaved best model to {best_path}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
             num_epochs: int = 100):
        """Main training loop"""
        for epoch in range(self.epoch, self.epoch + num_epochs):
            self.epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate if validation loader is provided
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                
                # Update learning rate scheduler
                self.scheduler.step(val_metrics['val_loss'])
                
                # Check if this is the best model so far
                is_best = val_metrics['val_loss'] < self.best_metric
                if is_best:
                    self.best_metric = val_metrics['val_loss']
            else:
                is_best = False
            
            # Save checkpoint
            self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_accuracy']:.2f}%")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"Val Acc: {val_metrics['val_accuracy']:.2f}%")
            
            # Update training history
            self.training_history.append({
                'epoch': epoch,
                **train_metrics,
                **val_metrics
            })

def create_data_loaders(config: Dict) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create data loaders with memory-efficient settings"""
    # Optimized data loading settings
    num_workers = 0  # Keep at 0 to avoid memory issues with multiprocessing
    pin_memory = False  # Disable pin_memory to save VRAM
    
    # Clear any existing file handles
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("\n=== Loading Datasets ===")
    print(f"Data directory: {config['data_dir']}")
    print(f"Label directory: {config.get('label_dir', 'Not specified')}")
    
    train_dataset = CustomDataset(
        data_dir=config['data_dir'],
        label_dir=config.get('label_dir'),
        transform=train_transform,
        is_train=True
    )
    
    print(f"\nFound {len(train_dataset)} total images")
    labeled_count = sum(1 for p in train_dataset.label_paths if p is not None)
    print(f"- Labeled images: {labeled_count}")
    print(f"- Unlabeled images: {len(train_dataset) - labeled_count}")
    
    # Create data loaders with memory optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=1 if num_workers > 0 else None,  # Only set prefetch_factor if using workers
        drop_last=True
    )
    
    # Create validation loader if validation directory is provided
    val_loader = None
    if 'val_data_dir' in config:
        val_dataset = CustomDataset(
            data_dir=config['val_data_dir'],
            label_dir=config.get('val_label_dir'),
            transform=val_transform,
            is_train=False
        )
        
        print(f"\nFound {len(val_dataset)} validation images")
        val_labeled_count = sum(1 for p in val_dataset.label_paths if p is not None)
        print(f"- Labeled validation images: {val_labeled_count}")
        print(f"- Unlabeled validation images: {len(val_dataset) - val_labeled_count}")
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,  # Only set prefetch_factor if using workers
            drop_last=False
        )
    
    return train_loader, val_loader

def optimize_gpu_performance():
    """Optimize GPU performance settings"""
    if torch.cuda.is_available():
        # Set CUDA device to the first available GPU
        device = torch.device("cuda:0")
        
        # Clear CUDA cache and set memory settings
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        # Enable TF32 for Ampere GPUs if available
        if torch.cuda.get_device_capability(device)[0] >= 8:  # Ampere or newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Set GPU to high performance mode
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)  # Limit GPU memory to 80%
            print("✓ GPU memory limited to 80% of available VRAM")
        except Exception as e:
            print(f"⚠ Could not set GPU memory fraction: {e}")
        
        # Set higher process priority (Windows specific)
        try:
            import ctypes
            import os
            import sys
            
            # Constants for Windows process priority
            ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            HIGH_PRIORITY_CLASS = 0x00000080
            IDLE_PRIORITY_CLASS = 0x00000040
            NORMAL_PRIORITY_CLASS = 0x00000020
            REALTIME_PRIORITY_CLASS = 0x00000100
            
            # Set high priority for the Python process
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                HIGH_PRIORITY_CLASS
            )
            
            # Set high priority for the Python thread
            THREAD_PRIORITY_HIGHEST = 2
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(),
                THREAD_PRIORITY_HIGHEST
            )
            
            print("✓ GPU performance optimizations applied")
        except Exception as e:
            print(f"⚠ Could not set process priority: {e}")
        
        return device
    else:
        print("⚠ No CUDA device available, using CPU")
        return torch.device("cpu")

def main():
    """Main function"""
    # Optimize GPU performance
    device = optimize_gpu_performance()
    
    # Configuration
    config = {
        # Data configuration
        'data_dir': r'c:\Users\lyler\OneDrive\Gambit\dataset\images',
        'label_dir': r'c:\Users\lyler\OneDrive\Gambit\dataset\labels',
        'val_data_dir': r'c:\Users\lyler\OneDrive\Gambit\dataset\images',  # Same as train for now
        'val_label_dir': r'c:\Users\lyler\OneDrive\Gambit\dataset\labels',
        
        # Model configuration (optimized for 4GB VRAM)
        'neural_architecture': 'custom_cnn',  # Using custom CNN architecture
        'num_classes': 10,  # Update this based on your number of classes
        'hidden_sizes': [128, 64],  # Reduced hidden sizes
        'use_symbolic_reasoning': True,  # Enable symbolic reasoning
        'memory_size': 1000,  # Memory buffer size
        'rule_set_size': 100,  # Number of symbolic rules
        'rule_embedding_dim': 64,  # Dimension of rule embeddings
        'meta_batch_size': 4,  # Batch size for meta-learning
        'inner_lr': 0.001,  # Inner loop learning rate
        'outer_lr': 0.0001,  # Outer loop learning rate
        
        # Training configuration (optimized for 4GB VRAM)
        'batch_size': 2,  # Further reduced batch size
        'learning_rate': 0.0005,  # Reduced learning rate for stability
        'weight_decay': 1e-5,
        'num_epochs': 20,  # More epochs but with smaller batch size
        'gradient_accumulation_steps': 2,  # Simulate larger batch size
        'mixed_precision': True,
        
        # Symbolic controller configuration
        'num_rules': 50,
        'controller_hidden_size': 256,
        'use_attention': True,
        'controller_input_size': 512  # Should match model's feature size
    }
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Initialize trainer with the optimized device
    config['device'] = device
    trainer = CustomTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader, num_epochs=config['num_epochs'])
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
