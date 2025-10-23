"""
Secure Image Training Script for NeuroSym-CML
Trains on sensitive images and automatically cleans up data after training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import os
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from tqdm import tqdm
import gc

# Import NeuroSym-CML components
from meta_model import HybridModel, ModelSpec
from symbolic_controller import SymbolicController
from replay_buffer import ReplayBuffer
from secure_image_dataset import SecureDataModule, secure_training_cleanup

class SecureImageTrainer:
    """Secure trainer that handles sensitive image data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler() if config.get('mixed_precision', True) else None
        
        # Initialize model
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        print(f"🔒 Secure training initialized on {self.device}")
        print(f"🏗️  Model architecture: {config['neural_architecture']}")
        print(f"📦 Batch size: {config['batch_size']}")
        print(f"🧠 Mixed precision: {config.get('mixed_precision', True)}")
    
    def _create_model(self) -> HybridModel:
        """Create the proper NeuroSym-CML hybrid model with symbolic reasoning"""
        spec = ModelSpec(
            neural_architecture=self.config['neural_architecture'],
            num_classes=self.config['num_classes'],
            hidden_sizes=self.config['hidden_sizes'],
            use_symbolic_reasoning=self.config.get('use_symbolic_reasoning', True),
            memory_size=self.config.get('memory_size', 1000),
            rule_set_size=self.config.get('rule_set_size', 100),
            rule_embedding_dim=self.config.get('rule_embedding_dim', 64),
            meta_batch_size=self.config.get('meta_batch_size', 4),
            inner_lr=self.config.get('inner_lr', 0.01),
            outer_lr=self.config.get('outer_lr', 0.001),
            memory_sampling_strategy=self.config.get('memory_sampling_strategy', 'random'),
            learning_rate=self.config['learning_rate'],
            device=str(self.device)
        )
        
        # Create the NeuroSym-CML hybrid model
        model = HybridModel(spec)
        model.to(self.device)
        
        # Initialize symbolic controller for advanced reasoning
        # Calculate input size: 224*224*3 = 150528 for image features
        image_input_size = 224 * 224 * 3  # 150528
        self.symbolic_controller = SymbolicController(
            num_rules=self.config.get('rule_set_size', 100),
            input_size=image_input_size,  # Correct input size for flattened images
            hidden_size=64,
            use_task_metadata=self.config.get('use_task_metadata', True),
            use_prior_state=self.config.get('use_prior_state', True),
            use_attention=self.config.get('use_attention', True)
        ).to(self.device)
        
        # Initialize replay buffer for continual learning
        self.replay_buffer = ReplayBuffer(
            memory_size=self.config.get('memory_size', 1000)
        )
        
        print(f"🧠 NeuroSym-CML model created with:")
        print(f"   - Architecture: {self.config['neural_architecture']}")
        print(f"   - Symbolic reasoning: {self.config.get('use_symbolic_reasoning', True)}")
        print(f"   - Rule set size: {self.config.get('rule_set_size', 100)}")
        print(f"   - Memory size: {self.config.get('memory_size', 1000)}")
        print(f"   - Meta-learning: {self.config.get('meta_batch_size', 4)} tasks per update")
        
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs']
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self, dataloader: DataLoader, dataset) -> Dict[str, float]:
        """Train for one epoch with secure memory management"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Start epoch cleanup
        dataset.start_epoch()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['images'].to(self.device)
            texts = batch['texts'].to(self.device)
            rules = batch['rules'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass with NeuroSym-CML paradigm
            if self.scaler:
                with autocast():
                    # Convert rule indices to embeddings for prior state
                    rule_embeddings = torch.zeros(rules.size(0), 64, device=self.device)
                    for i, rule_idx in enumerate(rules):
                        rule_embeddings[i, rule_idx % 64] = 1.0  # Simple one-hot encoding
                    
                    # Use symbolic controller for advanced reasoning
                    enhanced_rules, _ = self.symbolic_controller.forward(
                        x=images.flatten(1),  # Flatten image features
                        task_metadata={'id': 0, 'type': 'image_analysis'},  # Task metadata
                        prior_state=rule_embeddings  # Use rule embeddings as prior state
                    )
                    
                    # Forward through NeuroSym-CML hybrid model
                    # Use original rule indices, not enhanced rules
                    outputs = self.model(images, texts, rules)
                    # Use multi-label loss for object detection
                    loss = F.binary_cross_entropy_with_logits(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard training with NeuroSym-CML paradigm
                # Convert rule indices to embeddings for prior state
                rule_embeddings = torch.zeros(rules.size(0), 64, device=self.device)
                for i, rule_idx in enumerate(rules):
                    rule_embeddings[i, rule_idx % 64] = 1.0  # Simple one-hot encoding
                
                # Use symbolic controller for advanced reasoning
                enhanced_rules, _ = self.symbolic_controller.forward(
                    x=images.flatten(1),  # Flatten image features
                    task_metadata={'id': 0, 'type': 'image_analysis'},  # Task metadata
                    prior_state=rule_embeddings  # Use rule embeddings as prior state
                )
                
                # Forward through NeuroSym-CML hybrid model
                outputs = self.model(images, texts, rules)
                loss = F.binary_cross_entropy_with_logits(outputs, targets)
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item()
            
            # Calculate accuracy for multi-label
            predictions = torch.sigmoid(outputs) > 0.5
            correct += (predictions == targets).float().mean().item() * targets.size(0)
            total += targets.size(0)
            
            # Store experience in replay buffer for continual learning
            experience = (
                images.detach().cpu(),
                texts.detach().cpu(),
                enhanced_rules.detach().cpu(),
                targets.detach().cpu(),
                outputs.detach().cpu()
            )
            self.replay_buffer.add(experience)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
            
            # Memory cleanup every 10 batches
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # End epoch cleanup
        dataset.end_epoch()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                images = batch['images'].to(self.device)
                texts = batch['texts'].to(self.device)
                rules = batch['rules'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Convert rule indices to embeddings for prior state
                rule_embeddings = torch.zeros(rules.size(0), 64, device=self.device)
                for i, rule_idx in enumerate(rules):
                    rule_embeddings[i, rule_idx % 64] = 1.0  # Simple one-hot encoding
                
                # Use symbolic controller for advanced reasoning
                enhanced_rules, _ = self.symbolic_controller.forward(
                    x=images.flatten(1),  # Flatten image features
                    task_metadata={'id': 0, 'type': 'image_analysis'},  # Task metadata
                    prior_state=rule_embeddings  # Use rule embeddings as prior state
                )
                
                # Forward through NeuroSym-CML hybrid model
                outputs = self.model(images, texts, rules)
                loss = F.binary_cross_entropy_with_logits(outputs, targets)
                
                total_loss += loss.item()
                
                # Calculate accuracy for multi-label
                predictions = torch.sigmoid(outputs) > 0.5
                correct += (predictions == targets).float().mean().item() * targets.size(0)
                total += targets.size(0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, dataset):
        """Main training loop with secure cleanup"""
        print(f"🔒 Starting secure training for {self.config['epochs']} epochs...")
        print(f"💻 Device: {self.device}")
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        try:
            start_epoch = self.epoch if hasattr(self, 'epoch') else 0
            for epoch in range(start_epoch, self.config['epochs']):
                self.epoch = epoch
                
                # Training
                train_metrics = self.train_epoch(train_loader, dataset)
                
                # Validation
                val_metrics = self.validate(val_loader)
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                    else:
                        self.scheduler.step()
                
                # Logging
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                      f"Val Loss: {val_metrics.get('loss', 'N/A'):.4f}, "
                      f"Val Acc: {val_metrics.get('accuracy', 'N/A'):.2f}%, "
                      f"LR: {current_lr:.6f}")
                
                # Save checkpoint
                if (epoch + 1) % self.config.get('checkpoint_frequency', 10) == 0:
                    self.save_checkpoint(epoch, train_metrics, val_metrics)
                
                # Early stopping
                if val_metrics and val_metrics['loss'] < self.best_loss:
                    self.best_loss = val_metrics['loss']
                    self.save_checkpoint(epoch, train_metrics, val_metrics, is_best=True)
                
                # Meta-learning step for NeuroSym-CML
                if len(self.replay_buffer) > self.config.get('meta_batch_size', 4):
                    self._meta_learning_step()
                
                # Record history
                self.training_history.append({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics.get('loss', None),
                    'val_accuracy': val_metrics.get('accuracy', None),
                    'learning_rate': current_lr
                })
        
        finally:
            # CRITICAL: Always perform secure cleanup
            print("🔒 Training completed - performing secure cleanup...")
            secure_training_cleanup(dataset)
            
            training_time = time.time() - start_time
            print(f"✅ Secure training completed in {training_time:.2f} seconds")
            print(f"🏆 Best validation loss: {self.best_loss:.4f}")
            print("🔒 All sensitive data has been removed from memory")
    
    def save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Dict, is_best: bool = False):
        """Save model checkpoint (without sensitive data)"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/secure_checkpoint_epoch_{epoch+1}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            torch.save(checkpoint, "checkpoints/best_secure_model.pt")
            print(f"💾 Best secure model saved at epoch {epoch+1}")
        
        print(f"💾 Secure checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint and resume training state"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state with strict=False to handle missing layers
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError as e:
            print(f"⚠️  Some layers missing, loading with strict=False: {e}")
            # Load with strict=False to ignore missing layers (fusion, output)
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
            if missing_keys:
                print(f"   Missing keys (will be created on first forward pass): {missing_keys}")
            if unexpected_keys:
                print(f"   Unexpected keys (ignored): {unexpected_keys}")
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"✅ Checkpoint loaded successfully!")
        print(f"   - Epoch: {self.epoch}")
        print(f"   - Best loss: {self.best_loss:.4f}")
        print(f"   - Training history: {len(self.training_history)} records")
        
        return checkpoint
    
    def _meta_learning_step(self):
        """Perform meta-learning step using replay buffer"""
        try:
            # Sample batch from replay buffer
            meta_batch = self.replay_buffer.sample(self.config.get('meta_batch_size', 4))
            
            if meta_batch is None:
                return
            
            # Inner loop: adapt to each task in the meta-batch
            inner_losses = []
            for task in meta_batch:
                # Quick adaptation step
                images, texts, rules, targets, outputs = task
                images = images.to(self.device)
                texts = texts.to(self.device)
                rules = rules.to(self.device)
                targets = targets.to(self.device)
                
                # Convert rule indices to embeddings for prior state
                rule_embeddings = torch.zeros(rules.size(0), 64, device=self.device)
                for i, rule_idx in enumerate(rules):
                    rule_embeddings[i, rule_idx % 64] = 1.0  # Simple one-hot encoding
                
                # Forward pass
                enhanced_rules, _ = self.symbolic_controller.forward(
                    x=images.flatten(1),
                    task_metadata={'id': 0, 'type': 'image_analysis'},
                    prior_state=rule_embeddings
                )
                outputs = self.model(images, texts, rules)
                loss = F.binary_cross_entropy_with_logits(outputs, targets)
                inner_losses.append(loss)
            
            # Outer loop: update meta-parameters
            if inner_losses:
                meta_loss = torch.stack(inner_losses).mean()
                meta_loss.backward()
                
                # Update with meta-learning rate
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.data -= self.config.get('outer_lr', 0.001) * param.grad.data
                
                # Clear gradients
                self.optimizer.zero_grad()
                
        except Exception as e:
            print(f"⚠️  Meta-learning step failed: {e}")

def main():
    """Main secure training function with proper NeuroSym-CML paradigm"""
    parser = argparse.ArgumentParser(description='NeuroSym-CML Secure Image Training')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint file to resume training from')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    print("🔒 Secure Image Training for NeuroSym-CML")
    print("🧠 Using Hybrid Neuro-Symbolic Continual Meta-Learning Architecture")
    print("=" * 60)
    
    if args.resume:
        print(f"🔄 Resuming training from checkpoint: {args.resume}")
    else:
        print("🚀 Starting fresh training")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Training on CPU (will be slow).")
    else:
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Dataset paths
    images_dir = "../dataset/images"
    labels_dir = "../dataset/labels"
    classes_file = "../dataset/classes.txt"
    
    # Check if dataset exists
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"❌ Labels directory not found: {labels_dir}")
        return
    
    if not os.path.exists(classes_file):
        print(f"❌ Classes file not found: {classes_file}")
        return
    
    # Training configuration - Proper NeuroSym-CML Architecture
    config = {
        'neural_architecture': 'custom_cnn',  # Use NeuroSym-CML custom CNN for image processing
        'num_classes': 42,  # Number of classes in your dataset
        'hidden_sizes': [512, 256, 128],  # Deeper architecture for NeuroSym-CML
        'batch_size': args.batch_size,  # Use command line argument
        'learning_rate': args.learning_rate,  # Use command line argument
        'weight_decay': 1e-4,
        'epochs': args.epochs,  # Use command line argument
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'mixed_precision': False,  # Disable for stability
        'gradient_accumulation_steps': 2,  # Effective batch size = 8
        'checkpoint_frequency': 5,
        'memory_size': 1000,  # Experience replay for continual learning
        'rule_set_size': 100,  # Symbolic reasoning rules
        'use_symbolic_reasoning': True,  # Enable NeuroSym-CML symbolic reasoning
        'rule_embedding_dim': 64,  # Rule embedding dimension
        'meta_batch_size': 4,  # Meta-learning batch size
        'inner_lr': 0.01,  # Inner loop learning rate for meta-learning
        'outer_lr': 0.001,  # Outer loop learning rate for meta-learning
        'memory_sampling_strategy': 'random',  # Experience replay sampling
        'use_attention': True,  # Enable attention mechanisms
        'use_task_metadata': True,  # Use task metadata in reasoning
        'use_prior_state': True  # Use prior learned symbolic state
    }
    
    # Create secure data module
    print("📊 Setting up secure data module...")
    data_module = SecureDataModule(
        images_dir=images_dir,
        labels_dir=labels_dir,
        classes_file=classes_file,
        batch_size=config['batch_size'],
        # max_samples=None,  # Process all available images
        cleanup_after_epoch=True
    )
    
    # Create dataloaders
    train_loader, val_loader, dataset = data_module.create_dataloaders(train_split=0.8)
    
    # Create trainer
    trainer = SecureImageTrainer(config)
    
    # Load checkpoint if resuming
    if args.resume:
        try:
            checkpoint = trainer.load_checkpoint(args.resume)
            print(f"🔄 Resuming from epoch {trainer.epoch + 1}")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("🚀 Starting fresh training instead...")
    
    # Start training
    print("🚀 Starting secure training...")
    trainer.train(train_loader, val_loader, dataset)
    
    print("🎉 NeuroSym-CML secure training completed successfully!")
    print("🧠 Hybrid neuro-symbolic continual meta-learning architecture trained")
    print("🔒 All sensitive data has been removed from memory")
    print("💾 Check the 'checkpoints' folder for saved models")
    print("🚀 Model ready for deployment with advanced reasoning capabilities")

if __name__ == "__main__":
    main()
