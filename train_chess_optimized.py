#!/usr/bin/env python3
"""
Optimized Chess Training for NeuroSym-CML
Uses all modular components for maximum performance and world-class results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import numpy as np
import math

# Import modular components
from meta_model import HybridModel, ModelSpec
from modular_architecture import create_chess_trainer, TextOnlyAdapter
from modular_symbolic_controller import create_symbolic_controller
from modular_replay_buffer import create_replay_buffer
from train_chess import ChessDataset  # Reuse existing dataset
from custom_architecture_selector import CustomArchitectureSelector

class OptimizedChessTrainer:
    """World-class optimized chess trainer using all modular components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize architecture selector for optimal model selection
        self.arch_selector = CustomArchitectureSelector()
        optimal_arch = self.arch_selector.select_architecture(
            task_type="sequential_data",
            data_type="text",
            requirements=["memory", "sequential", "chess_strategy"]
        )
        config['neural_architecture'] = optimal_arch
        
        # Create optimized model
        self.model = self._create_optimized_model()
        
        # Initialize memory-efficient symbolic controller
        self.symbolic_controller = create_symbolic_controller(
            controller_type='production_rule',  # More memory efficient than neuro_symbolic
            num_rules=config.get('rule_set_size', 100),
            input_size=min(self.model.feature_dim, 512),  # Cap input size
            hidden_size=config.get('symbolic_hidden_size', 64)
        ).to(self.device)
        
        # Initialize specialized chess replay buffer
        self.replay_buffer = create_replay_buffer(
            buffer_type='text',
            memory_size=config.get('replay_buffer_size', 50000),  # Large buffer for chess
            device=str(self.device)
        )
        
        # Advanced optimizer with learning rate scheduling
        self.optimizer = self._create_advanced_optimizer()
        self.scheduler = self._create_lr_scheduler()
        
        # Mixed precision training for speed
        self.scaler = GradScaler('cuda') if config.get('mixed_precision', True) and torch.cuda.is_available() else None
        
        # Advanced loss functions
        self.criterion = self._create_advanced_loss()
        
        # Performance tracking
        self.best_accuracy = 0.0
        self.training_history = []
        self.chess_metrics = {
            'tactical_accuracy': 0.0,
            'positional_understanding': 0.0,
            'endgame_skill': 0.0,
            'opening_knowledge': 0.0
        }
        
        # Memory management for 4GB GPU
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        self.effective_batch_size = config['batch_size'] * self.gradient_accumulation_steps
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"🏆 Optimized Chess Trainer Initialized (4GB GPU Optimized)")
        print(f"   Device: {self.device}")
        print(f"   Architecture: {config['neural_architecture']}")
        print(f"   Symbolic Controller: production_rule (memory efficient)")
        print(f"   Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Batch Size: {config['batch_size']} (Effective: {self.effective_batch_size})")
        print(f"   Mixed Precision: {config.get('mixed_precision', True)}")
        print(f"   Memory Optimized: True")
    
    def _create_optimized_model(self) -> HybridModel:
        """Create optimized model for chess"""
        model_spec = ModelSpec(
            neural_architecture=self.config['neural_architecture'],
            num_classes=self.config.get('num_classes', 20),  # More classes for chess positions
            hidden_sizes=self.config.get('hidden_sizes', [256, 128]),  # Memory efficient
            input_shape=(self.config.get('max_length', 512),),  # Shorter sequences
            dropout_rate=self.config.get('dropout_rate', 0.1),  # Lower dropout for chess
            use_batch_norm=True,
            device=self.device,
            rule_set_size=self.config.get('rule_set_size', 200)
        )
        
        model = HybridModel(model_spec).to(self.device)
        
        # Apply advanced initialization
        self._apply_advanced_initialization(model)
        
        return model
    
    def _apply_advanced_initialization(self, model: nn.Module):
        """Apply advanced weight initialization for better convergence"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _create_advanced_optimizer(self) -> optim.Optimizer:
        """Create advanced optimizer with optimal settings"""
        # Use AdamW with optimal hyperparameters for chess
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 2e-4),  # Optimal LR for chess
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.get('weight_decay', 1e-4),
            amsgrad=True  # Better convergence
        )
    
    def _create_lr_scheduler(self):
        """Create learning rate scheduler"""
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart period
            eta_min=1e-6  # Minimum learning rate
        )
    
    def _create_advanced_loss(self) -> nn.Module:
        """Create advanced loss function for chess"""
        class ChessAdvancedLoss(nn.Module):
            def __init__(self, alpha=0.7, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ce_loss = nn.CrossEntropyLoss()
                self.focal_loss = self._focal_loss
            
            def _focal_loss(self, inputs, targets):
                """Focal loss for handling class imbalance in chess positions"""
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
            
            def forward(self, outputs, targets):
                # Combine cross-entropy and focal loss
                ce = self.ce_loss(outputs, targets)
                focal = self.focal_loss(outputs, targets)
                return 0.6 * ce + 0.4 * focal
        
        return ChessAdvancedLoss()
    
    def _calculate_chess_metrics(self, outputs: torch.Tensor, labels: torch.Tensor, 
                                metadata: List[Dict]) -> Dict[str, float]:
        """Calculate chess-specific performance metrics"""
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == labels).float()
        
        # Initialize metrics
        tactical_correct = []
        positional_correct = []
        endgame_correct = []
        opening_correct = []
        
        # Categorize based on metadata (if available)
        for i, meta in enumerate(metadata):
            if meta.get('position_type') == 'tactical':
                tactical_correct.append(correct[i].item())
            elif meta.get('position_type') == 'positional':
                positional_correct.append(correct[i].item())
            elif meta.get('position_type') == 'endgame':
                endgame_correct.append(correct[i].item())
            elif meta.get('position_type') == 'opening':
                opening_correct.append(correct[i].item())
        
        return {
            'tactical_accuracy': np.mean(tactical_correct) if tactical_correct else 0.0,
            'positional_understanding': np.mean(positional_correct) if positional_correct else 0.0,
            'endgame_skill': np.mean(endgame_correct) if endgame_correct else 0.0,
            'opening_knowledge': np.mean(opening_correct) if opening_correct else 0.0,
            'overall_accuracy': correct.mean().item()
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor], accumulation_step: int = 0) -> Dict[str, float]:
        """Optimized training step with all enhancements"""
        self.model.train()
        self.symbolic_controller.train()
        
        # Move to device
        text_encodings = batch['text_encoding'].to(self.device)
        labels = batch['labels'].to(self.device)
        metadata = batch.get('metadata', [{}] * len(labels))
        
        # Only zero gradients at the start of accumulation
        if accumulation_step == 0:
            self.optimizer.zero_grad()
        
        # Use mixed precision if available
        if self.scaler:
            with autocast('cuda'):
                # Generate symbolic rules
                text_encodings_float = text_encodings.float()
                rule_indices, symbolic_state = self.symbolic_controller(text_encodings_float)
                
                # Create dummy images for hybrid model
                batch_size = text_encodings.size(0)
                dummy_images = torch.zeros(batch_size, 3, 224, 224).to(self.device)
                
                # Forward pass
                outputs = self.model(dummy_images, text_encodings_float, rule_indices)
                
                # Calculate loss with gradient accumulation
                loss = self.criterion(outputs, labels) / self.gradient_accumulation_steps
            
            # Backward pass with scaling
            self.scaler.scale(loss).backward()
            
            # Only step optimizer after accumulating gradients
            if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Clear cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Standard precision training
            text_encodings_float = text_encodings.float()
            rule_indices, symbolic_state = self.symbolic_controller(text_encodings_float)
            
            batch_size = text_encodings.size(0)
            dummy_images = torch.zeros(batch_size, 3, 224, 224).to(self.device)
            
            outputs = self.model(dummy_images, text_encodings_float, rule_indices)
            loss = self.criterion(outputs, labels) / self.gradient_accumulation_steps
            
            loss.backward()
            
            # Only step optimizer after accumulating gradients
            if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Clear cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Calculate metrics
        with torch.no_grad():
            chess_metrics = self._calculate_chess_metrics(outputs, labels, metadata)
            
            # Store experience in replay buffer
            for i in range(len(text_encodings)):
                self.replay_buffer.add(
                    text_encoding=text_encodings[i].cpu(),
                    labels=labels[i:i+1].cpu(),
                    loss=loss.item(),
                    accuracy=chess_metrics['overall_accuracy'],
                    metadata=metadata[i] if i < len(metadata) else {}
                )
        
        return {
            'loss': loss.item(),
            **chess_metrics,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Optimized evaluation step"""
        self.model.eval()
        self.symbolic_controller.eval()
        
        with torch.no_grad():
            text_encodings = batch['text_encoding'].to(self.device)
            labels = batch['labels'].to(self.device)
            metadata = batch.get('metadata', [{}] * len(labels))
            
            # Forward pass
            text_encodings_float = text_encodings.float()
            rule_indices, symbolic_state = self.symbolic_controller(text_encodings_float)
            
            batch_size = text_encodings.size(0)
            dummy_images = torch.zeros(batch_size, 3, 224, 224).to(self.device)
            
            outputs = self.model(dummy_images, text_encodings_float, rule_indices)
            loss = self.criterion(outputs, labels)
            
            # Calculate chess metrics
            chess_metrics = self._calculate_chess_metrics(outputs, labels, metadata)
        
        return {
            'loss': loss.item(),
            **chess_metrics
        }
    
    def train(self, dataloader: DataLoader, num_epochs: int):
        """Main training loop with all optimizations"""
        print(f"🚀 Starting optimized chess training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = {'loss': 0.0, 'overall_accuracy': 0.0, 'tactical_accuracy': 0.0,
                           'positional_understanding': 0.0, 'endgame_skill': 0.0, 'opening_knowledge': 0.0}
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Chess Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch, batch_idx)
                
                # Accumulate metrics
                for key in train_metrics:
                    if key in metrics:
                        train_metrics[key] += metrics[key]
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{metrics['loss']:.4f}",
                    'Acc': f"{metrics['overall_accuracy']:.4f}",
                    'Tactical': f"{metrics['tactical_accuracy']:.4f}",
                    'LR': f"{metrics['lr']:.2e}"
                })
            
            # Average metrics
            for key in train_metrics:
                train_metrics[key] /= num_batches
            
            # Evaluation phase
            eval_metrics = {'loss': 0.0, 'overall_accuracy': 0.0, 'tactical_accuracy': 0.0,
                          'positional_understanding': 0.0, 'endgame_skill': 0.0, 'opening_knowledge': 0.0}
            eval_batches = 0
            
            for batch in dataloader:  # Using same data for eval (you can split this)
                metrics = self.evaluate_step(batch)
                for key in eval_metrics:
                    if key in metrics:
                        eval_metrics[key] += metrics[key]
                eval_batches += 1
            
            # Average eval metrics
            for key in eval_metrics:
                eval_metrics[key] /= eval_batches
            
            # Update learning rate
            self.scheduler.step()
            
            # Save epoch results
            epoch_results = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'eval': eval_metrics
            }
            self.training_history.append(epoch_results)
            
            # Print detailed results
            print(f"\nEpoch {epoch + 1}/{num_epochs} Results:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['overall_accuracy']:.4f}")
            print(f"  Train - Tactical: {train_metrics['tactical_accuracy']:.4f}, Positional: {train_metrics['positional_understanding']:.4f}")
            print(f"  Eval  - Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['overall_accuracy']:.4f}")
            print(f"  Eval  - Tactical: {eval_metrics['tactical_accuracy']:.4f}, Positional: {eval_metrics['positional_understanding']:.4f}")
            
            # Save best model
            if eval_metrics['overall_accuracy'] > self.best_accuracy:
                self.best_accuracy = eval_metrics['overall_accuracy']
                self.chess_metrics = {
                    'tactical_accuracy': eval_metrics['tactical_accuracy'],
                    'positional_understanding': eval_metrics['positional_understanding'],
                    'endgame_skill': eval_metrics['endgame_skill'],
                    'opening_knowledge': eval_metrics['opening_knowledge']
                }
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'symbolic_controller_state_dict': self.symbolic_controller.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'accuracy': self.best_accuracy,
                    'chess_metrics': self.chess_metrics,
                    'config': self.config,
                    'training_history': self.training_history
                }, 'best_chess_model_optimized.pt')
                
                print(f"  🏆 New best accuracy: {self.best_accuracy:.4f}")
        
        print(f"\n🎉 Chess training completed!")
        print(f"Best Overall Accuracy: {self.best_accuracy:.4f}")
        print(f"Chess Skill Breakdown:")
        print(f"  Tactical Accuracy: {self.chess_metrics['tactical_accuracy']:.4f}")
        print(f"  Positional Understanding: {self.chess_metrics['positional_understanding']:.4f}")
        print(f"  Endgame Skill: {self.chess_metrics['endgame_skill']:.4f}")
        print(f"  Opening Knowledge: {self.chess_metrics['opening_knowledge']:.4f}")

def main():
    """Main function with world-class configuration"""
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_classes': 10,  # Reduced for memory efficiency
        'max_length': 512,  # Reduced sequence length for memory
        'batch_size': 2,  # Much smaller batch size for 4GB GPU
        'learning_rate': 2e-4,  # Optimal learning rate
        'weight_decay': 1e-4,
        'num_epochs': 20,  # Reduced epochs for 4GB GPU
        'dropout_rate': 0.1,  # Lower dropout for chess precision
        'gradient_accumulation_steps': 4,  # Simulate larger batch size
        'use_batch_norm': True,
        'neural_architecture': 'custom_transformer',  # Will be optimized by selector
        'hidden_sizes': [256, 128],  # Smaller network for 4GB GPU
        'rule_set_size': 100,  # Reduced rules for memory
        'symbolic_hidden_size': 64,  # Smaller symbolic controller
        'replay_buffer_size': 10000,  # Smaller buffer for 4GB GPU
        'mixed_precision': True,
        'gradient_clipping': True
    }
    
    print("♟️ World-Class Optimized Chess Training")
    print("=" * 60)
    print("🎯 Target: Achieve world-class chess AI performance")
    print("🔧 Using all modular optimizations")
    print("=" * 60)
    
    # Create dataset
    dataset = ChessDataset()
    if len(dataset) == 0:
        print("❌ No chess data found!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Create optimized trainer
    trainer = OptimizedChessTrainer(config)
    
    # Train the model
    trainer.train(dataloader, config['num_epochs'])
    
    # Save final results
    results = {
        'best_accuracy': trainer.best_accuracy,
        'chess_metrics': trainer.chess_metrics,
        'config': config,
        'training_history': trainer.training_history
    }
    
    with open('chess_world_class_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🏆 World-class chess training completed!")
    print(f"📊 Results saved to chess_world_class_results.json")

if __name__ == "__main__":
    main()