#!/usr/bin/env python3
"""
4GB GPU Optimized Chess Training for NeuroSym-CML
Specifically designed for 4GB GPU constraints while maintaining high performance
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

# Import modular components
from meta_model import HybridModel, ModelSpec
from modular_symbolic_controller import create_symbolic_controller
from modular_replay_buffer import create_replay_buffer
from train_chess import ChessDataset

class MemoryEfficientChessTrainer:
    """Memory-efficient chess trainer optimized for 4GB GPU"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Clear GPU memory first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create lightweight model
        self.model = self._create_lightweight_model()
        
        # Use simple but effective symbolic controller
        self.symbolic_controller = create_symbolic_controller(
            controller_type='logic_based',  # Most memory efficient
            num_rules=config.get('rule_set_size', 50),  # Minimal rules
            input_size=256,  # Fixed smaller size
            hidden_size=32  # Very small hidden size
        ).to(self.device)
        
        # Small replay buffer
        self.replay_buffer = create_replay_buffer(
            buffer_type='text',
            memory_size=config.get('replay_buffer_size', 5000),
            device=str(self.device)
        )
        
        # Efficient optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Simple scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.8)
        
        # Mixed precision for memory efficiency
        self.scaler = GradScaler('cuda') if torch.cuda.is_available() else None
        
        # Multi-label loss for chess concepts
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Gradient accumulation for effective larger batch size
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 8)
        
        # Performance tracking
        self.best_accuracy = 0.0
        self.training_history = []
        
        print(f"♟️ Memory-Efficient Chess Trainer (4GB GPU)")
        print(f"   Device: {self.device}")
        print(f"   Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Batch Size: {config['batch_size']} (Effective: {config['batch_size'] * self.gradient_accumulation_steps})")
        print(f"   Memory Optimized: True")
    
    def _create_lightweight_model(self) -> nn.Module:
        """Create very lightweight model for 4GB GPU"""
        class LightweightChessModel(nn.Module):
            def __init__(self, vocab_size=30000, embed_dim=256, hidden_dim=128, num_classes=10):
                super().__init__()
                
                # Small embedding layer
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                
                # Single LSTM layer
                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.1)
                
                # Simple classifier
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, num_classes)
                )
                
                self.feature_dim = hidden_dim
            
            def forward(self, dummy_images, text_embeddings, rule_indices):
                # Ignore dummy images and rule indices for simplicity
                batch_size, seq_len = text_embeddings.shape[:2]
                
                # Treat text_embeddings as token indices if they're integers
                if text_embeddings.dtype in [torch.long, torch.int]:
                    # Clamp to vocabulary size
                    text_embeddings = torch.clamp(text_embeddings.long(), 0, 29999)
                    x = self.embedding(text_embeddings)
                else:
                    # If already embeddings, just use them
                    x = text_embeddings
                
                # LSTM forward
                lstm_out, _ = self.lstm(x)
                
                # Use last output
                last_output = lstm_out[:, -1, :]
                
                # Classify
                output = self.classifier(last_output)
                
                return output
        
        return LightweightChessModel(
            vocab_size=30000,
            embed_dim=256,
            hidden_dim=128,
            num_classes=self.config.get('num_classes', 10)
        ).to(self.device)
    
    def train_step(self, batch: Dict[str, torch.Tensor], accumulation_step: int) -> Dict[str, float]:
        """Memory-efficient training step"""
        self.model.train()
        
        # Move to device
        text_encodings = batch['text_encoding'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Ensure labels are the right shape for multi-label classification
        if labels.dim() == 1 and labels.size(0) == 10:
            # Single sample with 10 labels, add batch dimension
            labels = labels.unsqueeze(0)
        elif labels.dim() > 2:
            # Too many dimensions, squeeze appropriately
            labels = labels.squeeze()
            if labels.dim() == 1 and labels.size(0) == 10:
                labels = labels.unsqueeze(0)
        
        # Ensure batch size matches
        if labels.size(0) != text_encodings.size(0):
            # Repeat labels if needed
            labels = labels[:text_encodings.size(0)]
        
        # Only zero gradients at start of accumulation
        if accumulation_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()
        
        # Use mixed precision
        if self.scaler:
            with autocast('cuda'):
                # Generate simple rule indices
                rule_indices = torch.randint(0, 50, (text_encodings.size(0),)).to(self.device)
                
                # Create minimal dummy images
                dummy_images = torch.zeros(text_encodings.size(0), 3, 32, 32).to(self.device)
                
                # Forward pass
                outputs = self.model(dummy_images, text_encodings, rule_indices)
                
                # Calculate loss with accumulation
                loss = self.criterion(outputs, labels) / self.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Step optimizer after accumulation
            if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Clear cache
                torch.cuda.empty_cache()
        else:
            # CPU training
            rule_indices = torch.randint(0, 50, (text_encodings.size(0),))
            dummy_images = torch.zeros(text_encodings.size(0), 3, 32, 32)
            
            outputs = self.model(dummy_images, text_encodings, rule_indices)
            loss = self.criterion(outputs, labels) / self.gradient_accumulation_steps
            
            loss.backward()
            
            if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
        
        # Calculate metrics for multi-label classification
        with torch.no_grad():
            predictions = torch.sigmoid(outputs) > 0.5
            accuracy = (predictions == labels.bool()).float().mean().item()
        
        return {
            'loss': loss.item() * self.gradient_accumulation_steps,  # Unscale for display
            'accuracy': accuracy,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Memory-efficient evaluation step"""
        self.model.eval()
        
        with torch.no_grad():
            text_encodings = batch['text_encoding'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Handle labels for multi-label classification
            if labels.dim() == 1 and labels.size(0) == 10:
                labels = labels.unsqueeze(0)
            elif labels.dim() > 2:
                labels = labels.squeeze()
                if labels.dim() == 1 and labels.size(0) == 10:
                    labels = labels.unsqueeze(0)
            
            # Ensure batch size matches
            if labels.size(0) != text_encodings.size(0):
                labels = labels[:text_encodings.size(0)]
            
            # Simple forward pass
            rule_indices = torch.randint(0, 50, (text_encodings.size(0),)).to(self.device)
            dummy_images = torch.zeros(text_encodings.size(0), 3, 32, 32).to(self.device)
            
            outputs = self.model(dummy_images, text_encodings, rule_indices)
            loss = self.criterion(outputs, labels)
            
            predictions = torch.sigmoid(outputs) > 0.5
            accuracy = (predictions == labels.bool()).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def train(self, dataloader: DataLoader, num_epochs: int):
        """Memory-efficient training loop"""
        print(f"🚀 Starting memory-efficient chess training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = 0.0
            train_accuracy = 0.0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Chess Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch, batch_idx)
                
                train_loss += metrics['loss']
                train_accuracy += metrics['accuracy']
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f"{metrics['loss']:.4f}",
                    'Acc': f"{metrics['accuracy']:.4f}",
                    'LR': f"{metrics['lr']:.2e}"
                })
                
                # Clear cache every few batches
                if batch_idx % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Average metrics
            train_loss /= num_batches
            train_accuracy /= num_batches
            
            # Evaluation phase
            eval_loss = 0.0
            eval_accuracy = 0.0
            eval_batches = 0
            
            for batch in dataloader:
                metrics = self.evaluate_step(batch)
                eval_loss += metrics['loss']
                eval_accuracy += metrics['accuracy']
                eval_batches += 1
            
            eval_loss /= eval_batches
            eval_accuracy /= eval_batches
            
            # Update scheduler
            self.scheduler.step()
            
            # Save results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'eval_loss': eval_loss,
                'eval_accuracy': eval_accuracy
            }
            self.training_history.append(epoch_results)
            
            # Print results
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            print(f"  Eval  - Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")
            
            # Save best model
            if eval_accuracy > self.best_accuracy:
                self.best_accuracy = eval_accuracy
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': self.best_accuracy,
                    'config': self.config,
                    'training_history': self.training_history
                }, 'best_chess_model_4gb.pt')
                
                print(f"  🏆 New best accuracy: {self.best_accuracy:.4f}")
            
            # Clear cache at end of epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\n🎉 Chess training completed!")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")

def main():
    """Main function optimized for 4GB GPU"""
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_classes': 10,
        'max_length': 256,  # Short sequences
        'batch_size': 1,  # Very small batch size
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 15,
        'rule_set_size': 50,  # Minimal rules
        'replay_buffer_size': 5000,  # Small buffer
        'gradient_accumulation_steps': 8,  # Simulate batch size 8
        'mixed_precision': True
    }
    
    print("♟️ 4GB GPU Optimized Chess Training")
    print("=" * 50)
    print("🎯 Target: High performance on 4GB GPU")
    print("🔧 Memory optimizations enabled")
    print("=" * 50)
    
    # Create dataset
    dataset = ChessDataset()
    if len(dataset) == 0:
        print("❌ No chess data found!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # No multiprocessing to save memory
        pin_memory=False  # Disable pin memory to save GPU memory
    )
    
    # Create memory-efficient trainer
    trainer = MemoryEfficientChessTrainer(config)
    
    # Train the model
    trainer.train(dataloader, config['num_epochs'])
    
    # Save results
    results = {
        'best_accuracy': trainer.best_accuracy,
        'config': config,
        'training_history': trainer.training_history
    }
    
    with open('chess_4gb_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🏆 4GB GPU chess training completed!")
    print(f"📊 Results saved to chess_4gb_results.json")

if __name__ == "__main__":
    main()