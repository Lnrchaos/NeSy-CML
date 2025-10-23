#!/usr/bin/env python3
"""
Modular Chess Training for NeuroSym-CML
Uses the new modular architecture system
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import time

# Import existing components
from meta_model import HybridModel, ModelSpec
from modular_architecture import create_chess_trainer
from train_chess import ChessDataset  # Reuse existing dataset

def create_chess_model(config: Dict[str, Any]) -> HybridModel:
    """Create the chess model"""
    model_spec = ModelSpec(
        neural_architecture=config['neural_architecture'],
        num_classes=config['num_classes'],
        hidden_sizes=config.get('hidden_sizes', [256, 128]),
        input_shape=(config['max_length'],),
        dropout_rate=config.get('dropout_rate', 0.2),
        use_batch_norm=config.get('use_batch_norm', True),
        device=torch.device(config['device']),
        rule_set_size=config.get('rule_set_size', 100)
    )
    
    return HybridModel(model_spec)

def train_chess_model(config: Dict[str, Any]):
    """Train chess model using modular architecture"""
    
    print("♟️ Modular Chess Training with NeuroSym-CML")
    print("=" * 50)
    print(f"Using device: {config['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\nChess Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create dataset
    print("\nLoading chess dataset...")
    dataset = ChessDataset()
    
    if len(dataset) == 0:
        print("❌ No chess data found! Please check your dataset.")
        return None
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Create base model
    print("\nCreating chess model...")
    base_model = create_chess_model(config)
    
    # Create modular trainer
    trainer = create_chess_trainer(base_model, config)
    
    # Training loop
    print(f"\nStarting chess training for {config['num_epochs']} epochs...")
    
    best_accuracy = 0.0
    training_history = []
    
    for epoch in range(config['num_epochs']):
        epoch_metrics = {'epoch': epoch + 1, 'train_loss': 0.0, 'train_accuracy': 0.0}
        
        # Training phase
        pbar = tqdm(dataloader, desc=f"Chess Epoch {epoch + 1}")
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch in pbar:
            # Train step
            metrics = trainer.train_step(batch)
            
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{metrics['loss']:.4f}",
                'Acc': f"{metrics['accuracy']:.4f}"
            })
        
        # Calculate epoch averages
        epoch_metrics['train_loss'] = total_loss / num_batches
        epoch_metrics['train_accuracy'] = total_accuracy / num_batches
        
        # Evaluation phase (using same data for now)
        eval_loss = 0.0
        eval_accuracy = 0.0
        eval_batches = 0
        
        for batch in dataloader:
            eval_metrics = trainer.evaluate_step(batch)
            eval_loss += eval_metrics['loss']
            eval_accuracy += eval_metrics['accuracy']
            eval_batches += 1
        
        epoch_metrics['eval_loss'] = eval_loss / eval_batches
        epoch_metrics['eval_accuracy'] = eval_accuracy / eval_batches
        
        training_history.append(epoch_metrics)
        
        # Print epoch results
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}:")
        print(f"  Train Loss: {epoch_metrics['train_loss']:.4f}")
        print(f"  Train Accuracy: {epoch_metrics['train_accuracy']:.4f}")
        print(f"  Eval Loss: {epoch_metrics['eval_loss']:.4f}")
        print(f"  Eval Accuracy: {epoch_metrics['eval_accuracy']:.4f}")
        
        # Save best model
        if epoch_metrics['eval_accuracy'] > best_accuracy:
            best_accuracy = epoch_metrics['eval_accuracy']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'accuracy': best_accuracy,
                'config': config
            }, 'best_chess_model.pt')
            print(f"  ✅ New best accuracy: {best_accuracy:.4f}")
    
    print(f"\n🎉 Chess training completed!")
    print(f"Best accuracy: {best_accuracy:.4f}")
    
    return {
        'best_accuracy': best_accuracy,
        'training_history': training_history,
        'model': trainer.model
    }

def main():
    """Main training function"""
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_classes': 10,
        'max_length': 512,
        'batch_size': 4,  # Smaller batch size for stability
        'learning_rate': 1e-4,  # Lower learning rate
        'weight_decay': 1e-5,
        'num_epochs': 10,  # Fewer epochs for testing
        'dropout_rate': 0.2,
        'use_batch_norm': True,
        'neural_architecture': 'custom_transformer',
        'hidden_sizes': [256, 128],
        'rule_set_size': 100,
        'feature_dim': 512
    }
    
    results = train_chess_model(config)
    
    if results:
        print(f"\n📊 Final Results:")
        print(f"Chess Model Accuracy: {results['best_accuracy']:.4f}")
        
        # Save results
        import json
        with open('chess_training_results.json', 'w') as f:
            json.dump({
                'accuracy': results['best_accuracy'],
                'config': config,
                'training_history': results['training_history']
            }, f, indent=2)
        
        print("Results saved to chess_training_results.json")

if __name__ == "__main__":
    main()