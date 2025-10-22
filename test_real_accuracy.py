#!/usr/bin/env python3
"""
Test script to check the REAL accuracy of trained multimodal models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_multimodal_newson import MultiModalNeuroSym, MultiModalDataset, MultiModalTrainer
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_checkpoint(checkpoint_path, config):
    """Load a trained model from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = MultiModalNeuroSym(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
    logger.info(f"Checkpoint loss: {checkpoint['loss']:.6f}")
    
    return model, device

def test_model_accuracy(model, device, dataset_path):
    """Test the model's REAL accuracy with detailed debugging"""
    
    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = MultiModalDataset(dataset_path, tokenizer)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Testing on {len(dataset)} samples")
    
    # Label mapping
    label_to_idx = {
        'programming': 0, 'gambit_programming': 0, 'advanced_programming': 0,
        'legal_concept': 1, 'legal_ai': 1,
        'chess_strategy': 2, 'chess_ai': 2,
        'security_programming': 3, 'ai_integration': 3,
        'general': 4
    }
    
    model.eval()
    total_correct = 0
    total_samples = 0
    predictions_list = []
    targets_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, images)
            predictions = torch.argmax(outputs['classification'], dim=1)
            
            # Create proper targets
            targets = []
            for i in range(predictions.size(0)):
                if isinstance(batch['label'], list):
                    label = batch['label'][i]
                else:
                    label = 'general'
                target_idx = label_to_idx.get(label, 4)
                targets.append(target_idx)
            
            targets = torch.tensor(targets).to(device)
            
            # Calculate accuracy for this batch
            correct = (predictions == targets).sum().item()
            batch_accuracy = correct / targets.size(0)
            
            total_correct += correct
            total_samples += targets.size(0)
            
            # Store for detailed analysis
            predictions_list.extend(predictions.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
            
            # Print detailed batch info
            logger.info(f"Batch {batch_idx + 1}:")
            logger.info(f"  Predictions: {predictions.cpu().numpy()}")
            logger.info(f"  Targets:     {targets.cpu().numpy()}")
            logger.info(f"  Labels:      {batch['label'] if isinstance(batch['label'], list) else 'N/A'}")
            logger.info(f"  Batch Accuracy: {batch_accuracy:.4f} ({correct}/{targets.size(0)})")
            logger.info("-" * 50)
    
    # Calculate final accuracy
    final_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # Print summary
    logger.info("=" * 60)
    logger.info("FINAL RESULTS:")
    logger.info(f"Total Correct: {total_correct}")
    logger.info(f"Total Samples: {total_samples}")
    logger.info(f"REAL ACCURACY: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    logger.info("=" * 60)
    
    # Class-wise accuracy
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred, target in zip(predictions_list, targets_list):
        class_total[target] += 1
        if pred == target:
            class_correct[target] += 1
    
    logger.info("CLASS-WISE ACCURACY:")
    class_names = ['programming', 'legal', 'chess', 'security', 'general']
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            logger.info(f"  {class_name}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
    
    return final_accuracy

def main():
    """Test all available checkpoints"""
    
    # Configuration (same as training)
    config = {
        'neural_architecture': 'resnet18',
        'num_classes': 42,
        'hidden_sizes': [256, 128],
        'rule_set_size': 100,
        'rule_embedding_dim': 64,
        'fusion_dim': 512,
        'num_heads': 8,
        'num_output_classes': 5,
        'image_feature_dim': 512,
    }
    
    # Dataset path
    dataset_path = "../dataset"  # Adjust if needed
    
    # Test available checkpoints
    checkpoint_dir = Path("checkpoints")
    checkpoints = [
        "best_multimodal_newson.pt",
        "multimodal_newson_epoch_20.pt",
        "multimodal_newson_epoch_15.pt",
        "multimodal_newson_epoch_10.pt",
    ]
    
    for checkpoint_name in checkpoints:
        checkpoint_path = checkpoint_dir / checkpoint_name
        if checkpoint_path.exists():
            logger.info(f"\n{'='*60}")
            logger.info(f"TESTING: {checkpoint_name}")
            logger.info(f"{'='*60}")
            
            try:
                model, device = load_model_checkpoint(checkpoint_path, config)
                accuracy = test_model_accuracy(model, device, dataset_path)
                
                logger.info(f"CHECKPOINT {checkpoint_name}: {accuracy:.4f} accuracy")
                
            except Exception as e:
                logger.error(f"Failed to test {checkpoint_name}: {e}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

if __name__ == "__main__":
    main()