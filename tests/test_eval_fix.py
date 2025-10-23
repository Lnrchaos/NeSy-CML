#!/usr/bin/env python3
"""
Test script to verify the evaluation fix works
"""

import torch
from train_chess_improved import ImprovedChessTrainer
from train_chess import ChessDataset
from torch.utils.data import DataLoader

def test_evaluation_fix():
    """Test that the evaluation function works without shape errors"""
    print("🧪 Testing Evaluation Fix")
    print("=" * 40)
    
    # Create a minimal config
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_classes': 10,
        'max_length': 256,
        'batch_size': 1,  # Use batch size 1 to test edge cases
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'num_epochs': 1,
        'rule_set_size': 75,
        'replay_buffer_size': 1000,
        'gradient_accumulation_steps': 1,
        'mixed_precision': False  # Disable for testing
    }
    
    try:
        # Create dataset and dataloader
        dataset = ChessDataset()
        if len(dataset) == 0:
            print("❌ No chess data found!")
            return
        
        # Use a small subset for testing
        subset_size = min(3, len(dataset))
        subset_indices = list(range(subset_size))
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        
        dataloader = DataLoader(
            subset_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"📚 Testing with {subset_size} samples")
        
        # Create trainer
        trainer = ImprovedChessTrainer(config)
        
        # Test the comprehensive evaluation
        print("🔍 Testing comprehensive evaluation...")
        trainer._comprehensive_evaluation(dataloader)
        
        print("✅ Evaluation completed successfully!")
        print("🎉 Shape mismatch issue has been fixed!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluation_fix()