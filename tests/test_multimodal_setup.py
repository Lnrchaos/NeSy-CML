#!/usr/bin/env python3
"""
Test script to verify the multimodal training setup works correctly
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from meta_model import ModelSpec, HybridModel
from train_multimodal_newson import MultiModalNeuroSym, MultiModalDataset, MultiModalTrainer
from transformers import AutoTokenizer

def test_model_creation():
    """Test that we can create the models without errors"""
    print("Testing model creation...")
    
    # Test ModelSpec and HybridModel
    spec = ModelSpec(
        neural_architecture="resnet18",
        num_classes=10,
        hidden_sizes=[256, 128],
        use_symbolic_reasoning=True,
        rule_set_size=50,
        rule_embedding_dim=64
    )
    
    hybrid_model = HybridModel(spec)
    print(f"✓ HybridModel created successfully")
    print(f"  - Architecture: {spec.neural_architecture}")
    print(f"  - Feature dim: {hybrid_model.feature_dim}")
    
    # Test MultiModalNeuroSym
    config = {
        'neural_architecture': 'resnet18',
        'num_classes': 10,
        'hidden_sizes': [256, 128],
        'rule_set_size': 50,
        'rule_embedding_dim': 64,
        'fusion_dim': 512,
        'num_heads': 8,
        'num_output_classes': 5,
        'image_feature_dim': 512
    }
    
    multimodal_model = MultiModalNeuroSym(config)
    print(f"✓ MultiModalNeuroSym created successfully")
    
    return hybrid_model, multimodal_model, config

def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\nTesting forward pass...")
    
    hybrid_model, multimodal_model, config = test_model_creation()
    
    # Create dummy data
    batch_size = 2
    device = torch.device('cpu')  # Use CPU for testing
    
    # Test HybridModel forward pass
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_text_embeddings = torch.randn(batch_size, 768)  # BERT hidden size
    dummy_rule_indices = torch.randint(0, config['rule_set_size'], (batch_size,))
    
    try:
        hybrid_output = hybrid_model(dummy_images, dummy_text_embeddings, dummy_rule_indices)
        print(f"✓ HybridModel forward pass successful")
        print(f"  - Output shape: {hybrid_output.shape}")
    except Exception as e:
        print(f"✗ HybridModel forward pass failed: {e}")
        return False
    
    # Test MultiModalNeuroSym forward pass
    dummy_input_ids = torch.randint(0, 1000, (batch_size, 128))  # Token IDs
    dummy_attention_mask = torch.ones(batch_size, 128)
    
    try:
        multimodal_output = multimodal_model(dummy_input_ids, dummy_attention_mask, dummy_images)
        print(f"✓ MultiModalNeuroSym forward pass successful")
        print(f"  - Classification shape: {multimodal_output['classification'].shape}")
        print(f"  - Response features shape: {multimodal_output['response_features'].shape}")
        print(f"  - Fused features shape: {multimodal_output['fused_features'].shape}")
    except Exception as e:
        print(f"✗ MultiModalNeuroSym forward pass failed: {e}")
        return False
    
    return True

def test_dataset_creation():
    """Test dataset creation with dummy data"""
    print("\nTesting dataset creation...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Create a temporary dataset directory structure
        temp_dir = Path("temp_test_dataset")
        temp_dir.mkdir(exist_ok=True)
        
        # Create dummy subdirectories
        (temp_dir / "programming_data").mkdir(exist_ok=True)
        (temp_dir / "law_data").mkdir(exist_ok=True)
        (temp_dir / "Chess_data").mkdir(exist_ok=True)
        
        # Test dataset creation
        dataset = MultiModalDataset(str(temp_dir), tokenizer)
        print(f"✓ Dataset created successfully")
        print(f"  - Dataset size: {len(dataset)}")
        
        # Test getting a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  - Sample keys: {list(sample.keys())}")
            print(f"  - Input IDs shape: {sample['input_ids'].shape}")
            print(f"  - Image shape: {sample['image'].shape}")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False

def test_trainer_initialization():
    """Test trainer initialization"""
    print("\nTesting trainer initialization...")
    
    config = {
        'neural_architecture': 'resnet18',
        'num_classes': 10,
        'hidden_sizes': [256, 128],
        'rule_set_size': 50,
        'rule_embedding_dim': 64,
        'fusion_dim': 512,
        'num_heads': 8,
        'num_output_classes': 5,
        'image_feature_dim': 512,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 2,
        'epochs': 1,
        'max_text_length': 128
    }
    
    try:
        trainer = MultiModalTrainer(config)
        print(f"✓ Trainer initialized successfully")
        print(f"  - Device: {trainer.device}")
        print(f"  - Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
        return True
        
    except Exception as e:
        print(f"✗ Trainer initialization failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("MULTIMODAL NEUROSYM-CML SETUP TEST")
    print("=" * 60)
    
    tests = [
        test_model_creation,
        test_forward_pass,
        test_dataset_creation,
        test_trainer_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your multimodal setup is ready for training.")
        print("\nNext steps:")
        print("1. Prepare your dataset in the correct directory structure")
        print("2. Run: python train_multimodal_newson.py")
        print("3. Monitor training progress and adjust hyperparameters as needed")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()