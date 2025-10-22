#!/usr/bin/env python3
"""
Test script to verify checkpoint loading functionality
"""

import torch
import os
from meta_model import HybridModel, ModelSpec

def test_checkpoint_loading():
    """Test loading a checkpoint without running full training"""
    print("🧪 Testing checkpoint loading...")
    
    # Create a simple model spec
    spec = ModelSpec(
        neural_architecture='custom_cnn',
        num_classes=42,
        hidden_sizes=[512, 256, 128],
        rule_set_size=100,
        rule_embedding_dim=64
    )
    
    # Create model
    model = HybridModel(spec)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test loading checkpoint
    checkpoint_path = "checkpoints/secure_checkpoint_epoch_5.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"📊 Checkpoint info:")
    print(f"   - Epoch: {checkpoint['epoch']}")
    print(f"   - Keys: {list(checkpoint.keys())}")
    
    # Test loading model state
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("✅ Model state loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"⚠️  Strict loading failed: {e}")
        try:
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
            print("✅ Model state loaded successfully (strict=False)")
            if missing_keys:
                print(f"   Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"   Unexpected keys: {unexpected_keys}")
        except Exception as e2:
            print(f"❌ Failed to load model state: {e2}")
            return False
    
    print("🎉 Checkpoint loading test completed successfully!")
    return True

if __name__ == "__main__":
    test_checkpoint_loading()
