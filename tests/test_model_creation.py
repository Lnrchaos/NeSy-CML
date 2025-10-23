#!/usr/bin/env python3
"""
Test script to debug model creation issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from meta_model import HybridModel, ModelSpec
    print("✅ NeuroSym-CML components imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_model_creation():
    """Test creating the model step by step"""
    print("Testing model creation...")
    
    # Create model spec
    model_spec = ModelSpec(
        neural_architecture='custom_transformer',
        input_shape=(224, 224),
        num_classes=8,
        hidden_sizes=[256, 128, 64],
        dropout_rate=0.1,
        rule_set_size=50,
        use_symbolic_reasoning=True,
        learning_rate=1e-4,
        batch_size=2
    )
    print("✅ ModelSpec created")
    
    # Try to create the model
    try:
        model = HybridModel(model_spec)
        print("✅ HybridModel created successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_creation()
    if success:
        print("🎉 Model creation test passed!")
    else:
        print("💥 Model creation test failed!")

