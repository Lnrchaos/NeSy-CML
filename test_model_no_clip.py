#!/usr/bin/env python3
"""
Test script to create model without CLIP loading
"""

import sys
import os
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from meta_model import ModelSpec
    print("✅ ModelSpec imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class SimpleHybridModel(nn.Module):
    """Simplified HybridModel without CLIP for testing"""
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.device = torch.device(self.spec.device)
        
        # Simple backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 256)
        )
        self.feature_dim = 256
        
        # Simple text projection (no CLIP)
        self.text_projection = nn.Linear(512, self.feature_dim)
        
        # Rule embeddings
        self.rule_embeddings = nn.Embedding(
            num_embeddings=self.spec.rule_set_size,
            embedding_dim=self.spec.rule_embedding_dim
        )
        
        # Classifier
        self.classifier = nn.Linear(
            self.feature_dim + self.feature_dim + self.spec.rule_embedding_dim,
            self.spec.num_classes
        )
        
    def forward(self, x, text_embeddings, rule_indices):
        # Extract image features
        image_features = self.backbone(x)
        
        # Project text embeddings
        text_features = self.text_projection(text_embeddings)
        
        # Get rule embeddings
        rule_embs = self.rule_embeddings(rule_indices)
        
        # Concatenate all features
        fused = torch.cat([image_features, text_features, rule_embs], dim=1)
        
        # Classify
        return self.classifier(fused)

def test_simple_model():
    """Test creating the simplified model"""
    print("Testing simplified model creation...")
    
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
    
    # Create simplified model
    try:
        model = SimpleHybridModel(model_spec)
        print("✅ SimpleHybridModel created successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        text_embeddings = torch.randn(batch_size, 512)
        rule_indices = torch.randint(0, 50, (batch_size,))
        
        with torch.no_grad():
            output = model(x, text_embeddings, rule_indices)
            print(f"✅ Forward pass successful, output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_model()
    if success:
        print("🎉 Simplified model test passed!")
    else:
        print("💥 Simplified model test failed!")

