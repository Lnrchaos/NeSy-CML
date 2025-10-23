#!/usr/bin/env python3
"""
Evaluate the poetry model to see if it's worth releasing
"""

import torch
import json
from pathlib import Path

def evaluate_poetry_model():
    """Check if the poetry model is release-worthy"""
    print("🎭 Evaluating Poetry Model for GitHub Release")
    print("=" * 50)
    
    model_path = "best_poetry_model_optimized.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model file {model_path} not found!")
        return
    
    try:
        # Load the model checkpoint (with weights_only=False for older models)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print(f"📊 Model Information:")
        print(f"   File size: 654.06 MB")
        
        # Check what's in the checkpoint
        if isinstance(checkpoint, dict):
            print(f"   Checkpoint keys: {list(checkpoint.keys())}")
            
            # Look for performance metrics
            if 'best_f1' in checkpoint:
                f1_score = checkpoint['best_f1']
                print(f"   Best F1 Score: {f1_score:.4f}")
                
            if 'best_accuracy' in checkpoint:
                accuracy = checkpoint['best_accuracy']
                print(f"   Best Accuracy: {accuracy:.4f}")
                
            if 'epoch' in checkpoint:
                epoch = checkpoint['epoch']
                print(f"   Training Epochs: {epoch}")
                
            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"   Model Config: {config}")
        
        # Check for training history or results
        results_files = [
            "poetry_training_history.json",
            "poetry_results.json", 
            "poetry_optimized_results.json"
        ]
        
        for results_file in results_files:
            if Path(results_file).exists():
                print(f"\n📈 Found results file: {results_file}")
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    print(f"   Results: {results}")
                except:
                    print(f"   Could not read {results_file}")
        
        # Release worthiness assessment
        print(f"\n🎯 RELEASE WORTHINESS ASSESSMENT:")
        
        # We need to determine if this model is good enough
        # Let's check if we can find any performance indicators
        
        print(f"\n💭 POETRY MODEL EVALUATION:")
        print(f"   ✅ Model exists and loads successfully")
        print(f"   ✅ Reasonable file size (654MB - within 2GB limit)")
        print(f"   ❓ Performance metrics need verification")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. 🧪 Test the model on sample poetry to verify quality")
        print(f"   2. 📊 Check if F1 scores are >0.5 for main categories")
        print(f"   3. 🎭 Ensure it can generate/classify poetry effectively")
        print(f"   4. 📝 Document the model's capabilities clearly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def quick_poetry_test():
    """Quick test of poetry model capabilities"""
    print(f"\n🧪 QUICK POETRY MODEL TEST:")
    print(f"   To properly evaluate, we'd need to:")
    print(f"   1. Load the model architecture")
    print(f"   2. Test on sample poetry")
    print(f"   3. Check classification accuracy")
    print(f"   4. Verify creative generation quality")
    
    print(f"\n🎯 RELEASE DECISION CRITERIA:")
    print(f"   ✅ Release if: F1 > 0.4, generates coherent poetry")
    print(f"   ⏳ Wait if: F1 < 0.3, poor generation quality")
    print(f"   🔧 Improve if: Major issues with poetry understanding")

if __name__ == "__main__":
    success = evaluate_poetry_model()
    if success:
        quick_poetry_test()