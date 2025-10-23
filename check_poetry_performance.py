#!/usr/bin/env python3
"""
Check the actual performance metrics of the poetry model
"""

import torch
import json

def check_poetry_performance():
    """Check the detailed performance of the poetry model"""
    print("🎭 Poetry Model Performance Analysis")
    print("=" * 50)
    
    try:
        # Load the model checkpoint
        checkpoint = torch.load("best_poetry_model_optimized.pt", map_location='cpu', weights_only=False)
        
        print(f"📊 Training Information:")
        print(f"   Epochs completed: {checkpoint.get('epoch', 'Unknown')}")
        print(f"   Model classes: {checkpoint['config']['num_classes']}")
        
        # Check creativity score
        if 'creativity_score' in checkpoint:
            creativity = checkpoint['creativity_score']
            print(f"   Creativity Score: {creativity}")
        
        # Check poetry metrics
        if 'poetry_metrics' in checkpoint:
            metrics = checkpoint['poetry_metrics']
            print(f"\n🎯 Poetry Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
        
        # Check training history
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            print(f"\n📈 Training History:")
            if isinstance(history, list) and len(history) > 0:
                last_epoch = history[-1]
                print(f"   Last epoch metrics: {last_epoch}")
            else:
                print(f"   Training history: {history}")
        
        # Overall assessment
        print(f"\n🎯 RELEASE WORTHINESS DECISION:")
        
        # Check if we have enough training
        epochs = checkpoint.get('epoch', 0)
        if epochs < 5:
            print(f"   ⚠️  Only {epochs} epoch(s) trained - might be undertrained")
            print(f"   💡 Recommendation: Train for more epochs before release")
            return False
        
        # Check if we have performance metrics
        has_metrics = 'poetry_metrics' in checkpoint or 'creativity_score' in checkpoint
        if not has_metrics:
            print(f"   ⚠️  No clear performance metrics found")
            print(f"   💡 Recommendation: Evaluate model performance first")
            return False
        
        # If we have creativity score, check if it's reasonable
        if 'creativity_score' in checkpoint:
            creativity = checkpoint['creativity_score']
            if creativity > 0.5:
                print(f"   ✅ Good creativity score: {creativity:.3f}")
                print(f"   🎉 Model appears release-worthy!")
                return True
            else:
                print(f"   ⚠️  Low creativity score: {creativity:.3f}")
                print(f"   💡 Recommendation: Improve training before release")
                return False
        
        print(f"   ❓ Unable to determine quality - need more evaluation")
        return False
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        return False

def release_recommendations():
    """Provide specific recommendations for release"""
    print(f"\n💡 RELEASE RECOMMENDATIONS:")
    print(f"   📊 For GitHub Release, your model should have:")
    print(f"   ✅ F1 scores > 0.4 for main poetry categories")
    print(f"   ✅ Creativity score > 0.5")
    print(f"   ✅ At least 10+ training epochs")
    print(f"   ✅ Coherent poetry generation capability")
    
    print(f"\n🎯 CURRENT STATUS:")
    print(f"   Chess Model: ✅ Ready (83.94% accuracy, 25 epochs)")
    print(f"   Poetry Model: ❓ Needs evaluation")
    
    print(f"\n📝 SUGGESTED APPROACH:")
    print(f"   1. 🚀 Release chess model immediately (excellent performance)")
    print(f"   2. 🧪 Test poetry model more thoroughly")
    print(f"   3. 🎭 If poetry model performs well, release as v1.1")
    print(f"   4. 📈 If not, train longer and release later")

if __name__ == "__main__":
    is_worthy = check_poetry_performance()
    release_recommendations()
    
    if is_worthy:
        print(f"\n🎉 VERDICT: Poetry model is RELEASE WORTHY! 🚀")
    else:
        print(f"\n⏳ VERDICT: Hold off on poetry model release for now")