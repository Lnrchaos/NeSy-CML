#!/usr/bin/env python3
"""
Comprehensive test of the new chess labeling system
"""

import torch
from train_chess import ChessDataset

def comprehensive_chess_test():
    """Test all aspects of the new chess labeling system"""
    print("🧪 Comprehensive Chess Labeling Test")
    print("=" * 50)
    
    # Create dataset
    dataset = ChessDataset()
    
    if len(dataset) == 0:
        print("❌ No chess data found!")
        return
    
    print(f"📚 Found {len(dataset)} chess samples")
    
    # New comprehensive class names
    class_names = ['tactics', 'strategy', 'opening', 'endgame', 'pieces', 
                   'notation', 'middlegame', 'evaluation', 'checkmate', 'draw']
    
    class_counts = torch.zeros(10)
    total_samples = len(dataset)
    
    # Analyze all samples
    print(f"\n📊 Analyzing all {total_samples} samples...")
    
    for i in range(total_samples):
        sample = dataset[i]
        labels = sample['labels']
        class_counts += labels
    
    print(f"\n📈 Complete Label Distribution:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        percentage = (count / total_samples) * 100
        print(f"  {name:12}: {count:3.0f}/{total_samples} ({percentage:5.1f}%)")
    
    # Detailed analysis for key categories
    print(f"\n🎯 Key Category Analysis:")
    
    # Tactics analysis
    tactics_count = class_counts[0].item()
    print(f"📍 Tactics: {tactics_count}/{total_samples} samples")
    if tactics_count >= 3:
        print("  ✅ Good tactics detection (expected 3+ from tactics books)")
    else:
        print("  ⚠️  Low tactics detection - may need improvement")
    
    # Middlegame analysis
    middlegame_count = class_counts[6].item()
    print(f"📍 Middlegame: {middlegame_count}/{total_samples} samples")
    if middlegame_count >= 2:
        print("  ✅ Good middlegame detection")
    else:
        print("  ⚠️  Low middlegame detection")
    
    # Strategy analysis
    strategy_count = class_counts[1].item()
    print(f"📍 Strategy: {strategy_count}/{total_samples} samples")
    
    # Opening analysis
    opening_count = class_counts[2].item()
    print(f"📍 Opening: {opening_count}/{total_samples} samples")
    
    # Show samples with multiple labels (most comprehensive)
    print(f"\n🏆 Most Comprehensive Samples (multiple labels):")
    for i in range(min(5, total_samples)):
        sample = dataset[i]
        labels = sample['labels']
        active_labels = [class_names[j] for j, val in enumerate(labels) if val > 0]
        
        if len(active_labels) >= 3:  # Samples with 3+ labels
            text = sample.get('text', 'No text available')[:150]
            print(f"\n📚 Sample {i}: {len(active_labels)} labels")
            print(f"  Labels: {', '.join(active_labels)}")
            print(f"  Text: {text}...")
    
    # Overall assessment
    total_positive_labels = class_counts.sum().item()
    avg_labels_per_sample = total_positive_labels / total_samples
    
    print(f"\n📋 Overall Assessment:")
    print(f"  Total positive labels: {total_positive_labels}")
    print(f"  Average labels per sample: {avg_labels_per_sample:.2f}")
    print(f"  Label coverage: {(class_counts > 0).sum().item()}/10 categories used")
    
    if avg_labels_per_sample >= 2.0:
        print("  ✅ Excellent multi-label detection!")
    elif avg_labels_per_sample >= 1.5:
        print("  ✅ Good multi-label detection")
    else:
        print("  ⚠️  Could improve multi-label detection")

if __name__ == "__main__":
    comprehensive_chess_test()