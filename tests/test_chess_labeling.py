#!/usr/bin/env python3
"""
Test script to verify chess content labeling is working correctly
"""

import torch
from train_chess import ChessDataset

def test_chess_labeling():
    """Test the chess labeling system"""
    print("🧪 Testing Chess Content Labeling")
    print("=" * 40)
    
    # Create dataset
    dataset = ChessDataset()
    
    if len(dataset) == 0:
        print("❌ No chess data found!")
        return
    
    print(f"📚 Found {len(dataset)} chess samples")
    
    # Test a few samples
    class_names = ['tactics', 'strategy', 'opening', 'endgame', 'pieces', 
                   'notation', 'middlegame', 'evaluation', 'checkmate', 'draw']
    
    class_counts = torch.zeros(10)
    
    # Sample first 10 items to check labeling
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        labels = sample['labels']
        text = sample.get('text', 'No text available')[:200]  # First 200 chars
        
        print(f"\n📄 Sample {i+1}:")
        print(f"Text preview: {text}...")
        
        active_labels = []
        for j, (name, value) in enumerate(zip(class_names, labels)):
            if value > 0:
                active_labels.append(name)
                class_counts[j] += 1
        
        print(f"Labels: {active_labels if active_labels else 'None'}")
    
    print(f"\n📊 Overall Label Distribution:")
    total_samples = min(10, len(dataset))
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        percentage = (count / total_samples) * 100
        print(f"  {name:12}: {count:3.0f}/{total_samples} ({percentage:5.1f}%)")
    
    # Check specifically for middlegame
    middlegame_count = class_counts[6].item()  # middlegame is now index 6
    print(f"\n🎯 Middlegame Detection:")
    print(f"Found {middlegame_count} samples with middlegame content out of {total_samples} tested")
    
    if middlegame_count == 0:
        print("⚠️  No middlegame content detected - this might indicate:")
        print("   1. PDF text extraction issues")
        print("   2. Labeling logic needs improvement")
        print("   3. Content doesn't contain expected keywords")
    else:
        print("✅ Middlegame content successfully detected!")

if __name__ == "__main__":
    test_chess_labeling()