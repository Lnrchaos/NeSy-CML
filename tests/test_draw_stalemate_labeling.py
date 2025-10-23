#!/usr/bin/env python3
"""
Test script to verify enhanced draw and stalemate labeling
"""

import torch
from train_chess import ChessDataset

def test_draw_stalemate_labeling():
    """Test the enhanced draw and stalemate labeling system"""
    print("🧪 Testing Enhanced Draw & Stalemate Labeling")
    print("=" * 50)
    
    # Create dataset
    dataset = ChessDataset()
    
    if len(dataset) == 0:
        print("❌ No chess data found!")
        return
    
    print(f"📚 Found {len(dataset)} chess samples")
    
    # Enhanced class names
    class_names = ['tactics', 'strategy', 'opening', 'endgame', 'pieces', 
                   'notation', 'middlegame', 'evaluation', 'checkmate', 'draw']
    
    class_counts = torch.zeros(10)
    total_samples = len(dataset)
    
    # Test sample texts that should trigger draw/stalemate detection
    test_texts = [
        "The position is a stalemate - the king is not in check but has no legal moves",
        "This endgame is a theoretical draw due to insufficient material",
        "White can force a draw by perpetual check",
        "The fifty-move rule applies here, so it's a draw",
        "Black sets up a fortress to hold the draw",
        "This is a stalemate trap that White should avoid",
        "The position repeats three times, so it's a draw by repetition"
    ]
    
    print(f"\n🧪 Testing specific draw/stalemate concepts:")
    
    # Test our labeling function directly
    temp_dataset = dataset
    
    for i, test_text in enumerate(test_texts):
        labels = temp_dataset._create_labels(test_text)
        active_labels = [class_names[j] for j, val in enumerate(labels) if val > 0]
        
        print(f"\n📝 Test {i+1}: '{test_text[:60]}...'")
        print(f"   Detected labels: {active_labels}")
        
        # Check if draw was detected
        if labels[9] > 0:  # draw is index 9
            print(f"   ✅ Draw/stalemate correctly detected!")
        else:
            print(f"   ⚠️  Draw/stalemate NOT detected")
    
    # Analyze actual dataset
    print(f"\n📊 Analyzing all {total_samples} samples for draw content...")
    
    draw_samples = []
    checkmate_samples = []
    
    for i in range(total_samples):
        sample = dataset[i]
        labels = sample['labels']
        class_counts += labels
        
        # Collect samples with draw content
        if labels[9] > 0:  # draw is index 9
            text = sample.get('text', 'No text available')[:200]
            draw_samples.append((i, text))
        
        # Collect samples with checkmate content
        if labels[8] > 0:  # checkmate is index 8
            text = sample.get('text', 'No text available')[:200]
            checkmate_samples.append((i, text))
    
    print(f"\n📈 Complete Label Distribution:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        percentage = (count / total_samples) * 100
        print(f"  {name:12}: {count:3.0f}/{total_samples} ({percentage:5.1f}%)")
    
    print(f"\n🎯 Draw & Stalemate Analysis:")
    draw_count = class_counts[9].item()
    checkmate_count = class_counts[8].item()
    
    print(f"📍 Draw/Stalemate: {draw_count}/{total_samples} samples")
    print(f"📍 Checkmate: {checkmate_count}/{total_samples} samples")
    
    if draw_count > 0:
        print(f"\n✅ Found draw content in these samples:")
        for sample_idx, text in draw_samples[:3]:  # Show first 3
            print(f"  Sample {sample_idx}: {text}...")
    else:
        print(f"\n⚠️  No draw content detected in dataset")
    
    if checkmate_count > 0:
        print(f"\n✅ Found checkmate content in these samples:")
        for sample_idx, text in checkmate_samples[:3]:  # Show first 3
            print(f"  Sample {sample_idx}: {text}...")
    
    # Overall assessment
    print(f"\n📋 Enhanced Labeling Assessment:")
    if draw_count > 0:
        print(f"  ✅ Draw/stalemate detection is working!")
    else:
        print(f"  ⚠️  May need more draw-related content in dataset")
    
    if checkmate_count > 0:
        print(f"  ✅ Checkmate detection is working!")
    
    total_ending_concepts = draw_count + checkmate_count
    print(f"  📊 Total game ending concepts: {total_ending_concepts}/{total_samples}")

if __name__ == "__main__":
    test_draw_stalemate_labeling()