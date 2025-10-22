#!/usr/bin/env python3
"""
Check what's being extracted from the compilation book
"""

import torch
from train_chess import ChessDataset

def check_compilation_book():
    """Check the content from the compilation book"""
    print("🔍 Checking Compilation Book Content")
    print("=" * 50)
    
    dataset = ChessDataset()
    
    class_names = ['tactics', 'strategy', 'opening', 'endgame', 'pieces', 
                   'notation', 'middlegame', 'evaluation', 'checkmate', 'draw']
    
    print(f"📚 Total samples: {len(dataset)}")
    
    # Look for samples that might be from the compilation book
    compilation_samples = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        text = sample.get('text', '').lower()
        labels = sample['labels']
        
        # Look for indicators this might be from the compilation
        if any(word in text for word in ['compilation', 'collection', 'books', 'chess books']):
            compilation_samples.append((i, text[:300], labels))
    
    if compilation_samples:
        print(f"\n📖 Found {len(compilation_samples)} potential compilation samples:")
        for i, (sample_idx, text, labels) in enumerate(compilation_samples):
            active_labels = [class_names[j] for j, val in enumerate(labels) if val > 0]
            print(f"\nSample {sample_idx}:")
            print(f"  Text: {text}...")
            print(f"  Labels: {active_labels}")
    else:
        print(f"\n⚠️  No obvious compilation samples found")
        print(f"Let's check the last few samples (likely the newest):")
        
        for i in range(max(0, len(dataset)-3), len(dataset)):
            sample = dataset[i]
            text = sample.get('text', '')[:200]
            labels = sample['labels']
            active_labels = [class_names[j] for j, val in enumerate(labels) if val > 0]
            
            print(f"\nSample {i}:")
            print(f"  Text: {text}...")
            print(f"  Labels: {active_labels}")

if __name__ == "__main__":
    check_compilation_book()