#!/usr/bin/env python3
"""
Debug script to check tactics book processing specifically
"""

import torch
from train_chess import ChessDataset
import os

def debug_tactics_books():
    """Debug the tactics book processing"""
    print("🔍 Debugging Tactics Book Processing")
    print("=" * 50)
    
    # Create dataset
    dataset = ChessDataset()
    
    # Expected tactics books
    tactics_files = [
        "How to Calculate Chess Tactics-1.pdf",
        "Mastering Chess Tactics.pdf",
        "Chessbook - Jan Timman - The Art of Chess Analysis (1997).pdf"
    ]
    
    class_names = ['tactics', 'strategy', 'opening', 'endgame', 'pieces', 
                   'notation', 'middlegame', 'evaluation', 'checkmate', 'draw']
    
    print(f"📚 Total samples in dataset: {len(dataset)}")
    
    tactics_samples = []
    
    # Check all samples for tactics content
    for i in range(len(dataset)):
        sample = dataset[i]
        labels = sample['labels']
        
        # Check if this sample has tactics content
        if labels[0] > 0:  # tactics is now index 0
            tactics_samples.append(i)
            text = sample.get('text', 'No text available')
            print(f"\n📚 Found tactics content in sample {i}:")
            print(f"Text length: {len(text)} characters")
            print(f"First 300 characters:")
            print(text[:300])
            print(f"Active labels:")
            for j, (name, value) in enumerate(zip(class_names, labels)):
                if value > 0:
                    print(f"  - {name}")
            print("-" * 30)
    
    print(f"\n📊 Summary:")
    print(f"Found {len(tactics_samples)} samples with tactics content")
    print(f"Expected at least 3 (from the tactics books)")
    
    if len(tactics_samples) < 3:
        print(f"\n⚠️  Missing tactics detection! Let's check what's in the text...")
        
        # Sample a few more to see what terms are actually present
        print(f"\n🔍 Checking first 5 samples for tactics-related terms:")
        tactics_terms = ['tactics', 'tactical', 'pin', 'fork', 'skewer', 'discovered', 'deflection', 'decoy', 
                        'combination', 'calculate', 'analysis', 'pattern', 'motif', 'sacrifice', 'attack']
        
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            text = sample.get('text', '').lower()
            found_terms = [term for term in tactics_terms if term in text]
            
            print(f"\nSample {i}: Found terms: {found_terms}")
            if found_terms:
                print(f"  Text preview: {text[:200]}...")

if __name__ == "__main__":
    debug_tactics_books()