#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for PDF text extraction functionality
"""
import os
import sys
import io
import sys
from pathlib import Path

# Set stdout to use UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Import after setting up encoding
from train_chess_improved import ImprovedChessTrainer

def test_pdf_extraction(pdf_path: str):
    """Test PDF content extraction from a single file"""
    print(f"\n🔍 Testing PDF extraction for: {pdf_path}")
    
    # Test content extraction using FullChessDataset
    try:
        from pathlib import Path
        from train_chess_improved import FullChessDataset
        
        # Create a temporary dataset instance to test extraction
        dataset = FullChessDataset.__new__(FullChessDataset)  # Create without calling __init__
        dataset.chars_per_sample = 2048
        dataset.overlap = 512
        dataset.max_length = 256
        dataset.samples = []
        
        # Test extraction on single file
        samples, pages_processed = dataset._extract_all_book_content(Path(pdf_path))
        
        if not samples:
            print("❌ Error: No content extracted from PDF")
            return False
            
        print(f"✅ Successfully extracted {len(samples)} samples from {pages_processed} pages")
        
        # Show first few samples
        print("\nFirst 3 samples:")
        for i, sample in enumerate(samples[:3]):
            print(f"\nSample {i+1}:")
            if 'text' in sample:
                text_preview = sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text']
                print(f"Text: {text_preview}")
            if 'label' in sample:
                print(f"Label: {sample['label']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting content: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    if len(sys.argv) == 2:
        # If PDF path provided as argument
        pdf_path = sys.argv[1]
        if not os.path.exists(pdf_path):
            print(f"Error: File not found: {pdf_path}")
            return
        test_pdf_extraction(pdf_path)
    else:
        # Default: test with a chess PDF from your dataset
        chess_data_dir = "dataset/Chess_data"
        
        if os.path.exists(chess_data_dir):
            pdf_files = [f for f in os.listdir(chess_data_dir) if f.endswith('.pdf')]
            if pdf_files:
                pdf_path = os.path.join(chess_data_dir, pdf_files[0])
                print(f"🧪 Testing PDF processing with: {pdf_files[0]}")
                test_pdf_extraction(pdf_path)
            else:
                print("❌ No PDF files found in dataset/Chess_data/")
                print("Usage: python test_pdf_processing.py <path_to_pdf>")
        else:
            print("❌ Chess data directory not found")
            print("Usage: python test_pdf_processing.py <path_to_pdf>")

if __name__ == "__main__":
    import torch  # Import here to avoid circular imports
    main()
