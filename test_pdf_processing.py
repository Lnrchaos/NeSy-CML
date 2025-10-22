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
from train_legal_neurosym import LegalNeuroSymTrainer

def test_pdf_extraction(pdf_path: str):
    """Test PDF text extraction from a single file"""
    print(f"\n🔍 Testing PDF extraction for: {pdf_path}")
    
    # Initialize trainer with minimal config
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'learning_rate': 0.001,
        'batch_size': 4,
        'epochs': 10,
        'rule_set_size': 50,
        'memory_size': 1000,
    }
    trainer = LegalNeuroSymTrainer(config)
    
    # Test text extraction
    try:
        text = trainer._extract_pdf_text(pdf_path)
        if not text.strip():
            print("❌ Error: No text extracted from PDF")
            return False
            
        print(f"✅ Successfully extracted {len(text)} characters")
        print("\nFirst 500 characters:")
        print(text[:500] + "..." if len(text) > 500 else text)
        return True
        
    except Exception as e:
        print(f"❌ Error extracting text: {str(e)}")
        return False

def main():
    """Main test function"""
    if len(sys.argv) < 2:
        print("Usage: python test_pdf_processing.py <path_to_pdf>")
        return
        
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
        
    test_pdf_extraction(pdf_path)

if __name__ == "__main__":
    import torch  # Import here to avoid circular imports
    main()
