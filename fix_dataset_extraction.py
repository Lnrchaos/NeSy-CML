#!/usr/bin/env python3
"""
Fix the dataset to actually use the rich content from your books
"""

import torch
import PyPDF2
from pathlib import Path
from typing import List, Dict
import re

class ImprovedChessDataset:
    """Extract multiple samples per book instead of wasting 99.9% of content"""
    
    def __init__(self, max_samples_per_book=10, chars_per_sample=1024):
        self.max_samples_per_book = max_samples_per_book
        self.chars_per_sample = chars_per_sample
        self.samples = []
        self._load_all_books()
    
    def _load_all_books(self):
        """Load multiple samples from each book"""
        chess_data_dir = Path("dataset/Chess_data")
        
        for pdf_file in chess_data_dir.glob("*.pdf"):
            print(f"📖 Processing {pdf_file.name}...")
            book_samples = self._extract_book_samples(pdf_file)
            self.samples.extend(book_samples)
            print(f"   → Extracted {len(book_samples)} samples")
        
        print(f"\n📊 Total samples: {len(self.samples)}")
    
    def _extract_book_samples(self, pdf_path: Path) -> List[Dict]:
        """Extract multiple samples from a single book"""
        samples = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from multiple pages
                all_text = ""
                pages_to_sample = min(len(pdf_reader.pages), 50)  # Sample up to 50 pages
                
                for i in range(0, pages_to_sample, max(1, pages_to_sample // self.max_samples_per_book)):
                    try:
                        page_text = pdf_reader.pages[i].extract_text()
                        if page_text and len(page_text.strip()) > 100:
                            # Clean the text
                            clean_text = self._clean_text(page_text)
                            
                            if len(clean_text) >= self.chars_per_sample:
                                # Create sample from this page
                                sample_text = clean_text[:self.chars_per_sample]
                                labels = self._create_labels(sample_text)
                                
                                samples.append({
                                    'text': sample_text,
                                    'labels': labels,
                                    'source': pdf_path.name,
                                    'page': i
                                })
                                
                                if len(samples) >= self.max_samples_per_book:
                                    break
                    except:
                        continue
                        
        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")
        
        return samples
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers
        text = re.sub(r'Page \d+', '', text)
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)]', '', text)
        return text.strip()
    
    def _create_labels(self, text: str) -> torch.Tensor:
        """Create labels - copy from your existing system"""
        labels = torch.zeros(10)
        text_lower = text.lower()
        
        # Simplified version - you can use your full labeling system
        if any(term in text_lower for term in ['tactics', 'tactical', 'pin', 'fork', 'combination']):
            labels[0] = 1  # tactics
        if any(term in text_lower for term in ['strategy', 'positional', 'plan']):
            labels[1] = 1  # strategy
        if any(term in text_lower for term in ['opening', 'debut', 'development']):
            labels[2] = 1  # opening
        if any(term in text_lower for term in ['endgame', 'ending', 'opposition']):
            labels[3] = 1  # endgame
        if any(term in text_lower for term in ['pawn', 'king', 'queen', 'rook', 'bishop', 'knight']):
            labels[4] = 1  # pieces
        if any(term in text_lower for term in ['e4', 'd4', 'nf3', 'file', 'rank']):
            labels[5] = 1  # notation
        if any(term in text_lower for term in ['middlegame', 'middle game', 'attack']):
            labels[6] = 1  # middlegame
        if any(term in text_lower for term in ['good move', 'mistake', 'blunder']):
            labels[7] = 1  # evaluation
        if any(term in text_lower for term in ['checkmate', 'mate', 'mating']):
            labels[8] = 1  # checkmate
        if any(term in text_lower for term in ['draw', 'stalemate', 'repetition']):
            labels[9] = 1  # draw
            
        return labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def test_improved_dataset():
    """Test the improved dataset extraction"""
    print("🧪 Testing Improved Dataset Extraction")
    print("=" * 50)
    
    dataset = ImprovedChessDataset(max_samples_per_book=5, chars_per_sample=1024)
    
    print(f"\n📊 Results:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Expected improvement: {len(dataset) / 20:.1f}x more data")
    
    # Analyze label distribution
    class_names = ['tactics', 'strategy', 'opening', 'endgame', 'pieces', 
                   'notation', 'middlegame', 'evaluation', 'checkmate', 'draw']
    
    class_counts = torch.zeros(10)
    for i in range(len(dataset)):
        sample = dataset[i]
        labels = sample['labels']
        class_counts += labels
    
    print(f"\n📈 Label Distribution:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        percentage = (count / len(dataset)) * 100
        print(f"  {name:12}: {count:3.0f}/{len(dataset)} ({percentage:5.1f}%)")
    
    print(f"\n🎯 This should dramatically improve F1 scores!")

if __name__ == "__main__":
    test_improved_dataset()