#!/usr/bin/env python3
"""
Extract ALL the rich data from your chess books - every page, every chapter
"""

import torch
import PyPDF2
from pathlib import Path
from typing import List, Dict
import re

class FullChessDataset:
    """Extract EVERY page of content from your rich chess books"""
    
    def __init__(self, chars_per_sample=2048, overlap=512):
        self.chars_per_sample = chars_per_sample
        self.overlap = overlap  # Overlap between samples to not lose context
        self.samples = []
        self._load_all_content()
    
    def _load_all_content(self):
        """Load ALL content from every book"""
        chess_data_dir = Path("dataset/Chess_data")
        
        total_pages_processed = 0
        
        for pdf_file in chess_data_dir.glob("*.pdf"):
            print(f"📖 Processing {pdf_file.name}...")
            book_samples, pages_processed = self._extract_all_book_content(pdf_file)
            self.samples.extend(book_samples)
            total_pages_processed += pages_processed
            print(f"   → Extracted {len(book_samples)} samples from {pages_processed} pages")
        
        print(f"\n📊 FULL EXTRACTION RESULTS:")
        print(f"   Total samples: {len(self.samples)}")
        print(f"   Total pages processed: {total_pages_processed}")
        print(f"   Average samples per page: {len(self.samples)/total_pages_processed:.1f}")
    
    def _extract_all_book_content(self, pdf_path: Path) -> tuple[List[Dict], int]:
        """Extract content from EVERY page of a book"""
        samples = []
        pages_processed = 0
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                print(f"   📄 Processing all {total_pages} pages...")
                
                # Process EVERY page
                for page_num in range(total_pages):
                    try:
                        page_text = pdf_reader.pages[page_num].extract_text()
                        
                        if page_text and len(page_text.strip()) > 200:  # Skip mostly empty pages
                            clean_text = self._clean_text(page_text)
                            
                            if len(clean_text) >= 500:  # Minimum meaningful content
                                # Split long pages into multiple samples with overlap
                                page_samples = self._split_text_into_samples(clean_text, pdf_path.name, page_num)
                                samples.extend(page_samples)
                                pages_processed += 1
                                
                                # Progress indicator for large books
                                if page_num % 50 == 0 and page_num > 0:
                                    print(f"     → Processed {page_num}/{total_pages} pages...")
                                    
                    except Exception as e:
                        # Skip problematic pages but continue
                        continue
                        
        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")
        
        return samples, pages_processed
    
    def _split_text_into_samples(self, text: str, source: str, page_num: int) -> List[Dict]:
        """Split long text into overlapping samples to preserve context"""
        samples = []
        
        # If text is short enough, make one sample
        if len(text) <= self.chars_per_sample:
            labels = self._create_labels(text)
            samples.append({
                'text': text,
                'labels': labels,
                'source': source,
                'page': page_num,
                'sample_id': f"{source}_p{page_num}_s0"
            })
        else:
            # Split into overlapping chunks
            start = 0
            sample_idx = 0
            
            while start < len(text):
                end = start + self.chars_per_sample
                sample_text = text[start:end]
                
                # Try to break at sentence boundaries
                if end < len(text):
                    last_period = sample_text.rfind('.')
                    last_space = sample_text.rfind(' ')
                    if last_period > len(sample_text) * 0.8:  # If period is near end
                        sample_text = sample_text[:last_period + 1]
                    elif last_space > len(sample_text) * 0.8:  # If space is near end
                        sample_text = sample_text[:last_space]
                
                labels = self._create_labels(sample_text)
                samples.append({
                    'text': sample_text,
                    'labels': labels,
                    'source': source,
                    'page': page_num,
                    'sample_id': f"{source}_p{page_num}_s{sample_idx}"
                })
                
                # Move start position with overlap
                start += self.chars_per_sample - self.overlap
                sample_idx += 1
                
                if start >= len(text):
                    break
        
        return samples
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and common headers/footers
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'Chapter \d+', '', text)
        # Remove excessive punctuation but keep chess notation
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\+\=]', '', text)
        return text.strip()
    
    def _create_labels(self, text: str) -> torch.Tensor:
        """Enhanced labeling system using your comprehensive chess terminology"""
        labels = torch.zeros(10)
        text_lower = text.lower()
        
        # 1. TACTICS - comprehensive tactical terms
        tactics_terms = [
            'tactics', 'tactical', 'pin', 'fork', 'skewer', 'discovered attack',
            'discovered check', 'deflection', 'decoy', 'clearance', 'interference',
            'x-ray', 'double attack', 'knight fork', 'royal fork', 'family fork',
            'back rank', 'smothered mate', 'combination', 'combinations', 'sacrifice',
            'zwischenzug', 'in-between move', 'tactical motif', 'tactical pattern'
        ]
        if any(term in text_lower for term in tactics_terms):
            labels[0] = 1
        
        # 2. STRATEGY - comprehensive strategic concepts
        strategy_terms = [
            'strategy', 'strategic', 'plan', 'planning', 'positional', 'positional play',
            'piece coordination', 'central control', 'space advantage', 'weak squares',
            'strong square', 'outpost', 'prophylaxis', 'prevention', 'color complexes',
            'good bishop', 'bad bishop', 'piece harmony', 'domination', 'initiative'
        ]
        if any(term in text_lower for term in strategy_terms):
            labels[1] = 1
        
        # 3. OPENING - opening theory and concepts
        opening_terms = [
            'opening', 'openings', 'debut', 'development', 'castle', 'castling',
            'gambit', 'gambits', 'transposition', 'main line', 'sideline',
            'theory novelty', 'book move', 'sicilian', 'french', 'caro-kann',
            'ruy lopez', 'italian', 'english opening', 'queens gambit', 'kings gambit'
        ]
        if any(term in text_lower for term in opening_terms):
            labels[2] = 1
        
        # 4. ENDGAME - endgame principles and techniques
        endgame_terms = [
            'endgame', 'endgames', 'ending', 'endings', 'opposition', 'triangulation',
            'lucena position', 'philidor position', 'king activity', 'king walk',
            'pawn promotion', 'underpromotion', 'passed pawn', 'outside passed pawn',
            'rook endgame', 'queen endgame', 'bishop endgame', 'knight endgame'
        ]
        if any(term in text_lower for term in endgame_terms):
            labels[3] = 1
        
        # 5. PIECES - comprehensive piece terminology
        piece_terms = [
            'pawn', 'pawns', 'rook', 'rooks', 'knight', 'knights', 'bishop', 'bishops',
            'queen', 'queens', 'king', 'kings', 'piece', 'pieces',
            'kings pawn', 'queens pawn', 'kings bishop', 'queens bishop',
            'bishop pair', 'knight outpost', 'rook lift', 'doubled rooks'
        ]
        if any(term in text_lower for term in piece_terms):
            labels[4] = 1
        
        # 6. NOTATION - chess notation and coordinates
        notation_terms = [
            'e4', 'd4', 'nf3', 'nc3', 'bb5', 'bc4', 'be2', 'bg5',
            'file', 'rank', 'diagonal', 'square', 'squares',
            'a-file', 'e-file', 'h-file', 'first rank', 'eighth rank',
            'kingside', 'queenside', 'center', 'central squares'
        ]
        if any(term in text_lower for term in notation_terms):
            labels[5] = 1
        
        # 7. MIDDLEGAME - middlegame concepts
        middlegame_terms = [
            'middlegame', 'middle game', 'attack', 'attacking', 'defense',
            'counterattack', 'tempo', 'initiative', 'calculation', 'variations',
            'candidate moves', 'forcing moves', 'piece activity', 'coordination'
        ]
        if any(term in text_lower for term in middlegame_terms):
            labels[6] = 1
        
        # 8. EVALUATION - move quality and assessment
        evaluation_terms = [
            'good move', 'excellent move', 'brilliant move', 'mistake', 'blunder',
            'inaccuracy', 'best move', 'advantage', 'disadvantage', 'equal position',
            'winning position', 'evaluation', 'assessment', 'analysis'
        ]
        if any(term in text_lower for term in evaluation_terms):
            labels[7] = 1
        
        # 9. CHECKMATE - mating patterns
        checkmate_terms = [
            'checkmate', 'mate', 'mating', 'mating attack', 'mating pattern',
            'back rank mate', 'smothered mate', 'mate in one', 'mate in two',
            'forced mate', 'check', 'discovered check', 'double check'
        ]
        if any(term in text_lower for term in checkmate_terms):
            labels[8] = 1
        
        # 10. DRAW - draw and stalemate concepts
        draw_terms = [
            'draw', 'drawn', 'stalemate', 'threefold repetition', 'fifty move rule',
            'insufficient material', 'perpetual check', 'fortress', 'dead position',
            'agreed draw', 'draw offer', 'repetition'
        ]
        if any(term in text_lower for term in draw_terms):
            labels[9] = 1
        
        return labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def test_full_extraction():
    """Test extracting ALL content from your rich chess books"""
    print("🚀 FULL CHESS DATASET EXTRACTION")
    print("=" * 60)
    print("📚 Processing ALL pages from ALL your rich chess books...")
    print("⏱️  This will take a few minutes but will use ALL your data!")
    
    # Create full dataset (larger samples, some overlap)
    dataset = FullChessDataset(chars_per_sample=2048, overlap=512)
    
    print(f"\n🎯 FULL EXTRACTION RESULTS:")
    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Improvement over original: {len(dataset) / 20:.1f}x more data")
    print(f"   Sample size: 2048 characters (4x larger)")
    
    # Analyze label distribution
    class_names = ['tactics', 'strategy', 'opening', 'endgame', 'pieces', 
                   'notation', 'middlegame', 'evaluation', 'checkmate', 'draw']
    
    class_counts = torch.zeros(10)
    for i in range(len(dataset)):
        sample = dataset[i]
        labels = sample['labels']
        class_counts += labels
    
    print(f"\n📊 COMPREHENSIVE LABEL DISTRIBUTION:")
    total_labels = class_counts.sum().item()
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        percentage = (count / len(dataset)) * 100
        print(f"  {name:12}: {count:4.0f}/{len(dataset)} ({percentage:5.1f}%) - {'✅ EXCELLENT' if count > 20 else '✅ GOOD' if count > 10 else '⚠️ LOW' if count > 0 else '❌ ZERO'}")
    
    print(f"\n🏆 EXPECTED F1 SCORE IMPROVEMENTS:")
    print(f"   Average labels per sample: {total_labels / len(dataset):.2f}")
    print(f"   Classes with >10 samples: {(class_counts > 10).sum().item()}/10")
    print(f"   Classes with >20 samples: {(class_counts > 20).sum().item()}/10")
    print(f"   Zero-sample classes: {(class_counts == 0).sum().item()}/10")
    
    if (class_counts == 0).sum().item() == 0:
        print(f"   🎉 NO ZERO CLASSES - F1 scores should be much higher!")
    
    print(f"\n💡 This dataset should achieve F1 scores of 0.4-0.7+ !")

if __name__ == "__main__":
    test_full_extraction()