#!/usr/bin/env python3
"""
Analyze how many pages are in each chess book and how much content we're actually using
"""

import os
import PyPDF2
from pathlib import Path

def analyze_book_pages():
    """Analyze the actual page count and content of chess books"""
    print("📚 Chess Book Page Analysis")
    print("=" * 50)
    
    chess_data_dir = Path("dataset/Chess_data")
    
    if not chess_data_dir.exists():
        print("❌ Chess data directory not found!")
        return
    
    total_pages = 0
    book_info = []
    
    for pdf_file in chess_data_dir.glob("*.pdf"):
        try:
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                file_size = os.path.getsize(pdf_file) / (1024 * 1024)  # MB
                
                # Try to extract some text to see quality
                sample_text = ""
                if page_count > 0:
                    try:
                        sample_text = pdf_reader.pages[0].extract_text()[:200]
                    except:
                        sample_text = "Text extraction failed"
                
                book_info.append({
                    'name': pdf_file.name,
                    'pages': page_count,
                    'size_mb': file_size,
                    'sample_text': sample_text
                })
                
                total_pages += page_count
                
        except Exception as e:
            print(f"❌ Error reading {pdf_file.name}: {e}")
    
    # Sort by page count
    book_info.sort(key=lambda x: x['pages'], reverse=True)
    
    print(f"📊 Chess Book Collection Analysis:")
    print(f"   Total books: {len(book_info)}")
    print(f"   Total pages: {total_pages:,}")
    print(f"   Average pages per book: {total_pages/len(book_info):.1f}")
    
    print(f"\n📖 Individual Book Details:")
    for i, book in enumerate(book_info):
        print(f"\n{i+1:2d}. {book['name']}")
        print(f"    📄 Pages: {book['pages']:,}")
        print(f"    💾 Size: {book['size_mb']:.1f} MB")
        print(f"    📝 Sample: {book['sample_text'][:100]}...")
    
    # Calculate the waste
    print(f"\n🎯 THE REAL PROBLEM:")
    print(f"   Total content: {total_pages:,} pages")
    print(f"   Current usage: 512 characters per book")
    print(f"   Estimated chars per page: ~2000")
    print(f"   Total available chars: ~{total_pages * 2000:,}")
    print(f"   Actually used chars: ~{len(book_info) * 512:,}")
    print(f"   WASTE FACTOR: {(total_pages * 2000) / (len(book_info) * 512):.1f}x")
    print(f"   ❌ Using <0.1% of available content!")
    
    print(f"\n💡 SOLUTION:")
    print(f"   Instead of 1 sample per book (512 chars)")
    print(f"   Create multiple samples per book:")
    print(f"   - Extract text from each page/chapter")
    print(f"   - Create ~{total_pages//10} samples from {total_pages:,} pages")
    print(f"   - This would give you 100x more training data!")

if __name__ == "__main__":
    analyze_book_pages()