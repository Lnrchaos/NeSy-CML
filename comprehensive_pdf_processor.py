#!/usr/bin/env python3
"""
Comprehensive PDF processor for extracting ALL content from ALL PDFs
Processes every page of every PDF in each dataset category
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import PyPDF2
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensivePDFProcessor:
    """Extract ALL content from ALL PDFs in dataset"""
    
    def __init__(self, dataset_dir="dataset", chunk_size=1500, overlap=300):
        self.dataset_dir = Path(dataset_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.samples = []
        
    def process_all_datasets(self):
        """Process all PDF datasets comprehensively"""
        logger.info("🚀 Starting comprehensive PDF processing...")
        
        # Process each dataset category
        categories = {
            "programming_data": self._process_programming_pdfs,
            "law_data": self._process_legal_pdfs,
            "Chess_data": self._process_chess_pdfs,
            "poetry": self._process_poetry_pdfs,
            "Martial_Arts": self._process_martial_arts_pdfs
        }
        
        total_samples = 0
        
        for category, processor_func in categories.items():
            category_dir = self.dataset_dir / category
            if category_dir.exists():
                logger.info(f"📚 Processing {category}...")
                category_samples = processor_func(category_dir)
                self.samples.extend(category_samples)
                total_samples += len(category_samples)
                logger.info(f"✅ {category}: {len(category_samples)} samples extracted")
            else:
                logger.warning(f"⚠️  {category} directory not found")
        
        logger.info(f"🎉 Total samples extracted: {total_samples}")
        return self.samples
    
    def _process_programming_pdfs(self, prog_dir: Path) -> List[Dict[str, Any]]:
        """Process ALL programming PDFs comprehensively"""
        samples = []
        pdf_files = list(prog_dir.glob("*.pdf"))
        
        for pdf_file in tqdm(pdf_files, desc="Programming PDFs"):
            file_samples = self._extract_all_pages(pdf_file, "programming")
            samples.extend(file_samples)
            
        return samples
    
    def _process_legal_pdfs(self, law_dir: Path) -> List[Dict[str, Any]]:
        """Process ALL legal PDFs comprehensively"""
        samples = []
        pdf_files = list(law_dir.glob("*.pdf"))
        
        for pdf_file in tqdm(pdf_files, desc="Legal PDFs"):
            file_samples = self._extract_all_pages(pdf_file, "legal")
            samples.extend(file_samples)
            
        return samples
    
    def _process_chess_pdfs(self, chess_dir: Path) -> List[Dict[str, Any]]:
        """Process ALL chess PDFs comprehensively"""
        samples = []
        pdf_files = list(chess_dir.glob("*.pdf"))
        
        for pdf_file in tqdm(pdf_files, desc="Chess PDFs"):
            file_samples = self._extract_all_pages(pdf_file, "chess")
            samples.extend(file_samples)
            
        return samples
    
    def _process_poetry_pdfs(self, poetry_dir: Path) -> List[Dict[str, Any]]:
        """Process ALL poetry PDFs comprehensively"""
        samples = []
        pdf_files = list(poetry_dir.glob("*.pdf"))
        
        for pdf_file in tqdm(pdf_files, desc="Poetry PDFs"):
            file_samples = self._extract_all_pages(pdf_file, "poetry")
            samples.extend(file_samples)
            
        return samples
    
    def _process_martial_arts_pdfs(self, ma_dir: Path) -> List[Dict[str, Any]]:
        """Process ALL martial arts PDFs comprehensively"""
        samples = []
        pdf_files = list(ma_dir.glob("*.pdf"))
        
        for pdf_file in tqdm(pdf_files, desc="Martial Arts PDFs"):
            file_samples = self._extract_all_pages(pdf_file, "martial_arts")
            samples.extend(file_samples)
            
        return samples
    
    def _extract_all_pages(self, pdf_path: Path, domain: str) -> List[Dict[str, Any]]:
        """Extract content from EVERY page of a PDF"""
        samples = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num in range(total_pages):
                    try:
                        page_text = pdf_reader.pages[page_num].extract_text()
                        
                        if page_text and len(page_text.strip()) > 200:
                            clean_text = self._clean_text(page_text)
                            
                            if len(clean_text) >= 500:
                                # Split into overlapping chunks
                                page_samples = self._create_text_chunks(
                                    clean_text, pdf_path, page_num, domain
                                )
                                samples.extend(page_samples)
                                
                    except Exception as e:
                        logger.debug(f"Error processing page {page_num} of {pdf_path.name}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
        
        return samples
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'Chapter \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Clean up special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', ' ', text)
        
        return text.strip()
    
    def _create_text_chunks(self, text: str, pdf_path: Path, page_num: int, domain: str) -> List[Dict[str, Any]]:
        """Create overlapping text chunks for training"""
        chunks = []
        
        # Create overlapping chunks
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk = text[i:i + self.chunk_size]
            
            if len(chunk) >= 500:  # Minimum chunk size
                # Determine specific category
                category = self._categorize_content(pdf_path.name, chunk, domain)
                
                chunks.append({
                    "text": chunk,
                    "label": category,
                    "domain": domain,
                    "source_file": pdf_path.name,
                    "page_number": page_num,
                    "chunk_index": len(chunks),
                    "type": "pdf_content"
                })
        
        return chunks
    
    def _categorize_content(self, filename: str, content: str, domain: str) -> str:
        """Categorize content based on filename and content analysis"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        if domain == "programming":
            if "python" in filename_lower or "python" in content_lower:
                return "python_programming"
            elif "swift" in filename_lower or "swift" in content_lower:
                return "swift_programming"
            elif "assembly" in filename_lower or "mips" in filename_lower:
                return "assembly_programming"
            elif "machine_learning" in filename_lower or "ai" in filename_lower:
                return "ai_programming"
            elif "security" in filename_lower or "penetration" in filename_lower:
                return "security_programming"
            elif "language" in filename_lower:
                return "programming_languages"
            else:
                return "general_programming"
                
        elif domain == "legal":
            if "criminal" in filename_lower or "criminal" in content_lower:
                return "criminal_law"
            elif "civil" in filename_lower or "civil" in content_lower:
                return "civil_law"
            elif "west_virginia" in filename_lower:
                return "state_law"
            elif "conversation" in filename_lower:
                return "legal_conversation"
            elif "question" in filename_lower:
                return "legal_qa"
            elif "constitution" in content_lower:
                return "constitutional_law"
            else:
                return "general_law"
                
        elif domain == "chess":
            if "tactics" in filename_lower or "tactic" in content_lower:
                return "chess_tactics"
            elif "opening" in filename_lower or "opening" in content_lower:
                return "chess_openings"
            elif "endgame" in filename_lower or "ending" in filename_lower:
                return "chess_endgames"
            elif "strategy" in filename_lower or "strategy" in content_lower:
                return "chess_strategy"
            elif "analysis" in filename_lower:
                return "chess_analysis"
            else:
                return "general_chess"
                
        elif domain == "poetry":
            if "anthology" in filename_lower:
                return "poetry_anthology"
            elif "modern" in filename_lower or "children" in filename_lower:
                return "modern_poetry"
            elif "tragedy" in filename_lower:
                return "classical_poetry"
            else:
                return "general_poetry"
                
        elif domain == "martial_arts":
            if "hapkido" in filename_lower:
                return "hapkido"
            elif "karate" in filename_lower:
                return "karate"
            elif "kyokushin" in filename_lower:
                return "kyokushin"
            elif "aikido" in filename_lower:
                return "aikido"
            elif "judo" in filename_lower:
                return "judo"
            else:
                return "general_martial_arts"
        
        return domain
    
    def save_processed_data(self, output_file="processed_comprehensive_dataset.json"):
        """Save processed data to JSON file"""
        output_path = self.dataset_dir / output_file
        
        # Create summary statistics
        summary = {
            "total_samples": len(self.samples),
            "domains": {},
            "categories": {},
            "source_files": {}
        }
        
        for sample in self.samples:
            domain = sample["domain"]
            category = sample["label"]
            source = sample["source_file"]
            
            summary["domains"][domain] = summary["domains"].get(domain, 0) + 1
            summary["categories"][category] = summary["categories"].get(category, 0) + 1
            summary["source_files"][source] = summary["source_files"].get(source, 0) + 1
        
        # Save data
        data = {
            "summary": summary,
            "samples": self.samples
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved comprehensive dataset to {output_path}")
        
        # Print summary
        print("\n📊 Comprehensive Dataset Summary:")
        print(f"   Total Samples: {summary['total_samples']:,}")
        print(f"   Domains: {len(summary['domains'])}")
        for domain, count in summary['domains'].items():
            print(f"     - {domain}: {count:,} samples")
        print(f"   Categories: {len(summary['categories'])}")
        print(f"   Source Files: {len(summary['source_files'])}")

def main():
    """Main function"""
    processor = ComprehensivePDFProcessor()
    
    print("🚀 Comprehensive PDF Dataset Processor")
    print("=" * 50)
    print("This will extract ALL content from ALL PDFs in your dataset")
    print("Processing: Programming, Legal, Chess, Poetry, Martial Arts")
    print()
    
    # Process all datasets
    samples = processor.process_all_datasets()
    
    # Save processed data
    processor.save_processed_data()
    
    print(f"\n✅ Processing complete! Extracted {len(samples):,} samples")
    print("💡 This comprehensive dataset utilizes ALL your PDF content")

if __name__ == "__main__":
    main()