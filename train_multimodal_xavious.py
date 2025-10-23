#!/usr/bin/env python3
"""
Multi-Modal NewSon Training for Gambit
Trains NeuroSym-CML on text/code + image data for proper NLP understanding
"""

import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image
import transformers
from transformers import AutoTokenizer, AutoModel
import logging

# Suppress networkx warnings
warnings.filterwarnings("ignore", message=".*networkx backend defined more than once.*")

# Import your existing NeuroSym-CML components
from meta_model import HybridModel, ModelSpec
from symbolic_controller import SymbolicController
from replay_buffer import ReplayBuffer
from evaluator import evaluate
from data_module import DataModule, ContinualDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalDataset(Dataset):
    """Dataset for multi-modal training (text + images)"""
    
    def __init__(self, data_dir: str, tokenizer, max_length: int = 512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load training samples from various sources"""
        samples = []
        
        # 1. Programming data (from your programming_data folder)
        prog_data_dir = self.data_dir / "programming_data"
        if prog_data_dir.exists():
            samples.extend(self._load_programming_samples(prog_data_dir))
        
        # 2. Legal data (from your law_data folder)
        law_data_dir = self.data_dir / "law_data"
        if law_data_dir.exists():
            samples.extend(self._load_legal_samples(law_data_dir))
        
        # 3. Chess data (from your Chess_data folder)
        chess_data_dir = self.data_dir / "Chess_data"
        if chess_data_dir.exists():
            samples.extend(self._load_chess_samples(chess_data_dir))
        
        # 4. Create synthetic Gambit code samples
        samples.extend(self._create_gambit_samples())
        
        # 5. Create expanded programming examples
        samples.extend(self._create_expanded_programming_samples())
        
        # 6. Create security-focused examples
        samples.extend(self._create_security_samples())
        
        # 7. Create more diverse examples to prevent overfitting
        samples.extend(self._create_additional_samples())
        
        # 8. Load safe images for multimodal training
        samples.extend(self._load_safe_images())
        
        logger.info(f"Loaded {len(samples)} training samples")
        return samples
    
    def _load_programming_samples(self, prog_dir: Path) -> List[Dict[str, Any]]:
        """Load programming-related samples from actual PDF files"""
        samples = []
        
        try:
            import PyPDF2
            import re
            
            # Process actual programming PDF files
            pdf_files = list(prog_dir.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} programming PDF files to process")
            
            for pdf_file in pdf_files:  # Process ALL PDF files
                try:
                    with open(pdf_file, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        total_pages = len(pdf_reader.pages)
                        logger.info(f"Processing {pdf_file.name}: {total_pages} pages")
                        
                        # Extract text from ALL pages
                        for page_num in range(total_pages):
                            page_text = pdf_reader.pages[page_num].extract_text()
                            
                            if page_text and len(page_text.strip()) > 200:
                                # Clean the text
                                clean_text = re.sub(r'\s+', ' ', page_text)
                                clean_text = re.sub(r'Page \d+', '', clean_text)
                                clean_text = clean_text.strip()
                                
                                if len(clean_text) >= 300:
                                    # Determine category based on filename
                                    category = "programming"
                                    if "python" in pdf_file.name.lower():
                                        category = "python_programming"
                                    elif "swift" in pdf_file.name.lower():
                                        category = "swift_programming"
                                    elif "assembly" in pdf_file.name.lower():
                                        category = "assembly_programming"
                                    elif "machine_learning" in pdf_file.name.lower() or "ai" in pdf_file.name.lower():
                                        category = "ai_programming"
                                    elif "security" in pdf_file.name.lower() or "penetration" in pdf_file.name.lower():
                                        category = "security_programming"
                                    
                                    # Split into chunks for training
                                    chunk_size = 1000
                                    for i in range(0, len(clean_text), chunk_size):
                                        chunk = clean_text[i:i + chunk_size]
                                        if len(chunk) >= 200:
                                            samples.append({
                                                "text": f"Programming Document: {pdf_file.stem}\nContent: {chunk}",
                                                "label": category,
                                                "type": "programming_document",
                                                "source": pdf_file.name
                                            })
                                
                except Exception as e:
                    logger.warning(f"Error processing {pdf_file.name}: {e}")
                    continue
                    
        except ImportError:
            logger.warning("PyPDF2 not available, skipping programming PDF processing")
        except Exception as e:
            logger.error(f"Error loading programming samples: {e}")
        
        logger.info(f"Loaded {len(samples)} programming samples from real PDF files")
        return samples
    
    def _load_legal_samples(self, law_dir: Path) -> List[Dict[str, Any]]:
        """Load legal-related samples from actual PDF files"""
        samples = []
        
        try:
            import PyPDF2
            import re
            
            # Process actual legal PDF files
            pdf_files = list(law_dir.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} legal PDF files to process")
            
            for pdf_file in pdf_files[:10]:  # Limit to first 10 files to avoid memory issues
                try:
                    with open(pdf_file, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        
                        # Extract text from first few pages
                        for page_num in range(min(3, len(pdf_reader.pages))):
                            page_text = pdf_reader.pages[page_num].extract_text()
                            
                            if page_text and len(page_text.strip()) > 200:
                                # Clean the text
                                clean_text = re.sub(r'\s+', ' ', page_text)
                                clean_text = re.sub(r'Page \d+', '', clean_text)
                                clean_text = clean_text.strip()
                                
                                if len(clean_text) >= 300:
                                    # Determine category based on filename
                                    category = "legal_document"
                                    if "criminal" in pdf_file.name.lower():
                                        category = "criminal_law"
                                    elif "civil" in pdf_file.name.lower():
                                        category = "civil_law"
                                    elif "west_virginia" in pdf_file.name.lower():
                                        category = "state_law"
                                    elif "conversation" in pdf_file.name.lower():
                                        category = "legal_conversation"
                                    elif "question" in pdf_file.name.lower():
                                        category = "legal_qa"
                                    
                                    # Split into chunks for training
                                    chunk_size = 1000
                                    for i in range(0, len(clean_text), chunk_size):
                                        chunk = clean_text[i:i + chunk_size]
                                        if len(chunk) >= 200:
                                            samples.append({
                                                "text": f"Legal Document: {pdf_file.stem}\nContent: {chunk}",
                                                "label": category,
                                                "type": "legal_document",
                                                "source": pdf_file.name
                                            })
                                
                except Exception as e:
                    logger.warning(f"Error processing {pdf_file.name}: {e}")
                    continue
                    
        except ImportError:
            logger.warning("PyPDF2 not available, skipping legal PDF processing")
        except Exception as e:
            logger.error(f"Error loading legal samples: {e}")
        
        logger.info(f"Loaded {len(samples)} legal samples from real PDF files")
        return samples
    
    def _load_chess_samples(self, chess_dir: Path) -> List[Dict[str, Any]]:
        """Load chess-related samples from actual PDF files"""
        samples = []
        
        try:
            import PyPDF2
            import re
            
            # Process actual chess PDF files
            pdf_files = list(chess_dir.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} chess PDF files to process")
            
            for pdf_file in pdf_files[:8]:  # Limit to first 8 files to avoid memory issues
                try:
                    with open(pdf_file, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        
                        # Extract text from first few pages
                        for page_num in range(min(3, len(pdf_reader.pages))):
                            page_text = pdf_reader.pages[page_num].extract_text()
                            
                            if page_text and len(page_text.strip()) > 200:
                                # Clean the text
                                clean_text = re.sub(r'\s+', ' ', page_text)
                                clean_text = re.sub(r'Page \d+', '', clean_text)
                                clean_text = clean_text.strip()
                                
                                if len(clean_text) >= 300:
                                    # Determine category based on filename and content
                                    category = "chess_strategy"
                                    if "tactics" in pdf_file.name.lower():
                                        category = "chess_tactics"
                                    elif "opening" in pdf_file.name.lower():
                                        category = "chess_openings"
                                    elif "endgame" in pdf_file.name.lower() or "ending" in pdf_file.name.lower():
                                        category = "chess_endgames"
                                    elif "analysis" in pdf_file.name.lower():
                                        category = "chess_analysis"
                                    elif "compilation" in pdf_file.name.lower():
                                        category = "chess_collection"
                                    
                                    # Split into chunks for training
                                    chunk_size = 1000
                                    for i in range(0, len(clean_text), chunk_size):
                                        chunk = clean_text[i:i + chunk_size]
                                        if len(chunk) >= 200:
                                            samples.append({
                                                "text": f"Chess Guide: {pdf_file.stem}\nContent: {chunk}",
                                                "label": category,
                                                "type": "chess_document",
                                                "source": pdf_file.name
                                            })
                                
                except Exception as e:
                    logger.warning(f"Error processing {pdf_file.name}: {e}")
                    continue
                    
        except ImportError:
            logger.warning("PyPDF2 not available, skipping chess PDF processing")
        except Exception as e:
            logger.error(f"Error loading chess samples: {e}")
        
        logger.info(f"Loaded {len(samples)} chess samples from real PDF files")
        return samples
    
    def _create_gambit_samples(self) -> List[Dict[str, Any]]:
        """Create synthetic Gambit language samples"""
        samples = []
        
        gambit_examples = [
            {
                "description": "Create a simple hello world program",
                "code": "def main():\n    print \"Hello, Gambit!\"\n    return 0\n\nmain()",
                "explanation": "This creates a main function that prints a greeting and returns 0"
            },
            {
                "description": "Use flexible conditional statements",
                "code": "if user_authenticated then:\n    grant_access()\nhowever security_level < 5:\n    require_additional_auth()\nunless emergency_override:\n    log_access_attempt()",
                "explanation": "Gambit's flexible conditionals allow natural language-like logic flow"
            },
            {
                "description": "Implement security operations",
                "code": "firewall configure_rules()\nif threat_detected then:\n    generate table rainbow_attack\n    keylog capture_evidence()\nelse:\n    monitor_traffic()",
                "explanation": "Gambit has built-in security operations for cybersecurity tasks"
            },
            {
                "description": "Use NewSon AI integration",
                "code": "~ Press Ctrl+O to access NewSon\ninstruct newson:\n    \"Analyze this code for vulnerabilities\"\nteach newson:\n    security_best_practices",
                "explanation": "NewSon AI can be instructed and taught through Gambit code"
            }
        ]
        
        for example in gambit_examples:
            samples.append({
                "text": f"Task: {example['description']}\nCode:\n{example['code']}\nExplanation: {example['explanation']}",
                "label": "gambit_programming",
                "type": "code_example"
            })
        
        return samples
    
    def _create_expanded_programming_samples(self) -> List[Dict[str, Any]]:
        """Create expanded programming examples"""
        samples = []
        
        expanded_examples = [
            {
                "description": "Advanced function with error handling",
                "code": "def secure_login(username, password):\n    try:\n        if validate_credentials(username, password):\n            return grant_access(username)\n        else:\n            log_failed_attempt(username)\n            return \"Access denied\"\n    except SecurityException as e:\n        alert_security_team(e)\n        return \"Security alert triggered\"",
                "explanation": "This shows error handling and security practices in Gambit"
            },
            {
                "description": "Data structure manipulation",
                "code": "def process_user_data(users):\n    filtered_users = []\n    for user in users:\n        if user.is_active and user.permissions > 0:\n            filtered_users.append(user)\n    return filtered_users",
                "explanation": "Demonstrates list comprehension and data filtering"
            },
            {
                "description": "Async operations with NewSon",
                "code": "async def analyze_with_newson(code_snippet):\n    result = await newson.analyze_code(code_snippet)\n    if result.has_vulnerabilities:\n        return result.security_recommendations\n    return result.optimization_suggestions",
                "explanation": "Shows async programming with AI integration"
            }
        ]
        
        for example in expanded_examples:
            samples.append({
                "text": f"Advanced Task: {example['description']}\nCode:\n{example['code']}\nExplanation: {example['explanation']}",
                "label": "advanced_programming",
                "type": "advanced_code"
            })
        
        return samples
    
    def _create_security_samples(self) -> List[Dict[str, Any]]:
        """Create security-focused examples"""
        samples = []
        
        security_examples = [
            {
                "description": "Implement input validation",
                "code": "def validate_input(user_input):\n    if contains_sql_injection(user_input):\n        log_security_event(\"SQL injection attempt\")\n        return sanitize_input(user_input)\n    if contains_xss(user_input):\n        log_security_event(\"XSS attempt\")\n        return escape_html(user_input)\n    return user_input",
                "explanation": "Security validation prevents common attacks"
            },
            {
                "description": "Secure file operations",
                "code": "def secure_file_read(filepath):\n    if not is_safe_path(filepath):\n        raise SecurityException(\"Path traversal detected\")\n    if not user_has_permission(current_user, filepath):\n        raise PermissionException(\"Access denied\")\n    return read_file_safely(filepath)",
                "explanation": "Secure file handling with permission checks"
            },
            {
                "description": "Encryption and hashing",
                "code": "def secure_password_storage(password):\n    salt = generate_random_salt()\n    hashed = hash_password_with_salt(password, salt)\n    return store_securely(hashed, salt)",
                "explanation": "Proper password hashing with salt"
            }
        ]
        
        for example in security_examples:
            samples.append({
                "text": f"Security Task: {example['description']}\nCode:\n{example['code']}\nExplanation: {example['explanation']}",
                "label": "security_programming",
                "type": "security_code"
            })
        
        return samples
    
    def _create_additional_samples(self) -> List[Dict[str, Any]]:
        """Create additional diverse samples to prevent overfitting"""
        samples = []
        
        # More programming examples
        prog_examples = [
            {
                "description": "Create a loop in Gambit",
                "code": "for i in range(10):\n    print i\n    if i > 5:\n        break",
                "explanation": "Gambit supports standard loop constructs with break/continue"
            },
            {
                "description": "Error handling in Gambit",
                "code": "try:\n    risky_operation()\nexcept SecurityError:\n    log_security_breach()\nfinally:\n    cleanup_resources()",
                "explanation": "Comprehensive error handling with security awareness"
            },
            {
                "description": "Data structures in Gambit",
                "code": "users = {\"admin\": {\"level\": 5}, \"guest\": {\"level\": 1}}\nfor user, data in users.items():\n    grant_permissions(user, data[\"level\"])",
                "explanation": "Dictionary operations with security-aware permission handling"
            }
        ]
        
        # More legal examples
        legal_examples = [
            {
                "question": "What is due process?",
                "answer": "Due process ensures fair treatment through the judicial system",
                "category": "legal_concept"
            },
            {
                "question": "How does contract law apply to software?",
                "answer": "Software licenses are contracts governing usage rights and obligations",
                "category": "legal_concept"
            }
        ]
        
        # More chess examples
        chess_examples = [
            {
                "question": "What is a fork in chess?",
                "answer": "A fork attacks two or more pieces simultaneously",
                "category": "chess_strategy"
            },
            {
                "question": "How do you castle in chess?",
                "answer": "Move king two squares toward rook, then rook to square king crossed",
                "category": "chess_strategy"
            }
        ]
        
        # Add programming samples
        for example in prog_examples:
            samples.append({
                "text": f"Programming Task: {example['description']}\nCode:\n{example['code']}\nExplanation: {example['explanation']}",
                "label": "programming",
                "type": "code_example"
            })
        
        # Add legal samples
        for example in legal_examples:
            samples.append({
                "text": f"Legal Question: {example['question']}\nAnswer: {example['answer']}",
                "label": example['category'],
                "type": "legal_qa"
            })
        
        # Add chess samples
        for example in chess_examples:
            samples.append({
                "text": f"Chess Question: {example['question']}\nAnswer: {example['answer']}",
                "label": example['category'],
                "type": "chess_qa"
            })
        
        return samples
    
    def _load_safe_images(self) -> List[Dict[str, Any]]:
        """Load safe images for multimodal training"""
        samples = []
        
        try:
            from PIL import Image
            import torch
            from torchvision import transforms
            
            # Path to safe images
            safe_images_dir = self.data_dir / "safe_images"
            
            if not safe_images_dir.exists():
                logger.warning("Safe images directory not found - skipping image samples")
                return samples
            
            # Image preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Load images from each category
            categories = [
                "geometric", "colors", "numbers", "letters", 
                "chess_pieces", "programming", "patterns", 
                "charts", "educational"
            ]
            
            for category in categories:
                category_dir = safe_images_dir / category
                if category_dir.exists():
                    png_files = list(category_dir.glob("*.png"))
                    
                    for img_file in png_files[:5]:  # Limit to 5 per category
                        try:
                            # Load and process image
                            with Image.open(img_file) as img:
                                img = img.convert('RGB')
                                img_tensor = transform(img)
                                
                                # Create descriptive text based on category and filename
                                item_name = img_file.stem.split('_')[0]
                                
                                if category == "geometric":
                                    text = f"This is a geometric shape: {item_name}. Geometric shapes are fundamental elements in mathematics and design."
                                elif category == "colors":
                                    text = f"This represents the color {item_name}. Colors are important in visual perception and design."
                                elif category == "numbers":
                                    text = f"This is the number {item_name}. Numbers are used for counting, measuring, and mathematical operations."
                                elif category == "letters":
                                    text = f"This is the letter {item_name}. Letters form the alphabet used in written language."
                                elif category == "chess_pieces":
                                    text = f"This is a chess piece symbol. Chess is a strategic board game played worldwide."
                                elif category == "programming":
                                    text = f"This represents the programming keyword '{item_name}'. Programming keywords are reserved words in code."
                                elif category == "patterns":
                                    text = f"This shows a visual pattern: {item_name}. Patterns are recurring designs or sequences."
                                elif category == "charts":
                                    text = f"This is a data visualization chart. Charts help represent information graphically."
                                elif category == "educational":
                                    text = f"This is an educational symbol used in mathematics or science. Educational symbols help convey complex concepts."
                                else:
                                    text = f"This is an image from the {category} category showing {item_name}."
                                
                                samples.append({
                                    "text": text,
                                    "image_tensor": img_tensor,
                                    "image_path": str(img_file),
                                    "label": category,
                                    "type": "image_text_pair"
                                })
                                
                        except Exception as e:
                            logger.warning(f"Error loading image {img_file}: {e}")
                            continue
            
            # Also load some downloaded images
            downloaded_dir = safe_images_dir / "downloaded"
            if downloaded_dir.exists():
                for subdir in ["picsum", "nature", "objects"]:
                    subdir_path = downloaded_dir / subdir
                    if subdir_path.exists():
                        png_files = list(subdir_path.glob("*.png"))
                        
                        for img_file in png_files[:3]:  # Limit to 3 per subdirectory
                            try:
                                with Image.open(img_file) as img:
                                    img = img.convert('RGB')
                                    img_tensor = transform(img)
                                    
                                    if subdir == "picsum":
                                        text = "This is a photograph from a public domain image collection. Such images are useful for visual AI training."
                                    elif subdir == "nature":
                                        text = "This is a nature-inspired image showing natural elements like landscapes, sky, or organic forms."
                                    elif subdir == "objects":
                                        text = "This shows everyday objects that are commonly found in our environment."
                                    else:
                                        text = f"This is an image from the {subdir} collection."
                                    
                                    samples.append({
                                        "text": text,
                                        "image_tensor": img_tensor,
                                        "image_path": str(img_file),
                                        "label": f"downloaded_{subdir}",
                                        "type": "image_text_pair"
                                    })
                                    
                            except Exception as e:
                                logger.warning(f"Error loading downloaded image {img_file}: {e}")
                                continue
            
            logger.info(f"Loaded {len(samples)} safe image samples")
            
        except ImportError as e:
            logger.warning(f"Missing dependencies for image loading: {e}")
        except Exception as e:
            logger.error(f"Error loading safe images: {e}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            sample['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Use real image if available, otherwise generate synthetic
        if 'image_tensor' in sample:
            image = sample['image_tensor']
        else:
            image = self._generate_content_image(sample)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'image': image,
            'label': sample['label'],
            'text': sample['text']
        }
    
    def _generate_content_image(self, sample):
        """Generate meaningful images based on content type"""
        import numpy as np
        
        # Create structured patterns instead of random noise
        if 'programming' in sample['label'] or 'gambit' in sample['label']:
            # Code-like patterns: structured, grid-like
            image = self._create_code_pattern()
        elif 'legal' in sample['label']:
            # Document-like patterns: text blocks, formal structure
            image = self._create_document_pattern()
        elif 'chess' in sample['label']:
            # Chess board patterns: geometric, strategic
            image = self._create_chess_pattern()
        elif 'security' in sample['label']:
            # Network/security patterns: connected nodes, flows
            image = self._create_security_pattern()
        else:
            # General patterns: balanced, neutral
            image = self._create_general_pattern()
        
        return torch.tensor(image, dtype=torch.float32)
    
    def _create_code_pattern(self):
        """Create code-like visual patterns"""
        image = np.zeros((3, 224, 224))
        
        # Create code-like structure: lines, indentation, blocks
        for i in range(0, 224, 20):  # Horizontal lines (code lines)
            if np.random.random() > 0.3:  # Not every line
                indent = np.random.randint(0, 4) * 10  # Indentation
                line_length = np.random.randint(50, 180)  # Variable length
                image[0, i:i+2, indent:indent+line_length] = 0.8  # Blue channel
                
                # Add syntax highlighting colors
                if np.random.random() > 0.7:  # Keywords
                    image[1, i:i+2, indent:indent+30] = 0.6  # Green
                if np.random.random() > 0.8:  # Strings
                    image[2, i:i+2, indent+40:indent+80] = 0.7  # Red
        
        return image
    
    def _create_document_pattern(self):
        """Create document-like visual patterns"""
        image = np.zeros((3, 224, 224))
        
        # Create document structure: paragraphs, margins
        margin = 20
        for block in range(4):  # 4 paragraphs
            start_y = margin + block * 50
            end_y = start_y + 30
            
            # Paragraph block
            image[:, start_y:end_y, margin:224-margin] = 0.3
            
            # Title/header for first block
            if block == 0:
                image[:, start_y:start_y+5, margin:120] = 0.8
        
        return image
    
    def _create_chess_pattern(self):
        """Create chess-like visual patterns"""
        image = np.zeros((3, 224, 224))
        
        # Create checkerboard pattern
        square_size = 28  # 224/8 = 28
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    y_start, y_end = i * square_size, (i + 1) * square_size
                    x_start, x_end = j * square_size, (j + 1) * square_size
                    image[:, y_start:y_end, x_start:x_end] = 0.6
        
        # Add some "pieces" (random bright spots)
        for _ in range(12):
            i, j = np.random.randint(0, 8, 2)
            y_start, y_end = i * square_size, (i + 1) * square_size
            x_start, x_end = j * square_size, (j + 1) * square_size
            image[np.random.randint(0, 3), y_start:y_end, x_start:x_end] = 1.0
        
        return image
    
    def _create_security_pattern(self):
        """Create security/network-like visual patterns"""
        image = np.zeros((3, 224, 224))
        
        # Create network nodes
        nodes = [(np.random.randint(20, 204), np.random.randint(20, 204)) for _ in range(8)]
        
        # Draw nodes
        for x, y in nodes:
            image[:, max(0, y-5):min(224, y+5), max(0, x-5):min(224, x+5)] = 0.8
        
        # Connect nodes with lines
        for i in range(len(nodes)-1):
            x1, y1 = nodes[i]
            x2, y2 = nodes[i+1]
            # Simple line drawing (approximate)
            steps = max(abs(x2-x1), abs(y2-y1))
            if steps > 0:
                for step in range(steps):
                    x = int(x1 + (x2-x1) * step / steps)
                    y = int(y1 + (y2-y1) * step / steps)
                    if 0 <= x < 224 and 0 <= y < 224:
                        image[1, y, x] = 0.5  # Green connections
        
        return image
    
    def _create_general_pattern(self):
        """Create general visual patterns"""
        image = np.zeros((3, 224, 224))
        
        # Create gradient background
        for i in range(224):
            for j in range(224):
                image[0, i, j] = i / 224 * 0.3
                image[1, i, j] = j / 224 * 0.3
                image[2, i, j] = (i + j) / (2 * 224) * 0.3
        
        return image

class MultiModalNeuroSym(nn.Module):
    """Multi-modal NeuroSym-CML model for text and image understanding"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Text encoder (using your existing architecture)
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        text_dim = self.text_encoder.config.hidden_size
        
        # Image encoder (using your existing NeuroSym-CML)
        # Set num_classes to match the feature dimension we want
        model_spec = ModelSpec(
            neural_architecture=config.get('neural_architecture', 'resnet18'),
            num_classes=config.get('image_feature_dim', 512),  # Use feature dim as num_classes
            hidden_sizes=config.get('hidden_sizes', [256, 128]),
            use_symbolic_reasoning=True,
            rule_set_size=config.get('rule_set_size', 100),
            rule_embedding_dim=config.get('rule_embedding_dim', 64)
        )
        self.image_encoder = HybridModel(model_spec)
        image_dim = config.get('image_feature_dim', 512)
        
        # Fusion layer
        fusion_dim = config.get('fusion_dim', 512)
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.image_projection = nn.Linear(image_dim, fusion_dim)
        
        # Multi-modal fusion
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=config.get('num_heads', 8),
            batch_first=True
        )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, config.get('num_output_classes', 5))
        )
        
        # Response generation (simple for now)
        self.response_generator = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Linear(fusion_dim * 2, text_dim),
            nn.Tanh()
        )
        
        # Initialize SymbolicController properly to avoid loading errors
        from symbolic_controller import SymbolicController
        # Use text_dim (768) as input_size to match saved checkpoint
        symbolic_input_size = config.get('symbolic_controller_input_size', text_dim)  # Use text_dim instead of fusion_dim
        self.symbolic_controller = SymbolicController(
            num_rules=config.get('rule_set_size', 100),
            input_size=symbolic_input_size,
            hidden_size=64
        )
    
    def forward(self, input_ids, attention_mask, images):
        batch_size = input_ids.size(0)
        
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state.mean(dim=1)  # [batch, text_dim]
        text_projected = self.text_projection(text_features)  # [batch, fusion_dim]
        
        # Encode images (using your NeuroSym-CML)
        # Your HybridModel needs text_embeddings and rule_indices
        # Use symbolic controller to generate intelligent rule indices
        # Controller is now pre-initialized in __init__
        
        # Generate rule indices based on text features and task type
        task_metadata = {'type': 0, 'id': 0}  # Could be enhanced with actual task info
        rule_indices, symbolic_state = self.symbolic_controller(text_features, task_metadata)
        
        # Use text features as text_embeddings for the image encoder
        image_features = self.image_encoder(images, text_features, rule_indices)
        if isinstance(image_features, tuple):
            image_features = image_features[0]  # Take first element if tuple
        
        # Ensure image features are the right shape
        if len(image_features.shape) > 2:
            image_features = image_features.view(batch_size, -1)
        
        image_projected = self.image_projection(image_features)  # [batch, fusion_dim]
        
        # Multi-modal fusion using attention
        # Prepare for attention: [batch, seq_len, embed_dim]
        text_seq = text_projected.unsqueeze(1)  # [batch, 1, fusion_dim]
        image_seq = image_projected.unsqueeze(1)  # [batch, 1, fusion_dim]
        
        # Concatenate text and image sequences
        combined_seq = torch.cat([text_seq, image_seq], dim=1)  # [batch, 2, fusion_dim]
        
        # Apply attention
        fused_features, _ = self.fusion_layer(
            combined_seq, combined_seq, combined_seq
        )  # [batch, 2, fusion_dim]
        
        # Pool the fused features
        fused_pooled = fused_features.mean(dim=1)  # [batch, fusion_dim]
        
        # Generate outputs
        classification = self.classifier(fused_pooled)
        response_features = self.response_generator(fused_pooled)
        
        return {
            'classification': classification,
            'response_features': response_features,
            'fused_features': fused_pooled
        }

class MultiModalTrainer:
    """Trainer for multi-modal NeuroSym-CML model"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize model
        self.model = MultiModalNeuroSym(config).to(self.device)
        
        # Freeze most parameters to prevent overfitting
        if config.get('freeze_backbone', True):
            self._freeze_backbone_layers()
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Initialize loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.response_loss = nn.MSELoss()
        
        # Initialize replay buffer for experience replay
        self.replay_buffer = ReplayBuffer(memory_size=config.get('memory_size', 1000))
        self.use_replay = config.get('use_replay', True)
        
        # Initialize advanced data module for meta-learning
        self.data_module = None  # Will be initialized when needed
        self.use_meta_learning = config.get('use_meta_learning', False)
        
        logger.info(f"Initialized MultiModalTrainer on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Replay buffer enabled: {self.use_replay}")
        logger.info(f"Meta-learning enabled: {self.use_meta_learning}")
    
    def _freeze_backbone_layers(self):
        """Freeze backbone layers to prevent overfitting on small datasets"""
        # Freeze text encoder (BERT) layers except the last few
        for name, param in self.model.text_encoder.named_parameters():
            if 'encoder.layer.11' not in name and 'encoder.layer.10' not in name:
                param.requires_grad = False
        
        # Freeze most of the image encoder backbone
        if hasattr(self.model.image_encoder, 'neural_component'):
            for name, param in self.model.image_encoder.neural_component.named_parameters():
                if 'fc' not in name and 'classifier' not in name:  # Keep final layers trainable
                    param.requires_grad = False
        
        logger.info("Frozen backbone layers to prevent overfitting")
    
    def train(self, dataset_path: str, epochs: int = 10, batch_size: int = 8):
        """Train the multi-modal model"""
        
        # Create dataset and dataloader
        dataset = MultiModalDataset(dataset_path, self.tokenizer)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Batch size: {batch_size}")
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    images = batch['image'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask, images)
                    
                    # Create proper targets based on sample labels
                    batch_size = input_ids.size(0)
                    
                    # Map text labels to class indices
                    label_to_idx = {
                        # Programming related (class 0)
                        'programming': 0, 'gambit_programming': 0, 'advanced_programming': 0,
                        'python_programming': 0, 'swift_programming': 0, 'assembly_programming': 0,
                        'ai_programming': 0, 'security_programming': 0, 'programming_languages': 0,
                        'programming_document': 0,
                        
                        # Legal related (class 1)
                        'legal_concept': 1, 'legal_ai': 1, 'legal_document': 1, 'criminal_law': 1,
                        'civil_law': 1, 'state_law': 1, 'legal_conversation': 1, 'legal_qa': 1,
                        
                        # Chess related (class 2)
                        'chess_strategy': 2, 'chess_ai': 2, 'chess_tactics': 2, 'chess_openings': 2,
                        'chess_endgames': 2, 'chess_analysis': 2, 'chess_collection': 2, 'chess_document': 2,
                        'chess_pieces': 2,
                        
                        # Visual/Educational (class 3)
                        'geometric': 3, 'colors': 3, 'numbers': 3, 'letters': 3, 'patterns': 3,
                        'charts': 3, 'educational': 3, 'math': 3, 'science': 3,
                        
                        # Downloaded/Misc (class 4)
                        'downloaded_picsum': 4, 'downloaded_nature': 4, 'downloaded_objects': 4,
                        'downloaded_unsplash': 4, 'downloaded_pixabay_demo': 4, 'general': 4,
                        'ai_integration': 4
                    }
                    
                    classification_targets = []
                    for i in range(batch_size):
                        label = batch[f'label'][i] if isinstance(batch['label'], list) else 'general'
                        target_idx = label_to_idx.get(label, 4)  # Default to 'general'
                        classification_targets.append(target_idx)
                    
                    classification_targets = torch.tensor(classification_targets).to(self.device)
                    
                    # Calculate classification loss (main loss)
                    class_loss = self.classification_loss(
                        outputs['classification'], 
                        classification_targets
                    )
                    
                    # Use classification loss as main signal (remove confusing response loss)
                    total_loss_batch = class_loss
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    total_loss_batch.backward()
                    
                    # Gradient clipping (more aggressive)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    
                    self.optimizer.step()
                    
                    # Store experience in replay buffer
                    if self.use_replay:
                        experience = (
                            input_ids.cpu(),
                            attention_mask.cpu(), 
                            images.cpu(),
                            classification_targets.cpu(),
                            total_loss_batch.item()
                        )
                        self.replay_buffer.add(experience)
                        
                        # Replay experiences if buffer has enough samples
                        if len(self.replay_buffer) >= 32:
                            self._replay_experiences()
                    
                    total_loss += total_loss_batch.item()
                    num_batches += 1
                    
                    if batch_idx % 10 == 0:
                        # Memory monitoring
                        if torch.cuda.is_available():
                            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                            memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                            logger.info(
                                f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                                f"Loss: {total_loss_batch.item():.4f}, "
                                f"GPU Memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached"
                            )
                        else:
                            logger.info(
                                f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                                f"Loss: {total_loss_batch.item():.4f}"
                            )
                
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_loss = total_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
            
            # Evaluate model performance every epoch
            self._evaluate_model(dataset)
            
            # Save checkpoint every epoch
            self.save_checkpoint(epoch + 1, avg_loss)
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"multimodal_newson_epoch_{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Also save as best model if it's the first or has better loss
        best_model_path = checkpoint_dir / "best_multimodal_newson.pt"
        if not best_model_path.exists() or loss < getattr(self, 'best_loss', float('inf')):
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'config': self.config
            }, best_model_path)
            self.best_loss = loss
            logger.info(f"Best model updated: {best_model_path}")
    
    def _replay_experiences(self):
        """Replay stored experiences for better learning"""
        try:
            # Sample experiences from replay buffer
            experiences = self.replay_buffer.sample(16)  # Smaller replay batch
            
            for exp in experiences:
                input_ids, attention_mask, images, targets, _ = exp
                
                # Move to device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, images)
                
                # Calculate loss
                replay_loss = self.classification_loss(outputs['classification'], targets)
                
                # Backward pass with reduced learning rate
                self.optimizer.zero_grad()
                (replay_loss * 0.5).backward()  # Reduced weight for replay
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()
                
        except Exception as e:
            logger.warning(f"Replay experience failed: {e}")
    
    def _evaluate_model(self, dataset):
        """Evaluate model using the advanced evaluator"""
        try:
            logger.info("Running comprehensive model evaluation...")
            
            # Create a small evaluation dataloader
            eval_dataloader = DataLoader(
                dataset, 
                batch_size=8, 
                shuffle=False,
                num_workers=0
            )
            
            # Custom evaluation for multimodal model
            self.model.eval()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch in eval_dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    images = batch['image'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask, images)
                    
                    # Simple accuracy calculation
                    predictions = torch.argmax(outputs['classification'], dim=1)
                    
                    # Create proper targets based on batch labels
                    label_to_idx = {
                        # Programming related (class 0)
                        'programming': 0, 'gambit_programming': 0, 'advanced_programming': 0,
                        'python_programming': 0, 'swift_programming': 0, 'assembly_programming': 0,
                        'ai_programming': 0, 'security_programming': 0, 'programming_languages': 0,
                        'programming_document': 0,
                        
                        # Legal related (class 1)
                        'legal_concept': 1, 'legal_ai': 1, 'legal_document': 1, 'criminal_law': 1,
                        'civil_law': 1, 'state_law': 1, 'legal_conversation': 1, 'legal_qa': 1,
                        
                        # Chess related (class 2)
                        'chess_strategy': 2, 'chess_ai': 2, 'chess_tactics': 2, 'chess_openings': 2,
                        'chess_endgames': 2, 'chess_analysis': 2, 'chess_collection': 2, 'chess_document': 2,
                        'chess_pieces': 2,
                        
                        # Visual/Educational (class 3)
                        'geometric': 3, 'colors': 3, 'numbers': 3, 'letters': 3, 'patterns': 3,
                        'charts': 3, 'educational': 3, 'math': 3, 'science': 3,
                        
                        # Downloaded/Misc (class 4)
                        'downloaded_picsum': 4, 'downloaded_nature': 4, 'downloaded_objects': 4,
                        'downloaded_unsplash': 4, 'downloaded_pixabay_demo': 4, 'general': 4,
                        'ai_integration': 4
                    }
                    
                    # Debug: Print batch structure
                    if total_samples == 0:  # First batch only
                        logger.info(f"Batch keys: {list(batch.keys())}")
                        logger.info(f"Label type: {type(batch['label'])}")
                        if hasattr(batch['label'], 'shape'):
                            logger.info(f"Label shape: {batch['label'].shape}")
                    
                    # Handle different label formats
                    if isinstance(batch['label'], torch.Tensor):
                        # If labels are already tensors, use directly
                        targets = batch['label'].to(self.device)
                    elif isinstance(batch['label'], list):
                        # If labels are list of strings, convert
                        targets = []
                        for label in batch['label']:
                            target_idx = label_to_idx.get(label, 4)
                            targets.append(target_idx)
                        targets = torch.tensor(targets).to(self.device)
                    else:
                        # Fallback: create dummy targets for debugging
                        logger.warning(f"Unexpected label format: {type(batch['label'])}")
                        targets = torch.zeros(predictions.size(0), dtype=torch.long).to(self.device)
                    
                    correct = (predictions == targets).sum().item()
                    total_correct += correct
                    total_samples += targets.size(0)
                    
                    # Debug info for first batch
                    if total_samples <= 8:  # First batch
                        logger.info(f"Predictions: {predictions.cpu().numpy()}")
                        logger.info(f"Targets: {targets.cpu().numpy()}")
                        logger.info(f"Correct: {correct}/{targets.size(0)}")
            
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            logger.info(f"Evaluation Accuracy: {accuracy:.4f}")
            
            self.model.train()  # Back to training mode
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")

def main():
    """Main training function"""
    
    # Configuration
    config = {
        # Model architecture
        'neural_architecture': 'resnet18',  # Use your proven architecture
        'num_classes': 42,  # Match your trained model
        'hidden_sizes': [256, 128],
        'rule_set_size': 100,
        'rule_embedding_dim': 64,
        
        # Multi-modal settings
        'fusion_dim': 512,
        'num_heads': 8,
        'num_output_classes': 5,  # programming, legal, chess, security, general
        'image_feature_dim': 512,
        
        # Training settings
        'learning_rate': 5e-6,  # Much lower learning rate
        'weight_decay': 1e-4,   # Higher weight decay for regularization
        'batch_size': 4,  # Small batch size to avoid memory issues
        'epochs': 20,
        
        # Symbolic reasoning settings
        'use_symbolic_controller': True,
        'symbolic_hidden_size': 64,
        
        # Experience replay settings
        'use_replay': True,
        'memory_size': 1000,
        
        # Meta-learning settings
        'use_meta_learning': False,  # Can be enabled for advanced training
        'n_way': 5,  # Number of classes per meta-task
        'k_shot': 5,  # Number of examples per class
        'query_size': 10,  # Number of query examples per class
        
        # Data settings
        'max_text_length': 512,
    }
    
    # Initialize trainer
    trainer = MultiModalTrainer(config)
    
    # Start training
    dataset_path = "dataset"  # Path to your dataset folder
    trainer.train(
        dataset_path=dataset_path,
        epochs=config['epochs'],
        batch_size=config['batch_size']
    )
    
    logger.info("Training completed!")
    logger.info("You can now use the trained model with NewSon for proper NLP understanding!")

if __name__ == "__main__":
    main()