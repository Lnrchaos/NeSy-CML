#!/usr/bin/env python3
"""
Poetry Training for NeuroSym-CML
Trains NeuroSym-CML on poetry data with web scraping and custom architecture
"""

import os
import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
import random

# Import existing components
from meta_model import HybridModel, ModelSpec
from symbolic_controller import SymbolicController, ControllerConfig
from replay_buffer import ReplayBuffer

class PoetryWebScraper:
    """Handles web scraping of poetry from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
        })
        self.visited_urls = set()
        self.poem_links = set()
    
    def scrape_poetry_website(self, base_url: str, max_poems: int = 50) -> List[Dict]:
        """Scrape poetry from a given website"""
        try:
            print(f"Scraping poetry from: {base_url}")
            
            # Special handling for AllPoetry
            if 'allpoetry.com' in base_url:
                return self._scrape_allpoetry(base_url, max_poems)
            
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            poems = []
            
            # Look for common poetry containers
            poem_containers = soup.find_all(['div', 'article', 'section'], class_=re.compile(r'poem|verse|poetry', re.I))
            
            for container in poem_containers:
                if len(poems) >= max_poems:
                    break
                
                # Extract poem text
                poem_text = container.get_text(strip=True)
                if len(poem_text) > 50:  # Minimum length for a poem
                    poems.append({
                        'text': poem_text,
                        'source_url': base_url,
                        'title': self._extract_title(container),
                        'author': self._extract_author(container)
                    })
            
            # If no specific containers found, look for text blocks
            if not poems:
                text_blocks = soup.find_all('p')
                for block in text_blocks:
                    if len(poems) >= max_poems:
                        break
                    
                    text = block.get_text(strip=True)
                    if len(text) > 50 and self._looks_like_poetry(text):
                        poems.append({
                            'text': text,
                            'source_url': base_url,
                            'title': 'Untitled',
                            'author': 'Unknown'
                        })
            
            print(f"Successfully scraped {len(poems)} poems from {base_url}")
            return poems
            
        except Exception as e:
            print(f"Error scraping {base_url}: {str(e)}")
            return []
    
    def _scrape_allpoetry(self, base_url: str, max_poems: int = 50) -> List[Dict]:
        """Specialized scraper for AllPoetry.com"""
        try:
            print(f"Scraping AllPoetry from: {base_url}")
            response = self.session.get(base_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            poems = []
            
            # Debug: Print page title and some basic info
            title = soup.find('title')
            if title:
                print(f"Page title: {title.get_text()}")
            
            # First, find all poem links on the page
            poem_links = []
            
            # Look for poem links in various containers
            link_selectors = [
                'a[href*="/poem/"]',
                'a[href*="/poems/"]',
                '.poem-title a',
                '.poem-link a',
                'h3 a',
                'h4 a',
                '.poem-list a',
                '.poem-item a',
                '.poem-entry a',
                'div[class*="poem"] a',
                'li a[href*="/poem/"]'
            ]
            
            for selector in link_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href and '/poem/' in href:
                        if not href.startswith('http'):
                            href = 'https://allpoetry.com' + href
                        poem_links.append(href)
            
            # If no links found with selectors, try a more general approach
            if not poem_links:
                print("No poem links found with specific selectors, trying general approach...")
                all_links = soup.find_all('a', href=True)
                for link in all_links:
                    href = link.get('href')
                    if href and ('/poem/' in href or '/poems/' in href):
                        if not href.startswith('http'):
                            href = 'https://allpoetry.com' + href
                        poem_links.append(href)
            
            # Remove duplicates and limit
            poem_links = list(set(poem_links))[:max_poems]
            print(f"Found {len(poem_links)} poem links to scrape")
            
            # Check for pagination if we need more poems
            if len(poem_links) < max_poems:
                print("Checking for additional pages...")
                pagination_links = self._find_pagination_links(soup, base_url)
                for page_url in pagination_links[:3]:  # Limit to 3 additional pages
                    if len(poems) >= max_poems:
                        break
                    try:
                        print(f"Scraping additional page: {page_url}")
                        page_response = self.session.get(page_url, timeout=10)
                        page_soup = BeautifulSoup(page_response.text, 'html.parser')
                        additional_links = self._extract_poem_links_from_page(page_soup)
                        poem_links.extend(additional_links)
                        time.sleep(1)
                    except Exception as e:
                        print(f"Error scraping pagination page {page_url}: {e}")
                        continue
            
            # Remove duplicates and limit again
            poem_links = list(set(poem_links))[:max_poems]
            
            # Now scrape each individual poem
            for i, poem_url in enumerate(poem_links):
                if len(poems) >= max_poems:
                    break
                
                try:
                    print(f"Scraping poem {i+1}/{len(poem_links)}: {poem_url}")
                    poem_data = self._scrape_individual_poem(poem_url)
                    if poem_data:
                        poems.append(poem_data)
                    
                    # Be respectful to the server
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error scraping poem {poem_url}: {e}")
                    continue
            
            print(f"Successfully scraped {len(poems)} poems from AllPoetry")
            return poems
            
        except Exception as e:
            print(f"Error scraping AllPoetry {base_url}: {str(e)}")
            return []
    
    def _scrape_individual_poem(self, poem_url: str) -> Optional[Dict]:
        """Scrape an individual poem from AllPoetry"""
        try:
            response = self.session.get(poem_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract poem title
            title = "Untitled"
            title_elem = soup.find('h1') or soup.find('h2') or soup.find('h3')
            if title_elem:
                title = title_elem.get_text(strip=True)
            
            # Extract author
            author = "Unknown"
            author_elem = soup.find('a', class_=re.compile(r'author|user', re.I))
            if author_elem:
                author = author_elem.get_text(strip=True)
            
            # Extract poem text - look for various poem containers
            poem_text = ""
            poem_containers = [
                '.poem-text',
                '.poem-content',
                '.poem-body',
                '.poem',
                '.verse',
                '.poetry-content',
                'div[class*="poem"]',
                'div[class*="verse"]'
            ]
            
            for selector in poem_containers:
                container = soup.select_one(selector)
                if container:
                    poem_text = container.get_text(strip=True)
                    break
            
            # If no specific container found, look for text blocks
            if not poem_text:
                text_blocks = soup.find_all(['p', 'div'], string=re.compile(r'[A-Za-z]'))
                poem_lines = []
                for block in text_blocks:
                    text = block.get_text(strip=True)
                    if len(text) > 10 and self._looks_like_poetry(text):
                        poem_lines.append(text)
                
                if poem_lines:
                    poem_text = '\n'.join(poem_lines)
            
            # Clean up the poem text
            if poem_text:
                # Remove common website elements
                poem_text = re.sub(r'^\s*Share this poem\s*$', '', poem_text, flags=re.MULTILINE)
                poem_text = re.sub(r'^\s*Rate this poem\s*$', '', poem_text, flags=re.MULTILINE)
                poem_text = re.sub(r'^\s*Comments\s*$', '', poem_text, flags=re.MULTILINE)
                poem_text = re.sub(r'^\s*Add to favorites\s*$', '', poem_text, flags=re.MULTILINE)
                poem_text = re.sub(r'\s+', ' ', poem_text).strip()
                
                if len(poem_text) > 50:  # Minimum length for a poem
                    return {
                        'text': poem_text,
                        'source_url': poem_url,
                        'title': title,
                        'author': author
                    }
            
            return None
            
        except Exception as e:
            print(f"Error scraping individual poem {poem_url}: {e}")
            return None
    
    def _find_pagination_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Find pagination links for additional pages"""
        pagination_links = []
        
        # Look for common pagination patterns
        pagination_selectors = [
            'a[href*="page="]',
            'a[href*="p="]',
            '.pagination a',
            '.page-nav a',
            '.next-page a',
            'a[rel="next"]'
        ]
        
        for selector in pagination_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    if not href.startswith('http'):
                        href = 'https://allpoetry.com' + href
                    pagination_links.append(href)
        
        return pagination_links[:5]  # Limit to 5 pagination links
    
    def _extract_poem_links_from_page(self, soup: BeautifulSoup) -> List[str]:
        """Extract poem links from a page"""
        poem_links = []
        
        # Use the same selectors as the main method
        link_selectors = [
            'a[href*="/poem/"]',
            'a[href*="/poems/"]',
            '.poem-title a',
            '.poem-link a',
            'h3 a',
            'h4 a',
            '.poem-list a',
            '.poem-item a',
            '.poem-entry a',
            'div[class*="poem"] a',
            'li a[href*="/poem/"]'
        ]
        
        for selector in link_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href and '/poem/' in href:
                    if not href.startswith('http'):
                        href = 'https://allpoetry.com' + href
                    poem_links.append(href)
        
        return poem_links
    
    def _extract_title(self, container) -> str:
        """Extract poem title from container"""
        title_elem = container.find(['h1', 'h2', 'h3', 'h4'], class_=re.compile(r'title|poem', re.I))
        if title_elem:
            return title_elem.get_text(strip=True)
        return 'Untitled'
    
    def _extract_author(self, container) -> str:
        """Extract poem author from container"""
        author_elem = container.find(['span', 'div', 'p'], class_=re.compile(r'author|by', re.I))
        if author_elem:
            return author_elem.get_text(strip=True)
        return 'Unknown'
    
    def _looks_like_poetry(self, text: str) -> bool:
        """Check if text looks like poetry"""
        # Simple heuristics for poetry detection
        lines = text.split('\n')
        if len(lines) < 3:
            return False
        
        # Check for short lines (common in poetry)
        short_lines = sum(1 for line in lines if len(line.split()) < 10)
        if short_lines / len(lines) > 0.3:
            return True
        
        # Check for poetic patterns
        if re.search(r'\b(and|or|but)\b.*\n.*\b(and|or|but)\b', text, re.MULTILINE):
            return True
        
        return False

class PoetryDataset(Dataset):
    """Dataset for poetry training data"""
    
    def __init__(self, poems: List[Dict], max_length: int = 512):
        self.poems = poems
        self.max_length = max_length
        self.vocab = self._build_vocab()
        self.poetry_styles = self._extract_poetry_styles()
    
    def _build_vocab(self):
        """Build vocabulary from all poems"""
        vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        idx = 4
        
        for poem in self.poems:
            words = re.findall(r'\b\w+\b', poem['text'].lower())
            for word in words:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        
        return vocab
    
    def _extract_poetry_styles(self):
        """Extract different poetry styles and themes"""
        styles = {
            'sonnet': 0,
            'haiku': 1,
            'free_verse': 2,
            'rhyming': 3,
            'narrative': 4,
            'lyric': 5,
            'epic': 6,
            'elegy': 7,
            'ode': 8,
            'ballad': 9
        }
        return styles
    
    def _classify_poem_style(self, text: str) -> int:
        """Classify poem style based on content and structure"""
        text_lower = text.lower()
        lines = text.split('\n')
        
        # Check for specific forms
        if len(lines) == 14 and any('sonnet' in text_lower for _ in [1]):
            return self.poetry_styles['sonnet']
        elif len(lines) == 3 and all(len(line.split()) <= 5 for line in lines):
            return self.poetry_styles['haiku']
        elif re.search(r'\b(rhyme|rhyming)\b', text_lower):
            return self.poetry_styles['rhyming']
        elif re.search(r'\b(story|tale|narrative)\b', text_lower):
            return self.poetry_styles['narrative']
        elif re.search(r'\b(emotion|feeling|heart)\b', text_lower):
            return self.poetry_styles['lyric']
        elif re.search(r'\b(epic|heroic|grand)\b', text_lower):
            return self.poetry_styles['epic']
        elif re.search(r'\b(death|mourn|grief)\b', text_lower):
            return self.poetry_styles['elegy']
        elif re.search(r'\b(ode|praise|celebrate)\b', text_lower):
            return self.poetry_styles['ode']
        elif re.search(r'\b(ballad|song|sing)\b', text_lower):
            return self.poetry_styles['ballad']
        else:
            return self.poetry_styles['free_verse']
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to tensor using vocabulary"""
        words = re.findall(r'\b\w+\b', text.lower())
        encoded = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Add start and end tokens
        encoded = [self.vocab['<START>']] + encoded + [self.vocab['<END>']]
        
        # Pad or truncate to max_length
        if len(encoded) < self.max_length:
            encoded.extend([self.vocab['<PAD>']] * (self.max_length - len(encoded)))
        else:
            encoded = encoded[:self.max_length]
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.poems)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        poem = self.poems[idx]
        
        # Encode text
        text_encoding = self._encode_text(poem['text'])
        
        # Classify style
        style_label = self._classify_poem_style(poem['text'])
        
        # Create multi-label classification (poem could have multiple characteristics)
        labels = torch.zeros(len(self.poetry_styles))
        labels[style_label] = 1.0
        
        # Add additional characteristics
        text_lower = poem['text'].lower()
        if re.search(r'\b(love|heart|passion)\b', text_lower):
            labels[5] = 1.0  # lyric
        if re.search(r'\b(nature|tree|flower|mountain)\b', text_lower):
            labels[1] = 1.0  # haiku-like
        if re.search(r'\b(rhyme|rhyming)\b', text_lower):
            labels[3] = 1.0  # rhyming
        
        return {
            'text': poem['text'],
            'text_encoding': text_encoding,
            'labels': labels,
            'title': poem['title'],
            'author': poem['author'],
            'source_url': poem['source_url']
        }

class PoetryTrainer:
    """Trainer for poetry analysis using NeuroSym-CML"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Initialize symbolic controller
        controller_config = ControllerConfig(
            rule_set_size=config.get('rule_set_size', 200),
            max_rules_per_sample=config.get('max_rules_per_sample', 15),
            device=self.device
        )
        self.symbolic_controller = SymbolicController(controller_config)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.get('replay_buffer_size', 10000),
            device=self.device
        )
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Load checkpoint if available
        self._load_checkpoint()
        
        # Setup optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Create output directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def _create_model(self) -> nn.Module:
        """Create the model architecture"""
        model_spec = ModelSpec(
            neural_architecture=config['neural_architecture'],
            num_classes=config['num_classes'],
            hidden_sizes=config.get('hidden_sizes', [512, 256]),
            input_shape=(config['max_length'],),
            dropout_rate=config.get('dropout_rate', 0.3),
            use_batch_norm=config.get('use_batch_norm', True),
            device=self.device,
            rule_set_size=config.get('rule_set_size', 200)
        )
        
        model = HybridModel(model_spec)
        
        print("\nPoetry Model Architecture:")
        print(model)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTrainable parameters: {total_params:,}")
        
        return model
    
    def _load_checkpoint(self):
        """Load model from checkpoint if available"""
        checkpoint_path = os.path.join('checkpoints', 'best_poetry_model.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading poetry model from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
                print("Poetry model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load poetry model: {e}")
                print("Starting with fresh weights")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.symbolic_controller.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Poetry Epoch {epoch + 1}")
        for batch in pbar:
            text_encodings = batch['text_encoding'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass through symbolic controller
            rule_indices = self.symbolic_controller(text_encodings)
            
            # Forward pass through model
            outputs = self.model(text_encodings, rule_indices=rule_indices)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Store in replay buffer
            self.replay_buffer.add(
                state=text_encodings,
                action=rule_indices,
                reward=1.0 - loss.item(),
                next_state=text_encodings,
                done=torch.ones(text_encodings.size(0), dtype=torch.bool, device=self.device)
            )
            
            # Calculate metrics
            total_loss += loss.item() * text_encodings.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        return {
            'train_loss': total_loss / len(train_loader.dataset),
            'train_acc': 100. * correct / total
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'symbolic_controller_state': self.symbolic_controller.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = os.path.join('checkpoints', 'latest_poetry_checkpoint.pt')
        torch.save(state, latest_path)
        print(f"Saved poetry checkpoint to {latest_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join('checkpoints', 'best_poetry_model.pt')
            torch.save(state, best_path)
            print(f"New best poetry model saved to {best_path}")
            
        # Also save a copy with epoch number
        epoch_path = os.path.join('checkpoints', f'poetry_checkpoint_epoch_{epoch:03d}.pt')
        torch.save(state, epoch_path)
    
    def train(self, train_loader: DataLoader, num_epochs: int):
        """Main training loop"""
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            # Train for one epoch
            metrics = self.train_epoch(train_loader, epoch)
            
            # Save checkpoint
            is_best = metrics['train_acc'] > best_acc
            if is_best:
                best_acc = metrics['train_acc']
            
            self.save_checkpoint(epoch, metrics, is_best)
            
            print(f"Poetry Epoch {epoch + 1}/{num_epochs} - "
                  f"Loss: {metrics['train_loss']:.4f} - "
                  f"Accuracy: {metrics['train_acc']:.2f}%")
            
            # Update learning rate
            self.scheduler.step(metrics['train_loss'])

def main():
    """Main function for poetry training"""
    print("📝 Poetry Training with NeuroSym-CML")
    print("=" * 50)
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Configuration
    config = {
        'device': device,
        'num_classes': 10,  # Poetry styles
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_epochs': 25,
        'dropout_rate': 0.3,
        'use_batch_norm': True,
        'neural_architecture': 'custom_transformer',
        'hidden_sizes': [512, 256],
        'rule_set_size': 200,
        'max_rules_per_sample': 15,
        'replay_buffer_size': 10000
    }
    
    print("\nPoetry Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize scraper
    scraper = PoetryWebScraper()
    
    # Scrape poetry from various sources
    print("\nScraping poetry from web sources...")
    poetry_sources = [
        'https://allpoetry.com/Words_Of_Anthrax',  # Your specific request
        'https://www.poetryfoundation.org/poems',
        'https://www.poets.org/poetsorg/poems',
        'https://www.poemhunter.com/poems/',
        'https://www.poetry.com/poems',
        'https://www.allpoetry.com/poems'
    ]
    
    all_poems = []
    for source in poetry_sources:
        poems = scraper.scrape_poetry_website(source, max_poems=20)
        all_poems.extend(poems)
        time.sleep(2)  # Be respectful to servers
    
    print(f"Successfully scraped {len(all_poems)} poems")
    
    # Add some sample poems if scraping didn't work well
    if len(all_poems) < 10:
        print("Adding sample poems...")
        sample_poems = [
            {
                'text': 'Roses are red, violets are blue, sugar is sweet, and so are you.',
                'title': 'Classic Rhyme',
                'author': 'Traditional',
                'source_url': 'sample'
            },
            {
                'text': 'The road not taken, two paths diverged in a yellow wood, and I took the one less traveled by.',
                'title': 'The Road Not Taken',
                'author': 'Robert Frost',
                'source_url': 'sample'
            },
            {
                'text': 'I wandered lonely as a cloud, that floats on high o\'er vales and hills, when all at once I saw a crowd, a host of golden daffodils.',
                'title': 'Daffodils',
                'author': 'William Wordsworth',
                'source_url': 'sample'
            }
        ]
        all_poems.extend(sample_poems)
    
    # Create dataset
    print(f"\nCreating poetry dataset with {len(all_poems)} poems...")
    dataset = PoetryDataset(all_poems, max_length=config['max_length'])
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = PoetryTrainer(config)
    
    # Start training
    print("\nStarting poetry training...")
    trainer.train(dataloader, config['num_epochs'])
    
    print("\nPoetry training completed!")
    print(f"Best model saved to: {os.path.join('checkpoints', 'best_poetry_model.pt')}")

if __name__ == "__main__":
    main()
