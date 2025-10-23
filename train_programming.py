#!/usr/bin/env python3
"""
Programming Data Training for NeuroSym-CML
Trains NeuroSym-CML on programming-related PDF documents
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
import PyPDF2

# Import existing components
from meta_model import HybridModel, ModelSpec

class ProgrammingDataset(Dataset):
    """Dataset for loading and processing programming-related PDF documents"""
    
    def __init__(self, data_dir: str, max_length: int = 512):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.documents = []
        self._load_documents()
    
    def _load_documents(self):
        """Load and process PDF documents from the data directory"""
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
            try:
                with open(pdf_file, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    
                    # Basic text cleaning with special handling for code
                    text = re.sub(r'\s+', ' ', text).strip()
                    if text:
                        self.documents.append({
                            'filename': pdf_file.name,
                            'text': text,
                            'length': len(text)
                        })
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {str(e)}")
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        # Tokenization with special handling for code elements
        text = self.documents[idx]['text']
        
        # Split into words but preserve code-like tokens
        tokens = []
        for token in re.findall(r'\w+|[^\w\s]', text):
            if len(tokens) >= self.max_length:
                break
            tokens.append(token)
        
        # Simple vocabulary mapping (in practice, use a proper tokenizer)
        vocab = {word: i+1 for i, word in enumerate(set(tokens))}  # 0 for padding
        token_ids = [vocab.get(token, 0) for token in tokens]
        
        # Pad or truncate to max_length
        if len(token_ids) < self.max_length:
            token_ids += [0] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
            
        return torch.tensor(token_ids, dtype=torch.long)

class ProgrammingTrainer:
    """Handles training of the NeuroSym-CML model on programming data"""
    
    def __init__(self, model_config: Dict[str, Any], data_dir: str, output_dir: str):
        self.model_config = model_config
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model with larger vocabulary for programming data
        self.model_spec = ModelSpec(**model_config)
        self.model = HybridModel(self.model_spec)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Setup optimizer with gradient clipping for stability
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    def train(self, num_epochs: int = 10, batch_size: int = 4, grad_clip: float = 1.0):
        """Train the model on programming data"""
        # Load dataset
        dataset = ProgrammingDataset(self.data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Starting training on {len(dataset)} programming documents")
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Reshape for loss calculation
                outputs = outputs.view(-1, outputs.size(-1))
                targets = batch.view(-1)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize with gradient clipping
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            self._save_checkpoint(epoch, avg_loss)
    
    def _save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.model_config
        }
        
        checkpoint_path = self.output_dir / f"programming_checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

def main():
    # Configuration for the model (slightly larger for programming data)
    model_config = {
        'vocab_size': 50000,  # Larger vocabulary for programming terms
        'hidden_size': 768,
        'num_hidden_layers': 8,  # Slightly deeper model
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'initializer_range': 0.02
    }
    
    # Initialize and run trainer
    trainer = ProgrammingTrainer(
        model_config=model_config,
        data_dir=r"C:\Users\lyler\OneDrive\Gambit\dataset\programming_data",
        output_dir="./programming_checkpoints"
    )
    
    # Start training with gradient clipping
    trainer.train(num_epochs=10, batch_size=2, grad_clip=1.0)

if __name__ == "__main__":
    main()
