#!/usr/bin/env python3
"""
Improved Chess Training for NeuroSym-CML
Optimized for both 4GB GPU constraints AND high performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import numpy as np

# Import modular components
from meta_model import HybridModel, ModelSpec
from modular_symbolic_controller import create_symbolic_controller
from modular_replay_buffer import create_replay_buffer
import PyPDF2
import re
from tensor_adapter import create_symbolic_adapter, ReplayBufferAdapter, ModelOutputAdapter

class FullChessDataset:
    """Extract ALL the rich data from your chess books - every page, every chapter"""
    
    def __init__(self, chars_per_sample=2048, overlap=512, max_length=256):
        self.chars_per_sample = chars_per_sample
        self.overlap = overlap
        self.max_length = max_length
        self.samples = []
        self._load_all_content()
    
    def _load_all_content(self):
        """Load ALL content from every book"""
        chess_data_dir = Path("dataset/Chess_data")
        
        print("🚀 Loading ALL content from your rich chess books...")
        total_pages_processed = 0
        
        for pdf_file in tqdm(list(chess_data_dir.glob("*.pdf")), desc="Processing books"):
            book_samples, pages_processed = self._extract_all_book_content(pdf_file)
            self.samples.extend(book_samples)
            total_pages_processed += pages_processed
        
        print(f"\n📊 FULL EXTRACTION COMPLETE:")
        print(f"   Total samples: {len(self.samples)}")
        print(f"   Total pages processed: {total_pages_processed}")
        print(f"   Improvement over old method: {len(self.samples) / 20:.1f}x more data")
    
    def _extract_all_book_content(self, pdf_path: Path):
        """Extract content from EVERY page of a book"""
        samples = []
        pages_processed = 0
        
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
                                page_samples = self._split_text_into_samples(clean_text, pdf_path.name, page_num)
                                samples.extend(page_samples)
                                pages_processed += 1
                                    
                    except Exception:
                        continue
                        
        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")
        
        return samples, pages_processed
    
    def _split_text_into_samples(self, text: str, source: str, page_num: int):
        """Split long text into overlapping samples"""
        samples = []
        
        if len(text) <= self.chars_per_sample:
            labels = self._create_labels(text)
            text_encoding = self._encode_text(text)
            samples.append({
                'text': text,
                'text_encoding': text_encoding,
                'labels': labels
            })
        else:
            start = 0
            while start < len(text):
                end = start + self.chars_per_sample
                sample_text = text[start:end]
                
                if end < len(text):
                    last_period = sample_text.rfind('.')
                    if last_period > len(sample_text) * 0.8:
                        sample_text = sample_text[:last_period + 1]
                
                labels = self._create_labels(sample_text)
                text_encoding = self._encode_text(sample_text)
                samples.append({
                    'text': sample_text,
                    'text_encoding': text_encoding,
                    'labels': labels
                })
                
                start += self.chars_per_sample - self.overlap
                if start >= len(text):
                    break
        
        return samples
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\+\=]', '', text)
        return text.strip()
    
    def _encode_text(self, text: str):
        """Simple character-level encoding"""
        chars = list(text.lower())
        char_to_idx = {chr(i): i for i in range(32, 127)}
        char_to_idx[' '] = 0
        
        encoded = []
        for char in chars[:self.max_length]:
            encoded.append(char_to_idx.get(char, 1))
        
        while len(encoded) < self.max_length:
            encoded.append(0)
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def _create_labels(self, text: str):
        """Enhanced labeling with comprehensive chess terminology (draw class removed)"""
        labels = torch.zeros(9)
        text_lower = text.lower()
        
        # Comprehensive chess labeling system (no 'draw' class)
        tactics_terms = ['tactics', 'tactical', 'pin', 'fork', 'skewer', 'combination', 'sacrifice']
        strategy_terms = ['strategy', 'strategic', 'plan', 'positional', 'initiative', 'outpost']
        opening_terms = ['opening', 'development', 'castle', 'gambit', 'sicilian', 'french']
        endgame_terms = ['endgame', 'ending', 'opposition', 'promotion', 'passed pawn']
        piece_terms = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king', 'piece']
        notation_terms = ['e4', 'd4', 'nf3', 'file', 'rank', 'square', 'kingside', 'queenside']
        middlegame_terms = ['middlegame', 'middle game', 'attack', 'defense', 'calculation']
        evaluation_terms = ['good move', 'mistake', 'blunder', 'advantage', 'analysis']
        checkmate_terms = ['checkmate', 'mate', 'mating', 'check', 'forced mate']
        
        if any(term in text_lower for term in tactics_terms): labels[0] = 1
        if any(term in text_lower for term in strategy_terms): labels[1] = 1
        if any(term in text_lower for term in opening_terms): labels[2] = 1
        if any(term in text_lower for term in endgame_terms): labels[3] = 1
        if any(term in text_lower for term in piece_terms): labels[4] = 1
        if any(term in text_lower for term in notation_terms): labels[5] = 1
        if any(term in text_lower for term in middlegame_terms): labels[6] = 1
        if any(term in text_lower for term in evaluation_terms): labels[7] = 1
        if any(term in text_lower for term in checkmate_terms): labels[8] = 1
        
        return labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class ImprovedChessTrainer:
    """High-performance chess trainer optimized for 4GB GPU with better learning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create improved model (lightweight enough for 4GB)
        self.model = self._create_improved_model()
        
        # Use custom tensor adapter for symbolic controller
        self.symbolic_adapter = create_symbolic_adapter(
            text_size=config.get('text_encoding_size', 256),  # match dataset encoding length
            controller_size=256
        ).to(self.device)
        
        # Use ProductionRuleController with adapter (kept small)
        self.symbolic_controller = create_symbolic_controller(
            controller_type='production_rule',
            num_rules=config.get('rule_set_size', 75),
            input_size=256,  # Fixed size after adapter
            hidden_size=64
        ).to(self.device)
        
        # Use text replay buffer properly for experience replay (modest size)
        self.replay_buffer = create_replay_buffer(
            buffer_type='text',
            memory_size=config.get('replay_buffer_size', 6000),
            device=str(self.device)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 3e-4),
            betas=(0.9, 0.999),
            weight_decay=config.get('weight_decay', 1e-4),
            amsgrad=True
        )
        
        # Scheduler with correct steps_per_epoch for stability/memory predictability
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.get('learning_rate', 3e-4),
            epochs=config['num_epochs'],
            steps_per_epoch=max(1, config.get('steps_per_epoch', 100)),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if torch.cuda.is_available() and config.get('mixed_precision', True) else None
        
        # Class weights for BCE (improves F1 on imbalanced data)
        self.pos_weights = None
        if config.get('pos_weights') is not None:
            pw = torch.tensor(config['pos_weights'], dtype=torch.float32)
            self.pos_weights = pw.to(self.device)
        
        # Improved loss with focal + pos_weight support
        self.criterion = self._create_improved_loss(self.pos_weights)
        
        # Thresholds for multi-label decisions (calibrated on eval)
        self.thresholds = torch.full((config.get('num_classes', 10),), 0.5, dtype=torch.float32, device=self.device)
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        
        # Performance tracking
        self.best_accuracy = 0.0
        self.best_f1_score = 0.0
        self.training_history = []
        
        print(f"♟️ Improved Chess Trainer (High Performance + 4GB Optimized)")
        print(f"   Device: {self.device}")
        print(f"   Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Batch Size: {config['batch_size']} (Effective: {config['batch_size'] * self.gradient_accumulation_steps})")
        print(f"   Learning Rate: {config.get('learning_rate', 3e-4)}")
        print(f"   F1 Mode: focal loss + pos_weight, per-class thresholds")
        print(f"   Advanced Features: Enabled")
    
    def _create_improved_model(self) -> nn.Module:
        """Create improved model with better architecture"""
        class ImprovedChessModel(nn.Module):
            def __init__(self, vocab_size=30000, embed_dim=384, hidden_dim=256, num_classes=10):
                super().__init__()
                
                # Better embedding with positional encoding
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim) * 0.02)
                
                # Multi-layer LSTM with residual connections
                self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.2)
                self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=0.2)
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
                
                # Advanced classifier with layer norm (better for small batches)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim // 2, num_classes)
                )
                
                self.feature_dim = hidden_dim
                
                # Initialize weights properly
                self._init_weights()
            
            def _init_weights(self):
                """Proper weight initialization"""
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.Embedding):
                        nn.init.normal_(module.weight, 0, 0.02)
                    elif isinstance(module, nn.LSTM):
                        for name, param in module.named_parameters():
                            if 'weight' in name:
                                nn.init.orthogonal_(param)
                            elif 'bias' in name:
                                nn.init.zeros_(param)
            
            def forward(self, dummy_images, text_embeddings, rule_indices):
                batch_size, seq_len = text_embeddings.shape[:2]
                
                # Handle input
                if text_embeddings.dtype in [torch.long, torch.int]:
                    text_embeddings = torch.clamp(text_embeddings.long(), 0, 29999)
                    x = self.embedding(text_embeddings)
                    
                    # Add positional encoding
                    if seq_len <= self.pos_encoding.size(0):
                        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
                else:
                    x = text_embeddings
                
                # Multi-layer LSTM with residual connections
                lstm1_out, _ = self.lstm1(x)
                lstm2_out, _ = self.lstm2(lstm1_out)
                
                # Residual connection
                if lstm1_out.size(-1) == lstm2_out.size(-1):
                    lstm_out = lstm1_out + lstm2_out
                else:
                    lstm_out = lstm2_out
                
                # Self-attention
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                
                # Global average pooling
                pooled = torch.mean(attn_out, dim=1)
                
                # Classification
                output = self.classifier(pooled)
                
                return output
        
        return ImprovedChessModel(
            vocab_size=30000,
            embed_dim=384,
            hidden_dim=256,
            num_classes=self.config.get('num_classes', 10)
        ).to(self.device)
    
    def _create_improved_loss(self, pos_weights: Optional[torch.Tensor]) -> nn.Module:
        """Create improved loss function for high accuracy under class imbalance"""
        class ImprovedChessLoss(nn.Module):
            def __init__(self, pos_weights: Optional[torch.Tensor] = None):
                super().__init__()
                self.alpha = 0.25  # focal alpha
                self.gamma = 2.0   # focal gamma
                self.pos_weights = pos_weights
            
            def focal_loss(self, inputs, targets):
                # BCE with logits supports class-wise pos_weight for imbalance
                bce_loss = F.binary_cross_entropy_with_logits(
                    inputs, targets.float(), pos_weight=self.pos_weights, reduction='none'
                )
                pt = torch.exp(-bce_loss)
                focal = self.alpha * (1 - pt) ** self.gamma * bce_loss
                return focal.mean()
            
            def forward(self, outputs, targets):
                return self.focal_loss(outputs, targets)
        
        return ImprovedChessLoss(pos_weights)
    
    def _calculate_advanced_metrics(self, outputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Calculate advanced metrics for high accuracy"""
        probs = torch.sigmoid(outputs)
        # Use per-class thresholds if available
        thr = self.thresholds.to(probs.device)
        while thr.dim() < probs.dim():
            thr = thr.unsqueeze(0)
        predictions = probs > thr
        labels_bool = labels.bool()
        
        # Calculate per-class metrics
        tp = (predictions & labels_bool).float().sum(dim=0)
        fp = (predictions & ~labels_bool).float().sum(dim=0)
        fn = (~predictions & labels_bool).float().sum(dim=0)
        
        # Precision, Recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Overall metrics
        accuracy = (predictions == labels_bool).float().mean().item()
        avg_f1 = f1.mean().item()
        avg_precision = precision.mean().item()
        avg_recall = recall.mean().item()
        
        return {
            'accuracy': accuracy,
            'f1_score': avg_f1,
            'precision': avg_precision,
            'recall': avg_recall
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor], accumulation_step: int) -> Dict[str, float]:
        """Improved training step with better learning"""
        self.model.train()
        
        # Move to device
        text_encodings = batch['text_encoding'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Handle labels properly
        num_classes = self.config.get('num_classes', 9)
        if labels.dim() == 1 and labels.size(0) == num_classes:
            labels = labels.unsqueeze(0)
        elif labels.dim() > 2:
            labels = labels.squeeze()
            if labels.dim() == 1 and labels.size(0) == num_classes:
                labels = labels.unsqueeze(0)
        
        # Only zero gradients at start of accumulation
        if accumulation_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()
        
        # Use mixed precision
        if self.scaler:
            with autocast('cuda'):
                # Use custom tensor adapter for symbolic controller
                adapted_input = self.symbolic_adapter.adapt_text_for_controller(text_encodings.float())
                rule_indices, symbolic_state = self.symbolic_controller(adapted_input)
                rule_indices = self.symbolic_adapter.adapt_rule_indices(rule_indices, text_encodings.size(0))
                
                # Create minimal dummy images
                dummy_images = torch.zeros(text_encodings.size(0), 3, 32, 32).to(self.device)
                
                # Forward pass
                outputs = self.model(dummy_images, text_encodings, rule_indices)
                
                # Calculate improved loss with adapter
                adapted_outputs, adapted_labels = ModelOutputAdapter.adapt_for_loss(outputs, labels)
                loss = self.criterion(adapted_outputs, adapted_labels) / self.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Step optimizer after accumulation
            if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()  # Step scheduler
                
                # Clear cache
                torch.cuda.empty_cache()
        else:
            # CPU training
            rule_indices = torch.randint(0, 75, (text_encodings.size(0),))
            dummy_images = torch.zeros(text_encodings.size(0), 3, 32, 32)
            
            outputs = self.model(dummy_images, text_encodings, rule_indices)
            adapted_outputs, adapted_labels = ModelOutputAdapter.adapt_for_loss(outputs, labels)
            loss = self.criterion(adapted_outputs, adapted_labels) / self.gradient_accumulation_steps
            
            loss.backward()
            
            if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
        
        # Calculate advanced metrics
        with torch.no_grad():
            metrics = self._calculate_advanced_metrics(outputs, labels)
            
            # Store experience in replay buffer for experience replay
            self.replay_buffer.add(
                text_encoding=text_encodings[0].cpu(),
                labels=labels[0].cpu(),
                loss=loss.item() * self.gradient_accumulation_steps,
                accuracy=metrics['accuracy'],
                metadata={'f1_score': metrics['f1_score']}
            )
        
        return {
            'loss': loss.item() * self.gradient_accumulation_steps,
            **metrics,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Improved evaluation step with advanced metrics"""
        self.model.eval()
        
        with torch.no_grad():
            text_encodings = batch['text_encoding'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Handle labels
            if labels.dim() == 1 and labels.size(0) == self.config.get('num_classes', 10):
                labels = labels.unsqueeze(0)
            elif labels.dim() > 2:
                labels = labels.squeeze()
                if labels.dim() == 1 and labels.size(0) == self.config.get('num_classes', 10):
                    labels = labels.unsqueeze(0)
            
            # Forward pass with custom tensor adapter
            adapted_input = self.symbolic_adapter.adapt_text_for_controller(text_encodings.float())
            rule_indices, symbolic_state = self.symbolic_controller(adapted_input)
            rule_indices = self.symbolic_adapter.adapt_rule_indices(rule_indices, text_encodings.size(0))
            dummy_images = torch.zeros(text_encodings.size(0), 3, 32, 32).to(self.device)
            
            outputs = self.model(dummy_images, text_encodings, rule_indices)
            loss = self.criterion(outputs, labels)
            
            # Calculate advanced metrics
            metrics = self._calculate_advanced_metrics(outputs, labels)
            
            # Add symbolic reasoning metrics
            symbolic_metrics = self._analyze_symbolic_reasoning(symbolic_state, labels)
            metrics.update(symbolic_metrics)
        
        return {
            'loss': loss.item(),
            **metrics
        }
    
    def _analyze_symbolic_reasoning(self, symbolic_state: Dict, labels: torch.Tensor) -> Dict[str, float]:
        """Analyze the quality of symbolic reasoning"""
        metrics = {}
        
        if 'rule_probabilities' in symbolic_state:
            rule_probs = symbolic_state['rule_probabilities']
            
            # Rule diversity - how many different rules are being used
            rule_entropy = -torch.sum(rule_probs * torch.log(rule_probs + 1e-8), dim=-1).mean()
            metrics['rule_diversity'] = rule_entropy.item()
            
            # Rule confidence - how confident the controller is
            max_rule_prob = torch.max(rule_probs, dim=-1)[0].mean()
            metrics['rule_confidence'] = max_rule_prob.item()
        
        if 'condition_activations' in symbolic_state:
            conditions = symbolic_state['condition_activations']
            
            # Condition activation rate
            activation_rate = (conditions > 0.5).float().mean()
            metrics['condition_activation_rate'] = activation_rate.item()
        
        return metrics
    
    def _calibrate_thresholds(self, eval_loader: DataLoader, max_batches: int = 2) -> None:
        """Calibrate per-class thresholds to maximize macro-F1 on a small eval subset."""
        self.model.eval()
        sigmoids, all_labels = [], []
        batches = 0
        with torch.no_grad():
            for batch in eval_loader:
                text_encodings = batch['text_encoding'].to(self.device)
                labels = batch['labels'].to(self.device)
                if labels.dim() == 1 and labels.size(0) == self.config.get('num_classes', 10):
                    labels = labels.unsqueeze(0)
                elif labels.dim() > 2:
                    labels = labels.squeeze()
                    if labels.dim() == 1 and labels.size(0) == self.config.get('num_classes', 10):
                        labels = labels.unsqueeze(0)
                adapted_input = self.symbolic_adapter.adapt_text_for_controller(text_encodings.float())
                rule_indices, _ = self.symbolic_controller(adapted_input)
                rule_indices = self.symbolic_adapter.adapt_rule_indices(rule_indices, text_encodings.size(0))
                dummy_images = torch.zeros(text_encodings.size(0), 3, 32, 32).to(self.device)
                outputs = self.model(dummy_images, text_encodings, rule_indices)
                sigmoids.append(torch.sigmoid(outputs).detach().cpu())
                all_labels.append(labels.detach().cpu())
                batches += 1
                if batches >= max_batches:
                    break
        if not sigmoids:
            return
        probs = torch.cat(sigmoids, dim=0)
        labels = torch.cat(all_labels, dim=0).bool()
        num_classes = probs.size(1)
        best_thr = torch.full((num_classes,), 0.5)
        thr_grid = torch.linspace(0.3, 0.7, steps=9)
        for c in range(num_classes):
            best_f1 = -1.0
            p = probs[:, c]
            y = labels[:, c]
            for t in thr_grid:
                preds = p > t
                tp = (preds & y).sum().item()
                fp = (preds & ~y).sum().item()
                fn = ((~preds) & y).sum().item()
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr[c] = t
        self.thresholds = best_thr.to(self.device)
    
    def train(self, dataloader: DataLoader, num_epochs: int, eval_loader: Optional[DataLoader] = None):
        """Improved training loop with better monitoring"""
        print(f"🚀 Starting improved chess training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = {'loss': 0.0, 'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Chess Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(pbar):
                # Regular training step
                metrics = self.train_step(batch, batch_idx)
                
                # Experience replay - learn from stored experiences
                if len(self.replay_buffer) > 32 and batch_idx % 2 == 0:  # Every other batch
                    replay_batch = self.replay_buffer.sample(2)  # Small replay batch
                    if replay_batch and 'text_encodings' in replay_batch:
                        # Create replay batch in same format
                        replay_data = {
                            'text_encoding': replay_batch['text_encodings'],
                            'labels': replay_batch['labels']
                        }
                        # Train on replay data
                        _ = self.train_step(replay_data, batch_idx + 1000)  # Offset for accumulation
                
                for key in train_metrics:
                    if key in metrics:
                        train_metrics[key] += metrics[key]
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f"{metrics['loss']:.4f}",
                    'Acc': f"{metrics['accuracy']:.4f}",
                    'F1': f"{metrics['f1_score']:.4f}",
                    'LR': f"{metrics['lr']:.2e}"
                })
                
                # Clear cache periodically
                if batch_idx % 3 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Average metrics
            for key in train_metrics:
                train_metrics[key] /= max(1, num_batches)
            
            # Evaluation phase
            eval_metrics = {'loss': 0.0, 'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}
            eval_batches = 0
            
            eval_iter = eval_loader if eval_loader is not None else dataloader
            for batch in eval_iter:
                metrics = self.evaluate_step(batch)
                for key in eval_metrics:
                    if key in metrics:
                        eval_metrics[key] += metrics[key]
                eval_batches += 1
            
            for key in eval_metrics:
                eval_metrics[key] /= max(1, eval_batches)
            
            # Optional: calibrate thresholds for better F1
            if self.config.get('calibrate_thresholds', True):
                self._calibrate_thresholds(eval_iter if eval_loader is not None else dataloader, max_batches=2)
                # Recompute eval metrics with calibrated thresholds (quick pass)
                eval_metrics = {'loss': 0.0, 'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}
                eval_batches = 0
                for batch in eval_iter:
                    metrics = self.evaluate_step(batch)
                    for key in eval_metrics:
                        if key in metrics:
                            eval_metrics[key] += metrics[key]
                    eval_batches += 1
                for key in eval_metrics:
                    eval_metrics[key] /= max(1, eval_batches)
            
            # Save results
            epoch_results = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'eval': eval_metrics
            }
            self.training_history.append(epoch_results)
            
            # Print detailed results
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_score']:.4f}")
            print(f"  Eval  - Loss: {eval_metrics['loss']:.4f}, Acc: {eval_metrics['accuracy']:.4f}, F1: {eval_metrics['f1_score']:.4f}")
            
            # Save best model based on F1 score
            if eval_metrics['f1_score'] > self.best_f1_score:
                self.best_f1_score = eval_metrics['f1_score']
                self.best_accuracy = eval_metrics['accuracy']
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'f1_score': self.best_f1_score,
                    'accuracy': self.best_accuracy,
                    'thresholds': self.thresholds.detach().cpu().tolist(),
                    'config': self.config,
                    'training_history': self.training_history
                }, 'best_chess_model_improved.pt')
                
                print(f"  🏆 New best F1 score: {self.best_f1_score:.4f} (Acc: {self.best_accuracy:.4f})")
            
            # Clear cache at end of epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final comprehensive evaluation
        print(f"\n🔍 Running comprehensive final evaluation...")
        try:
            self._comprehensive_evaluation(eval_loader if eval_loader is not None else dataloader)
        except Exception as e:
            print(f"Comprehensive evaluation failed: {e}")
        
        # Analyze class balance
        self._analyze_class_balance(eval_loader if eval_loader is not None else dataloader)
        
        print(f"\n🎉 Improved chess training completed!")
        print(f"Best F1 Score: {self.best_f1_score:.4f}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")
    
    def _analyze_class_balance(self, dataloader: DataLoader):
        """Analyze class distribution to understand F1 score"""
        print(f"\n📊 Class Balance Analysis:")
        print(f"=" * 40)
        
        num_classes = self.config.get('num_classes', 9)
        class_counts = torch.zeros(num_classes)
        total_samples = 0
        
        for batch in dataloader:
            labels = batch['labels'].to(self.device)
            if labels.dim() == 1 and labels.size(0) == num_classes:
                labels = labels.unsqueeze(0)
            elif labels.dim() > 2:
                labels = labels.squeeze()
                if labels.dim() == 1 and labels.size(0) == num_classes:
                    labels = labels.unsqueeze(0)
            
            class_counts += labels.sum(dim=0).cpu()
            total_samples += labels.size(0)
        
        class_names = ['tactics', 'strategy', 'opening', 'endgame', 'pieces', 
                      'notation', 'middlegame', 'evaluation', 'checkmate']
        
        print("Class Distribution:")
        for i, (name, count) in enumerate(zip(class_names, class_counts)):
            percentage = (count / max(1, total_samples)) * 100
            print(f"  {name:12}: {count:3.0f}/{total_samples} ({percentage:5.1f}%)")
        
        avg_positive_rate = (class_counts.sum() / (max(1, total_samples) * num_classes)) * 100
        print(f"\nOverall positive rate: {avg_positive_rate:.1f}%")
        print(f"This explains why F1 is low - very imbalanced classes!")
        print(f"=" * 40)
    
    def _comprehensive_evaluation(self, dataloader: DataLoader):
        """Run comprehensive evaluation with detailed analysis"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_symbolic_states = []
        
        with torch.no_grad():
            for batch in dataloader:
                text_encodings = batch['text_encoding'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Handle labels
                num_classes = self.config.get('num_classes', 9)
                if labels.dim() == 1 and labels.size(0) == num_classes:
                    labels = labels.unsqueeze(0)
                elif labels.dim() > 2:
                    labels = labels.squeeze()
                    if labels.dim() == 1 and labels.size(0) == num_classes:
                        labels = labels.unsqueeze(0)
                
                # Forward pass with proper shape handling
                # text_encodings should be token indices [batch_size, seq_len] for the model
                # But we need embeddings for the symbolic controller
                
                # For symbolic controller, create pooled embeddings
                if text_encodings.dtype in [torch.long, torch.int]:
                    # If we have token indices, convert to embeddings for pooling
                    with torch.no_grad():
                        temp_embeddings = self.model.embedding(torch.clamp(text_encodings.long(), 0, 29999))
                        pooled_text = temp_embeddings.float().mean(dim=1)  # [batch_size, hidden_size]
                else:
                    # If we already have embeddings
                    pooled_text = text_encodings.float().mean(dim=1)  # [batch_size, hidden_size]
                
                # Ensure pooled_text has correct shape for symbolic controller
                if pooled_text.dim() == 1:
                    pooled_text = pooled_text.unsqueeze(0)
                
                # Ensure pooled_text has the expected feature size (256)
                if pooled_text.size(-1) != 256:
                    # Pad or truncate to match expected input size
                    current_size = pooled_text.size(-1)
                    if current_size < 256:
                        # Pad with zeros
                        padding = torch.zeros(pooled_text.size(0), 256 - current_size).to(pooled_text.device)
                        pooled_text = torch.cat([pooled_text, padding], dim=-1)
                    else:
                        # Truncate
                        pooled_text = pooled_text[:, :256]
                
                rule_indices, symbolic_state = self.symbolic_controller(pooled_text)
                dummy_images = torch.zeros(text_encodings.size(0), 3, 32, 32).to(self.device)
                
                # Ensure rule_indices has correct shape
                if rule_indices.dim() == 0:
                    rule_indices = rule_indices.unsqueeze(0)
                elif rule_indices.dim() == 1 and rule_indices.size(0) != text_encodings.size(0):
                    rule_indices = rule_indices.expand(text_encodings.size(0))
                
                outputs = self.model(dummy_images, text_encodings, rule_indices)
                predictions = torch.sigmoid(outputs) > 0.5
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_symbolic_states.append(symbolic_state)
        
        # Combine all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Calculate comprehensive metrics
        print(f"\n📊 Comprehensive Chess AI Analysis:")
        print(f"=" * 50)
        
        # Per-class analysis (draw removed)
        class_names = ['tactics', 'strategy', 'opening', 'endgame', 'pieces', 
                      'notation', 'middlegame', 'evaluation', 'checkmate']
        
        for i, class_name in enumerate(class_names):
            if i < all_labels.size(1):
                class_labels = all_labels[:, i].bool()
                class_preds = all_predictions[:, i].bool()
                
                tp = (class_preds & class_labels).sum().item()
                fp = (class_preds & ~class_labels).sum().item()
                fn = (~class_preds & class_labels).sum().item()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                print(f"{class_name:12}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        # Overall symbolic reasoning analysis
        if all_symbolic_states:
            avg_rule_diversity = np.mean([s.get('rule_diversity', 0) for s in all_symbolic_states if isinstance(s, dict)])
            print(f"\n🧠 Symbolic Reasoning Analysis:")
            print(f"Average Rule Diversity: {avg_rule_diversity:.3f}")
        
        print(f"=" * 50)

def _compute_pos_weights(ds) -> List[float]:
    # Compute class-wise pos_weight = neg/pos
    if len(ds) == 0:
        return []
    num_classes = ds[0]['labels'].numel()
    counts = torch.zeros(num_classes)
    for i in range(len(ds)):
        labels = ds[i]['labels']
        counts += labels
    pos = counts
    neg = len(ds) - counts
    pos_weights = []
    for i in range(num_classes):
        if pos[i] > 0:
            pos_weights.append((neg[i] / pos[i]).item())
        else:
            pos_weights.append(1.0)
    return pos_weights


def _make_weighted_sampler(ds) -> WeightedRandomSampler:
    # Weight each sample by the rarity of its positive classes (min rarity among its labels)
    if len(ds) == 0:
        return WeightedRandomSampler(weights=[1.0], num_samples=1, replacement=True)
    num_classes = ds[0]['labels'].numel()
    counts = torch.zeros(num_classes)
    for i in range(len(ds)):
        counts += ds[i]['labels']
    total = len(ds)
    sample_weights = []
    for i in range(len(ds)):
        labels = ds[i]['labels']
        if labels.sum() > 0:
            active = (labels > 0).nonzero().flatten()
            rarest = counts[active].min().item()
            w = total / (rarest + 1)
        else:
            w = 1.0
        sample_weights.append(w)
    return WeightedRandomSampler(weights=sample_weights, num_samples=max(total, 1) * 2, replacement=True)


def main():
    """Main function with improved configuration"""
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
'num_classes': 9,
        'max_length': 256,
        'batch_size': 2,  # Small for 4GB GPU
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'num_epochs': 25,
        'rule_set_size': 75,
        'replay_buffer_size': 6000,
        'gradient_accumulation_steps': 4,
        'mixed_precision': True,
        'calibrate_thresholds': True,
        'text_encoding_size': 256
    }
    
    print("♟️ Improved High-Performance Chess Training (4GB GPU)")
    print("=" * 60)
    print("🎯 Target: High accuracy with advanced techniques")
    print("🔧 Advanced optimizations: Focal loss, pos_weight, attention, F1 tracking")
    print("=" * 60)
    
    # Create FULL dataset using ALL your rich chess content
    dataset = FullChessDataset(chars_per_sample=2048, overlap=512, max_length=config['max_length'])
    if len(dataset) == 0:
        print("❌ No chess data found!")
        return
    
    # Split into train/val (80/20)
    indices = torch.randperm(len(dataset)).tolist()
    val_size = max(1, int(0.2 * len(indices)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    
    # Compute class weights on train set (Subset -> use underlying dataset)
    # Build a lightweight view to iterate labels
    class _View:
        def __init__(self, subset):
            self.subset = subset
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, i):
            return self.subset.dataset[self.subset.indices[i]]
    train_view = _View(train_ds)

    pos_weights = _compute_pos_weights(train_view)
    config['pos_weights'] = pos_weights
    
    # Optional weighted sampler for train
    sampler = _make_weighted_sampler(train_view)

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=0,
        pin_memory=(config['device'] == 'cuda')
    )
    eval_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=(config['device'] == 'cuda')
    )

    # steps_per_epoch for scheduler
    config['steps_per_epoch'] = max(1, len(train_loader))
    
    # Create improved trainer
    trainer = ImprovedChessTrainer(config)
    
    # Train the model
    trainer.train(train_loader, config['num_epochs'], eval_loader=eval_loader)
    
    # Save results
    results = {
        'best_f1_score': trainer.best_f1_score,
        'best_accuracy': trainer.best_accuracy,
        'config': config,
        'training_history': trainer.training_history
    }
    
    with open('chess_improved_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🏆 Improved chess training completed!")
    print(f"📊 Results saved to chess_improved_results.json")

if __name__ == "__main__":
    main()
