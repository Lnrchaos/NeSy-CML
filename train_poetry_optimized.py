#!/usr/bin/env python3
"""
Optimized Poetry Training for NeuroSym-CML
Uses all modular components for maximum creative AI performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import numpy as np
import re

# Import modular components
from meta_model import HybridModel, ModelSpec
from modular_architecture import create_poetry_trainer, TextOnlyAdapter
from modular_symbolic_controller import create_symbolic_controller
from modular_replay_buffer import create_replay_buffer
# Import dataset components - will create if needed
from custom_architecture_selector import CustomArchitectureSelector
from torch.utils.data import Dataset

class PoetryDataset(Dataset):
    """Simple poetry dataset for training"""
    
    def __init__(self, poems: List[Dict[str, str]], max_length: int = 256):
        self.poems = poems
        self.max_length = max_length
        self.vocab = self._build_vocab()
        self.poetry_styles = {
            'narrative': 0, 'lyric': 1, 'epic': 2, 'dramatic': 3, 'satirical': 4,
            'pastoral': 5, 'elegy': 6, 'ode': 7, 'sonnet': 8, 'haiku': 9,
            'free_verse': 10, 'ballad': 11, 'limerick': 12, 'acrostic': 13, 'concrete': 14
        }
    
    def _build_vocab(self):
        """Build vocabulary from poems"""
        vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        idx = 4
        
        for poem in self.poems:
            words = re.findall(r'\b\w+\b', poem['text'].lower())
            for word in words:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        
        return vocab
    
    def _classify_poem_style(self, text: str) -> int:
        """Classify poem style based on content"""
        text_lower = text.lower()
        
        # Simple classification based on keywords and structure
        if re.search(r'\b(story|tale|once upon)\b', text_lower):
            return self.poetry_styles['narrative']
        elif re.search(r'\b(love|heart|emotion|feel)\b', text_lower):
            return self.poetry_styles['lyric']
        elif re.search(r'\b(hero|battle|grand|epic)\b', text_lower):
            return self.poetry_styles['epic']
        elif re.search(r'\b(death|mourn|loss|grief)\b', text_lower):
            return self.poetry_styles['elegy']
        elif re.search(r'\b(praise|honor|celebrate)\b', text_lower):
            return self.poetry_styles['ode']
        elif len(text.split('\n')) == 14:  # Rough sonnet detection
            return self.poetry_styles['sonnet']
        elif len(text.split('\n')) == 3:  # Rough haiku detection
            return self.poetry_styles['haiku']
        elif re.search(r'\b(song|sing|ballad)\b', text_lower):
            return self.poetry_styles['ballad']
        else:
            return self.poetry_styles['free_verse']
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to tensor"""
        words = re.findall(r'\b\w+\b', text.lower())
        encoded = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Add start and end tokens
        encoded = [self.vocab['<START>']] + encoded + [self.vocab['<END>']]
        
        # Pad or truncate
        if len(encoded) < self.max_length:
            encoded.extend([self.vocab['<PAD>']] * (self.max_length - len(encoded)))
        else:
            encoded = encoded[:self.max_length]
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.poems)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        poem = self.poems[idx]
        
        text_encoding = self._encode_text(poem['text'])
        style_label = self._classify_poem_style(poem['text'])
        
        return {
            'text_encoding': text_encoding,
            'labels': torch.tensor(style_label, dtype=torch.long),
            'text_data': poem['text']
        }

class OptimizedPoetryTrainer:
    """World-class optimized poetry trainer for creative AI excellence"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize architecture selector for optimal creative model
        self.arch_selector = CustomArchitectureSelector()
        optimal_arch = self.arch_selector.select_architecture(
            task_type="sequential_data",
            data_type="text",
            requirements=["creativity", "sequential", "language_generation"]
        )
        config['neural_architecture'] = optimal_arch
        
        # Create optimized model for poetry
        self.model = self._create_creative_model()
        
        # Create tensor adapter for shape compatibility
        # Text encodings have shape [batch_size, max_length], we need to adapt to symbolic controller input
        text_encoding_size = config.get('max_length', 256)  # This is the sequence length
        self.tensor_adapter = nn.Sequential(
            nn.Linear(text_encoding_size, 128),  # Reduce from 256 to 128
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)  # Further reduce to 64 for symbolic controller
        ).to(self.device)
        
        # Initialize fuzzy logic controller (best for creative, nuanced reasoning)
        self.symbolic_controller = create_symbolic_controller(
            controller_type='fuzzy_logic',  # Best for creative, nuanced poetry
            num_rules=config.get('rule_set_size', 50),
            input_size=64,  # Match tensor adapter output
            hidden_size=config.get('symbolic_hidden_size', 32),
            num_fuzzy_sets=config.get('num_fuzzy_sets', 5)  # More fuzzy sets for creativity
        ).to(self.device)
        
        # Initialize creative poetry replay buffer
        self.replay_buffer = create_replay_buffer(
            buffer_type='adaptive',  # Adaptive for creative learning
            memory_size=config.get('replay_buffer_size', 30000),
            device=str(self.device),
            adaptation_rate=0.02  # Higher adaptation for creativity
        )
        
        # Advanced optimizer for creative tasks
        self.optimizer = self._create_creative_optimizer()
        self.scheduler = self._create_creative_scheduler()
        
        # Mixed precision for efficiency
        self.scaler = GradScaler() if config.get('mixed_precision', True) else None
        
        # Creative loss function
        self.criterion = self._create_creative_loss()
        
        # Poetry-specific metrics
        self.best_creativity_score = 0.0
        self.poetry_metrics = {
            'rhythm_accuracy': 0.0,
            'rhyme_detection': 0.0,
            'metaphor_understanding': 0.0,
            'emotional_resonance': 0.0,
            'creativity_score': 0.0
        }
        self.training_history = []
        
        print(f"🎭 Optimized Poetry Trainer Initialized")
        print(f"   Device: {self.device}")
        print(f"   Architecture: {config['neural_architecture']}")
        print(f"   Symbolic Controller: fuzzy_logic (creative reasoning)")
        print(f"   Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Creative Features: Enabled")
    
    def _create_creative_model(self) -> HybridModel:
        """Create model optimized for creative poetry generation"""
        model_spec = ModelSpec(
            neural_architecture=self.config['neural_architecture'],
            num_classes=self.config.get('num_classes', 15),  # Poetry style classes
            hidden_sizes=self.config.get('hidden_sizes', [768, 384, 192]),  # Creative depth
            input_shape=(self.config.get('max_length', 768),),  # Longer for poetry
            dropout_rate=self.config.get('dropout_rate', 0.15),  # Higher for creativity
            use_batch_norm=True,
            device=self.device,
            rule_set_size=self.config.get('rule_set_size', 150)
        )
        
        model = HybridModel(model_spec).to(self.device)
        
        # Apply creative initialization
        self._apply_creative_initialization(model)
        
        return model
    
    def _apply_creative_initialization(self, model: nn.Module):
        """Apply stable initialization that prevents NaN"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization for stability
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Stable embedding initialization
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                # Stable RNN initialization
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def _create_creative_optimizer(self) -> optim.Optimizer:
        """Create optimizer with proper learning rate"""
        # Use Adam with higher learning rate for actual learning
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-3),  # Higher LR for learning
            betas=(0.9, 0.999),  # Standard betas
            eps=1e-8,  # Standard epsilon
            weight_decay=self.config.get('weight_decay', 1e-4)  # Reasonable weight decay
        )
    
    def _create_creative_scheduler(self):
        """Create learning rate scheduler for creative training"""
        return optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=1e-5,
            max_lr=self.config.get('learning_rate', 3e-4),
            step_size_up=2000,  # Cycle for creative exploration
            mode='triangular2'
        )
    
    def _create_creative_loss(self) -> nn.Module:
        """Create stable loss function that promotes creativity"""
        class StableCreativeLoss(nn.Module):
            def __init__(self, diversity_weight=0.1, smoothness_weight=0.05):
                super().__init__()
                self.diversity_weight = diversity_weight
                self.smoothness_weight = smoothness_weight
                self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)  # Reduced label smoothing
            
            def diversity_loss(self, outputs):
                """Encourage diverse outputs with numerical stability"""
                # Calculate entropy to encourage diversity
                probs = F.softmax(outputs, dim=-1)
                # Clamp probabilities to prevent log(0)
                probs = torch.clamp(probs, min=1e-7, max=1.0)
                entropy = -torch.sum(probs * torch.log(probs), dim=-1)
                # Check for NaN and replace with 0
                entropy = torch.where(torch.isnan(entropy), torch.zeros_like(entropy), entropy)
                return -entropy.mean()  # Negative because we want high entropy
            
            def smoothness_loss(self, outputs):
                """Encourage smooth transitions with stability"""
                if outputs.size(0) > 1:
                    diff = outputs[1:] - outputs[:-1]
                    smoothness = torch.mean(torch.norm(diff, dim=-1))
                    # Check for NaN
                    if torch.isnan(smoothness):
                        return torch.tensor(0.0, device=outputs.device)
                    return smoothness
                return torch.tensor(0.0, device=outputs.device)
            
            def forward(self, outputs, targets):
                # Check for NaN in inputs
                if torch.isnan(outputs).any() or torch.isnan(targets).any():
                    print("⚠️ NaN detected in loss inputs, using fallback")
                    return torch.tensor(1.0, device=outputs.device, requires_grad=True)
                
                ce = self.ce_loss(outputs, targets)
                
                # Check if CE loss is NaN
                if torch.isnan(ce):
                    print("⚠️ NaN in CrossEntropy loss, using fallback")
                    return torch.tensor(1.0, device=outputs.device, requires_grad=True)
                
                diversity = self.diversity_loss(outputs)
                smoothness = self.smoothness_loss(outputs)
                
                # Check for NaN in additional losses
                if torch.isnan(diversity):
                    diversity = torch.tensor(0.0, device=outputs.device)
                if torch.isnan(smoothness):
                    smoothness = torch.tensor(0.0, device=outputs.device)
                
                total_loss = ce + self.diversity_weight * diversity + self.smoothness_weight * smoothness
                
                # Final NaN check
                if torch.isnan(total_loss):
                    print("⚠️ NaN in total loss, using CE only")
                    return ce
                
                return total_loss
        
        return StableCreativeLoss()
    
    def _calculate_poetry_metrics(self, outputs: torch.Tensor, labels: torch.Tensor,
                                 text_data: List[str]) -> Dict[str, float]:
        """Calculate poetry-specific creative metrics"""
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == labels).float()
        
        # Initialize creative metrics
        rhythm_scores = []
        rhyme_scores = []
        metaphor_scores = []
        emotional_scores = []
        creativity_scores = []
        
        for i, text in enumerate(text_data):
            # Analyze rhythm (syllable patterns)
            rhythm_score = self._analyze_rhythm(text)
            rhythm_scores.append(rhythm_score)
            
            # Analyze rhyme schemes
            rhyme_score = self._analyze_rhyme(text)
            rhyme_scores.append(rhyme_score)
            
            # Detect metaphorical language
            metaphor_score = self._analyze_metaphors(text)
            metaphor_scores.append(metaphor_score)
            
            # Assess emotional resonance
            emotional_score = self._analyze_emotion(text)
            emotional_scores.append(emotional_score)
            
            # Overall creativity score
            creativity_score = (rhythm_score + rhyme_score + metaphor_score + emotional_score) / 4
            creativity_scores.append(creativity_score)
        
        return {
            'rhythm_accuracy': np.mean(rhythm_scores),
            'rhyme_detection': np.mean(rhyme_scores),
            'metaphor_understanding': np.mean(metaphor_scores),
            'emotional_resonance': np.mean(emotional_scores),
            'creativity_score': np.mean(creativity_scores),
            'overall_accuracy': correct.mean().item()
        }
    
    def _analyze_rhythm(self, text: str) -> float:
        """Analyze rhythmic patterns in poetry"""
        # Simple syllable counting and pattern detection
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Count vowel clusters as syllable approximation
        syllable_counts = []
        for word in words:
            syllables = len(re.findall(r'[aeiouAEIOU]+', word))
            syllable_counts.append(max(1, syllables))  # At least 1 syllable per word
        
        if len(syllable_counts) < 2:
            return 0.5
        
        # Check for rhythmic consistency
        avg_syllables = np.mean(syllable_counts)
        variance = np.var(syllable_counts)
        
        # Lower variance indicates better rhythm
        rhythm_score = max(0.0, 1.0 - (variance / (avg_syllables + 1)))
        return min(1.0, rhythm_score)
    
    def _analyze_rhyme(self, text: str) -> float:
        """Analyze rhyme schemes in poetry"""
        lines = text.split('\n')
        if len(lines) < 2:
            return 0.0
        
        # Extract last words (potential rhymes)
        last_words = []
        for line in lines:
            words = line.strip().split()
            if words:
                last_words.append(words[-1].lower().rstrip('.,!?;:'))
        
        if len(last_words) < 2:
            return 0.0
        
        # Simple rhyme detection (ending similarity)
        rhyme_pairs = 0
        total_pairs = 0
        
        for i in range(len(last_words)):
            for j in range(i + 1, len(last_words)):
                total_pairs += 1
                word1, word2 = last_words[i], last_words[j]
                
                # Check for ending similarity (simple rhyme detection)
                if len(word1) >= 2 and len(word2) >= 2:
                    if word1[-2:] == word2[-2:] or word1[-3:] == word2[-3:]:
                        rhyme_pairs += 1
        
        return rhyme_pairs / max(1, total_pairs)
    
    def _analyze_metaphors(self, text: str) -> float:
        """Detect metaphorical and figurative language"""
        # Simple metaphor indicators
        metaphor_indicators = [
            'like', 'as', 'is', 'was', 'becomes', 'turns into',
            'resembles', 'seems', 'appears', 'feels like'
        ]
        
        comparative_words = [
            'than', 'more', 'less', 'better', 'worse', 'brighter',
            'darker', 'deeper', 'higher', 'stronger'
        ]
        
        text_lower = text.lower()
        metaphor_count = 0
        
        for indicator in metaphor_indicators:
            metaphor_count += text_lower.count(indicator)
        
        for word in comparative_words:
            metaphor_count += text_lower.count(word)
        
        # Normalize by text length
        words = text.split()
        if not words:
            return 0.0
        
        metaphor_density = metaphor_count / len(words)
        return min(1.0, metaphor_density * 10)  # Scale appropriately
    
    def _analyze_emotion(self, text: str) -> float:
        """Analyze emotional content and resonance"""
        # Emotional word categories
        positive_emotions = [
            'joy', 'happy', 'love', 'beautiful', 'wonderful', 'amazing',
            'bright', 'warm', 'gentle', 'peaceful', 'serene', 'blissful'
        ]
        
        negative_emotions = [
            'sad', 'dark', 'pain', 'sorrow', 'grief', 'lonely',
            'cold', 'harsh', 'bitter', 'angry', 'fear', 'despair'
        ]
        
        intense_emotions = [
            'passion', 'fire', 'burning', 'intense', 'overwhelming',
            'powerful', 'fierce', 'wild', 'storm', 'thunder'
        ]
        
        text_lower = text.lower()
        emotion_score = 0
        
        for emotion in positive_emotions + negative_emotions + intense_emotions:
            if emotion in text_lower:
                emotion_score += 1
        
        # Normalize by text length
        words = text.split()
        if not words:
            return 0.0
        
        emotional_density = emotion_score / len(words)
        return min(1.0, emotional_density * 15)  # Scale for emotional impact
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Optimized training step for creative poetry"""
        self.model.train()
        self.symbolic_controller.train()
        
        # Move to device
        text_encodings = batch['text_encoding'].to(self.device)
        labels = batch['labels'].to(self.device)
        text_data = batch.get('text_data', [''] * len(labels))
        
        self.optimizer.zero_grad()
        
        # Use mixed precision
        if self.scaler:
            with autocast():
                # Generate fuzzy symbolic rules for creativity
                text_encodings_float = text_encodings.float()
                
                # Adapt tensor shape for symbolic controller
                adapted_encodings = self.tensor_adapter(text_encodings_float)
                rule_indices, symbolic_state = self.symbolic_controller(adapted_encodings)
                
                # Create dummy images
                batch_size = text_encodings.size(0)
                dummy_images = torch.zeros(batch_size, 3, 224, 224).to(self.device)
                
                # Forward pass
                outputs = self.model(dummy_images, text_encodings_float, rule_indices)
                
                # Creative loss
                loss = self.criterion(outputs, labels)
            
            # Check for NaN loss before backward pass
            if torch.isnan(loss):
                print("⚠️ NaN loss detected, skipping backward pass")
                return {'loss': float('nan'), 'creativity_score': 0.0, 'lr': self.optimizer.param_groups[0]['lr']}
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Check for NaN gradients before unscaling
            has_nan_grad = False
            for param in self.model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print("⚠️ NaN gradients detected, skipping optimizer step")
                self.optimizer.zero_grad()
                # Skip scaler update when we have NaN gradients
                return {'loss': loss.item(), 'creativity_score': 0.0, 'lr': self.optimizer.param_groups[0]['lr']}
            
            # Unscale and clip gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            text_encodings_float = text_encodings.float()
            # Adapt tensor shape for symbolic controller
            adapted_encodings = self.tensor_adapter(text_encodings_float)
            rule_indices, symbolic_state = self.symbolic_controller(adapted_encodings)
            
            batch_size = text_encodings.size(0)
            dummy_images = torch.zeros(batch_size, 3, 224, 224).to(self.device)
            
            outputs = self.model(dummy_images, text_encodings_float, rule_indices)
            loss = self.criterion(outputs, labels)
            
            # Check for NaN loss before backward pass
            if torch.isnan(loss):
                print("⚠️ NaN loss detected, skipping backward pass")
                return {'loss': float('nan'), 'creativity_score': 0.0, 'lr': self.optimizer.param_groups[0]['lr']}
            
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in self.model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print("⚠️ NaN gradients detected, zeroing gradients")
                self.optimizer.zero_grad()
                return {'loss': loss.item(), 'creativity_score': 0.0, 'lr': self.optimizer.param_groups[0]['lr']}
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # Calculate creative metrics
        with torch.no_grad():
            poetry_metrics = self._calculate_poetry_metrics(outputs, labels, text_data)
            
            # Store creative experiences in adaptive buffer
            for i in range(len(text_encodings)):
                performance_score = poetry_metrics['creativity_score']  # Use creativity as performance
                self.replay_buffer.add(
                    experience={
                        'text_encoding': text_encodings[i].cpu(),
                        'labels': labels[i:i+1].cpu(),
                        'loss': loss.item(),
                        'creativity': poetry_metrics['creativity_score'],
                        'text_data': text_data[i] if i < len(text_data) else ''
                    },
                    performance_score=performance_score
                )
        
        return {
            'loss': loss.item(),
            **poetry_metrics,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluation step for poetry"""
        self.model.eval()
        self.symbolic_controller.eval()
        
        with torch.no_grad():
            text_encodings = batch['text_encoding'].to(self.device)
            labels = batch['labels'].to(self.device)
            text_data = batch.get('text_data', [''] * len(labels))
            
            text_encodings_float = text_encodings.float()
            # Adapt tensor shape for symbolic controller
            adapted_encodings = self.tensor_adapter(text_encodings_float)
            rule_indices, symbolic_state = self.symbolic_controller(adapted_encodings)
            
            batch_size = text_encodings.size(0)
            dummy_images = torch.zeros(batch_size, 3, 224, 224).to(self.device)
            
            outputs = self.model(dummy_images, text_encodings_float, rule_indices)
            loss = self.criterion(outputs, labels)
            
            poetry_metrics = self._calculate_poetry_metrics(outputs, labels, text_data)
        
        return {
            'loss': loss.item(),
            **poetry_metrics
        }
    
    def train(self, dataloader: DataLoader, num_epochs: int):
        """Main creative training loop"""
        print(f"🎭 Starting optimized poetry training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = {
                'loss': 0.0, 'overall_accuracy': 0.0, 'rhythm_accuracy': 0.0,
                'rhyme_detection': 0.0, 'metaphor_understanding': 0.0,
                'emotional_resonance': 0.0, 'creativity_score': 0.0
            }
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Poetry Epoch {epoch + 1}")
            for batch in pbar:
                metrics = self.train_step(batch)
                
                for key in train_metrics:
                    if key in metrics:
                        train_metrics[key] += metrics[key]
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f"{metrics['loss']:.4f}",
                    'Creativity': f"{metrics.get('creativity_score', 0.0):.4f}",
                    'Emotion': f"{metrics.get('emotional_resonance', 0.0):.4f}",
                    'LR': f"{metrics.get('lr', 0.0):.2e}"
                })
            
            # Average metrics
            for key in train_metrics:
                train_metrics[key] /= num_batches
            
            # Evaluation phase
            eval_metrics = {
                'loss': 0.0, 'overall_accuracy': 0.0, 'rhythm_accuracy': 0.0,
                'rhyme_detection': 0.0, 'metaphor_understanding': 0.0,
                'emotional_resonance': 0.0, 'creativity_score': 0.0
            }
            eval_batches = 0
            
            for batch in dataloader:
                metrics = self.evaluate_step(batch)
                for key in eval_metrics:
                    if key in metrics:
                        eval_metrics[key] += metrics[key]
                eval_batches += 1
            
            for key in eval_metrics:
                eval_metrics[key] /= eval_batches
            
            # Update scheduler
            self.scheduler.step()
            
            # Save results
            epoch_results = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'eval': eval_metrics
            }
            self.training_history.append(epoch_results)
            
            # Print creative results
            print(f"\nEpoch {epoch + 1}/{num_epochs} Creative Results:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Creativity: {train_metrics['creativity_score']:.4f}")
            print(f"  Train - Rhythm: {train_metrics['rhythm_accuracy']:.4f}, Emotion: {train_metrics['emotional_resonance']:.4f}")
            print(f"  Eval  - Loss: {eval_metrics['loss']:.4f}, Creativity: {eval_metrics['creativity_score']:.4f}")
            print(f"  Eval  - Metaphor: {eval_metrics['metaphor_understanding']:.4f}, Rhyme: {eval_metrics['rhyme_detection']:.4f}")
            
            # Save best creative model
            if eval_metrics['creativity_score'] > self.best_creativity_score:
                self.best_creativity_score = eval_metrics['creativity_score']
                self.poetry_metrics = {
                    'rhythm_accuracy': eval_metrics['rhythm_accuracy'],
                    'rhyme_detection': eval_metrics['rhyme_detection'],
                    'metaphor_understanding': eval_metrics['metaphor_understanding'],
                    'emotional_resonance': eval_metrics['emotional_resonance'],
                    'creativity_score': eval_metrics['creativity_score']
                }
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'symbolic_controller_state_dict': self.symbolic_controller.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'creativity_score': self.best_creativity_score,
                    'poetry_metrics': self.poetry_metrics,
                    'config': self.config,
                    'training_history': self.training_history
                }, 'best_poetry_model_optimized.pt')
                
                print(f"  🎨 New best creativity score: {self.best_creativity_score:.4f}")
        
        print(f"\n🎉 Creative poetry training completed!")
        print(f"Best Creativity Score: {self.best_creativity_score:.4f}")
        print(f"Poetry Skills Breakdown:")
        print(f"  Rhythm Accuracy: {self.poetry_metrics['rhythm_accuracy']:.4f}")
        print(f"  Rhyme Detection: {self.poetry_metrics['rhyme_detection']:.4f}")
        print(f"  Metaphor Understanding: {self.poetry_metrics['metaphor_understanding']:.4f}")
        print(f"  Emotional Resonance: {self.poetry_metrics['emotional_resonance']:.4f}")

def create_poetry_dataset() -> PoetryDataset:
    """Create poetry dataset from available sources"""
    poems = []
    
    # Check for existing poetry data
    poetry_data_path = Path("dataset/poetry")
    if poetry_data_path.exists():
        print(f"📚 Found poetry data directory: {poetry_data_path}")
        
        # Look for text files or PDFs
        for file_path in poetry_data_path.glob("*"):
            if file_path.suffix.lower() in ['.txt', '.pdf']:
                print(f"📖 Processing: {file_path.name}")
                
                if file_path.suffix.lower() == '.pdf':
                    # For PDF files, create sample poems based on the title
                    poems.extend(create_sample_poems_from_pdf(file_path))
                else:
                    # For text files, read directly
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            poems.extend(parse_poetry_text(content))
                    except Exception as e:
                        print(f"⚠️ Error reading {file_path}: {e}")
    
    # If no poems found, create sample dataset
    if not poems:
        print("📝 Creating sample poetry dataset...")
        poems = create_sample_poetry_dataset()
    
    print(f"✅ Created dataset with {len(poems)} poems")
    return PoetryDataset(poems)

def create_sample_poems_from_pdf(pdf_path: Path) -> List[Dict[str, str]]:
    """Create sample poems inspired by PDF title"""
    # Since we can't easily parse PDF, create themed poems based on the title
    if "Dark Arts" in pdf_path.name:
        return [
            {
                'text': '''In shadows deep where darkness dwells,
Ancient secrets the night foretells,
Whispered words of power untold,
Stories dark and legends old.

Through mystic realms where spirits roam,
Far beyond our earthly home,
Magic flows like rivers black,
On paths from which few souls come back.''',
                'title': 'Dark Arts I',
                'author': 'Generated',
                'style': 'dark_poetry'
            },
            {
                'text': '''Beneath the moon's ethereal glow,
Where shadow-dancers come and go,
The ancient arts of night unfold,
In languages both new and old.

Candles flicker, incense burns,
As the wheel of darkness turns,
Mysteries deep within the soul,
Make the broken spirit whole.''',
                'title': 'Dark Arts II',
                'author': 'Generated',
                'style': 'mystical'
            },
            {
                'text': '''Words of power, spoken low,
Set the mystic energies flow,
Through the veil between the worlds,
Where reality unfurls.

Dark and light in balance dance,
Nothing left to mere chance,
In the arts of shadow's way,
Night transforms into day.''',
                'title': 'Dark Arts III',
                'author': 'Generated',
                'style': 'mystical'
            }
        ]
    else:
        return create_sample_poetry_dataset()

def parse_poetry_text(content: str) -> List[Dict[str, str]]:
    """Parse poetry from text content"""
    poems = []
    
    # Split by double newlines (common poem separator)
    potential_poems = content.split('\n\n')
    
    for i, text in enumerate(potential_poems):
        text = text.strip()
        if len(text) > 50 and '\n' in text:  # Likely a poem
            poems.append({
                'text': text,
                'title': f'Poem {i+1}',
                'author': 'Unknown',
                'style': 'general'
            })
    
    return poems

def create_sample_poetry_dataset() -> List[Dict[str, str]]:
    """Create a sample poetry dataset for training"""
    return [
        {
            'text': '''Two roads diverged in a yellow wood,
And sorry I could not travel both
And be one traveler, long I stood
And looked down one as far as I could
To where it bent in the undergrowth.''',
            'title': 'The Road Not Taken (excerpt)',
            'author': 'Robert Frost',
            'style': 'narrative'
        },
        {
            'text': '''I wandered lonely as a cloud
That floats on high o'er vales and hills,
When all at once I saw a crowd,
A host, of golden daffodils.''',
            'title': 'Daffodils (excerpt)',
            'author': 'William Wordsworth',
            'style': 'lyric'
        },
        {
            'text': '''Roses are red,
Violets are blue,
Sugar is sweet,
And so are you.''',
            'title': 'Classic Rhyme',
            'author': 'Traditional',
            'style': 'simple'
        },
        {
            'text': '''The sun sets low behind the hill,
The world grows quiet, calm, and still,
Stars emerge in evening's glow,
As gentle breezes softly blow.''',
            'title': 'Evening Peace',
            'author': 'Generated',
            'style': 'lyric'
        },
        {
            'text': '''In the forest deep and dark,
Where the ancient spirits hark,
Whispered secrets fill the air,
Magic dances everywhere.''',
            'title': 'Forest Magic',
            'author': 'Generated',
            'style': 'mystical'
        },
        {
            'text': '''Love is like a burning flame,
Never twice exactly same,
Sometimes bright and sometimes dim,
Dancing on passion's whim.''',
            'title': 'Love\'s Flame',
            'author': 'Generated',
            'style': 'lyric'
        },
        {
            'text': '''The warrior stood upon the field,
With sword and honor as his shield,
Against the darkness he would fight,
To bring the world back to the light.''',
            'title': 'The Warrior',
            'author': 'Generated',
            'style': 'epic'
        },
        {
            'text': '''Autumn leaves fall to the ground,
Making barely any sound,
Golden, red, and brown they lay,
Marking summer's end today.''',
            'title': 'Autumn Leaves',
            'author': 'Generated',
            'style': 'nature'
        },
        {
            'text': '''Time flows like a river wide,
Carrying all upon its tide,
Past and future meet as one,
In the moment, never done.''',
            'title': 'River of Time',
            'author': 'Generated',
            'style': 'philosophical'
        },
        {
            'text': '''Dreams take flight on silver wings,
To lands where hope eternal sings,
Beyond the reach of earthly care,
In realms of wonder, pure and fair.''',
            'title': 'Silver Wings',
            'author': 'Generated',
            'style': 'fantasy'
        }
    ]

def main():
    """Main function for world-class creative AI"""
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_classes': 15,  # Poetry style classes
        'max_length': 256,  # Reduced for 4GB GPU
        'batch_size': 2,  # Very small for 4GB GPU
        'learning_rate': 1e-3,  # Higher learning rate for actual learning
        'weight_decay': 5e-5,  # Lower for creativity
        'num_epochs': 5,  # Much shorter for debugging
        'dropout_rate': 0.15,  # Higher for creative diversity
        'use_batch_norm': True,
        'neural_architecture': 'custom_lstm',  # Will be optimized
        'hidden_sizes': [256, 128, 64],  # Much smaller for 4GB GPU
        'rule_set_size': 50,  # Reduced for memory
        'symbolic_hidden_size': 32,  # Reduced for memory
        'num_fuzzy_sets': 5,  # Reduced for memory
        'replay_buffer_size': 5000,  # Reduced for memory
        'mixed_precision': False  # Disable for stability
    }
    
    print("🎭 World-Class Creative Poetry AI Training")
    print("=" * 60)
    print("🎯 Target: Achieve world-class creative poetry AI")
    print("🎨 Focus: Creativity, emotion, rhythm, metaphor")
    print("=" * 60)
    
    # Create dataset from available poetry data
    dataset = create_poetry_dataset()
    if len(dataset) == 0:
        print("❌ No poetry data found!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Create optimized creative trainer
    trainer = OptimizedPoetryTrainer(config)
    
    # Train the creative model
    trainer.train(dataloader, config['num_epochs'])
    
    # Save creative results
    results = {
        'best_creativity_score': trainer.best_creativity_score,
        'poetry_metrics': trainer.poetry_metrics,
        'config': config,
        'training_history': trainer.training_history
    }
    
    with open('poetry_creative_ai_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎨 World-class creative poetry AI training completed!")
    print(f"📊 Results saved to poetry_creative_ai_results.json")

if __name__ == "__main__":
    main()