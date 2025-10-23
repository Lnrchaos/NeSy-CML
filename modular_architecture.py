"""
Modular Architecture System for NeuroSym-CML
Adapts model expectations to different data types and training scenarios
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod

class ModalityAdapter(ABC):
    """Base class for modality-specific adapters"""
    
    @abstractmethod
    def process_input(self, data: Any) -> Dict[str, torch.Tensor]:
        """Process input data into standard format"""
        pass
    
    @abstractmethod
    def get_expected_output_shape(self, batch_size: int) -> tuple:
        """Get expected output shape for this modality"""
        pass

class TextOnlyAdapter(ModalityAdapter):
    """Adapter for text-only training (chess, poetry, programming)"""
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
    
    def process_input(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert text data to multimodal format"""
        text_encodings = data['text_encoding']
        batch_size = text_encodings.size(0)
        device = text_encodings.device
        
        # Convert text encodings to embeddings if needed
        if text_encodings.dtype == torch.long:
            # If token IDs, create simple embeddings
            text_embeddings = torch.randn(batch_size, self.feature_dim).to(device)
        else:
            text_embeddings = text_encodings.float()
        
        # Create minimal dummy images (not used in text-only training)
        dummy_images = torch.zeros(batch_size, 3, 64, 64).to(device)
        
        # Create simple rule indices
        rule_indices = torch.randint(0, 100, (batch_size,)).to(device)
        
        return {
            'images': dummy_images,
            'text_embeddings': text_embeddings,
            'rule_indices': rule_indices,
            'labels': data.get('labels', torch.zeros(batch_size).long().to(device))
        }
    
    def get_expected_output_shape(self, batch_size: int) -> tuple:
        return (batch_size, 10)  # Assuming 10 classes

class ImageOnlyAdapter(ModalityAdapter):
    """Adapter for image-only training"""
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
    
    def process_input(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert image data to multimodal format"""
        images = data['image']
        batch_size = images.size(0)
        device = images.device
        
        # Create dummy text embeddings
        text_embeddings = torch.zeros(batch_size, self.feature_dim).to(device)
        
        # Create simple rule indices
        rule_indices = torch.randint(0, 100, (batch_size,)).to(device)
        
        return {
            'images': images,
            'text_embeddings': text_embeddings,
            'rule_indices': rule_indices,
            'labels': data.get('labels', torch.zeros(batch_size).long().to(device))
        }
    
    def get_expected_output_shape(self, batch_size: int) -> tuple:
        return (batch_size, 5)  # Assuming 5 classes for images

class MultiModalAdapter(ModalityAdapter):
    """Adapter for full multimodal training"""
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
    
    def process_input(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process full multimodal data"""
        return {
            'images': data['image'],
            'text_embeddings': data['text_embeddings'],
            'rule_indices': data['rule_indices'],
            'labels': data['labels']
        }
    
    def get_expected_output_shape(self, batch_size: int) -> tuple:
        return (batch_size, 5)  # Multimodal classes

class ModularModel(nn.Module):
    """Modular wrapper that adapts to different input types"""
    
    def __init__(self, base_model: nn.Module, adapter: ModalityAdapter):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.feature_dim = getattr(base_model, 'feature_dim', 512)
    
    def forward(self, **kwargs) -> torch.Tensor:
        """Forward pass with automatic adaptation"""
        # Process inputs through adapter
        processed_data = self.adapter.process_input(kwargs)
        
        # Forward through base model
        return self.base_model(
            processed_data['images'],
            processed_data['text_embeddings'],
            processed_data['rule_indices']
        )

class AdaptiveTrainer:
    """Adaptive trainer that works with any modality"""
    
    def __init__(self, base_model: nn.Module, modality: str, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Select appropriate adapter
        if modality == 'text_only':
            self.adapter = TextOnlyAdapter(config.get('feature_dim', 512))
        elif modality == 'image_only':
            self.adapter = ImageOnlyAdapter(config.get('feature_dim', 512))
        elif modality == 'multimodal':
            self.adapter = MultiModalAdapter(config.get('feature_dim', 512))
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Create modular model
        self.model = ModularModel(base_model, self.adapter).to(self.device)
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        print(f"✅ Adaptive trainer initialized for {modality} modality")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step that works with any data format"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Get labels (adapter ensures they exist)
        processed_data = self.adapter.process_input(batch)
        labels = processed_data['labels']
        
        # Calculate loss
        loss = self.criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step"""
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            
            # Get labels
            processed_data = self.adapter.process_input(batch)
            labels = processed_data['labels']
            
            # Calculate metrics
            loss = self.criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == labels).float().mean()
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy.item()
            }

def create_adaptive_trainer(base_model: nn.Module, modality: str, config: Dict[str, Any]) -> AdaptiveTrainer:
    """Factory function to create adaptive trainer"""
    return AdaptiveTrainer(base_model, modality, config)

# Convenience functions for each training type
def create_chess_trainer(base_model: nn.Module, config: Dict[str, Any]) -> AdaptiveTrainer:
    """Create trainer specifically for chess data"""
    return create_adaptive_trainer(base_model, 'text_only', config)

def create_poetry_trainer(base_model: nn.Module, config: Dict[str, Any]) -> AdaptiveTrainer:
    """Create trainer specifically for poetry data"""
    return create_adaptive_trainer(base_model, 'text_only', config)

def create_programming_trainer(base_model: nn.Module, config: Dict[str, Any]) -> AdaptiveTrainer:
    """Create trainer specifically for programming data"""
    return create_adaptive_trainer(base_model, 'text_only', config)

def create_image_trainer(base_model: nn.Module, config: Dict[str, Any]) -> AdaptiveTrainer:
    """Create trainer specifically for image data"""
    return create_adaptive_trainer(base_model, 'image_only', config)

def create_multimodal_trainer(base_model: nn.Module, config: Dict[str, Any]) -> AdaptiveTrainer:
    """Create trainer for full multimodal data"""
    return create_adaptive_trainer(base_model, 'multimodal', config)