"""
Custom Tensor Adapter for NeuroSym-CML
Handles shape mismatches and tensor transformations between different components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

class TensorAdapter(nn.Module):
    """Universal tensor adapter for handling shape mismatches"""
    
    def __init__(self, input_size: int, output_size: int, adapter_type: str = 'linear'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.adapter_type = adapter_type
        
        if adapter_type == 'linear':
            self.adapter = nn.Linear(input_size, output_size)
        elif adapter_type == 'mlp':
            self.adapter = nn.Sequential(
                nn.Linear(input_size, (input_size + output_size) // 2),
                nn.ReLU(),
                nn.Linear((input_size + output_size) // 2, output_size)
            )
        elif adapter_type == 'conv1d':
            self.adapter = nn.Conv1d(input_size, output_size, kernel_size=1)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt tensor shape"""
        original_shape = x.shape
        
        # Handle different input shapes
        if x.dim() == 3:  # [batch, seq, features]
            # Pool sequence dimension
            x = x.mean(dim=1)  # [batch, features]
        elif x.dim() == 4:  # [batch, channels, height, width]
            # Global average pooling
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif x.dim() == 1:  # [features]
            # Add batch dimension
            x = x.unsqueeze(0)
        
        # Ensure we have the right input size
        if x.size(-1) != self.input_size:
            if x.size(-1) > self.input_size:
                # Truncate
                x = x[..., :self.input_size]
            else:
                # Pad with zeros
                padding = self.input_size - x.size(-1)
                x = F.pad(x, (0, padding))
        
        # Apply adapter
        if self.adapter_type == 'conv1d':
            x = x.unsqueeze(-1)  # Add dimension for conv1d
            x = self.adapter(x.transpose(-1, -2)).transpose(-1, -2).squeeze(-1)
        else:
            x = self.adapter(x)
        
        return x

class SymbolicControllerAdapter(nn.Module):
    """Specialized adapter for symbolic controllers"""
    
    def __init__(self, text_encoding_size: int = 512, controller_input_size: int = 256):
        super().__init__()
        self.text_adapter = TensorAdapter(text_encoding_size, controller_input_size, 'mlp')
        self.controller_input_size = controller_input_size
    
    def adapt_text_for_controller(self, text_encodings: torch.Tensor) -> torch.Tensor:
        """Adapt text encodings for symbolic controller input"""
        return self.text_adapter(text_encodings)
    
    def adapt_rule_indices(self, rule_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Ensure rule indices have correct batch size"""
        if rule_indices.dim() == 0:  # Scalar
            rule_indices = rule_indices.unsqueeze(0).expand(batch_size)
        elif rule_indices.size(0) != batch_size:
            # Repeat or truncate to match batch size
            if rule_indices.size(0) == 1:
                rule_indices = rule_indices.expand(batch_size)
            else:
                rule_indices = rule_indices[:batch_size]
        
        return rule_indices

class ReplayBufferAdapter:
    """Adapter for replay buffer data"""
    
    @staticmethod
    def adapt_batch_for_replay(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt batch data for replay buffer storage"""
        adapted_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Ensure tensors are on CPU for storage
                adapted_batch[key] = value.cpu()
                
                # Handle different tensor shapes
                if key == 'text_encoding' and value.dim() == 3:
                    # Pool sequence dimension for storage efficiency
                    adapted_batch[key] = value.mean(dim=1).cpu()
                elif key == 'labels' and value.dim() > 2:
                    # Flatten extra dimensions
                    adapted_batch[key] = value.squeeze().cpu()
            else:
                adapted_batch[key] = value
        
        return adapted_batch
    
    @staticmethod
    def adapt_replay_for_training(replay_batch: Dict[str, torch.Tensor], 
                                 device: torch.device) -> Dict[str, torch.Tensor]:
        """Adapt replay buffer data back for training"""
        adapted_batch = {}
        
        for key, value in replay_batch.items():
            if isinstance(value, torch.Tensor):
                adapted_batch[key] = value.to(device)
                
                # Restore expected shapes
                if key == 'text_encoding' and value.dim() == 2:
                    # Add sequence dimension back
                    adapted_batch[key] = value.unsqueeze(1).to(device)
                elif key == 'labels' and value.dim() == 1:
                    # Add batch dimension if needed
                    if value.size(0) == 10:  # Chess has 10 classes
                        adapted_batch[key] = value.unsqueeze(0).to(device)
            else:
                adapted_batch[key] = value
        
        return adapted_batch

class ModelOutputAdapter:
    """Adapter for model outputs to handle different evaluation formats"""
    
    @staticmethod
    def adapt_for_loss(outputs: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adapt outputs and labels for loss calculation"""
        # Ensure outputs and labels have compatible shapes
        if outputs.dim() != labels.dim():
            if labels.dim() == 1 and labels.size(0) == outputs.size(1):
                # Single sample with multiple classes
                labels = labels.unsqueeze(0)
            elif outputs.dim() == 2 and labels.dim() == 1:
                # Multiple samples, single class each
                if labels.size(0) != outputs.size(0):
                    # Adjust batch size
                    min_batch = min(outputs.size(0), labels.size(0))
                    outputs = outputs[:min_batch]
                    labels = labels[:min_batch]
        
        return outputs, labels
    
    @staticmethod
    def adapt_for_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adapt outputs and labels for metric calculation"""
        # Convert to same format for metric calculation
        if outputs.dim() == 2 and labels.dim() == 2:
            # Multi-label classification
            predictions = torch.sigmoid(outputs) > 0.5
            labels_bool = labels.bool()
        else:
            # Single-label classification
            predictions = torch.argmax(outputs, dim=-1)
            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=-1)
        
        return predictions, labels

# Convenience functions
def create_symbolic_adapter(text_size: int = 512, controller_size: int = 256) -> SymbolicControllerAdapter:
    """Create a symbolic controller adapter"""
    return SymbolicControllerAdapter(text_size, controller_size)

def adapt_tensor_shape(tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """General purpose tensor shape adapter"""
    current_shape = tensor.shape
    
    # Handle common shape adaptations
    if len(target_shape) == 2 and len(current_shape) == 3:
        # Sequence to batch: pool sequence dimension
        tensor = tensor.mean(dim=1)
    elif len(target_shape) == 3 and len(current_shape) == 2:
        # Batch to sequence: add sequence dimension
        tensor = tensor.unsqueeze(1)
    elif len(target_shape) == 1 and len(current_shape) == 2:
        # Batch to single: take first sample
        tensor = tensor[0]
    elif len(target_shape) == 2 and len(current_shape) == 1:
        # Single to batch: add batch dimension
        tensor = tensor.unsqueeze(0)
    
    # Handle size mismatches in last dimension
    if tensor.size(-1) != target_shape[-1]:
        if tensor.size(-1) > target_shape[-1]:
            # Truncate
            tensor = tensor[..., :target_shape[-1]]
        else:
            # Pad
            padding = target_shape[-1] - tensor.size(-1)
            tensor = F.pad(tensor, (0, padding))
    
    return tensor

# Example usage and testing
if __name__ == "__main__":
    print("🔧 Testing Tensor Adapter System")
    print("=" * 40)
    
    # Test basic adapter
    adapter = TensorAdapter(512, 256, 'mlp')
    test_input = torch.randn(2, 10, 512)  # [batch, seq, features]
    output = adapter(test_input)
    print(f"✅ Basic adapter: {test_input.shape} -> {output.shape}")
    
    # Test symbolic controller adapter
    symbolic_adapter = create_symbolic_adapter()
    adapted_text = symbolic_adapter.adapt_text_for_controller(test_input)
    print(f"✅ Symbolic adapter: {test_input.shape} -> {adapted_text.shape}")
    
    # Test rule indices adaptation
    rule_indices = torch.tensor([5])
    adapted_rules = symbolic_adapter.adapt_rule_indices(rule_indices, batch_size=2)
    print(f"✅ Rule indices: {rule_indices.shape} -> {adapted_rules.shape}")
    
    print("🎉 Tensor adapter system ready!")