import torch
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

class SymbolicController(nn.Module):
    """Advanced controller for managing symbolic rule indices generation and routing.
    
    This controller uses a combination of neural and symbolic reasoning to dynamically
    generate rule indices based on input features, task metadata, and learned symbolic states.
    """
    
    def __init__(self, 
                 num_rules: int,
                 input_size: int,
                 hidden_size: int = 64,
                 use_task_metadata: bool = True,
                 use_prior_state: bool = True,
                 use_attention: bool = True):
        super().__init__()
        """Initialize the symbolic controller.
        
        Args:
            num_rules: Total number of available rules
            input_size: Size of the input features
            hidden_size: Size of the hidden state
            use_task_metadata: Whether to use task metadata in rule selection
            use_prior_state: Whether to use prior learned symbolic state
            use_attention: Whether to use attention mechanism for rule selection
        """
        self.num_rules = num_rules
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_task_metadata = use_task_metadata
        self.use_prior_state = use_prior_state
        self.use_attention = use_attention
        
        # Feature processing components
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Attention mechanism for rule selection
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
            self.attention_norm = nn.LayerNorm(hidden_size)
            
        # Initialize weights
        self._init_weights()
        
        # Task metadata handling
        if use_task_metadata:
            self.task_metadata_encoder = nn.Sequential(
                nn.Embedding(1000, hidden_size),  # Larger embedding space
                nn.LayerNorm(hidden_size)
            )
        
        # Rule selection components
        self.rule_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_rules)
        )
        
        # State transition for learning rule dependencies
        self.state_transition = nn.Linear(hidden_size, hidden_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
    
    def to(self, device):
        """Move model components to specified device."""
        self.feature_extractor.to(device)
        if self.use_attention:
            self.attention.to(device)
            self.attention_norm.to(device)
        if self.use_task_metadata:
            self.task_metadata_encoder.to(device)
        self.rule_selector.to(device)
        self.state_transition.to(device)
        self.device = device
        return self
    
    def _process_task_metadata(self, task_metadata: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process task metadata into useful representations."""
        if not self.use_task_metadata:
            return torch.zeros(1, self.hidden_size, device=self.device)
            
        # Extract task ID and other relevant metadata
        task_id = task_metadata.get('id', 0)
        task_type = task_metadata.get('type', 0)
        
        # Encode metadata
        metadata_tensor = torch.tensor([task_id], device=self.device)
        metadata_embed = self.task_metadata_encoder(metadata_tensor)
        
        return metadata_embed
    
    def _apply_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism to refine feature representation."""
        if not self.use_attention:
            return features
            
        # Self-attention to capture feature dependencies
        attn_output, _ = self.attention(
            features.unsqueeze(0),
            features.unsqueeze(0),
            features.unsqueeze(0)
        )
        
        return self.attention_norm(attn_output.squeeze(0))
    
    def generate_rule_indices(self, 
                            x: torch.Tensor, 
                            task_metadata: Optional[Dict[str, Any]] = None,
                            prior_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate rule indices for input batch.
        
        Args:
            x: Input features [batch_size, feature_size]
            task_metadata: Optional task-specific metadata
            prior_state: Optional prior learned symbolic state
            
        Returns:
            tuple: (rule_indices, new_state) where
                rule_indices: Tensor of shape [batch_size] containing selected rule indices
                new_state: Tensor of shape [batch_size, hidden_size] containing updated state
        """
        batch_size = x.size(0)
        
        # Extract features from input
        features = self.feature_extractor(x)
        
        # Process task metadata
        if self.use_task_metadata and task_metadata is not None:
            metadata_embed = self._process_task_metadata(task_metadata)
            features = features + metadata_embed
        
        # Apply attention mechanism
        features = self._apply_attention(features)
        
        # Incorporate prior state
        if self.use_prior_state and prior_state is not None:
            features = features + prior_state
        
        # Generate rule probabilities
        rule_logits = self.rule_selector(features)
        rule_probs = F.softmax(rule_logits, dim=-1)
        
        # Sample rule indices
        rule_indices = torch.multinomial(rule_probs, 1).squeeze(-1)
        
        # Update state based on selected rules
        new_state = self.state_transition(features)
        if prior_state is not None:
            new_state = new_state + prior_state
        
        return rule_indices, new_state
    
    def forward(self, 
               x: torch.Tensor, 
               task_metadata: Optional[Dict[str, Any]] = None,
               prior_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the controller."""
        return self.generate_rule_indices(x, task_metadata, prior_state)

@dataclass
class ControllerConfig:
    """Configuration for the symbolic controller."""
    num_rules: int
    input_size: int
    hidden_size: int = 64
    use_task_metadata: bool = True
    use_prior_state: bool = True
    use_attention: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

