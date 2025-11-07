"""
Sparse Mixture of Experts (MoE) - Advanced implementation
Includes Switch Transformer, GShard, and other sparse expert routing mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class RoutingStrategy(Enum):
    """Expert routing strategies"""
    TOP_K = "top_k"  # Top-k routing
    SWITCH = "switch"  # Switch Transformer routing
    BASE = "base"  # Base MoE routing
    LOAD_BALANCED = "load_balanced"  # Load-balanced routing


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts"""
    num_experts: int = 8
    expert_capacity: int = 4
    hidden_size: int = 512
    expert_hidden_size: Optional[int] = None
    num_layers_per_expert: int = 2
    routing_strategy: RoutingStrategy = RoutingStrategy.TOP_K
    top_k: int = 2  # Number of experts to route to
    load_balance_loss_weight: float = 0.01
    aux_loss_weight: float = 0.01
    dropout: float = 0.1
    use_expert_bias: bool = True
    use_layer_norm: bool = True
    activation: str = "gelu"


class Expert(nn.Module):
    """Individual expert network"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        expert_hidden = config.expert_hidden_size or config.hidden_size
        
        layers = []
        for i in range(config.num_layers_per_expert):
            layers.append(nn.Linear(
                config.hidden_size if i == 0 else expert_hidden,
                expert_hidden
            ))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(expert_hidden))
            layers.append(self._get_activation(config.activation))
            layers.append(nn.Dropout(config.dropout))
        
        # Output projection
        layers.append(nn.Linear(expert_hidden, config.hidden_size))
        
        self.expert = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert"""
        return self.expert(x)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            return nn.GELU()


class TopKRouter(nn.Module):
    """Top-k expert router"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.num_experts = config.num_experts
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_experts)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts
        
        Args:
            x: Input tokens [batch_size, seq_len, hidden_size]
            
        Returns:
            router_logits: Router logits [batch_size, seq_len, num_experts]
            router_probs: Router probabilities [batch_size, seq_len, num_experts]
            expert_mask: Expert assignment mask [batch_size, seq_len, num_experts]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute router logits
        router_logits = self.router(x)  # [batch, seq_len, num_experts]
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Create expert mask
        expert_mask = torch.zeros_like(router_logits)
        expert_mask.scatter_(-1, top_k_indices, 1.0)
        
        # Compute probabilities for top-k only
        router_probs = F.softmax(top_k_logits, dim=-1)
        
        # Expand probabilities to full expert dimension
        router_probs_full = torch.zeros_like(router_logits)
        router_probs_full.scatter_(-1, top_k_indices, router_probs)
        
        return router_logits, router_probs_full, expert_mask


class SwitchRouter(nn.Module):
    """Switch Transformer router (single expert per token)"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_experts)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to single expert (Switch Transformer)
        
        Returns:
            router_logits: Router logits
            router_probs: Router probabilities
            expert_mask: Expert assignment mask
        """
        # Compute router logits
        router_logits = self.router(x)  # [batch, seq_len, num_experts]
        
        # Get top-1 expert
        top_1_logits, top_1_indices = torch.topk(router_logits, 1, dim=-1)
        
        # Create expert mask
        expert_mask = torch.zeros_like(router_logits)
        expert_mask.scatter_(-1, top_1_indices, 1.0)
        
        # Compute probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        return router_logits, router_probs, expert_mask


class LoadBalancedRouter(nn.Module):
    """Load-balanced router with auxiliary loss"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.num_experts = config.num_experts
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_experts)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens with load balancing
        
        Returns:
            router_logits: Router logits
            router_probs: Router probabilities
            expert_mask: Expert assignment mask
            load_balance_loss: Load balancing loss
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute router logits
        router_logits = self.router(x)
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Create expert mask
        expert_mask = torch.zeros_like(router_logits)
        expert_mask.scatter_(-1, top_k_indices, 1.0)
        
        # Compute probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Compute load balancing loss
        # Fraction of tokens routed to each expert
        expert_load = expert_mask.mean(dim=[0, 1])  # [num_experts]
        
        # Desired load (uniform distribution)
        desired_load = torch.ones(self.num_experts, device=x.device) / self.num_experts
        
        # Load balancing loss (coefficient of variation)
        load_balance_loss = torch.std(expert_load) / (torch.mean(expert_load) + 1e-10)
        
        return router_logits, router_probs, expert_mask, load_balance_loss


class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts layer"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(config.num_experts)
        ])
        
        # Create router based on strategy
        if config.routing_strategy == RoutingStrategy.TOP_K:
            self.router = TopKRouter(config)
        elif config.routing_strategy == RoutingStrategy.SWITCH:
            self.router = SwitchRouter(config)
        elif config.routing_strategy == RoutingStrategy.LOAD_BALANCED:
            self.router = LoadBalancedRouter(config)
        else:
            self.router = TopKRouter(config)  # Default
        
        # Layer norm
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.layer_norm = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through MoE layer
        
        Args:
            x: Input tokens [batch_size, seq_len, hidden_size]
            
        Returns:
            output: Output tokens [batch_size, seq_len, hidden_size]
            aux_info: Auxiliary information (losses, statistics)
        """
        batch_size, seq_len, hidden_size = x.shape
        original_x = x
        
        # Apply layer norm
        x = self.layer_norm(x)
        
        # Route tokens to experts
        if isinstance(self.router, LoadBalancedRouter):
            router_logits, router_probs, expert_mask, load_balance_loss = self.router(x)
        else:
            router_logits, router_probs, expert_mask = self.router(x)
            load_balance_loss = None
        
        # Process through experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Get tokens assigned to this expert
            expert_assignments = expert_mask[:, :, i:i+1]  # [batch, seq_len, 1]
            
            # Apply expert to all tokens (will be masked)
            expert_output = expert(x)  # [batch, seq_len, hidden_size]
            
            # Mask output based on assignment
            expert_output = expert_output * expert_assignments
            
            # Weight by router probability
            router_weight = router_probs[:, :, i:i+1]  # [batch, seq_len, 1]
            expert_output = expert_output * router_weight
            
            expert_outputs.append(expert_output)
        
        # Combine expert outputs
        output = sum(expert_outputs)  # [batch, seq_len, hidden_size]
        
        # Residual connection
        output = output + original_x
        
        # Prepare auxiliary information
        aux_info = {
            'router_logits': router_logits,
            'router_probs': router_probs,
            'expert_mask': expert_mask,
            'expert_utilization': expert_mask.mean(dim=[0, 1])  # [num_experts]
        }
        
        if load_balance_loss is not None:
            aux_info['load_balance_loss'] = load_balance_loss
        
        # Compute auxiliary loss (expert diversity)
        aux_loss = self._compute_aux_loss(router_probs, expert_mask)
        aux_info['aux_loss'] = aux_loss
        
        return output, aux_info
    
    def _compute_aux_loss(self, router_probs: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss for expert diversity"""
        # Encourage diverse expert usage
        expert_usage = expert_mask.mean(dim=[0, 1])  # [num_experts]
        desired_usage = torch.ones_like(expert_usage) / self.config.num_experts
        
        # KL divergence between actual and desired usage
        aux_loss = F.kl_div(
            F.log_softmax(expert_usage, dim=0),
            desired_usage,
            reduction='sum'
        )
        
        return aux_loss


class SparseMoEBlock(nn.Module):
    """Complete MoE block with attention and MoE layers"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        
        # MoE layer
        self.moe = SparseMoELayer(config)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through MoE block
        
        Args:
            x: Input tokens [batch_size, seq_len, hidden_size]
            
        Returns:
            output: Output tokens
            aux_info: Auxiliary information
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.attention_norm(x + attn_out)
        
        # MoE layer
        output, aux_info = self.moe(x)
        
        return output, aux_info


class SparseMixtureOfExperts(nn.Module):
    """
    Sparse Mixture of Experts Model
    
    Implements a sparse MoE architecture where tokens are routed to a subset of experts,
    enabling efficient scaling to large numbers of parameters.
    """
    
    def __init__(self, config: MoEConfig, vocab_size: int = 30000, 
                 max_seq_len: int = 512, num_layers: int = 6):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, config.hidden_size)
        
        # MoE blocks
        self.moe_blocks = nn.ModuleList([
            SparseMoEBlock(config) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, vocab_size)
        
        # Layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through MoE model
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            aux_info: Auxiliary information from all layers
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Collect auxiliary information
        all_aux_info = []
        
        # Pass through MoE blocks
        for block in self.moe_blocks:
            x, aux_info = block(x)
            all_aux_info.append(aux_info)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Output projection
        logits = self.output_proj(x)
        
        # Aggregate auxiliary information
        aggregated_aux = self._aggregate_aux_info(all_aux_info)
        
        return logits, aggregated_aux
    
    def _aggregate_aux_info(self, all_aux_info: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate auxiliary information from all layers"""
        aggregated = {
            'total_aux_loss': sum(info.get('aux_loss', torch.tensor(0.0)) for info in all_aux_info),
            'total_load_balance_loss': sum(info.get('load_balance_loss', torch.tensor(0.0)) 
                                          for info in all_aux_info),
            'expert_utilization': torch.stack([info['expert_utilization'] for info in all_aux_info]).mean(dim=0),
            'num_layers': len(all_aux_info)
        }
        return aggregated


def create_sparse_moe(num_experts: int = 8, hidden_size: int = 512,
                     top_k: int = 2, routing_strategy: str = "top_k",
                     num_layers: int = 6) -> SparseMixtureOfExperts:
    """
    Factory function to create a sparse MoE model
    
    Args:
        num_experts: Number of experts
        hidden_size: Hidden dimension size
        top_k: Number of experts to route to
        routing_strategy: Routing strategy ("top_k", "switch", "load_balanced")
        num_layers: Number of MoE layers
        
    Returns:
        SparseMixtureOfExperts instance
    """
    config = MoEConfig(
        num_experts=num_experts,
        hidden_size=hidden_size,
        top_k=top_k,
        routing_strategy=RoutingStrategy(routing_strategy.lower())
    )
    return SparseMixtureOfExperts(config, num_layers=num_layers)

