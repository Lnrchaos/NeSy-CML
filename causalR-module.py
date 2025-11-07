"""
Causal Reasoning Module - Advanced causal inference and reasoning
Implements causal discovery, counterfactual reasoning, and causal effect estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any, List, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np


class CausalRelationType(Enum):
    """Types of causal relations"""
    DIRECT = "direct"  # Direct causal effect
    INDIRECT = "indirect"  # Indirect causal effect through mediator
    CONFOUNDED = "confounded"  # Confounded by common cause
    COLLIDER = "collider"  # Collider structure


@dataclass
class CausalConfig:
    """Configuration for causal reasoning module"""
    hidden_size: int = 512
    num_causal_factors: int = 10
    num_layers: int = 3
    use_attention: bool = True
    use_graph_structure: bool = True
    dropout: float = 0.1
    temperature: float = 1.0  # Temperature for causal strength


class CausalGraph(nn.Module):
    """Learnable causal graph structure"""
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        self.num_factors = config.num_causal_factors
        
        # Adjacency matrix (learnable)
        self.adjacency = nn.Parameter(
            torch.randn(config.num_causal_factors, config.num_causal_factors) * 0.1
        )
        
        # Edge strengths
        self.edge_strengths = nn.Parameter(
            torch.ones(config.num_causal_factors, config.num_causal_factors) * 0.5
        )
        
        # Causal mechanisms (MLPs for each edge)
        self.mechanisms = nn.ModuleDict()
        for i in range(config.num_causal_factors):
            for j in range(config.num_causal_factors):
                if i != j:
                    key = f"{i}_{j}"
                    self.mechanisms[key] = nn.Sequential(
                        nn.Linear(config.hidden_size, config.hidden_size),
                        nn.LayerNorm(config.hidden_size),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.hidden_size, config.hidden_size)
                    )
        
    def forward(self, factors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply causal graph structure
        
        Args:
            factors: Causal factors [batch_size, num_factors, hidden_size]
            
        Returns:
            updated_factors: Updated factors after causal propagation
            adjacency_weights: Learned adjacency weights
        """
        batch_size, num_factors, hidden_size = factors.shape
        
        # Get adjacency weights (sigmoid to keep in [0, 1])
        adjacency_weights = torch.sigmoid(self.adjacency) * torch.sigmoid(self.edge_strengths)
        
        # Apply causal mechanisms
        updated_factors = factors.clone()
        
        for i in range(num_factors):
            causal_effects = []
            for j in range(num_factors):
                if i != j:
                    key = f"{j}_{i}"
                    if key in self.mechanisms:
                        # Compute causal effect from j to i
                        effect = self.mechanisms[key](factors[:, j, :])  # [batch, hidden_size]
                        weight = adjacency_weights[j, i]
                        causal_effects.append(weight * effect)
            
            if causal_effects:
                # Aggregate causal effects
                total_effect = sum(causal_effects)
                updated_factors[:, i, :] = updated_factors[:, i, :] + total_effect
        
        return updated_factors, adjacency_weights


class CausalAttention(nn.Module):
    """Attention mechanism for causal reasoning"""
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Causal mask (lower triangular)
        self.register_buffer('causal_mask', None)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply causal attention
        
        Args:
            x: Input [batch_size, seq_len, hidden_size]
            mask: Optional attention mask
            
        Returns:
            output: Attended output
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Create causal mask if not exists
        if self.causal_mask is None or self.causal_mask.shape[0] != seq_len:
            self.causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            self.causal_mask = self.causal_mask.to(x.device)
        
        # Apply causal mask
        if mask is not None:
            mask = mask & ~self.causal_mask
        else:
            mask = ~self.causal_mask
        
        # Attention
        attn_out, attn_weights = self.attention(x, x, x, attn_mask=mask)
        
        # Residual and norm
        output = self.layer_norm(x + attn_out)
        
        return output


class CounterfactualReasoner(nn.Module):
    """Counterfactual reasoning module"""
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        
        # Intervention network
        self.intervention_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Abduction network (inferring unobserved causes)
        self.abduction_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Action network (applying interventions)
        self.action_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
    def forward(self, observed: torch.Tensor, intervention: Optional[torch.Tensor] = None,
                intervention_target: Optional[int] = None) -> torch.Tensor:
        """
        Perform counterfactual reasoning
        
        Args:
            observed: Observed state [batch_size, num_factors, hidden_size]
            intervention: Intervention value [batch_size, hidden_size]
            intervention_target: Index of factor to intervene on
            
        Returns:
            counterfactual: Counterfactual state
        """
        batch_size, num_factors, hidden_size = observed.shape
        
        # Abduction: infer unobserved causes
        abducted = self.abduction_net(observed)
        
        # Action: apply intervention if provided
        if intervention is not None and intervention_target is not None:
            # Prepare intervention
            intervention_expanded = intervention.unsqueeze(1).expand(-1, num_factors, -1)
            combined = torch.cat([abducted, intervention_expanded], dim=-1)
            intervened = self.action_net(combined)
            
            # Apply intervention to target factor
            intervened[:, intervention_target, :] = intervention
        else:
            intervened = abducted
        
        # Prediction: compute counterfactual outcome
        counterfactual = self.intervention_net(
            torch.cat([observed, intervened], dim=-1)
        )
        
        return counterfactual


class CausalEffectEstimator(nn.Module):
    """Estimate causal effects (ATE, ITE, etc.)"""
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        
        # Treatment effect network
        self.treatment_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Outcome prediction
        self.outcome_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1)  # Single outcome
        )
        
    def forward(self, factors: torch.Tensor, treatment: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estimate causal effects
        
        Args:
            factors: Causal factors [batch_size, num_factors, hidden_size]
            treatment: Treatment indicator [batch_size, hidden_size]
            
        Returns:
            Dictionary with:
                - ate: Average Treatment Effect
                - ite: Individual Treatment Effect
                - outcome_treated: Outcome under treatment
                - outcome_control: Outcome under control
        """
        batch_size = factors.shape[0]
        
        # Compute outcomes under treatment and control
        treatment_expanded = treatment.unsqueeze(1).expand(-1, self.config.num_causal_factors, -1)
        control_expanded = torch.zeros_like(treatment_expanded)
        
        # Treatment effect
        treated_factors = self.treatment_net(torch.cat([factors, treatment_expanded], dim=-1))
        control_factors = self.treatment_net(torch.cat([factors, control_expanded], dim=-1))
        
        # Aggregate factors for outcome prediction
        treated_aggregated = treated_factors.mean(dim=1)  # [batch, hidden_size]
        control_aggregated = control_factors.mean(dim=1)
        
        # Predict outcomes
        outcome_treated = self.outcome_net(treated_aggregated)  # [batch, 1]
        outcome_control = self.outcome_net(control_aggregated)  # [batch, 1]
        
        # Compute effects
        ite = outcome_treated - outcome_control  # Individual Treatment Effect
        ate = ite.mean()  # Average Treatment Effect
        
        return {
            'ate': ate,
            'ite': ite.squeeze(-1),  # [batch]
            'outcome_treated': outcome_treated.squeeze(-1),
            'outcome_control': outcome_control.squeeze(-1)
        }


class CausalReasoningModule(nn.Module):
    """
    Causal Reasoning Module
    
    Implements causal discovery, counterfactual reasoning, and causal effect estimation
    using neural networks and graph structures.
    """
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        
        # Factor extraction
        self.factor_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size * config.num_causal_factors)
        )
        
        # Causal graph
        if config.use_graph_structure:
            self.causal_graph = CausalGraph(config)
        
        # Causal attention
        if config.use_attention:
            self.causal_attention = CausalAttention(config)
        
        # Counterfactual reasoner
        self.counterfactual_reasoner = CounterfactualReasoner(config)
        
        # Causal effect estimator
        self.effect_estimator = CausalEffectEstimator(config)
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, x: torch.Tensor, 
                intervention: Optional[torch.Tensor] = None,
                intervention_target: Optional[int] = None,
                treatment: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through causal reasoning module
        
        Args:
            x: Input features [batch_size, seq_len, hidden_size]
            intervention: Optional intervention value
            intervention_target: Optional intervention target index
            treatment: Optional treatment indicator for effect estimation
            
        Returns:
            Dictionary with:
                - output: Causal reasoning output
                - factors: Extracted causal factors
                - adjacency: Learned causal graph adjacency
                - counterfactual: Counterfactual state (if intervention provided)
                - effects: Causal effects (if treatment provided)
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Extract causal factors
        factor_flat = self.factor_extractor(x.mean(dim=1))  # [batch, hidden_size * num_factors]
        factors = factor_flat.view(batch_size, self.config.num_causal_factors, hidden_size)
        
        # Apply causal attention if enabled
        if self.config.use_attention:
            factors = self.causal_attention(factors)
        
        # Apply causal graph structure if enabled
        if self.config.use_graph_structure:
            factors, adjacency = self.causal_graph(factors)
        else:
            adjacency = None
        
        # Counterfactual reasoning if intervention provided
        counterfactual = None
        if intervention is not None:
            counterfactual = self.counterfactual_reasoner(
                factors, intervention, intervention_target
            )
        
        # Causal effect estimation if treatment provided
        effects = None
        if treatment is not None:
            effects = self.effect_estimator(factors, treatment)
        
        # Output projection
        output = self.output_proj(factors.mean(dim=1))  # [batch, hidden_size]
        
        result = {
            'output': output,
            'factors': factors,
            'adjacency': adjacency
        }
        
        if counterfactual is not None:
            result['counterfactual'] = counterfactual
        
        if effects is not None:
            result['effects'] = effects
        
        return result
    
    def discover_causal_structure(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discover causal structure from data
        
        Args:
            x: Input data [batch_size, seq_len, hidden_size]
            
        Returns:
            adjacency: Discovered causal graph adjacency matrix
        """
        with torch.no_grad():
            result = self.forward(x)
            return result['adjacency']
    
    def estimate_ate(self, x: torch.Tensor, treatment: torch.Tensor) -> float:
        """
        Estimate Average Treatment Effect
        
        Args:
            x: Input features
            treatment: Treatment indicator
            
        Returns:
            ATE value
        """
        with torch.no_grad():
            result = self.forward(x, treatment=treatment)
            return result['effects']['ate'].item()
    
    def compute_counterfactual(self, x: torch.Tensor, intervention: torch.Tensor,
                               intervention_target: int) -> torch.Tensor:
        """
        Compute counterfactual outcome
        
        Args:
            x: Observed state
            intervention: Intervention value
            intervention_target: Target factor index
            
        Returns:
            Counterfactual state
        """
        with torch.no_grad():
            result = self.forward(x, intervention=intervention, 
                               intervention_target=intervention_target)
            return result['counterfactual']


def create_causal_reasoner(hidden_size: int = 512, num_causal_factors: int = 10,
                          use_graph_structure: bool = True,
                          use_attention: bool = True) -> CausalReasoningModule:
    """
    Factory function to create a causal reasoning module
    
    Args:
        hidden_size: Hidden dimension size
        num_causal_factors: Number of causal factors
        use_graph_structure: Whether to use learnable causal graph
        use_attention: Whether to use causal attention
        
    Returns:
        CausalReasoningModule instance
    """
    config = CausalConfig(
        hidden_size=hidden_size,
        num_causal_factors=num_causal_factors,
        use_graph_structure=use_graph_structure,
        use_attention=use_attention
    )
    return CausalReasoningModule(config)

