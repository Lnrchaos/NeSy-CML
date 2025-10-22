"""
Modular Symbolic Controller System for NeuroSym-CML
Provides multiple symbolic reasoning approaches that developers can choose from
All implementations written from scratch without external symbolic reasoning imports
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
import math
import numpy as np

class BaseSymbolicController(ABC, nn.Module):
    """Abstract base class for all symbolic controllers"""
    
    def __init__(self, num_rules: int, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.num_rules = num_rules
        self.input_size = input_size
        self.hidden_size = hidden_size
    
    @abstractmethod
    def forward(self, x: torch.Tensor, task_metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """Generate rule indices and symbolic state"""
        pass
    
    @abstractmethod
    def get_controller_type(self) -> str:
        """Return the type of symbolic controller"""
        pass

class LogicBasedController(BaseSymbolicController):
    """
    Logic-based symbolic controller using propositional logic rules
    Implements basic logical operations: AND, OR, NOT, IMPLIES
    """
    
    def __init__(self, num_rules: int, input_size: int, hidden_size: int = 64):
        super().__init__(num_rules, input_size, hidden_size)
        
        # Feature extraction for logical predicates
        self.predicate_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh()
        )
        
        # Logical operation networks
        self.and_gate = nn.Linear(hidden_size, hidden_size // 2)
        self.or_gate = nn.Linear(hidden_size, hidden_size // 2)
        self.not_gate = nn.Linear(hidden_size, hidden_size // 2)
        self.implies_gate = nn.Linear(hidden_size, hidden_size // 2)
        
        # Rule activation network
        self.rule_activator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_rules),
            nn.Softmax(dim=-1)
        )
        
        # Logic state tracker
        self.logic_state = nn.Parameter(torch.randn(1, hidden_size))
    
    def _apply_logical_operations(self, predicates: torch.Tensor) -> torch.Tensor:
        """Apply logical operations to predicates"""
        batch_size = predicates.size(0)
        
        # Apply logical gates
        and_result = torch.sigmoid(self.and_gate(predicates))
        or_result = torch.sigmoid(self.or_gate(predicates))
        not_result = torch.sigmoid(self.not_gate(predicates))
        implies_result = torch.sigmoid(self.implies_gate(predicates))
        
        # Combine logical operations
        logic_features = torch.cat([and_result, or_result, not_result, implies_result], dim=-1)
        
        return logic_features
    
    def forward(self, x: torch.Tensor, task_metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.size(0)
        
        # Extract logical predicates from input
        predicates = self.predicate_extractor(x)
        
        # Apply logical operations
        logic_features = self._apply_logical_operations(predicates)
        
        # Combine with logic state
        logic_state_expanded = self.logic_state.expand(batch_size, -1)
        combined_features = torch.cat([logic_features, logic_state_expanded], dim=-1)
        
        # Generate rule activations
        rule_probabilities = self.rule_activator(combined_features)
        
        # Select top rules (convert probabilities to indices)
        rule_indices = torch.multinomial(rule_probabilities, num_samples=1).squeeze(-1)
        
        symbolic_state = {
            'predicates': predicates,
            'logic_features': logic_features,
            'rule_probabilities': rule_probabilities,
            'controller_type': 'logic_based'
        }
        
        return rule_indices, symbolic_state
    
    def get_controller_type(self) -> str:
        return "logic_based"

class FuzzyLogicController(BaseSymbolicController):
    """
    Fuzzy logic-based symbolic controller
    Implements fuzzy sets, membership functions, and fuzzy inference
    """
    
    def __init__(self, num_rules: int, input_size: int, hidden_size: int = 64, num_fuzzy_sets: int = 5):
        super().__init__(num_rules, input_size, hidden_size)
        self.num_fuzzy_sets = num_fuzzy_sets
        
        # Fuzzy membership function parameters
        self.fuzzy_centers = nn.Parameter(torch.randn(num_fuzzy_sets, hidden_size))
        self.fuzzy_widths = nn.Parameter(torch.ones(num_fuzzy_sets, hidden_size))
        
        # Feature extraction for fuzzification
        self.fuzzifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Fuzzy rule base
        self.fuzzy_rules = nn.Parameter(torch.randn(num_rules, num_fuzzy_sets))
        
        # Defuzzification network
        self.defuzzifier = nn.Sequential(
            nn.Linear(num_fuzzy_sets, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_rules),
            nn.Softmax(dim=-1)
        )
    
    def _gaussian_membership(self, x: torch.Tensor, center: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian membership function"""
        return torch.exp(-0.5 * ((x.unsqueeze(1) - center.unsqueeze(0)) / width.unsqueeze(0)) ** 2)
    
    def _triangular_membership(self, x: torch.Tensor, center: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        """Compute triangular membership function"""
        distance = torch.abs(x.unsqueeze(1) - center.unsqueeze(0))
        membership = torch.clamp(1.0 - distance / width.unsqueeze(0), min=0.0)
        return membership
    
    def forward(self, x: torch.Tensor, task_metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.size(0)
        
        # Fuzzify input
        fuzzy_input = self.fuzzifier(x)
        
        # Compute membership degrees using Gaussian membership functions
        membership_degrees = self._gaussian_membership(
            fuzzy_input, self.fuzzy_centers, torch.abs(self.fuzzy_widths)
        )
        
        # Aggregate membership degrees across features
        aggregated_membership = torch.mean(membership_degrees, dim=-1)  # [batch_size, num_fuzzy_sets]
        
        # Apply fuzzy inference (simplified Mamdani inference)
        # Ensure compatible shapes for matrix multiplication
        if aggregated_membership.dim() == 1:
            aggregated_membership = aggregated_membership.unsqueeze(0)
        
        rule_strengths = torch.matmul(aggregated_membership, self.fuzzy_rules.T)  # [batch_size, num_rules]
        
        # Defuzzification
        rule_probabilities = self.defuzzifier(aggregated_membership)
        
        # Select rules based on fuzzy inference
        rule_indices = torch.multinomial(rule_probabilities, num_samples=1).squeeze(-1)
        
        symbolic_state = {
            'fuzzy_input': fuzzy_input,
            'membership_degrees': membership_degrees,
            'aggregated_membership': aggregated_membership,
            'rule_strengths': rule_strengths,
            'rule_probabilities': rule_probabilities,
            'controller_type': 'fuzzy_logic'
        }
        
        return rule_indices, symbolic_state
    
    def get_controller_type(self) -> str:
        return "fuzzy_logic"

class ProductionRuleController(BaseSymbolicController):
    """
    Production rule-based symbolic controller
    Implements IF-THEN rules with forward and backward chaining
    """
    
    def __init__(self, num_rules: int, input_size: int, hidden_size: int = 64):
        super().__init__(num_rules, input_size, hidden_size)
        
        # Condition evaluation network
        self.condition_evaluator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Rule condition matrices (IF part)
        self.rule_conditions = nn.Parameter(torch.randn(num_rules, hidden_size))
        self.condition_thresholds = nn.Parameter(torch.ones(num_rules) * 0.5)
        
        # Rule action matrices (THEN part)
        self.rule_actions = nn.Parameter(torch.randn(num_rules, hidden_size))
        
        # Working memory for rule chaining
        self.working_memory = nn.Parameter(torch.zeros(1, hidden_size))
        
        # Rule priority weights
        self.rule_priorities = nn.Parameter(torch.ones(num_rules))
    
    def _evaluate_conditions(self, features: torch.Tensor) -> torch.Tensor:
        """Evaluate rule conditions"""
        batch_size = features.size(0)
        
        # Compute similarity between features and rule conditions
        # Ensure compatible shapes for matrix multiplication
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        condition_similarities = torch.matmul(features, self.rule_conditions.T)  # [batch_size, num_rules]
        
        # Apply thresholds
        condition_met = torch.sigmoid(condition_similarities - self.condition_thresholds.unsqueeze(0))
        
        return condition_met
    
    def _forward_chaining(self, condition_activations: torch.Tensor) -> torch.Tensor:
        """Apply forward chaining inference"""
        batch_size = condition_activations.size(0)
        
        # Apply rule priorities
        weighted_activations = condition_activations * self.rule_priorities.unsqueeze(0)
        
        # Select rules that fire (above threshold)
        firing_threshold = 0.5
        firing_rules = (weighted_activations > firing_threshold).float()
        
        return firing_rules
    
    def forward(self, x: torch.Tensor, task_metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.size(0)
        
        # Extract features for condition evaluation
        condition_features = self.condition_evaluator(x)
        
        # Evaluate rule conditions
        condition_activations = self._evaluate_conditions(condition_features)
        
        # Apply forward chaining
        firing_rules = self._forward_chaining(condition_activations)
        
        # Update working memory
        # Ensure compatible shapes for matrix multiplication
        if firing_rules.dim() == 1:
            firing_rules = firing_rules.unsqueeze(0)
        
        memory_update = torch.matmul(firing_rules, self.rule_actions)
        updated_memory = self.working_memory + torch.mean(memory_update, dim=0, keepdim=True)
        
        # Select most activated rule
        rule_scores = condition_activations * self.rule_priorities.unsqueeze(0)
        rule_indices = torch.argmax(rule_scores, dim=-1)
        
        symbolic_state = {
            'condition_features': condition_features,
            'condition_activations': condition_activations,
            'firing_rules': firing_rules,
            'working_memory': updated_memory,
            'rule_scores': rule_scores,
            'controller_type': 'production_rule'
        }
        
        return rule_indices, symbolic_state
    
    def get_controller_type(self) -> str:
        return "production_rule"

class NeuroSymbolicController(BaseSymbolicController):
    """
    Neuro-symbolic controller that combines neural networks with symbolic reasoning
    Implements differentiable symbolic operations
    """
    
    def __init__(self, num_rules: int, input_size: int, hidden_size: int = 64):
        super().__init__(num_rules, input_size, hidden_size)
        
        # Neural feature extraction
        self.neural_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(0.1)
        )
        
        # Symbolic reasoning modules
        self.symbol_embeddings = nn.Parameter(torch.randn(num_rules, hidden_size))
        self.relation_matrix = nn.Parameter(torch.randn(num_rules, num_rules))
        
        # Attention mechanism for symbol selection
        self.symbol_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        
        # Differentiable symbolic operations
        self.symbolic_ops = nn.ModuleDict({
            'compose': nn.Linear(hidden_size * 2, hidden_size),
            'abstract': nn.Linear(hidden_size, hidden_size),
            'instantiate': nn.Linear(hidden_size, hidden_size),
            'generalize': nn.Linear(hidden_size, hidden_size)
        })
        
        # Rule synthesis network
        self.rule_synthesizer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_rules),
            nn.Softmax(dim=-1)
        )
    
    def _differentiable_symbolic_ops(self, features: torch.Tensor, symbols: torch.Tensor) -> torch.Tensor:
        """Apply differentiable symbolic operations"""
        
        # Composition: combine features with symbols
        composed = self.symbolic_ops['compose'](torch.cat([features, symbols], dim=-1))
        
        # Abstraction: extract abstract patterns
        abstracted = self.symbolic_ops['abstract'](composed)
        
        # Instantiation: ground abstract concepts
        instantiated = self.symbolic_ops['instantiate'](abstracted)
        
        # Generalization: create general rules
        generalized = self.symbolic_ops['generalize'](instantiated)
        
        return generalized
    
    def forward(self, x: torch.Tensor, task_metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.size(0)
        
        # Neural encoding
        neural_features = self.neural_encoder(x)
        
        # Symbol attention
        symbols = self.symbol_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        attended_symbols, attention_weights = self.symbol_attention(
            neural_features.unsqueeze(1), symbols, symbols
        )
        attended_symbols = attended_symbols.squeeze(1)
        
        # Apply differentiable symbolic operations
        symbolic_features = self._differentiable_symbolic_ops(neural_features, attended_symbols)
        
        # Synthesize rules
        rule_probabilities = self.rule_synthesizer(symbolic_features)
        
        # Select rules
        rule_indices = torch.multinomial(rule_probabilities, num_samples=1).squeeze(-1)
        
        symbolic_state = {
            'neural_features': neural_features,
            'attended_symbols': attended_symbols,
            'attention_weights': attention_weights,
            'symbolic_features': symbolic_features,
            'rule_probabilities': rule_probabilities,
            'controller_type': 'neuro_symbolic'
        }
        
        return rule_indices, symbolic_state
    
    def get_controller_type(self) -> str:
        return "neuro_symbolic"

class GraphBasedController(BaseSymbolicController):
    """
    Graph-based symbolic controller using graph neural networks for symbolic reasoning
    Implements graph traversal and reasoning over symbolic knowledge graphs
    """
    
    def __init__(self, num_rules: int, input_size: int, hidden_size: int = 64, num_graph_layers: int = 3):
        super().__init__(num_rules, input_size, hidden_size)
        self.num_graph_layers = num_graph_layers
        
        # Node feature transformation
        self.node_encoder = nn.Linear(input_size, hidden_size)
        
        # Graph neural network layers
        self.graph_layers = nn.ModuleList([
            GraphLayer(hidden_size) for _ in range(num_graph_layers)
        ])
        
        # Edge type embeddings
        self.edge_types = nn.Parameter(torch.randn(5, hidden_size))  # 5 edge types
        
        # Graph attention
        self.graph_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        
        # Rule graph adjacency (learnable)
        self.rule_adjacency = nn.Parameter(torch.randn(num_rules, num_rules))
        
        # Graph readout
        self.graph_readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_rules),
            nn.Softmax(dim=-1)
        )
    
    def _create_rule_graph(self, batch_size: int) -> torch.Tensor:
        """Create rule graph structure"""
        # Normalize adjacency matrix
        adjacency = torch.sigmoid(self.rule_adjacency)
        
        # Expand for batch
        batch_adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        
        return batch_adjacency
    
    def forward(self, x: torch.Tensor, task_metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.size(0)
        
        # Encode input as initial node features
        node_features = self.node_encoder(x).unsqueeze(1).expand(-1, self.num_rules, -1)
        
        # Create rule graph
        adjacency = self._create_rule_graph(batch_size)
        
        # Apply graph neural network layers
        graph_features = node_features
        layer_outputs = []
        
        for layer in self.graph_layers:
            graph_features = layer(graph_features, adjacency)
            layer_outputs.append(graph_features)
        
        # Graph attention pooling
        pooled_features, attention_weights = self.graph_attention(
            graph_features, graph_features, graph_features
        )
        
        # Global graph representation
        global_graph_features = torch.mean(pooled_features, dim=1)
        
        # Generate rule probabilities
        rule_probabilities = self.graph_readout(global_graph_features)
        
        # Select rules
        rule_indices = torch.multinomial(rule_probabilities, num_samples=1).squeeze(-1)
        
        symbolic_state = {
            'node_features': node_features,
            'graph_features': graph_features,
            'adjacency': adjacency,
            'layer_outputs': layer_outputs,
            'attention_weights': attention_weights,
            'global_features': global_graph_features,
            'rule_probabilities': rule_probabilities,
            'controller_type': 'graph_based'
        }
        
        return rule_indices, symbolic_state
    
    def get_controller_type(self) -> str:
        return "graph_based"

class GraphLayer(nn.Module):
    """Graph neural network layer for graph-based controller"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.message_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, feature_dim = node_features.shape
        
        # Message passing
        messages = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_weight = adjacency[:, i, j].unsqueeze(-1)
                    message_input = torch.cat([node_features[:, i], node_features[:, j]], dim=-1)
                    message = self.message_net(message_input) * edge_weight
                    messages.append(message)
        
        # Aggregate messages
        if messages:
            aggregated_messages = torch.stack(messages, dim=1)
            aggregated_messages = torch.sum(aggregated_messages.view(batch_size, num_nodes, -1, feature_dim), dim=2)
        else:
            aggregated_messages = torch.zeros_like(node_features)
        
        # Update node features
        update_input = torch.cat([node_features, aggregated_messages], dim=-1)
        updated_features = self.update_net(update_input)
        
        # Residual connection and layer norm
        output = self.layer_norm(node_features + updated_features)
        
        return output

class ModularSymbolicControllerFactory:
    """Factory class for creating different types of symbolic controllers"""
    
    CONTROLLER_TYPES = {
        'logic_based': LogicBasedController,
        'fuzzy_logic': FuzzyLogicController,
        'production_rule': ProductionRuleController,
        'neuro_symbolic': NeuroSymbolicController,
        'graph_based': GraphBasedController
    }
    
    @classmethod
    def create_controller(cls, controller_type: str, num_rules: int, input_size: int, 
                         hidden_size: int = 64, **kwargs) -> BaseSymbolicController:
        """
        Create a symbolic controller of the specified type
        
        Args:
            controller_type: Type of controller ('logic_based', 'fuzzy_logic', etc.)
            num_rules: Number of symbolic rules
            input_size: Size of input features
            hidden_size: Hidden layer size
            **kwargs: Additional controller-specific parameters
        
        Returns:
            Symbolic controller instance
        """
        if controller_type not in cls.CONTROLLER_TYPES:
            raise ValueError(f"Unknown controller type: {controller_type}. "
                           f"Available types: {list(cls.CONTROLLER_TYPES.keys())}")
        
        controller_class = cls.CONTROLLER_TYPES[controller_type]
        
        # Pass additional kwargs to specific controllers
        if controller_type == 'fuzzy_logic':
            return controller_class(num_rules, input_size, hidden_size, 
                                  num_fuzzy_sets=kwargs.get('num_fuzzy_sets', 5))
        elif controller_type == 'graph_based':
            return controller_class(num_rules, input_size, hidden_size,
                                  num_graph_layers=kwargs.get('num_graph_layers', 3))
        else:
            return controller_class(num_rules, input_size, hidden_size)
    
    @classmethod
    def list_available_controllers(cls) -> List[str]:
        """List all available controller types"""
        return list(cls.CONTROLLER_TYPES.keys())
    
    @classmethod
    def get_controller_info(cls, controller_type: str) -> Dict[str, str]:
        """Get information about a specific controller type"""
        info = {
            'logic_based': {
                'description': 'Logic-based controller using propositional logic (AND, OR, NOT, IMPLIES)',
                'best_for': 'Rule-based reasoning, logical inference tasks',
                'complexity': 'Medium'
            },
            'fuzzy_logic': {
                'description': 'Fuzzy logic controller with membership functions and fuzzy inference',
                'best_for': 'Uncertain reasoning, continuous rule activation',
                'complexity': 'Medium'
            },
            'production_rule': {
                'description': 'Production rule system with IF-THEN rules and forward chaining',
                'best_for': 'Expert systems, procedural knowledge',
                'complexity': 'Low'
            },
            'neuro_symbolic': {
                'description': 'Hybrid controller combining neural networks with symbolic operations',
                'best_for': 'Learning symbolic patterns, differentiable reasoning',
                'complexity': 'High'
            },
            'graph_based': {
                'description': 'Graph neural network for reasoning over symbolic knowledge graphs',
                'best_for': 'Relational reasoning, structured knowledge',
                'complexity': 'High'
            }
        }
        
        return info.get(controller_type, {'description': 'Unknown controller type'})

# Convenience function for easy controller creation
def create_symbolic_controller(controller_type: str, num_rules: int, input_size: int, 
                             hidden_size: int = 64, **kwargs) -> BaseSymbolicController:
    """
    Convenience function to create a symbolic controller
    
    Example usage:
        controller = create_symbolic_controller('fuzzy_logic', 100, 512, num_fuzzy_sets=7)
    """
    return ModularSymbolicControllerFactory.create_controller(
        controller_type, num_rules, input_size, hidden_size, **kwargs
    )

# Example usage and testing
if __name__ == "__main__":
    print("🧠 Modular Symbolic Controller System")
    print("=" * 50)
    
    # List available controllers
    print("Available Controllers:")
    for controller_type in ModularSymbolicControllerFactory.list_available_controllers():
        info = ModularSymbolicControllerFactory.get_controller_info(controller_type)
        print(f"  • {controller_type}: {info['description']}")
    
    print("\n" + "=" * 50)
    
    # Test each controller type
    batch_size = 4
    input_size = 512
    num_rules = 100
    hidden_size = 64
    
    test_input = torch.randn(batch_size, input_size)
    
    for controller_type in ModularSymbolicControllerFactory.list_available_controllers():
        print(f"\nTesting {controller_type} controller...")
        
        try:
            controller = create_symbolic_controller(controller_type, num_rules, input_size, hidden_size)
            rule_indices, symbolic_state = controller(test_input)
            
            print(f"  ✅ {controller_type}: Output shape {rule_indices.shape}")
            print(f"     Symbolic state keys: {list(symbolic_state.keys())}")
            
        except Exception as e:
            print(f"  ❌ {controller_type}: Error - {e}")
    
    print(f"\n🎉 Modular symbolic controller system ready!")