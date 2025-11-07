"""
Continual Learning Module - Advanced continual learning strategies
Implements Elastic Weight Consolidation (EWC), Progressive Neural Networks, 
Experience Replay, and other continual learning techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque


class ContinualLearningStrategy(Enum):
    """Continual learning strategies"""
    EWC = "ewc"  # Elastic Weight Consolidation
    PROGRESSIVE = "progressive"  # Progressive Neural Networks
    REPLAY = "replay"  # Experience Replay
    GEM = "gem"  # Gradient Episodic Memory
    A_GEM = "agem"  # Averaged Gradient Episodic Memory
    PACKNET = "packnet"  # PackNet (pruning-based)
    MAS = "mas"  # Memory Aware Synapses


@dataclass
class ContinualConfig:
    """Configuration for continual learning"""
    strategy: ContinualLearningStrategy = ContinualLearningStrategy.EWC
    memory_size: int = 1000
    ewc_lambda: float = 0.4  # EWC regularization strength
    gem_margin: float = 0.5  # GEM margin
    replay_alpha: float = 0.5  # Replay mixing coefficient
    fisher_samples: int = 100  # Samples for Fisher information
    use_task_embeddings: bool = True
    task_embedding_dim: int = 64
    max_tasks: int = 10


class ExperienceReplayBuffer:
    """Experience replay buffer for continual learning"""
    
    def __init__(self, capacity: int, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
    def add(self, experience: Dict[str, torch.Tensor]):
        """Add experience to buffer"""
        # Move to device
        experience = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in experience.items()}
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Sample batch of experiences"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation (EWC) for continual learning"""
    
    def __init__(self, model: nn.Module, config: ContinualConfig):
        self.model = model
        self.config = config
        self.fisher_information = {}
        self.optimal_params = {}
        
    def compute_fisher_information(self, dataloader: torch.utils.data.DataLoader,
                                  criterion: nn.Module, device: str = "cuda"):
        """Compute Fisher information matrix for EWC"""
        self.model.eval()
        fisher = {}
        
        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
        
        # Sample from dataloader
        num_samples = 0
        for batch_idx, batch in enumerate(dataloader):
            if num_samples >= self.config.fisher_samples:
                break
            
            # Get inputs and targets
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch['inputs'], batch['targets']
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # Compute gradients
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate Fisher information (squared gradients)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            
            num_samples += inputs.size(0)
        
        # Normalize
        for name in fisher:
            fisher[name] /= num_samples
        
        self.fisher_information = fisher
        
        # Store optimal parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss"""
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and name in self.optimal_params:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
        
        return self.config.ewc_lambda * ewc_loss


class GradientEpisodicMemory:
    """Gradient Episodic Memory (GEM) for continual learning"""
    
    def __init__(self, config: ContinualConfig, memory_size: int = 100):
        self.config = config
        self.memory_size = memory_size
        self.memory = []
        self.gradients = []
        
    def add_memory(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Add example to episodic memory"""
        if len(self.memory) >= self.memory_size:
            # Remove oldest
            self.memory.pop(0)
            self.gradients.pop(0)
        
        self.memory.append((inputs, targets))
    
    def compute_gradient_constraints(self, model: nn.Module, criterion: nn.Module) -> List[torch.Tensor]:
        """Compute gradient constraints from memory"""
        if len(self.memory) == 0:
            return []
        
        constraints = []
        model.eval()
        
        for inputs, targets in self.memory:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
            constraints.append([g.clone() for g in grads])
        
        return constraints
    
    def project_gradients(self, current_grads: List[torch.Tensor],
                          constraints: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Project gradients to satisfy constraints"""
        if len(constraints) == 0:
            return current_grads
        
        # Convert to tensors
        current_vec = torch.cat([g.flatten() for g in current_grads])
        constraint_vecs = [torch.cat([c.flatten() for c in constraint]) 
                          for constraint in constraints]
        
        # Check if constraints are violated
        violated = []
        for constraint_vec in constraint_vecs:
            dot_product = (current_vec * constraint_vec).sum()
            if dot_product < 0:
                violated.append(constraint_vec)
        
        if len(violated) == 0:
            return current_grads
        
        # Project to satisfy constraints
        projected = current_vec.clone()
        for constraint_vec in violated:
            # Project onto constraint
            projected = projected - (projected * constraint_vec).sum() / (constraint_vec ** 2).sum() * constraint_vec
        
        # Reshape back to gradients
        projected_grads = []
        idx = 0
        for grad in current_grads:
            size = grad.numel()
            projected_grads.append(projected[idx:idx+size].view(grad.shape))
            idx += size
        
        return projected_grads


class ProgressiveNeuralNetwork(nn.Module):
    """Progressive Neural Network for continual learning"""
    
    def __init__(self, base_model: nn.Module, config: ContinualConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.task_columns = nn.ModuleList([base_model])  # First column is base
        self.lateral_connections = nn.ModuleList()
        
    def add_task_column(self, input_size: int, hidden_size: int, output_size: int):
        """Add a new task-specific column"""
        # Create new column (similar architecture to base)
        new_column = self._create_column(input_size, hidden_size, output_size)
        self.task_columns.append(new_column)
        
        # Create lateral connections from previous columns
        lateral = nn.ModuleList()
        for prev_column in self.task_columns[:-1]:
            lateral.append(nn.Linear(
                prev_column.hidden_size if hasattr(prev_column, 'hidden_size') else hidden_size,
                hidden_size
            ))
        self.lateral_connections.append(lateral)
    
    def _create_column(self, input_size: int, hidden_size: int, output_size: int) -> nn.Module:
        """Create a new column network"""
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass through task-specific column"""
        # Base column output
        base_output = self.task_columns[0](x)
        
        # Task-specific column
        task_output = self.task_columns[task_id + 1](x)
        
        # Lateral connections from previous columns
        lateral_output = torch.zeros_like(task_output)
        for i, lateral in enumerate(self.lateral_connections[task_id]):
            prev_output = self.task_columns[i](x)
            lateral_output += lateral(prev_output)
        
        # Combine
        output = task_output + lateral_output
        return output


class ContinualLearningModule(nn.Module):
    """
    Continual Learning Module
    
    Implements various continual learning strategies to prevent catastrophic forgetting
    when learning multiple tasks sequentially.
    """
    
    def __init__(self, model: nn.Module, config: ContinualConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.current_task = 0
        
        # Strategy-specific components
        if config.strategy == ContinualLearningStrategy.EWC:
            self.ewc = ElasticWeightConsolidation(model, config)
        elif config.strategy == ContinualLearningStrategy.GEM:
            self.gem = GradientEpisodicMemory(config, config.memory_size)
        elif config.strategy == ContinualLearningStrategy.REPLAY:
            self.replay_buffer = ExperienceReplayBuffer(config.memory_size)
        elif config.strategy == ContinualLearningStrategy.PROGRESSIVE:
            self.progressive_net = ProgressiveNeuralNetwork(model, config)
        
        # Task embeddings
        if config.use_task_embeddings:
            self.task_embeddings = nn.Embedding(config.max_tasks, config.task_embedding_dim)
            self.task_adapter = nn.Linear(
                config.task_embedding_dim + self._get_model_output_size(),
                self._get_model_output_size()
            )
        
    def _get_model_output_size(self) -> int:
        """Get output size of model"""
        # Try to infer from last layer
        for module in reversed(list(self.model.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        return 512  # Default
    
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass"""
        if self.config.strategy == ContinualLearningStrategy.PROGRESSIVE:
            task_id = task_id if task_id is not None else self.current_task
            return self.progressive_net(x, task_id)
        
        output = self.model(x)
        
        # Add task embedding if enabled
        if self.config.use_task_embeddings and task_id is not None:
            task_emb = self.task_embeddings(torch.tensor(task_id, device=x.device))
            task_emb = task_emb.unsqueeze(0).expand(x.size(0), -1)
            combined = torch.cat([output, task_emb], dim=-1)
            output = self.task_adapter(combined)
        
        return output
    
    def compute_continual_loss(self, loss: torch.Tensor, 
                              inputs: Optional[torch.Tensor] = None,
                              targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute continual learning loss"""
        total_loss = loss
        
        if self.config.strategy == ContinualLearningStrategy.EWC:
            ewc_loss = self.ewc.compute_ewc_loss()
            total_loss = total_loss + ewc_loss
        
        elif self.config.strategy == ContinualLearningStrategy.GEM:
            if inputs is not None and targets is not None:
                # Add to memory
                self.gem.add_memory(inputs, targets)
        
        return total_loss
    
    def update_after_task(self, dataloader: torch.utils.data.DataLoader,
                         criterion: nn.Module, device: str = "cuda"):
        """Update continual learning components after task completion"""
        if self.config.strategy == ContinualLearningStrategy.EWC:
            self.ewc.compute_fisher_information(dataloader, criterion, device)
        
        self.current_task += 1
    
    def get_replay_batch(self, batch_size: int) -> Optional[List[Dict[str, torch.Tensor]]]:
        """Get batch from replay buffer"""
        if self.config.strategy == ContinualLearningStrategy.REPLAY:
            return self.replay_buffer.sample(batch_size)
        return None
    
    def add_to_replay(self, experience: Dict[str, torch.Tensor]):
        """Add experience to replay buffer"""
        if self.config.strategy == ContinualLearningStrategy.REPLAY:
            self.replay_buffer.add(experience)
    
    def project_gradients(self, grads: List[torch.Tensor], 
                          criterion: nn.Module) -> List[torch.Tensor]:
        """Project gradients for GEM"""
        if self.config.strategy == ContinualLearningStrategy.GEM:
            constraints = self.gem.compute_gradient_constraints(self.model, criterion)
            return self.gem.project_gradients(grads, constraints)
        return grads


def create_continual_learner(model: nn.Module, strategy: str = "ewc",
                              memory_size: int = 1000,
                              ewc_lambda: float = 0.4) -> ContinualLearningModule:
    """
    Factory function to create a continual learning module
    
    Args:
        model: Base model to protect from forgetting
        strategy: Continual learning strategy ("ewc", "gem", "replay", "progressive")
        memory_size: Size of memory buffer
        ewc_lambda: EWC regularization strength
        
    Returns:
        ContinualLearningModule instance
    """
    config = ContinualConfig(
        strategy=ContinualLearningStrategy(strategy.lower()),
        memory_size=memory_size,
        ewc_lambda=ewc_lambda
    )
    return ContinualLearningModule(model, config)

