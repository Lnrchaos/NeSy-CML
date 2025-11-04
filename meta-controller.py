import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

class MetaLearningType(Enum):
    MAML = "maml"
    REPTILE = "reptile"
    PROTO_NET = "proto_net"
    ANIL = "anil"  # Almost No Inner Loop
    META_SGD = "meta_sgd"
    META_CURVATURE = "meta_curvature"

@dataclass
class MetaLearnerConfig:
    inner_lr: float = 0.01
    meta_lr: float = 0.001
    num_inner_steps: int = 1
    first_order: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    adaptation_steps: int = 1
    adaptation_lr: Optional[float] = None
    learn_inner_lr: bool = False
    use_multi_step_loss: bool = False

class MetaLearner(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        config: MetaLearnerConfig,
        meta_learning_type: MetaLearningType = MetaLearningType.MAML
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.meta_learning_type = meta_learning_type
        self.inner_optimizer = None
        self.meta_optimizer = None
        self.device = torch.device(config.device)
        
        # Initialize inner learning rates if learning them
        if config.learn_inner_lr:
            self.inner_lrs = nn.ParameterDict({
                name: nn.Parameter(torch.ones_like(p) * config.inner_lr)
                for name, p in self.model.named_parameters() if p.requires_grad
            })
        else:
            self.inner_lrs = {name: config.inner_lr for name, _ in self.model.named_parameters()}
        
        # Special initialization for Meta-SGD
        if meta_learning_type == MetaLearningType.META_SGD:
            self.meta_optimizer = optim.SGD([
                {'params': self.model.parameters()},
                {'params': self.inner_lrs.parameters() if config.learn_inner_lr else []}
            ], lr=config.meta_lr)
        else:
            self.meta_optimizer = optim.Adam(
                list(self.model.parameters()) + 
                (list(self.inner_lrs.parameters()) if config.learn_inner_lr else []),
                lr=config.meta_lr
            )
        
        # Special handling for Meta-Curvature
        if meta_learning_type == MetaLearningType.META_CURVATURE:
            self.metric_tensors = nn.ParameterDict({
                name: nn.Parameter(torch.eye(p.numel(), device=self.device))
                for name, p in self.model.named_parameters() if p.requires_grad
            })

    def forward(self, support_set: Tuple[torch.Tensor, torch.Tensor], 
               query_set: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
               return_metrics: bool = False):
        """
        Args:
            support_set: Tuple of (support_inputs, support_targets)
            query_set: Optional tuple of (query_inputs, query_targets)
            return_metrics: If True, returns additional metrics
        """
        if self.meta_learning_type == MetaLearningType.MAML:
            return self._maml_forward(support_set, query_set, return_metrics)
        elif self.meta_learning_type == MetaLearningType.REPTILE:
            return self._reptile_forward(support_set, query_set, return_metrics)
        elif self.meta_learning_type == MetaLearningType.PROTO_NET:
            return self._protonet_forward(support_set, query_set, return_metrics)
        elif self.meta_learning_type == MetaLearningType.ANIL:
            return self._anil_forward(support_set, query_set, return_metrics)
        elif self.meta_learning_type == MetaLearningType.META_SGD:
            return self._meta_sgd_forward(support_set, query_set, return_metrics)
        elif self.meta_learning_type == MetaLearningType.META_CURVATURE:
            return self._meta_curvature_forward(support_set, query_set, return_metrics)
        else:
            raise ValueError(f"Unsupported meta-learning type: {self.meta_learning_type}")

    def _maml_forward(self, support_set, query_set, return_metrics):
        """Model-Agnostic Meta-Learning (MAML)"""
        support_inputs, support_targets = support_set
        fast_weights = {n: p.clone() for n, p in self.model.named_parameters()}
        
        # Inner loop adaptation
        for _ in range(self.config.num_inner_steps):
            outputs = self._forward_with_params(support_inputs, fast_weights)
            loss = F.cross_entropy(outputs, support_targets)
            
            grads = torch.autograd.grad(
                loss, 
                fast_weights.values(), 
                create_graph=not self.config.first_order
            )
            
            # Update fast weights
            fast_weights = {
                n: w - self.inner_lrs.get(n, self.config.inner_lr) * g
                for (n, w), g in zip(fast_weights.items(), grads)
            }
        
        # Evaluate on query set
        if query_set is not None:
            query_inputs, query_targets = query_set
            query_outputs = self._forward_with_params(query_inputs, fast_weights)
            query_loss = F.cross_entropy(query_outputs, query_targets)
            query_acc = (query_outputs.argmax(dim=1) == query_targets).float().mean()
            
            if return_metrics:
                return {
                    'loss': query_loss,
                    'accuracy': query_acc,
                    'fast_weights': fast_weights
                }
            return query_loss
        
        return None

    def _reptile_forward(self, support_set, query_set, return_metrics):
        """Reptile meta-learning algorithm"""
        support_inputs, support_targets = support_set
        fast_weights = {n: p.clone() for n, p in self.model.named_parameters()}
        
        # Inner loop adaptation
        for _ in range(self.config.num_inner_steps):
            outputs = self._forward_with_params(support_inputs, fast_weights)
            loss = F.cross_entropy(outputs, support_targets)
            
            grads = torch.autograd.grad(loss, fast_weights.values())
            fast_weights = {
                n: w - self.config.inner_lr * g
                for (n, w), g in zip(fast_weights.items(), grads)
            }
        
        # Reptile update
        self.reptile_update(fast_weights)
        
        # Evaluate on query set
        if query_set is not None:
            query_inputs, query_targets = query_set
            with torch.no_grad():
                query_outputs = self.model(query_inputs)
                query_loss = F.cross_entropy(query_outputs, query_targets)
                query_acc = (query_outputs.argmax(dim=1) == query_targets).float().mean()
            
            if return_metrics:
                return {
                    'loss': query_loss,
                    'accuracy': query_acc
                }
            return query_loss
        
        return None

    def _protonet_forward(self, support_set, query_set, return_metrics):
        """Prototypical Networks for Few-shot Learning"""
        support_inputs, support_targets = support_set
        query_inputs, query_targets = query_set
        
        # Get embeddings
        support_embeddings = self.model.forward_features(support_inputs)
        query_embeddings = self.model.forward_features(query_inputs)
        
        # Calculate prototypes (mean of support embeddings per class)
        classes = torch.unique(support_targets)
        prototypes = torch.stack([
            support_embeddings[support_targets == c].mean(0)
            for c in classes
        ])
        
        # Calculate distances to prototypes
        dists = torch.cdist(query_embeddings.unsqueeze(0), 
                           prototypes.unsqueeze(0)).squeeze(0)
        log_p_y = F.log_softmax(-dists, dim=1)
        
        # Calculate loss and accuracy
        loss = F.nll_loss(log_p_y, query_targets)
        preds = log_p_y.argmax(dim=1)
        acc = (preds == query_targets).float().mean()
        
        if return_metrics:
            return {
                'loss': loss,
                'accuracy': acc,
                'prototypes': prototypes
            }
        return loss

    def _anil_forward(self, support_set, query_set, return_metrics):
        """Almost No Inner Loop (ANIL) meta-learning"""
        support_inputs, support_targets = support_set
        query_inputs, query_targets = query_set
        
        # Only adapt the final layer
        fast_weights = {n: p.clone() for n, p in self.model.named_parameters() 
                       if 'classifier' in n or 'fc' in n}
        
        # Inner loop - only adapt the head
        for _ in range(self.config.num_inner_steps):
            outputs = self._forward_with_params(support_inputs, fast_weights, freeze_features=True)
            loss = F.cross_entropy(outputs, support_targets)
            
            grads = torch.autograd.grad(loss, fast_weights.values(), 
                                      create_graph=not self.config.first_order)
            
            fast_weights = {
                n: w - self.config.inner_lr * g
                for (n, w), g in zip(fast_weights.items(), grads)
            }
        
        # Evaluate on query set
        query_outputs = self._forward_with_params(query_inputs, fast_weights, freeze_features=True)
        query_loss = F.cross_entropy(query_outputs, query_targets)
        query_acc = (query_outputs.argmax(dim=1) == query_targets).float().mean()
        
        if return_metrics:
            return {
                'loss': query_loss,
                'accuracy': query_acc,
                'fast_weights': fast_weights
            }
        return query_loss

    def _meta_sgd_forward(self, support_set, query_set, return_metrics):
        """Meta-SGD: Learning to Learn for Few-Shot Learning"""
        support_inputs, support_targets = support_set
        query_inputs, query_targets = query_set
        
        # Initialize fast weights
        fast_weights = {n: p.clone() for n, p in self.model.named_parameters()}
        
        # Inner loop
        for _ in range(self.config.num_inner_steps):
            outputs = self._forward_with_params(support_inputs, fast_weights)
            loss = F.cross_entropy(outputs, support_targets)
            
            grads = torch.autograd.grad(loss, fast_weights.values(), 
                                      create_graph=not self.config.first_order)
            
            # Use parameter-specific learning rates
            fast_weights = {
                n: w - self.inner_lrs[n] * g
                for (n, w), g in zip(fast_weights.items(), grads)
            }
        
        # Evaluate on query set
        query_outputs = self._forward_with_params(query_inputs, fast_weights)
        query_loss = F.cross_entropy(query_outputs, query_targets)
        query_acc = (query_outputs.argmax(dim=1) == query_targets).float().mean()
        
        if return_metrics:
            return {
                'loss': query_loss,
                'accuracy': query_acc,
                'fast_weights': fast_weights
            }
        return query_loss

    def _meta_curvature_forward(self, support_set, query_set, return_metrics):
        """Meta-Curvature: Learning General Optimization-Based Meta-Learning"""
        support_inputs, support_targets = support_set
        query_inputs, query_targets = query_set
        
        # Initialize fast weights
        fast_weights = {n: p.clone() for n, p in self.model.named_parameters()}
        
        # Inner loop
        for _ in range(self.config.num_inner_steps):
            outputs = self._forward_with_params(support_inputs, fast_weights)
            loss = F.cross_entropy(outputs, support_targets)
            
            # Compute gradients with respect to the metric tensor
            grads = torch.autograd.grad(loss, fast_weights.values(), 
                                      create_graph=not self.config.first_order)
            
            # Apply metric tensor transformation
            transformed_grads = []
            for (n, g) in zip(fast_weights.keys(), grads):
                if n in self.metric_tensors:
                    g_flat = g.view(-1)
                    metric = self.metric_tensors[n]
                    transformed_g = torch.matmul(metric, g_flat)
                    transformed_g = transformed_g.view_as(g)
                    transformed_grads.append(transformed_g)
                else:
                    transformed_grads.append(g)
            
            # Update fast weights
            fast_weights = {
                n: w - self.config.inner_lr * g
                for (n, w), g in zip(fast_weights.items(), transformed_grads)
            }
        
        # Evaluate on query set
        query_outputs = self._forward_with_params(query_inputs, fast_weights)
        query_loss = F.cross_entropy(query_outputs, query_targets)
        query_acc = (query_outputs.argmax(dim=1) == query_targets).float().mean()
        
        if return_metrics:
            return {
                'loss': query_loss,
                'accuracy': query_acc,
                'fast_weights': fast_weights
            }
        return query_loss

    def _forward_with_params(self, x, params_dict, freeze_features=False):
        """Helper function to run forward pass with custom parameters"""
        if freeze_features:
            # Only compute gradients for the head
            with torch.no_grad():
                features = self.model.forward_features(x)
            return self.model.forward_head(features, params_dict)
        else:
            # Compute gradients for all parameters
            return self.model.forward_with_params(x, params_dict)

    def reptile_update(self, fast_weights):
        """Reptile update step"""
        with torch.no_grad():
            for (n, p), fast_p in zip(self.model.named_parameters(), fast_weights.values()):
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.add_(p - fast_p)

    def meta_update(self, loss):
        """Perform meta-update step"""
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()

    def adapt(self, support_set, num_steps=None):
        """Adapt the model to a new task using the support set"""
        if num_steps is None:
            num_steps = self.config.adaptation_steps
        
        fast_weights = {n: p.clone() for n, p in self.model.named_parameters()}
        support_inputs, support_targets = support_set
        
        for _ in range(num_steps):
            outputs = self._forward_with_params(support_inputs, fast_weights)
            loss = F.cross_entropy(outputs, support_targets)
            
            grads = torch.autograd.grad(loss, fast_weights.values(), 
                                      create_graph=not self.config.first_order)
            
            # Update fast weights
            fast_weights = {
                n: w - self.inner_lrs.get(n, self.config.inner_lr) * g
                for (n, w), g in zip(fast_weights.items(), grads)
            }
        
        return fast_weights

    def predict(self, x, fast_weights=None):
        """Make predictions using either the base model or fast weights"""
        if fast_weights is None:
            return self.model(x)
        else:
            return self._forward_with_params(x, fast_weights)

# Example usage:
if __name__ == "__main__":
    # Example model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=100, hidden_dim=64, num_classes=5):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.classifier = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x):
            features = self.features(x)
            return self.classifier(features)
        
        def forward_features(self, x):
            return self.features(x)
        
        def forward_head(self, features, params=None):
            if params is None:
                return self.classifier(features)
            else:
                return F.linear(features, params['classifier.weight'], params['classifier.bias'])
        
        def forward_with_params(self, x, params):
            # Custom forward pass using provided parameters
            x = F.linear(x, params['features.0.weight'], params['features.0.bias'])
            x = F.relu(x)
            x = F.linear(x, params['features.2.weight'], params['features.2.bias'])
            x = F.relu(x)
            return F.linear(x, params['classifier.weight'], params['classifier.bias'])

    # Create model and meta-learner
    model = SimpleModel()
    config = MetaLearnerConfig(
        inner_lr=0.01,
        meta_lr=0.001,
        num_inner_steps=5,
        first_order=False
    )
    
    # Initialize meta-learner with MAML
    meta_learner = MetaLearner(
        model=model,
        config=config,
        meta_learning_type=MetaLearningType.MAML
    )
    
    # Example training loop
    num_tasks = 100
    num_epochs = 10
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        for task in range(num_tasks):
            # Generate random task (in practice, you'd use a proper task sampler)
            support_inputs = torch.randn(5, 100)  # 5-shot, 100-dim input
            support_targets = torch.randint(0, 5, (5,))  # 5-way classification
            query_inputs = torch.randn(15, 100)  # 15 query examples
            query_targets = torch.randint(0, 5, (15,))
            
            # Meta-update
            metrics = meta_learner(
                support_set=(support_inputs, support_targets),
                query_set=(query_inputs, query_targets),
                return_metrics=True
            )
            
            # Update meta-parameters
            meta_learner.meta_update(metrics['loss'])
            
            # Track metrics
            epoch_loss += metrics['loss'].item()
            epoch_acc += metrics['accuracy'].item()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/num_tasks:.4f}, "
              f"Accuracy: {epoch_acc/num_tasks*100:.2f}%")