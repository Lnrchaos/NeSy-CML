"""
Modular Replay Buffer System for NeuroSym-CML
Supports various types of experiences and models with different replay strategies
"""

import torch
import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from collections import deque, namedtuple
import pickle
import os

# Define different experience types
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
MultiModalExperience = namedtuple('MultiModalExperience', ['text', 'image', 'action', 'reward', 'next_text', 'next_image', 'done'])
SymbolicExperience = namedtuple('SymbolicExperience', ['features', 'rules', 'symbolic_state', 'reward', 'next_features', 'done'])
TextExperience = namedtuple('TextExperience', ['text_encoding', 'labels', 'loss', 'accuracy', 'metadata'])
ImageExperience = namedtuple('ImageExperience', ['image', 'labels', 'predictions', 'confidence', 'metadata'])

class BaseReplayBuffer(ABC):
    """Abstract base class for all replay buffers"""
    
    def __init__(self, memory_size: int, device: str = 'cpu'):
        self.memory_size = memory_size
        self.device = device
        self.buffer = []
        self.position = 0
    
    @abstractmethod
    def add(self, experience: Any) -> None:
        """Add an experience to the buffer"""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> Any:
        """Sample experiences from the buffer"""
        pass
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear the buffer"""
        self.buffer.clear()
        self.position = 0
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.buffer) >= self.memory_size
    
    def save(self, filepath: str) -> None:
        """Save buffer to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load(self, filepath: str) -> None:
        """Load buffer from file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.buffer = pickle.load(f)
                self.position = len(self.buffer) % self.memory_size

class StandardReplayBuffer(BaseReplayBuffer):
    """Standard replay buffer for RL experiences (state, action, reward, next_state, done)"""
    
    def add(self, state: torch.Tensor, action: Union[int, torch.Tensor], reward: float, 
            next_state: torch.Tensor, done: bool) -> None:
        """Add a standard RL experience"""
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.memory_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.memory_size
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of standard experiences"""
        if len(self.buffer) < batch_size:
            experiences = self.buffer.copy()
        else:
            experiences = random.sample(self.buffer, batch_size)
        
        # Unpack experiences
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.tensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }

class MultiModalReplayBuffer(BaseReplayBuffer):
    """Replay buffer for multimodal experiences (text + image)"""
    
    def add(self, text: torch.Tensor, image: torch.Tensor, action: Union[int, torch.Tensor], 
            reward: float, next_text: torch.Tensor, next_image: torch.Tensor, done: bool) -> None:
        """Add a multimodal experience"""
        experience = MultiModalExperience(text, image, action, reward, next_text, next_image, done)
        
        if len(self.buffer) < self.memory_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.memory_size
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of multimodal experiences"""
        if len(self.buffer) < batch_size:
            experiences = self.buffer.copy()
        else:
            experiences = random.sample(self.buffer, batch_size)
        
        # Unpack experiences
        texts = torch.stack([exp.text for exp in experiences]).to(self.device)
        images = torch.stack([exp.image for exp in experiences]).to(self.device)
        actions = torch.tensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32).to(self.device)
        next_texts = torch.stack([exp.next_text for exp in experiences]).to(self.device)
        next_images = torch.stack([exp.next_image for exp in experiences]).to(self.device)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool).to(self.device)
        
        return {
            'texts': texts,
            'images': images,
            'actions': actions,
            'rewards': rewards,
            'next_texts': next_texts,
            'next_images': next_images,
            'dones': dones
        }

class SymbolicReplayBuffer(BaseReplayBuffer):
    """Replay buffer for symbolic reasoning experiences"""
    
    def add(self, features: torch.Tensor, rules: torch.Tensor, symbolic_state: Dict, 
            reward: float, next_features: torch.Tensor, done: bool) -> None:
        """Add a symbolic experience"""
        experience = SymbolicExperience(features, rules, symbolic_state, reward, next_features, done)
        
        if len(self.buffer) < self.memory_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.memory_size
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch of symbolic experiences"""
        if len(self.buffer) < batch_size:
            experiences = self.buffer.copy()
        else:
            experiences = random.sample(self.buffer, batch_size)
        
        # Unpack experiences
        features = torch.stack([exp.features for exp in experiences]).to(self.device)
        rules = torch.stack([exp.rules for exp in experiences]).to(self.device)
        symbolic_states = [exp.symbolic_state for exp in experiences]
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32).to(self.device)
        next_features = torch.stack([exp.next_features for exp in experiences]).to(self.device)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool).to(self.device)
        
        return {
            'features': features,
            'rules': rules,
            'symbolic_states': symbolic_states,
            'rewards': rewards,
            'next_features': next_features,
            'dones': dones
        }

class TextReplayBuffer(BaseReplayBuffer):
    """Replay buffer for text-based training experiences (chess, poetry, programming)"""
    
    def add(self, text_encoding: torch.Tensor, labels: torch.Tensor, loss: float, 
            accuracy: float, metadata: Optional[Dict] = None) -> None:
        """Add a text training experience"""
        experience = TextExperience(text_encoding, labels, loss, accuracy, metadata or {})
        
        if len(self.buffer) < self.memory_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.memory_size
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch of text experiences"""
        if len(self.buffer) < batch_size:
            experiences = self.buffer.copy()
        else:
            experiences = random.sample(self.buffer, batch_size)
        
        # Unpack experiences
        text_encodings = torch.stack([exp.text_encoding for exp in experiences]).to(self.device)
        labels = torch.stack([exp.labels for exp in experiences]).to(self.device)
        losses = torch.tensor([exp.loss for exp in experiences], dtype=torch.float32).to(self.device)
        accuracies = torch.tensor([exp.accuracy for exp in experiences], dtype=torch.float32).to(self.device)
        metadata = [exp.metadata for exp in experiences]
        
        return {
            'text_encodings': text_encodings,
            'labels': labels,
            'losses': losses,
            'accuracies': accuracies,
            'metadata': metadata
        }

class ImageReplayBuffer(BaseReplayBuffer):
    """Replay buffer for image-based training experiences"""
    
    def add(self, image: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor, 
            confidence: float, metadata: Optional[Dict] = None) -> None:
        """Add an image training experience"""
        experience = ImageExperience(image, labels, predictions, confidence, metadata or {})
        
        if len(self.buffer) < self.memory_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.memory_size
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch of image experiences"""
        if len(self.buffer) < batch_size:
            experiences = self.buffer.copy()
        else:
            experiences = random.sample(self.buffer, batch_size)
        
        # Unpack experiences
        images = torch.stack([exp.image for exp in experiences]).to(self.device)
        labels = torch.stack([exp.labels for exp in experiences]).to(self.device)
        predictions = torch.stack([exp.predictions for exp in experiences]).to(self.device)
        confidences = torch.tensor([exp.confidence for exp in experiences], dtype=torch.float32).to(self.device)
        metadata = [exp.metadata for exp in experiences]
        
        return {
            'images': images,
            'labels': labels,
            'predictions': predictions,
            'confidences': confidences,
            'metadata': metadata
        }

class PrioritizedReplayBuffer(BaseReplayBuffer):
    """Prioritized replay buffer that samples based on experience importance"""
    
    def __init__(self, memory_size: int, device: str = 'cpu', alpha: float = 0.6, beta: float = 0.4):
        super().__init__(memory_size, device)
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.priorities = np.zeros(memory_size, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self, experience: Any, priority: Optional[float] = None) -> None:
        """Add experience with priority"""
        if priority is None:
            priority = self.max_priority
        
        if len(self.buffer) < self.memory_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.max_priority = max(self.max_priority, priority)
        self.position = (self.position + 1) % self.memory_size
    
    def sample(self, batch_size: int) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        """Sample experiences based on priorities"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=True)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class AdaptiveReplayBuffer(BaseReplayBuffer):
    """Adaptive replay buffer that adjusts sampling strategy based on performance"""
    
    def __init__(self, memory_size: int, device: str = 'cpu', adaptation_rate: float = 0.01):
        super().__init__(memory_size, device)
        self.adaptation_rate = adaptation_rate
        self.experience_scores = []
        self.sampling_weights = None
    
    def add(self, experience: Any, performance_score: float = 0.0) -> None:
        """Add experience with performance score"""
        if len(self.buffer) < self.memory_size:
            self.buffer.append(experience)
            self.experience_scores.append(performance_score)
        else:
            self.buffer[self.position] = experience
            self.experience_scores[self.position] = performance_score
        
        self.position = (self.position + 1) % self.memory_size
        self._update_sampling_weights()
    
    def _update_sampling_weights(self) -> None:
        """Update sampling weights based on experience scores"""
        if len(self.experience_scores) == 0:
            return
        
        scores = np.array(self.experience_scores[:len(self.buffer)])
        # Convert scores to probabilities (higher score = higher probability)
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        self.sampling_weights = exp_scores / np.sum(exp_scores)
    
    def sample(self, batch_size: int) -> List[Any]:
        """Sample experiences using adaptive weights"""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        
        if self.sampling_weights is not None:
            indices = np.random.choice(len(self.buffer), batch_size, p=self.sampling_weights, replace=True)
            return [self.buffer[idx] for idx in indices]
        else:
            return random.sample(self.buffer, batch_size)

class ModularReplayBufferFactory:
    """Factory for creating different types of replay buffers"""
    
    BUFFER_TYPES = {
        'standard': StandardReplayBuffer,
        'multimodal': MultiModalReplayBuffer,
        'symbolic': SymbolicReplayBuffer,
        'text': TextReplayBuffer,
        'image': ImageReplayBuffer,
        'prioritized': PrioritizedReplayBuffer,
        'adaptive': AdaptiveReplayBuffer
    }
    
    @classmethod
    def create_buffer(cls, buffer_type: str, memory_size: int, device: str = 'cpu', **kwargs) -> BaseReplayBuffer:
        """
        Create a replay buffer of the specified type
        
        Args:
            buffer_type: Type of buffer ('standard', 'multimodal', etc.)
            memory_size: Maximum buffer size
            device: Device to store tensors on
            **kwargs: Additional buffer-specific parameters
        
        Returns:
            Replay buffer instance
        """
        if buffer_type not in cls.BUFFER_TYPES:
            raise ValueError(f"Unknown buffer type: {buffer_type}. "
                           f"Available types: {list(cls.BUFFER_TYPES.keys())}")
        
        buffer_class = cls.BUFFER_TYPES[buffer_type]
        
        # Pass additional kwargs to specific buffers
        if buffer_type == 'prioritized':
            return buffer_class(memory_size, device, 
                              alpha=kwargs.get('alpha', 0.6),
                              beta=kwargs.get('beta', 0.4))
        elif buffer_type == 'adaptive':
            return buffer_class(memory_size, device,
                              adaptation_rate=kwargs.get('adaptation_rate', 0.01))
        else:
            return buffer_class(memory_size, device)
    
    @classmethod
    def list_available_buffers(cls) -> List[str]:
        """List all available buffer types"""
        return list(cls.BUFFER_TYPES.keys())
    
    @classmethod
    def get_buffer_info(cls, buffer_type: str) -> Dict[str, str]:
        """Get information about a specific buffer type"""
        info = {
            'standard': {
                'description': 'Standard RL replay buffer (state, action, reward, next_state, done)',
                'best_for': 'Reinforcement learning tasks',
                'complexity': 'Low'
            },
            'multimodal': {
                'description': 'Replay buffer for multimodal experiences (text + image)',
                'best_for': 'Multimodal learning tasks',
                'complexity': 'Medium'
            },
            'symbolic': {
                'description': 'Replay buffer for symbolic reasoning experiences',
                'best_for': 'Symbolic AI and reasoning tasks',
                'complexity': 'Medium'
            },
            'text': {
                'description': 'Replay buffer for text-based training experiences',
                'best_for': 'NLP tasks (chess, poetry, programming)',
                'complexity': 'Low'
            },
            'image': {
                'description': 'Replay buffer for image-based training experiences',
                'best_for': 'Computer vision tasks',
                'complexity': 'Low'
            },
            'prioritized': {
                'description': 'Prioritized replay buffer that samples based on importance',
                'best_for': 'Tasks requiring focused learning on important experiences',
                'complexity': 'High'
            },
            'adaptive': {
                'description': 'Adaptive replay buffer that adjusts sampling based on performance',
                'best_for': 'Dynamic learning environments',
                'complexity': 'High'
            }
        }
        
        return info.get(buffer_type, {'description': 'Unknown buffer type'})

# Convenience function for easy buffer creation
def create_replay_buffer(buffer_type: str, memory_size: int, device: str = 'cpu', **kwargs) -> BaseReplayBuffer:
    """
    Convenience function to create a replay buffer
    
    Example usage:
        buffer = create_replay_buffer('text', 10000, 'cuda')
        prioritized_buffer = create_replay_buffer('prioritized', 10000, 'cuda', alpha=0.7)
    """
    return ModularReplayBufferFactory.create_buffer(buffer_type, memory_size, device, **kwargs)

# Example usage and testing
if __name__ == "__main__":
    print("🔄 Modular Replay Buffer System")
    print("=" * 50)
    
    # List available buffers
    print("Available Replay Buffers:")
    for buffer_type in ModularReplayBufferFactory.list_available_buffers():
        info = ModularReplayBufferFactory.get_buffer_info(buffer_type)
        print(f"  • {buffer_type}: {info['description']}")
    
    print("\n" + "=" * 50)
    
    # Test text buffer (most relevant for your training)
    print("\nTesting Text Replay Buffer...")
    text_buffer = create_replay_buffer('text', 1000, 'cpu')
    
    # Add some dummy text experiences
    for i in range(5):
        text_encoding = torch.randn(512)
        labels = torch.randint(0, 10, (1,))
        loss = np.random.random()
        accuracy = np.random.random()
        metadata = {'epoch': i, 'batch': i}
        
        text_buffer.add(text_encoding, labels, loss, accuracy, metadata)
    
    # Sample from buffer
    batch = text_buffer.sample(3)
    print(f"  ✅ Text buffer: Sampled batch with keys: {list(batch.keys())}")
    print(f"     Text encodings shape: {batch['text_encodings'].shape}")
    print(f"     Labels shape: {batch['labels'].shape}")
    
    print(f"\n🎉 Modular replay buffer system ready!")