"""
Enhanced Replay Buffer with Modular Support
Maintains backward compatibility while adding new modular features
"""

import torch
import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any, Union
from modular_replay_buffer import (
    create_replay_buffer, 
    ModularReplayBufferFactory,
    TextReplayBuffer,
    ImageReplayBuffer,
    MultiModalReplayBuffer
)

class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling past experiences
    """
    def __init__(self, memory_size: int):
        """
        Initialize the replay buffer
        
        Args:
            memory_size: Maximum number of experiences to store
        """
        self.memory_size = memory_size
        self.buffer = []
        self.position = 0

    def add(self, sample: Tuple):
        """
        Add a new experience to the buffer
        
        Args:
            sample: Tuple containing (state, action, reward, next_state, done)
        """
        if len(self.buffer) < self.memory_size:
            self.buffer.append(None)
        self.buffer[self.position] = sample
        self.position = (self.position + 1) % self.memory_size

    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample a batch of experiences from the buffer
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return [exp for exp in self.buffer if exp is not None]
        return random.sample([exp for exp in self.buffer if exp is not None], batch_size)

    def __len__(self):
        """
        Return the current size of internal memory
        """
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer for storing and sampling past experiences
    with priority-based sampling
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize the prioritized replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform sampling, 1 = full priority)
            beta: Importance sampling correction exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """
        Sample a batch of experiences from the buffer using prioritized sampling
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Extract experiences
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), indices, weights)

    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled experiences
        
        Args:
            indices: Indices of experiences to update
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """
        Return the current size of internal memory
        """
        return len(self.buffer)


# Enhanced Modular Replay Buffer Classes
class EnhancedReplayBuffer(ReplayBuffer):
    """Enhanced version of ReplayBuffer with modular support"""
    
    def __init__(self, memory_size: int, buffer_type: str = 'standard', device: str = 'cpu'):
        super().__init__(memory_size)
        self.buffer_type = buffer_type
        self.device = device
        
        # Create modular buffer for advanced features
        if buffer_type != 'legacy':
            self.modular_buffer = create_replay_buffer(buffer_type, memory_size, device)
    
    def add_experience(self, **kwargs) -> None:
        """Add experience using modular buffer"""
        if hasattr(self, 'modular_buffer'):
            if self.buffer_type == 'text':
                self.modular_buffer.add(
                    kwargs['text_encoding'],
                    kwargs['labels'], 
                    kwargs['loss'],
                    kwargs['accuracy'],
                    kwargs.get('metadata', {})
                )
            elif self.buffer_type == 'image':
                self.modular_buffer.add(
                    kwargs['image'],
                    kwargs['labels'],
                    kwargs['predictions'],
                    kwargs['confidence'],
                    kwargs.get('metadata', {})
                )
            elif self.buffer_type == 'multimodal':
                self.modular_buffer.add(
                    kwargs['text'],
                    kwargs['image'],
                    kwargs['action'],
                    kwargs['reward'],
                    kwargs['next_text'],
                    kwargs['next_image'],
                    kwargs['done']
                )
    
    def sample_batch(self, batch_size: int) -> Any:
        """Sample batch using modular buffer"""
        if hasattr(self, 'modular_buffer'):
            return self.modular_buffer.sample(batch_size)
        else:
            return self.sample(batch_size)
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get information about the buffer"""
        return {
            'type': self.buffer_type,
            'size': len(self),
            'capacity': self.memory_size,
            'device': self.device,
            'is_full': len(self) >= self.memory_size
        }

# Convenience functions for creating specialized buffers
def create_chess_buffer(memory_size: int = 10000, device: str = 'cpu') -> EnhancedReplayBuffer:
    """Create a replay buffer optimized for chess training"""
    return EnhancedReplayBuffer(memory_size, 'text', device)

def create_poetry_buffer(memory_size: int = 10000, device: str = 'cpu') -> EnhancedReplayBuffer:
    """Create a replay buffer optimized for poetry training"""
    return EnhancedReplayBuffer(memory_size, 'text', device)

def create_programming_buffer(memory_size: int = 10000, device: str = 'cpu') -> EnhancedReplayBuffer:
    """Create a replay buffer optimized for programming training"""
    return EnhancedReplayBuffer(memory_size, 'text', device)

def create_image_buffer(memory_size: int = 10000, device: str = 'cpu') -> EnhancedReplayBuffer:
    """Create a replay buffer optimized for image training"""
    return EnhancedReplayBuffer(memory_size, 'image', device)

def create_multimodal_buffer(memory_size: int = 10000, device: str = 'cpu') -> EnhancedReplayBuffer:
    """Create a replay buffer optimized for multimodal training"""
    return EnhancedReplayBuffer(memory_size, 'multimodal', device)

# Factory function for easy buffer creation
def get_optimal_buffer(training_type: str, memory_size: int = 10000, device: str = 'cpu') -> EnhancedReplayBuffer:
    """
    Get the optimal replay buffer for a specific training type
    
    Args:
        training_type: Type of training ('chess', 'poetry', 'programming', 'image', 'multimodal')
        memory_size: Buffer capacity
        device: Device to store tensors on
    
    Returns:
        Optimized replay buffer for the training type
    """
    buffer_mapping = {
        'chess': create_chess_buffer,
        'poetry': create_poetry_buffer,
        'programming': create_programming_buffer,
        'image': create_image_buffer,
        'multimodal': create_multimodal_buffer
    }
    
    if training_type in buffer_mapping:
        return buffer_mapping[training_type](memory_size, device)
    else:
        # Default to standard buffer
        return EnhancedReplayBuffer(memory_size, 'standard', device)

# Example usage
if __name__ == "__main__":
    print("🔄 Enhanced Modular Replay Buffer System")
    print("=" * 50)
    
    # Test different buffer types
    buffer_types = ['chess', 'poetry', 'programming', 'image', 'multimodal']
    
    for buffer_type in buffer_types:
        buffer = get_optimal_buffer(buffer_type, 1000, 'cpu')
        info = buffer.get_buffer_info()
        print(f"✅ {buffer_type.title()} Buffer: {info}")
    
    print("\n🎉 Enhanced replay buffer system ready!")