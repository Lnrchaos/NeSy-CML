"""
Memory-Augmented Neural Networks (MANN) - Advanced implementations
Includes Neural Turing Machine (NTM), Stack-Augmented RNN, and other memory architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class MemoryType(Enum):
    """Types of memory architectures"""
    NTM = "ntm"  # Neural Turing Machine
    STACK = "stack"  # Stack-Augmented RNN
    QUEUE = "queue"  # Queue-Augmented RNN
    DEQUE = "deque"  # Deque-Augmented RNN
    ASSOCIATIVE = "associative"  # Associative Memory


@dataclass
class MemoryConfig:
    """Configuration for memory-augmented networks"""
    memory_size: int = 128
    memory_dim: int = 64
    num_read_heads: int = 1
    num_write_heads: int = 1
    controller_hidden_size: int = 256
    controller_layers: int = 2
    memory_type: MemoryType = MemoryType.NTM
    use_content_addressing: bool = True
    use_location_addressing: bool = True
    sharpening_factor: float = 1.0
    dropout: float = 0.1


class ContentAddressingModule(nn.Module):
    """Content-based addressing for memory access"""
    
    def __init__(self, memory_dim: int, key_size: int):
        super().__init__()
        self.memory_dim = memory_dim
        self.key_size = key_size
        
    def forward(self, memory: torch.Tensor, keys: torch.Tensor, 
                beta: torch.Tensor) -> torch.Tensor:
        """
        Compute content-based addressing weights
        
        Args:
            memory: Memory matrix [batch_size, memory_size, memory_dim]
            keys: Query keys [batch_size, num_heads, key_size]
            beta: Key strength [batch_size, num_heads]
            
        Returns:
            Content weights [batch_size, num_heads, memory_size]
        """
        # Normalize memory and keys
        memory_norm = F.normalize(memory, p=2, dim=-1)
        keys_norm = F.normalize(keys, p=2, dim=-1)
        
        # Compute cosine similarity
        # Expand for broadcasting: [batch, 1, memory_size, memory_dim] x [batch, num_heads, 1, key_size]
        memory_expanded = memory_norm.unsqueeze(1)  # [batch, 1, memory_size, memory_dim]
        keys_expanded = keys_norm.unsqueeze(2)  # [batch, num_heads, 1, key_size]
        
        # Ensure dimensions match
        if memory_norm.shape[-1] != keys_norm.shape[-1]:
            # Project keys to memory dimension if needed
            if not hasattr(self, 'key_proj'):
                self.key_proj = nn.Linear(keys_norm.shape[-1], memory_norm.shape[-1]).to(keys.device)
            keys_expanded = self.key_proj(keys_expanded)
        
        # Compute similarity
        similarity = torch.sum(memory_expanded * keys_expanded, dim=-1)  # [batch, num_heads, memory_size]
        
        # Apply beta (key strength)
        beta_expanded = beta.unsqueeze(-1)  # [batch, num_heads, 1]
        weighted_similarity = similarity * beta_expanded
        
        # Softmax to get content weights
        content_weights = F.softmax(weighted_similarity, dim=-1)
        
        return content_weights


class LocationAddressingModule(nn.Module):
    """Location-based addressing with interpolation and shifting"""
    
    def __init__(self, memory_size: int):
        super().__init__()
        self.memory_size = memory_size
        
    def forward(self, content_weights: torch.Tensor, prev_weights: torch.Tensor,
                g: torch.Tensor, s: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Compute location-based addressing
        
        Args:
            content_weights: Content-based weights [batch_size, num_heads, memory_size]
            prev_weights: Previous weights [batch_size, num_heads, memory_size]
            g: Interpolation gate [batch_size, num_heads]
            s: Shift distribution [batch_size, num_heads, shift_range]
            gamma: Sharpening factor [batch_size, num_heads]
            
        Returns:
            Location-based weights [batch_size, num_heads, memory_size]
        """
        # Interpolation between content and previous weights
        g_expanded = g.unsqueeze(-1)  # [batch, num_heads, 1]
        interpolated = g_expanded * content_weights + (1 - g_expanded) * prev_weights
        
        # Circular convolution (shift)
        shifted = self._circular_convolution(interpolated, s)
        
        # Sharpening
        gamma_expanded = gamma.unsqueeze(-1)  # [batch, num_heads, 1]
        sharpened = shifted ** gamma_expanded
        sharpened = sharpened / (sharpened.sum(dim=-1, keepdim=True) + 1e-10)
        
        return sharpened
    
    def _circular_convolution(self, weights: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Apply circular convolution for shifting"""
        batch_size, num_heads, memory_size = weights.shape
        shift_range = shift.shape[-1]
        shift_center = shift_range // 2
        
        shifted = torch.zeros_like(weights)
        
        for i in range(memory_size):
            for j in range(shift_range):
                shift_amount = j - shift_center
                idx = (i - shift_amount) % memory_size
                shifted[:, :, i] += shift[:, :, j] * weights[:, :, idx]
        
        return shifted


class NTMMemory(nn.Module):
    """Neural Turing Machine memory module"""
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.memory_size = config.memory_size
        self.memory_dim = config.memory_dim
        
        # Initialize memory
        self.register_buffer('memory_init', torch.zeros(1, config.memory_size, config.memory_dim))
        
        # Addressing modules
        self.content_addressing = ContentAddressingModule(config.memory_dim, config.memory_dim)
        self.location_addressing = LocationAddressingModule(config.memory_size)
        
    def forward(self, read_keys: torch.Tensor, read_beta: torch.Tensor,
                write_keys: torch.Tensor, write_beta: torch.Tensor,
                write_vectors: torch.Tensor, erase_vectors: torch.Tensor,
                g: torch.Tensor, s: torch.Tensor, gamma: torch.Tensor,
                memory: torch.Tensor, prev_read_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform NTM memory operations
        
        Returns:
            updated_memory: Updated memory matrix
            read_vectors: Read vectors [batch_size, num_read_heads, memory_dim]
            read_weights: Read weights [batch_size, num_read_heads, memory_size]
        """
        batch_size = memory.shape[0]
        
        # Content-based addressing for reading
        read_content_weights = self.content_addressing(memory, read_keys, read_beta)
        
        # Location-based addressing for reading
        read_weights = self.location_addressing(
            read_content_weights, prev_read_weights, g, s, gamma
        )
        
        # Read from memory
        read_weights_expanded = read_weights.unsqueeze(-1)  # [batch, num_read_heads, memory_size, 1]
        memory_expanded = memory.unsqueeze(1)  # [batch, 1, memory_size, memory_dim]
        read_vectors = (memory_expanded * read_weights_expanded).sum(dim=2)  # [batch, num_read_heads, memory_dim]
        
        # Content-based addressing for writing
        write_content_weights = self.content_addressing(memory, write_keys, write_beta)
        
        # Location-based addressing for writing
        write_weights = self.location_addressing(
            write_content_weights, prev_read_weights, g, s, gamma
        )
        
        # Erase memory
        erase_weights = write_weights.sum(dim=1, keepdim=True)  # [batch, 1, memory_size]
        erase_expanded = erase_vectors.unsqueeze(-1)  # [batch, num_write_heads, 1]
        erase_matrix = (erase_weights * erase_expanded).sum(dim=1, keepdim=True)  # [batch, 1, memory_size]
        memory = memory * (1 - erase_matrix.unsqueeze(-1))
        
        # Write to memory
        write_expanded = write_vectors.unsqueeze(2)  # [batch, num_write_heads, 1, memory_dim]
        write_weights_expanded = write_weights.unsqueeze(-1)  # [batch, num_write_heads, memory_size, 1]
        write_additions = (write_expanded * write_weights_expanded).sum(dim=1)  # [batch, memory_size, memory_dim]
        memory = memory + write_additions
        
        return memory, read_vectors, read_weights


class StackMemory(nn.Module):
    """Stack-augmented memory for sequential operations"""
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.stack_size = config.memory_size
        self.stack_dim = config.memory_dim
        
    def forward(self, push_values: torch.Tensor, pop_weights: torch.Tensor,
                stack: torch.Tensor, stack_pointer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform stack operations
        
        Args:
            push_values: Values to push [batch_size, stack_dim]
            pop_weights: Pop weights [batch_size] (0 = no pop, 1 = pop)
            stack: Current stack [batch_size, stack_size, stack_dim]
            stack_pointer: Current stack pointer [batch_size]
            
        Returns:
            updated_stack: Updated stack
            popped_values: Popped values [batch_size, stack_dim]
            updated_pointer: Updated stack pointer
        """
        batch_size = stack.shape[0]
        
        # Pop operation
        popped_values = torch.zeros(batch_size, self.stack_dim, device=stack.device, dtype=stack.dtype)
        for b in range(batch_size):
            if stack_pointer[b] > 0 and pop_weights[b] > 0.5:
                popped_values[b] = stack[b, int(stack_pointer[b]) - 1]
                stack_pointer[b] = max(0, stack_pointer[b] - 1)
        
        # Push operation
        push_weights = 1 - pop_weights
        for b in range(batch_size):
            if stack_pointer[b] < self.stack_size - 1 and push_weights[b] > 0.5:
                stack[b, int(stack_pointer[b])] = push_values[b]
                stack_pointer[b] = min(self.stack_size - 1, stack_pointer[b] + 1)
        
        return stack, popped_values, stack_pointer


class QueueMemory(nn.Module):
    """Queue-augmented memory for FIFO operations"""
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.queue_size = config.memory_size
        self.queue_dim = config.memory_dim
        
    def forward(self, enqueue_values: torch.Tensor, dequeue_weights: torch.Tensor,
                queue: torch.Tensor, front_pointer: torch.Tensor, 
                rear_pointer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform queue operations (FIFO)
        
        Returns:
            updated_queue: Updated queue
            dequeued_values: Dequeued values [batch_size, queue_dim]
            updated_front: Updated front pointer
            updated_rear: Updated rear pointer
        """
        batch_size = queue.shape[0]
        dequeued_values = torch.zeros(batch_size, self.queue_dim, device=queue.device, dtype=queue.dtype)
        
        # Dequeue operation
        for b in range(batch_size):
            if front_pointer[b] != rear_pointer[b] and dequeue_weights[b] > 0.5:
                dequeued_values[b] = queue[b, int(front_pointer[b])]
                front_pointer[b] = (front_pointer[b] + 1) % self.queue_size
        
        # Enqueue operation
        enqueue_weights = 1 - dequeue_weights
        for b in range(batch_size):
            if (rear_pointer[b] + 1) % self.queue_size != front_pointer[b] and enqueue_weights[b] > 0.5:
                queue[b, int(rear_pointer[b])] = enqueue_values[b]
                rear_pointer[b] = (rear_pointer[b] + 1) % self.queue_size
        
        return queue, dequeued_values, front_pointer, rear_pointer


class MemoryController(nn.Module):
    """LSTM controller for memory-augmented networks"""
    
    def __init__(self, config: MemoryConfig, input_size: int, output_size: int):
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        
        # Controller network
        self.controller = nn.LSTM(
            input_size=input_size + config.num_read_heads * config.memory_dim,
            hidden_size=config.controller_hidden_size,
            num_layers=config.controller_layers,
            batch_first=True,
            dropout=config.dropout if config.controller_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.controller_hidden_size, output_size)
        
        # Memory interface size depends on memory type
        if config.memory_type == MemoryType.NTM:
            interface_size = (
                config.num_read_heads * config.memory_dim +  # Read keys
                config.num_read_heads +  # Read beta
                config.num_write_heads * config.memory_dim +  # Write keys
                config.num_write_heads * config.memory_dim +  # Write vectors
                config.num_write_heads +  # Erase vectors
                config.num_write_heads +  # Write beta
                config.num_read_heads +  # Interpolation gate
                config.num_read_heads * 3 +  # Shift distribution
                config.num_read_heads  # Sharpening factor
            )
        elif config.memory_type == MemoryType.STACK:
            interface_size = (
                config.memory_dim +  # Push values
                1  # Pop weights
            )
        elif config.memory_type == MemoryType.QUEUE:
            interface_size = (
                config.memory_dim +  # Enqueue values
                1  # Dequeue weights
            )
        else:
            interface_size = config.memory_dim * 2
        
        self.interface_proj = nn.Linear(config.controller_hidden_size, interface_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, prev_read_vectors: torch.Tensor,
                prev_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through controller
        
        Returns:
            output: Controller output
            hidden: LSTM hidden state
            interface: Interface parameters
        """
        batch_size, seq_len, _ = x.shape
        
        # Prepare input with previous read vectors
        if prev_read_vectors is not None:
            read_flat = prev_read_vectors.view(batch_size, 1, -1)
            read_expanded = read_flat.expand(-1, seq_len, -1)
            controller_input = torch.cat([x, read_expanded], dim=-1)
        else:
            read_zeros = torch.zeros(batch_size, seq_len,
                                   self.config.num_read_heads * self.config.memory_dim,
                                   device=x.device, dtype=x.dtype)
            controller_input = torch.cat([x, read_zeros], dim=-1)
        
        # LSTM forward
        controller_out, hidden = self.controller(controller_input, prev_hidden)
        controller_out = self.dropout(controller_out)
        
        # Project to output
        output = self.output_proj(controller_out)
        
        # Generate interface parameters
        interface_flat = self.interface_proj(controller_out)
        
        # Parse interface based on memory type
        interface = self._parse_interface(interface_flat)
        
        return output, hidden, interface
    
    def _parse_interface(self, interface_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse interface parameters based on memory type"""
        batch_size, seq_len, _ = interface_flat.shape
        config = self.config
        idx = 0
        interface = {}
        
        if config.memory_type == MemoryType.NTM:
            # Read keys
            read_keys_size = config.num_read_heads * config.memory_dim
            interface['read_keys'] = interface_flat[:, :, idx:idx+read_keys_size].view(
                batch_size, seq_len, config.num_read_heads, config.memory_dim)
            idx += read_keys_size
            
            # Read beta
            interface['read_beta'] = F.softplus(interface_flat[:, :, idx:idx+config.num_read_heads]) + 1.0
            idx += config.num_read_heads
            
            # Write keys
            write_keys_size = config.num_write_heads * config.memory_dim
            interface['write_keys'] = interface_flat[:, :, idx:idx+write_keys_size].view(
                batch_size, seq_len, config.num_write_heads, config.memory_dim)
            idx += write_keys_size
            
            # Write vectors
            write_vectors_size = config.num_write_heads * config.memory_dim
            interface['write_vectors'] = interface_flat[:, :, idx:idx+write_vectors_size].view(
                batch_size, seq_len, config.num_write_heads, config.memory_dim)
            idx += write_vectors_size
            
            # Erase vectors
            interface['erase_vectors'] = torch.sigmoid(interface_flat[:, :, idx:idx+config.num_write_heads])
            idx += config.num_write_heads
            
            # Write beta
            interface['write_beta'] = F.softplus(interface_flat[:, :, idx:idx+config.num_write_heads]) + 1.0
            idx += config.num_write_heads
            
            # Interpolation gate
            interface['g'] = torch.sigmoid(interface_flat[:, :, idx:idx+config.num_read_heads])
            idx += config.num_read_heads
            
            # Shift distribution
            shift_size = config.num_read_heads * 3
            interface['s'] = F.softmax(interface_flat[:, :, idx:idx+shift_size].view(
                batch_size, seq_len, config.num_read_heads, 3), dim=-1)
            idx += shift_size
            
            # Sharpening factor
            interface['gamma'] = F.softplus(interface_flat[:, :, idx:idx+config.num_read_heads]) + 1.0
            idx += config.num_read_heads
            
        elif config.memory_type == MemoryType.STACK:
            # Push values
            interface['push_values'] = interface_flat[:, :, idx:idx+config.memory_dim]
            idx += config.memory_dim
            
            # Pop weights
            interface['pop_weights'] = torch.sigmoid(interface_flat[:, :, idx:idx+1])
            idx += 1
            
        elif config.memory_type == MemoryType.QUEUE:
            # Enqueue values
            interface['enqueue_values'] = interface_flat[:, :, idx:idx+config.memory_dim]
            idx += config.memory_dim
            
            # Dequeue weights
            interface['dequeue_weights'] = torch.sigmoid(interface_flat[:, :, idx:idx+1])
            idx += 1
        
        return interface


class MemoryAugmentedNetwork(nn.Module):
    """
    Memory-Augmented Neural Network
    
    Supports multiple memory architectures:
    - Neural Turing Machine (NTM)
    - Stack-Augmented RNN
    - Queue-Augmented RNN
    """
    
    def __init__(self, config: MemoryConfig, input_size: int, output_size: int):
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        
        # Controller
        self.controller = MemoryController(config, input_size, output_size)
        
        # Memory modules
        if config.memory_type == MemoryType.NTM:
            self.memory = NTMMemory(config)
        elif config.memory_type == MemoryType.STACK:
            self.memory = StackMemory(config)
        elif config.memory_type == MemoryType.QUEUE:
            self.memory = QueueMemory(config)
        else:
            raise ValueError(f"Unsupported memory type: {config.memory_type}")
        
    def forward(self, x: torch.Tensor, 
                prev_state: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through memory-augmented network
        
        Args:
            x: Input [batch_size, seq_len, input_size]
            prev_state: Previous state dictionary
            
        Returns:
            output: Network output [batch_size, seq_len, output_size]
            state: Updated state dictionary
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize state if not provided
        if prev_state is None:
            state = self._init_state(batch_size, x.device, x.dtype)
        else:
            state = prev_state.copy()
        
        outputs = []
        
        # Process sequence step by step
        for t in range(seq_len):
            x_t = x[:, t:t+1, :]
            
            # Controller forward
            prev_read_vectors = state.get('read_vectors')
            prev_hidden = state.get('hidden')
            
            controller_out, hidden, interface = self.controller(
                x_t, prev_read_vectors, prev_hidden
            )
            
            state['hidden'] = hidden
            
            # Memory operations based on type
            if self.config.memory_type == MemoryType.NTM:
                state['memory'], read_vectors, read_weights = self.memory(
                    interface['read_keys'][:, 0, :, :],
                    interface['read_beta'][:, 0, :],
                    interface['write_keys'][:, 0, :, :],
                    interface['write_beta'][:, 0, :],
                    interface['write_vectors'][:, 0, :, :],
                    interface['erase_vectors'][:, 0, :],
                    interface['g'][:, 0, :],
                    interface['s'][:, 0, :, :],
                    interface['gamma'][:, 0, :],
                    state['memory'],
                    state.get('read_weights', torch.zeros(batch_size, self.config.num_read_heads,
                                                         self.config.memory_size,
                                                         device=x.device, dtype=x.dtype))
                )
                state['read_weights'] = read_weights
                
            elif self.config.memory_type == MemoryType.STACK:
                state['stack'], popped_values, state['stack_pointer'] = self.memory(
                    interface['push_values'][:, 0, :],
                    interface['pop_weights'][:, 0, 0],
                    state['stack'],
                    state['stack_pointer']
                )
                read_vectors = popped_values.unsqueeze(1)  # [batch, 1, memory_dim]
                
            elif self.config.memory_type == MemoryType.QUEUE:
                state['queue'], dequeued_values, state['front_pointer'], state['rear_pointer'] = self.memory(
                    interface['enqueue_values'][:, 0, :],
                    interface['dequeue_weights'][:, 0, 0],
                    state['queue'],
                    state['front_pointer'],
                    state['rear_pointer']
                )
                read_vectors = dequeued_values.unsqueeze(1)  # [batch, 1, memory_dim]
            
            state['read_vectors'] = read_vectors
            
            # Combine controller output with read vectors
            read_flat = read_vectors.view(batch_size, 1, -1)
            output = torch.cat([controller_out, read_flat], dim=-1)
            
            # Project to output size
            if not hasattr(self, 'final_proj'):
                self.final_proj = nn.Linear(
                    self.output_size + self.config.num_read_heads * self.config.memory_dim,
                    self.output_size
                ).to(x.device)
            
            output = self.final_proj(output)
            outputs.append(output)
        
        # Concatenate outputs
        output = torch.cat(outputs, dim=1)
        
        return output, state
    
    def _init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Initialize network state"""
        state = {
            'hidden': None,
            'read_vectors': torch.zeros(batch_size, self.config.num_read_heads, self.config.memory_dim,
                                      device=device, dtype=dtype)
        }
        
        if self.config.memory_type == MemoryType.NTM:
            state['memory'] = torch.zeros(batch_size, self.config.memory_size, self.config.memory_dim,
                                        device=device, dtype=dtype)
            state['read_weights'] = torch.zeros(batch_size, self.config.num_read_heads, self.config.memory_size,
                                              device=device, dtype=dtype)
        elif self.config.memory_type == MemoryType.STACK:
            state['stack'] = torch.zeros(batch_size, self.config.memory_size, self.config.memory_dim,
                                       device=device, dtype=dtype)
            state['stack_pointer'] = torch.zeros(batch_size, device=device, dtype=dtype)
        elif self.config.memory_type == MemoryType.QUEUE:
            state['queue'] = torch.zeros(batch_size, self.config.memory_size, self.config.memory_dim,
                                       device=device, dtype=dtype)
            state['front_pointer'] = torch.zeros(batch_size, device=device, dtype=dtype)
            state['rear_pointer'] = torch.zeros(batch_size, device=device, dtype=dtype)
        
        return state


def create_mann(memory_type: str = "ntm", input_size: int = 512, output_size: int = 256,
                memory_size: int = 128, memory_dim: int = 64) -> MemoryAugmentedNetwork:
    """
    Factory function to create a memory-augmented network
    
    Args:
        memory_type: Type of memory ("ntm", "stack", "queue")
        input_size: Size of input features
        output_size: Size of output features
        memory_size: Number of memory locations
        memory_dim: Dimension of each memory location
        
    Returns:
        MemoryAugmentedNetwork instance
    """
    config = MemoryConfig(
        memory_size=memory_size,
        memory_dim=memory_dim,
        memory_type=MemoryType(memory_type.lower())
    )
    return MemoryAugmentedNetwork(config, input_size, output_size)

