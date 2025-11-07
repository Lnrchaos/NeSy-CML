"""
Differentiable Neural Computer (DNC) - Advanced Memory-Augmented Neural Network
Implements a fully differentiable external memory system with content-based and location-based addressing.
Based on the original DNC architecture from DeepMind with modern improvements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DNCConfig:
    """Configuration for DNC architecture"""
    input_size: int = 512
    hidden_size: int = 256
    memory_size: int = 128  # Number of memory locations
    word_size: int = 64  # Size of each memory word
    num_read_heads: int = 4  # Number of read heads
    num_write_heads: int = 1  # Number of write heads
    controller_hidden_size: int = 256
    controller_layers: int = 2
    clip_value: float = 20.0  # Gradient clipping value
    dropout: float = 0.1
    use_layer_norm: bool = True


class ContentAddressing(nn.Module):
    """Content-based addressing mechanism for memory access"""
    
    def __init__(self, word_size: int, num_heads: int):
        super().__init__()
        self.word_size = word_size
        self.num_heads = num_heads
        
    def forward(self, memory: torch.Tensor, keys: torch.Tensor, strengths: torch.Tensor) -> torch.Tensor:
        """
        Compute content-based addressing weights
        
        Args:
            memory: Memory matrix [batch_size, memory_size, word_size]
            keys: Query keys [batch_size, num_heads, word_size]
            strengths: Key strengths [batch_size, num_heads]
            
        Returns:
            Content weights [batch_size, num_heads, memory_size]
        """
        batch_size, memory_size, word_size = memory.shape
        
        # Normalize memory and keys
        memory_norm = F.normalize(memory, p=2, dim=-1)
        keys_norm = F.normalize(keys, p=2, dim=-1)
        
        # Compute cosine similarity
        # memory_norm: [batch, memory_size, word_size]
        # keys_norm: [batch, num_heads, word_size]
        # Expand for broadcasting
        memory_expanded = memory_norm.unsqueeze(1)  # [batch, 1, memory_size, word_size]
        keys_expanded = keys_norm.unsqueeze(2)  # [batch, num_heads, 1, word_size]
        
        # Compute dot product similarity
        similarity = torch.sum(memory_expanded * keys_expanded, dim=-1)  # [batch, num_heads, memory_size]
        
        # Apply key strengths (temperature scaling)
        strengths_expanded = strengths.unsqueeze(-1)  # [batch, num_heads, 1]
        weighted_similarity = similarity * strengths_expanded
        
        # Softmax to get content weights
        content_weights = F.softmax(weighted_similarity, dim=-1)
        
        return content_weights


class TemporalLinkMatrix(nn.Module):
    """Temporal link matrix for tracking memory write order"""
    
    def __init__(self, memory_size: int):
        super().__init__()
        self.memory_size = memory_size
        
    def forward(self, link_matrix: torch.Tensor, precedence: torch.Tensor, 
                write_weights: torch.Tensor) -> torch.Tensor:
        """
        Update temporal link matrix
        
        Args:
            link_matrix: Current link matrix [batch_size, memory_size, memory_size]
            precedence: Precedence weights [batch_size, memory_size]
            write_weights: Write weights [batch_size, memory_size]
            
        Returns:
            Updated link matrix [batch_size, memory_size, memory_size]
        """
        batch_size = link_matrix.shape[0]
        
        # Reset links to written locations
        write_mask = write_weights.unsqueeze(-1)  # [batch, memory_size, 1]
        link_matrix = link_matrix * (1 - write_mask * write_weights.unsqueeze(1))
        
        # Add new links based on precedence
        precedence_expanded = precedence.unsqueeze(-1)  # [batch, memory_size, 1]
        write_expanded = write_weights.unsqueeze(1)  # [batch, 1, memory_size]
        new_links = precedence_expanded * write_expanded
        
        # Update link matrix
        link_matrix = link_matrix + new_links
        
        return link_matrix


class LocationAddressing(nn.Module):
    """Location-based addressing using temporal links"""
    
    def __init__(self, memory_size: int):
        super().__init__()
        self.memory_size = memory_size
        
    def forward(self, content_weights: torch.Tensor, link_matrix: torch.Tensor,
                prev_read_weights: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """
        Compute location-based addressing
        
        Args:
            content_weights: Content-based weights [batch_size, num_heads, memory_size]
            link_matrix: Temporal link matrix [batch_size, memory_size, memory_size]
            prev_read_weights: Previous read weights [batch_size, num_heads, memory_size]
            shift: Shift distribution [batch_size, num_heads, 3] (left, no shift, right)
            
        Returns:
            Location-based read weights [batch_size, num_heads, memory_size]
        """
        batch_size, num_heads, memory_size = content_weights.shape
        
        # Interpolation between content and location-based addressing
        # For now, use content weights as base
        location_weights = content_weights.clone()
        
        # Apply temporal links
        if prev_read_weights is not None:
            # Forward links: where did we read from last time?
            forward_weights = torch.bmm(prev_read_weights, link_matrix)  # [batch, num_heads, memory_size]
            
            # Backward links: where did we come from?
            backward_weights = torch.bmm(prev_read_weights, link_matrix.transpose(1, 2))
            
            # Combine forward and backward
            temporal_weights = 0.5 * forward_weights + 0.5 * backward_weights
            
            # Interpolate with content weights
            location_weights = 0.5 * location_weights + 0.5 * temporal_weights
        
        # Apply shift operation (circular convolution)
        if shift is not None:
            shifted_weights = self._apply_shift(location_weights, shift)
            location_weights = shifted_weights
        
        return location_weights
    
    def _apply_shift(self, weights: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Apply circular shift to weights"""
        batch_size, num_heads, memory_size = weights.shape
        
        # shift: [batch, num_heads, 3] -> probabilities for left, no shift, right
        shifted = torch.zeros_like(weights)
        
        for i in range(memory_size):
            # Left shift
            left_idx = (i - 1) % memory_size
            shifted[:, :, i] += shift[:, :, 0] * weights[:, :, left_idx]
            
            # No shift
            shifted[:, :, i] += shift[:, :, 1] * weights[:, :, i]
            
            # Right shift
            right_idx = (i + 1) % memory_size
            shifted[:, :, i] += shift[:, :, 2] * weights[:, :, right_idx]
        
        return shifted


class MemoryController(nn.Module):
    """LSTM-based controller for DNC"""
    
    def __init__(self, config: DNCConfig):
        super().__init__()
        self.config = config
        
        # Controller network
        self.controller = nn.LSTM(
            input_size=config.input_size + config.num_read_heads * config.word_size,
            hidden_size=config.controller_hidden_size,
            num_layers=config.controller_layers,
            batch_first=True,
            dropout=config.dropout if config.controller_layers > 1 else 0
        )
        
        # Output projections
        self.output_size = config.hidden_size
        self.output_proj = nn.Linear(config.controller_hidden_size, self.output_size)
        
        # Memory interface
        interface_size = (
            config.num_read_heads * config.word_size +  # Read keys
            config.num_read_heads +  # Read strengths
            config.num_write_heads * config.word_size +  # Write keys
            config.num_write_heads * config.word_size +  # Write vectors
            config.num_write_heads +  # Erase vectors
            config.num_write_heads +  # Write strengths
            config.num_write_heads +  # Free gates
            config.num_read_heads * 3 +  # Shift vectors
            config.num_read_heads +  # Sharpening factors
            config.num_write_heads  # Allocation gates
        )
        
        self.interface_proj = nn.Linear(config.controller_hidden_size, interface_size)
        
        # Layer norm
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.controller_hidden_size)
        else:
            self.layer_norm = nn.Identity()
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, prev_read_vectors: torch.Tensor,
                prev_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through controller
        
        Args:
            x: Input [batch_size, seq_len, input_size]
            prev_read_vectors: Previous read vectors [batch_size, num_read_heads, word_size]
            prev_hidden: Previous LSTM hidden state
            
        Returns:
            output: Controller output [batch_size, seq_len, output_size]
            hidden: LSTM hidden state
            interface: Dictionary of interface parameters
        """
        batch_size, seq_len, _ = x.shape
        
        # Prepare input with previous read vectors
        if prev_read_vectors is not None:
            read_flat = prev_read_vectors.view(batch_size, 1, -1)  # [batch, 1, num_read_heads * word_size]
            read_expanded = read_flat.expand(-1, seq_len, -1)
            controller_input = torch.cat([x, read_expanded], dim=-1)
        else:
            # Initialize read vectors as zeros
            read_zeros = torch.zeros(batch_size, seq_len, 
                                    self.config.num_read_heads * self.config.word_size,
                                    device=x.device, dtype=x.dtype)
            controller_input = torch.cat([x, read_zeros], dim=-1)
        
        # LSTM forward
        controller_out, hidden = self.controller(controller_input, prev_hidden)
        
        # Apply layer norm and dropout
        controller_out = self.layer_norm(controller_out)
        controller_out = self.dropout(controller_out)
        
        # Project to output
        output = self.output_proj(controller_out)
        
        # Generate interface parameters
        interface_flat = self.interface_proj(controller_out)
        
        # Parse interface parameters
        interface = self._parse_interface(interface_flat)
        
        return output, hidden, interface
    
    def _parse_interface(self, interface_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse flat interface vector into structured parameters"""
        batch_size, seq_len, _ = interface_flat.shape
        config = self.config
        idx = 0
        
        interface = {}
        
        # Read keys
        read_keys_size = config.num_read_heads * config.word_size
        interface['read_keys'] = interface_flat[:, :, idx:idx+read_keys_size].view(
            batch_size, seq_len, config.num_read_heads, config.word_size)
        idx += read_keys_size
        
        # Read strengths
        interface['read_strengths'] = F.softplus(interface_flat[:, :, idx:idx+config.num_read_heads])
        idx += config.num_read_heads
        
        # Write keys
        write_keys_size = config.num_write_heads * config.word_size
        interface['write_keys'] = interface_flat[:, :, idx:idx+write_keys_size].view(
            batch_size, seq_len, config.num_write_heads, config.word_size)
        idx += write_keys_size
        
        # Write vectors
        write_vectors_size = config.num_write_heads * config.word_size
        interface['write_vectors'] = interface_flat[:, :, idx:idx+write_vectors_size].view(
            batch_size, seq_len, config.num_write_heads, config.word_size)
        idx += write_vectors_size
        
        # Erase vectors
        interface['erase_vectors'] = torch.sigmoid(interface_flat[:, :, idx:idx+config.num_write_heads])
        idx += config.num_write_heads
        
        # Write strengths
        interface['write_strengths'] = F.softplus(interface_flat[:, :, idx:idx+config.num_write_heads])
        idx += config.num_write_heads
        
        # Free gates
        interface['free_gates'] = torch.sigmoid(interface_flat[:, :, idx:idx+config.num_read_heads])
        idx += config.num_read_heads
        
        # Shift vectors
        shift_size = config.num_read_heads * 3
        interface['shift'] = F.softmax(interface_flat[:, :, idx:idx+shift_size].view(
            batch_size, seq_len, config.num_read_heads, 3), dim=-1)
        idx += shift_size
        
        # Sharpening factors
        interface['sharpening'] = F.softplus(interface_flat[:, :, idx:idx+config.num_read_heads]) + 1.0
        idx += config.num_read_heads
        
        # Allocation gates
        interface['allocation_gates'] = torch.sigmoid(interface_flat[:, :, idx:idx+config.num_write_heads])
        idx += config.num_write_heads
        
        return interface


class ExternalMemory(nn.Module):
    """External memory matrix for DNC"""
    
    def __init__(self, config: DNCConfig):
        super().__init__()
        self.config = config
        self.memory_size = config.memory_size
        self.word_size = config.word_size
        
        # Initialize memory
        self.register_buffer('memory_init', torch.zeros(1, config.memory_size, config.word_size))
        
        # Usage vector
        self.register_buffer('usage_init', torch.zeros(1, config.memory_size))
        
        # Temporal link matrix
        self.register_buffer('link_matrix_init', torch.zeros(1, config.memory_size, config.memory_size))
        
        # Precedence vector
        self.register_buffer('precedence_init', torch.zeros(1, config.memory_size))
        
    def forward(self, write_weights: torch.Tensor, write_vectors: torch.Tensor,
                erase_vectors: torch.Tensor, read_weights: torch.Tensor,
                memory: torch.Tensor, usage: torch.Tensor,
                link_matrix: torch.Tensor, precedence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform memory operations
        
        Args:
            write_weights: Write weights [batch_size, num_write_heads, memory_size]
            write_vectors: Write vectors [batch_size, num_write_heads, word_size]
            erase_vectors: Erase vectors [batch_size, num_write_heads]
            read_weights: Read weights [batch_size, num_read_heads, memory_size]
            memory: Current memory [batch_size, memory_size, word_size]
            usage: Current usage vector [batch_size, memory_size]
            link_matrix: Current link matrix [batch_size, memory_size, memory_size]
            precedence: Current precedence [batch_size, memory_size]
            
        Returns:
            updated_memory: Updated memory matrix
            read_vectors: Read vectors [batch_size, num_read_heads, word_size]
            updated_usage: Updated usage vector
            updated_link_matrix: Updated link matrix
            updated_precedence: Updated precedence vector
        """
        batch_size = memory.shape[0]
        
        # Erase memory
        erase_weights = write_weights.sum(dim=1, keepdim=True)  # [batch, 1, memory_size]
        erase_expanded = erase_vectors.unsqueeze(-1)  # [batch, num_write_heads, 1]
        erase_matrix = (erase_weights * erase_expanded).sum(dim=1, keepdim=True)  # [batch, 1, memory_size]
        memory = memory * (1 - erase_matrix.unsqueeze(-1) * write_weights.sum(dim=1, keepdim=True).unsqueeze(-1))
        
        # Write to memory
        write_expanded = write_vectors.unsqueeze(2)  # [batch, num_write_heads, 1, word_size]
        write_weights_expanded = write_weights.unsqueeze(-1)  # [batch, num_write_heads, memory_size, 1]
        write_additions = (write_expanded * write_weights_expanded).sum(dim=1)  # [batch, memory_size, word_size]
        memory = memory + write_additions
        
        # Read from memory
        read_weights_expanded = read_weights.unsqueeze(-1)  # [batch, num_read_heads, memory_size, 1]
        memory_expanded = memory.unsqueeze(1)  # [batch, 1, memory_size, word_size]
        read_vectors = (memory_expanded * read_weights_expanded).sum(dim=2)  # [batch, num_read_heads, word_size]
        
        # Update usage vector
        write_total = write_weights.sum(dim=1)  # [batch, memory_size]
        usage = usage * (1 - write_total) + write_total
        
        # Update precedence
        write_total_expanded = write_total.unsqueeze(-1)  # [batch, memory_size, 1]
        precedence = precedence * (1 - write_total) + write_total
        
        # Update link matrix (simplified)
        # In full DNC, this uses temporal link matrix module
        link_matrix = link_matrix * (1 - write_total_expanded * write_total.unsqueeze(1))
        precedence_expanded = precedence.unsqueeze(-1)
        write_expanded = write_total.unsqueeze(1)
        link_matrix = link_matrix + precedence_expanded * write_expanded
        
        return memory, read_vectors, usage, link_matrix, precedence


class DNC(nn.Module):
    """
    Differentiable Neural Computer - Complete implementation
    
    A memory-augmented neural network with external memory that can be read from and written to
    using content-based and location-based addressing mechanisms.
    """
    
    def __init__(self, config: DNCConfig):
        super().__init__()
        self.config = config
        
        # Controller
        self.controller = MemoryController(config)
        
        # Memory components
        self.memory = ExternalMemory(config)
        self.content_addressing = ContentAddressing(config.word_size, config.num_read_heads)
        self.location_addressing = LocationAddressing(config.memory_size)
        self.temporal_link = TemporalLinkMatrix(config.memory_size)
        
        # Allocation addressing
        self.allocation_addressing = self._allocation_addressing
        
    def forward(self, x: torch.Tensor, 
                prev_state: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through DNC
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            prev_state: Previous state dictionary containing:
                - memory: Previous memory [batch_size, memory_size, word_size]
                - usage: Previous usage [batch_size, memory_size]
                - link_matrix: Previous link matrix [batch_size, memory_size, memory_size]
                - precedence: Previous precedence [batch_size, memory_size]
                - read_vectors: Previous read vectors [batch_size, num_read_heads, word_size]
                - read_weights: Previous read weights [batch_size, num_read_heads, memory_size]
                - hidden: Previous controller hidden state
                
        Returns:
            output: DNC output [batch_size, seq_len, output_size]
            state: Updated state dictionary
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize state if not provided
        if prev_state is None:
            state = self._init_state(batch_size, x.device, x.dtype)
        else:
            state = prev_state.copy()
        
        outputs = []
        read_vectors_list = []
        
        # Process sequence step by step
        for t in range(seq_len):
            x_t = x[:, t:t+1, :]  # [batch, 1, input_size]
            
            # Controller forward
            prev_read_vectors = state.get('read_vectors')
            prev_hidden = state.get('hidden')
            
            controller_out, hidden, interface = self.controller(
                x_t, prev_read_vectors, prev_hidden
            )
            
            state['hidden'] = hidden
            
            # Compute write weights (allocation + content-based)
            allocation_weights = self.allocation_addressing(
                state['usage'],
                interface['allocation_gates'][:, 0, :],
                interface['write_strengths'][:, 0, :]
            )
            
            content_write_weights = self.content_addressing(
                state['memory'],
                interface['write_keys'][:, 0, :, :],
                interface['write_strengths'][:, 0, :]
            )
            
            # Combine allocation and content-based write weights
            allocation_gate = interface['allocation_gates'][:, 0, :].unsqueeze(-1)
            write_weights = (
                allocation_gate * allocation_weights.unsqueeze(1) +
                (1 - allocation_gate) * content_write_weights
            )
            
            # Compute read weights (content + location-based)
            content_read_weights = self.content_addressing(
                state['memory'],
                interface['read_keys'][:, 0, :, :],
                interface['read_strengths'][:, 0, :]
            )
            
            prev_read_weights = state.get('read_weights')
            read_weights = self.location_addressing(
                content_read_weights,
                state['link_matrix'],
                prev_read_weights,
                interface['shift'][:, 0, :, :]
            )
            
            # Apply sharpening
            sharpening = interface['sharpening'][:, 0, :].unsqueeze(-1)
            read_weights = read_weights ** sharpening
            read_weights = read_weights / (read_weights.sum(dim=-1, keepdim=True) + 1e-10)
            
            # Memory operations
            state['memory'], read_vectors, state['usage'], state['link_matrix'], state['precedence'] = \
                self.memory(
                    write_weights,
                    interface['write_vectors'][:, 0, :, :],
                    interface['erase_vectors'][:, 0, :],
                    read_weights,
                    state['memory'],
                    state['usage'],
                    state['link_matrix'],
                    state['precedence']
                )
            
            # Update precedence
            write_total = write_weights.sum(dim=1)  # [batch, memory_size]
            state['precedence'] = state['precedence'] * (1 - write_total) + write_total
            
            state['read_vectors'] = read_vectors
            state['read_weights'] = read_weights
            
            # Combine controller output with read vectors
            read_flat = read_vectors.view(batch_size, 1, -1)
            output = torch.cat([controller_out, read_flat], dim=-1)
            
            # Project to output size
            if not hasattr(self, 'final_proj'):
                self.final_proj = nn.Linear(
                    self.config.hidden_size + self.config.num_read_heads * self.config.word_size,
                    self.config.hidden_size
                ).to(x.device)
            
            output = self.final_proj(output)
            outputs.append(output)
            read_vectors_list.append(read_vectors)
        
        # Concatenate outputs
        output = torch.cat(outputs, dim=1)  # [batch, seq_len, output_size]
        
        return output, state
    
    def _init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Initialize DNC state"""
        return {
            'memory': self.memory.memory_init.expand(batch_size, -1, -1).to(device).to(dtype),
            'usage': self.memory.usage_init.expand(batch_size, -1).to(device).to(dtype),
            'link_matrix': self.memory.link_matrix_init.expand(batch_size, -1, -1).to(device).to(dtype),
            'precedence': self.memory.precedence_init.expand(batch_size, -1).to(device).to(dtype),
            'read_vectors': torch.zeros(batch_size, self.config.num_read_heads, self.config.word_size,
                                       device=device, dtype=dtype),
            'read_weights': torch.zeros(batch_size, self.config.num_read_heads, self.config.memory_size,
                                      device=device, dtype=dtype),
            'hidden': None
        }
    
    def _allocation_addressing(self, usage: torch.Tensor, allocation_gates: torch.Tensor,
                              write_strengths: torch.Tensor) -> torch.Tensor:
        """
        Compute allocation-based write weights
        
        Args:
            usage: Usage vector [batch_size, memory_size]
            allocation_gates: Allocation gates [batch_size, num_write_heads]
            write_strengths: Write strengths [batch_size, num_write_heads]
            
        Returns:
            Allocation weights [batch_size, num_write_heads, memory_size]
        """
        batch_size, memory_size = usage.shape
        num_write_heads = allocation_gates.shape[1]
        
        # Sort usage to find free locations
        sorted_usage, free_list = torch.sort(usage, dim=-1)
        
        # Compute allocation weights
        allocation_weights = torch.zeros(batch_size, num_write_heads, memory_size,
                                       device=usage.device, dtype=usage.dtype)
        
        for h in range(num_write_heads):
            # Compute cumulative product of (1 - sorted_usage)
            cumprod = torch.cumprod(1 - sorted_usage, dim=-1)
            
            # Allocation weights
            alloc = sorted_usage * cumprod
            
            # Scatter back to original indices
            allocation_weights[:, h, :] = alloc.gather(1, free_list.argsort(dim=-1))
        
        return allocation_weights


def create_dnc(input_size: int = 512, hidden_size: int = 256, 
               memory_size: int = 128, word_size: int = 64,
               num_read_heads: int = 4, num_write_heads: int = 1) -> DNC:
    """
    Factory function to create a DNC model
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden/output features
        memory_size: Number of memory locations
        word_size: Size of each memory word
        num_read_heads: Number of read heads
        num_write_heads: Number of write heads
        
    Returns:
        DNC model instance
    """
    config = DNCConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        memory_size=memory_size,
        word_size=word_size,
        num_read_heads=num_read_heads,
        num_write_heads=num_write_heads
    )
    return DNC(config)

