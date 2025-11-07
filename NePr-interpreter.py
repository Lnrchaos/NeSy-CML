"""
Neural Program Interpreter - Execute and interpret neural programs
Implements a differentiable program interpreter that can execute symbolic programs
using neural networks, enabling program synthesis and neural-symbolic integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import re


class InstructionType(Enum):
    """Types of program instructions"""
    ASSIGN = "assign"  # Variable assignment
    ADD = "add"  # Addition
    SUB = "sub"  # Subtraction
    MUL = "mul"  # Multiplication
    DIV = "div"  # Division
    IF = "if"  # Conditional
    LOOP = "loop"  # Loop
    CALL = "call"  # Function call
    RETURN = "return"  # Return value
    COMPARE = "compare"  # Comparison
    LOGIC = "logic"  # Logical operations


@dataclass
class ProgramConfig:
    """Configuration for neural program interpreter"""
    hidden_size: int = 256
    max_variables: int = 20
    max_instructions: int = 100
    max_loop_iterations: int = 10
    use_attention: bool = True
    use_memory: bool = True
    dropout: float = 0.1


class VariableMemory(nn.Module):
    """Neural memory for storing program variables"""
    
    def __init__(self, config: ProgramConfig):
        super().__init__()
        self.config = config
        self.max_variables = config.max_variables
        self.hidden_size = config.hidden_size
        
        # Variable storage
        self.register_buffer('variable_memory', 
                           torch.zeros(1, config.max_variables, config.hidden_size))
        
        # Variable names (for symbolic access)
        self.variable_names = {}
        self.variable_counter = 0
        
    def allocate_variable(self, name: str) -> int:
        """Allocate a new variable slot"""
        if name in self.variable_names:
            return self.variable_names[name]
        
        if self.variable_counter >= self.max_variables:
            raise ValueError("Maximum variables reached")
        
        idx = self.variable_counter
        self.variable_names[name] = idx
        self.variable_counter += 1
        return idx
    
    def read(self, name: str, memory: torch.Tensor) -> torch.Tensor:
        """Read variable value"""
        if name not in self.variable_names:
            return torch.zeros(self.hidden_size, device=memory.device, dtype=memory.dtype)
        
        idx = self.variable_names[name]
        return memory[:, idx, :]
    
    def write(self, name: str, value: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Write variable value"""
        idx = self.allocate_variable(name)
        memory = memory.clone()
        memory[:, idx, :] = value
        return memory


class InstructionEncoder(nn.Module):
    """Encode program instructions into neural representations"""
    
    def __init__(self, config: ProgramConfig):
        super().__init__()
        self.config = config
        
        # Instruction type embeddings
        num_instruction_types = len(InstructionType)
        self.instruction_embedding = nn.Embedding(num_instruction_types, config.hidden_size)
        
        # Instruction encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
    def forward(self, instruction_type: InstructionType, 
                operands: List[torch.Tensor]) -> torch.Tensor:
        """Encode instruction"""
        # Get instruction type embedding
        type_idx = list(InstructionType).index(instruction_type)
        type_emb = self.instruction_embedding(torch.tensor(type_idx))
        
        # Encode operands
        if len(operands) == 0:
            operand_emb = torch.zeros(self.config.hidden_size)
        elif len(operands) == 1:
            operand_emb = operands[0]
        else:
            # Concatenate and project
            operand_emb = torch.cat(operands[:2], dim=-1)  # Take first two
            if operand_emb.shape[-1] != self.config.hidden_size * 2:
                # Pad if needed
                pad_size = self.config.hidden_size * 2 - operand_emb.shape[-1]
                operand_emb = F.pad(operand_emb, (0, pad_size))
        
        # Combine
        combined = torch.cat([type_emb, operand_emb], dim=-1)
        if combined.shape[-1] != self.config.hidden_size * 2:
            combined = F.pad(combined, (0, self.config.hidden_size * 2 - combined.shape[-1]))
        
        encoded = self.encoder(combined)
        return encoded


class ArithmeticUnit(nn.Module):
    """Neural arithmetic operations"""
    
    def __init__(self, config: ProgramConfig):
        super().__init__()
        self.config = config
        
        # Arithmetic operations
        self.add_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        self.sub_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        self.mul_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        self.div_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Neural addition"""
        combined = torch.cat([a, b], dim=-1)
        return self.add_net(combined)
    
    def sub(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Neural subtraction"""
        combined = torch.cat([a, b], dim=-1)
        return self.sub_net(combined)
    
    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Neural multiplication"""
        combined = torch.cat([a, b], dim=-1)
        return self.mul_net(combined)
    
    def div(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Neural division"""
        combined = torch.cat([a, b], dim=-1)
        return self.div_net(combined)


class ControlFlowUnit(nn.Module):
    """Neural control flow operations (if, loop)"""
    
    def __init__(self, config: ProgramConfig):
        super().__init__()
        self.config = config
        
        # Condition evaluator
        self.condition_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Comparison operations
        self.compare_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
    
    def evaluate_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """Evaluate condition (returns probability of true)"""
        return self.condition_net(condition)
    
    def compare(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compare two values"""
        combined = torch.cat([a, b], dim=-1)
        return self.compare_net(combined)


class ProgramExecutor(nn.Module):
    """Execute neural program instructions"""
    
    def __init__(self, config: ProgramConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.variable_memory = VariableMemory(config)
        self.instruction_encoder = InstructionEncoder(config)
        self.arithmetic = ArithmeticUnit(config)
        self.control_flow = ControlFlowUnit(config)
        
        # Program counter
        self.register_buffer('program_counter', torch.zeros(1, dtype=torch.long))
        
    def execute_instruction(self, instruction: Dict[str, Any], 
                         memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute a single instruction
        
        Args:
            instruction: Instruction dictionary with 'type' and 'operands'
            memory: Current variable memory
            
        Returns:
            updated_memory: Updated memory
            result: Instruction result
        """
        inst_type = instruction['type']
        operands = instruction.get('operands', [])
        
        result = None
        
        if inst_type == InstructionType.ASSIGN:
            # Assign value to variable
            var_name = instruction['variable']
            value = operands[0] if operands else torch.zeros(self.config.hidden_size, device=memory.device)
            memory = self.variable_memory.write(var_name, value, memory)
            result = value
            
        elif inst_type == InstructionType.ADD:
            # Addition
            a = operands[0] if len(operands) > 0 else torch.zeros(self.config.hidden_size, device=memory.device)
            b = operands[1] if len(operands) > 1 else torch.zeros(self.config.hidden_size, device=memory.device)
            result = self.arithmetic.add(a, b)
            
        elif inst_type == InstructionType.SUB:
            # Subtraction
            a = operands[0] if len(operands) > 0 else torch.zeros(self.config.hidden_size, device=memory.device)
            b = operands[1] if len(operands) > 1 else torch.zeros(self.config.hidden_size, device=memory.device)
            result = self.arithmetic.sub(a, b)
            
        elif inst_type == InstructionType.MUL:
            # Multiplication
            a = operands[0] if len(operands) > 0 else torch.zeros(self.config.hidden_size, device=memory.device)
            b = operands[1] if len(operands) > 1 else torch.zeros(self.config.hidden_size, device=memory.device)
            result = self.arithmetic.mul(a, b)
            
        elif inst_type == InstructionType.DIV:
            # Division
            a = operands[0] if len(operands) > 0 else torch.zeros(self.config.hidden_size, device=memory.device)
            b = operands[1] if len(operands) > 1 else torch.zeros(self.config.hidden_size, device=memory.device)
            result = self.arithmetic.div(a, b)
            
        elif inst_type == InstructionType.COMPARE:
            # Comparison
            a = operands[0] if len(operands) > 0 else torch.zeros(self.config.hidden_size, device=memory.device)
            b = operands[1] if len(operands) > 1 else torch.zeros(self.config.hidden_size, device=memory.device)
            result = self.control_flow.compare(a, b)
            
        else:
            result = torch.zeros(self.config.hidden_size, device=memory.device)
        
        return memory, result


class NeuralProgramInterpreter(nn.Module):
    """
    Neural Program Interpreter
    
    Executes symbolic programs using neural networks, enabling differentiable
    program execution and neural-symbolic integration.
    """
    
    def __init__(self, config: ProgramConfig):
        super().__init__()
        self.config = config
        
        # Program executor
        self.executor = ProgramExecutor(config)
        
        # Attention mechanism for instruction sequencing
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=4,
                dropout=config.dropout,
                batch_first=True
            )
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, program: List[Dict[str, Any]], 
                inputs: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Execute a neural program
        
        Args:
            program: List of instruction dictionaries
            inputs: Optional input variables
            
        Returns:
            Dictionary with:
                - output: Program output
                - memory: Final variable memory
                - execution_trace: Execution trace
        """
        batch_size = 1  # Default batch size
        device = next(self.parameters()).device
        
        # Initialize memory
        memory = torch.zeros(batch_size, self.config.max_variables, 
                           self.config.hidden_size, device=device)
        
        # Initialize input variables
        if inputs:
            for var_name, value in inputs.items():
                memory = self.executor.variable_memory.write(var_name, value, memory)
        
        # Execute program
        execution_trace = []
        results = []
        
        for instruction in program:
            # Read operands from memory if they're variable names
            operands = []
            if 'operands' in instruction:
                for op in instruction['operands']:
                    if isinstance(op, str):
                        # Variable name - read from memory
                        var_value = self.executor.variable_memory.read(op, memory)
                        operands.append(var_value.squeeze(0))  # Remove batch dim
                    else:
                        operands.append(op)
            
            instruction['operands'] = operands
            
            # Execute instruction
            memory, result = self.executor.execute_instruction(instruction, memory)
            
            execution_trace.append({
                'instruction': instruction,
                'result': result
            })
            results.append(result)
        
        # Aggregate results
        if results:
            # Use attention if enabled
            if self.config.use_attention and len(results) > 1:
                results_tensor = torch.stack(results).unsqueeze(0)  # [1, seq_len, hidden_size]
                attended, _ = self.attention(results_tensor, results_tensor, results_tensor)
                output = attended[:, -1, :]  # Take last attended result
            else:
                output = results[-1] if results else torch.zeros(self.config.hidden_size, device=device)
        else:
            output = torch.zeros(self.config.hidden_size, device=device)
        
        # Project output
        output = self.output_proj(output)
        
        return {
            'output': output,
            'memory': memory,
            'execution_trace': execution_trace
        }
    
    def parse_program(self, program_text: str) -> List[Dict[str, Any]]:
        """
        Parse text program into instruction list
        
        Args:
            program_text: Text representation of program
            
        Returns:
            List of instruction dictionaries
        """
        instructions = []
        lines = program_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Simple parser for basic operations
            if '=' in line:
                # Assignment: x = value or x = a + b
                parts = line.split('=')
                var_name = parts[0].strip()
                expr = parts[1].strip()
                
                # Parse expression
                if '+' in expr:
                    ops = expr.split('+')
                    instructions.append({
                        'type': InstructionType.ASSIGN,
                        'variable': var_name,
                        'operands': [op.strip() for op in ops]
                    })
                elif '-' in expr:
                    ops = expr.split('-')
                    instructions.append({
                        'type': InstructionType.ASSIGN,
                        'variable': var_name,
                        'operands': [op.strip() for op in ops]
                    })
                else:
                    instructions.append({
                        'type': InstructionType.ASSIGN,
                        'variable': var_name,
                        'operands': [expr]
                    })
            elif line.startswith('if'):
                # Conditional
                condition = line[2:].strip()
                instructions.append({
                    'type': InstructionType.IF,
                    'condition': condition
                })
            elif line.startswith('return'):
                # Return
                value = line[6:].strip()
                instructions.append({
                    'type': InstructionType.RETURN,
                    'operands': [value]
                })
        
        return instructions
    
    def execute_text_program(self, program_text: str, 
                            inputs: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Execute program from text"""
        program = self.parse_program(program_text)
        return self.forward(program, inputs)


def create_neural_interpreter(hidden_size: int = 256, max_variables: int = 20,
                              use_attention: bool = True) -> NeuralProgramInterpreter:
    """
    Factory function to create a neural program interpreter
    
    Args:
        hidden_size: Hidden dimension size
        max_variables: Maximum number of variables
        use_attention: Whether to use attention for instruction sequencing
        
    Returns:
        NeuralProgramInterpreter instance
    """
    config = ProgramConfig(
        hidden_size=hidden_size,
        max_variables=max_variables,
        use_attention=use_attention
    )
    return NeuralProgramInterpreter(config)

