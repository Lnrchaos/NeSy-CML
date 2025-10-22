"""
Custom Architecture Selector for NeuroSym-CML
Provides intelligent selection of the appropriate custom architecture based on task type
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class CustomArchitectureInfo:
    """Information about a custom NeuroSym-CML architecture"""
    name: str
    description: str
    feature_dim: int
    best_for: List[str]
    use_cases: List[str]
    advantages: List[str]
    architecture_type: str

class CustomArchitectureSelector:
    """Intelligent selector for NeuroSym-CML custom architectures"""
    
    def __init__(self):
        self.architectures = self._initialize_custom_architectures()
    
    def _initialize_custom_architectures(self) -> Dict[str, CustomArchitectureInfo]:
        """Initialize custom architectures with their properties"""
        return {
            "custom_cnn": CustomArchitectureInfo(
                name="Custom CNN",
                description="Specialized CNN architecture optimized for image processing with NeuroSym-CML integration",
                feature_dim=512,
                best_for=["image_classification", "object_detection", "computer_vision"],
                use_cases=["secure_image_training", "medical_imaging", "surveillance_analysis"],
                advantages=["Fast inference", "Memory efficient", "Excellent for spatial features"],
                architecture_type="convolutional"
            ),
            "custom_transformer": CustomArchitectureInfo(
                name="Custom Transformer",
                description="Advanced transformer architecture with attention mechanisms for multimodal NeuroSym-CML tasks",
                feature_dim=512,
                best_for=["multimodal_reasoning", "attention_based_tasks", "complex_patterns"],
                use_cases=["validation_testing", "multimodal_analysis", "attention_heavy_tasks"],
                advantages=["Attention mechanisms", "Parallel processing", "Excellent for sequences"],
                architecture_type="transformer"
            ),
            "custom_lstm": CustomArchitectureInfo(
                name="Custom LSTM",
                description="Specialized LSTM architecture for sequential data processing in NeuroSym-CML",
                feature_dim=512,
                best_for=["sequential_data", "code_analysis", "time_series"],
                use_cases=["gambit_code_training", "sequential_analysis", "temporal_patterns"],
                advantages=["Sequential memory", "Temporal understanding", "Code structure analysis"],
                architecture_type="recurrent"
            )
        }
    
    def select_architecture(self, task_type: str, data_type: str, requirements: List[str]) -> str:
        """Select the best custom architecture based on task requirements"""
        
        # Task-based selection logic
        if task_type in ["image_classification", "object_detection", "computer_vision"]:
            return "custom_cnn"
        elif task_type in ["multimodal_reasoning", "attention_based_tasks", "validation"]:
            return "custom_transformer"
        elif task_type in ["sequential_data", "code_analysis", "time_series"]:
            return "custom_lstm"
        
        # Data type-based selection
        if data_type in ["images", "visual_data"]:
            return "custom_cnn"
        elif data_type in ["multimodal", "text_and_images"]:
            return "custom_transformer"
        elif data_type in ["sequences", "code", "temporal"]:
            return "custom_lstm"
        
        # Requirements-based selection
        if "attention" in requirements or "parallel" in requirements:
            return "custom_transformer"
        elif "sequential" in requirements or "memory" in requirements:
            return "custom_lstm"
        elif "spatial" in requirements or "convolution" in requirements:
            return "custom_cnn"
        
        # Default fallback
        return "custom_transformer"
    
    def get_architecture_info(self, architecture_name: str) -> CustomArchitectureInfo:
        """Get detailed information about a specific architecture"""
        if architecture_name not in self.architectures:
            raise ValueError(f"Unknown architecture: {architecture_name}")
        return self.architectures[architecture_name]
    
    def list_architectures(self) -> List[str]:
        """List all available custom architectures"""
        return list(self.architectures.keys())
    
    def get_recommended_config(self, architecture_name: str, task_specific: bool = True) -> Dict:
        """Get recommended configuration for a specific architecture"""
        arch_info = self.get_architecture_info(architecture_name)
        
        base_config = {
            'neural_architecture': architecture_name,
            'hidden_sizes': [512, 256, 128],
            'use_symbolic_reasoning': True,
            'rule_set_size': 100,
            'rule_embedding_dim': 64,
            'memory_size': 1000,
            'meta_batch_size': 4,
            'inner_lr': 0.01,
            'outer_lr': 0.001,
            'memory_sampling_strategy': 'random',
            'use_attention': True,
            'use_task_metadata': True,
            'use_prior_state': True
        }
        
        # Task-specific optimizations
        if task_specific:
            if architecture_name == "custom_cnn":
                base_config.update({
                    'batch_size': 8,  # CNN can handle larger batches
                    'learning_rate': 0.001,
                    'mixed_precision': True  # CNNs benefit from mixed precision
                })
            elif architecture_name == "custom_transformer":
                base_config.update({
                    'batch_size': 4,  # Transformers need smaller batches
                    'learning_rate': 0.0005,  # Lower LR for stability
                    'gradient_accumulation_steps': 2
                })
            elif architecture_name == "custom_lstm":
                base_config.update({
                    'batch_size': 6,  # LSTM middle ground
                    'learning_rate': 0.001,
                    'gradient_clipping': 1.0  # LSTMs benefit from gradient clipping
                })
        
        return base_config
    
    def create_training_config(self, task_type: str, data_type: str, 
                             requirements: List[str] = None) -> Dict:
        """Create a complete training configuration with the best architecture"""
        if requirements is None:
            requirements = []
        
        # Select best architecture
        architecture = self.select_architecture(task_type, data_type, requirements)
        
        # Get recommended config
        config = self.get_recommended_config(architecture, task_specific=True)
        
        # Add task-specific parameters
        config.update({
            'task_type': task_type,
            'data_type': data_type,
            'requirements': requirements,
            'selected_architecture': architecture
        })
        
        return config

def demonstrate_architecture_selection():
    """Demonstrate the architecture selection process"""
    selector = CustomArchitectureSelector()
    
    print("🧠 NeuroSym-CML Custom Architecture Selector")
    print("=" * 60)
    
    # Example scenarios
    scenarios = [
        {
            'task_type': 'image_classification',
            'data_type': 'images',
            'requirements': ['spatial_features', 'fast_inference']
        },
        {
            'task_type': 'multimodal_reasoning',
            'data_type': 'multimodal',
            'requirements': ['attention', 'parallel_processing']
        },
        {
            'task_type': 'code_analysis',
            'data_type': 'sequences',
            'requirements': ['sequential_memory', 'temporal_understanding']
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}:")
        print(f"   Task: {scenario['task_type']}")
        print(f"   Data: {scenario['data_type']}")
        print(f"   Requirements: {scenario['requirements']}")
        
        # Select architecture
        architecture = selector.select_architecture(
            scenario['task_type'],
            scenario['data_type'],
            scenario['requirements']
        )
        
        # Get architecture info
        arch_info = selector.get_architecture_info(architecture)
        
        print(f"   🎯 Selected: {arch_info.name}")
        print(f"   📝 Description: {arch_info.description}")
        print(f"   ✅ Advantages: {', '.join(arch_info.advantages)}")
        
        # Get recommended config
        config = selector.get_recommended_config(architecture)
        print(f"   ⚙️  Recommended batch size: {config.get('batch_size', 'N/A')}")
        print(f"   ⚙️  Recommended learning rate: {config.get('learning_rate', 'N/A')}")

if __name__ == "__main__":
    demonstrate_architecture_selection()
