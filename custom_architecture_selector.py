"""
Custom Architecture Selector for NeSy-CML
Comprehensive intelligent selection system for all available architectures and components
Updated to include all new modules: advanced losses, threshold optimization, meta-learning
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn

@dataclass
class ArchitectureInfo:
    """Information about a NeSy-CML architecture"""
    name: str
    description: str
    feature_dim: int
    best_for: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    advantages: List[str] = field(default_factory=list)
    architecture_type: str = ""
    memory_efficient: bool = False
    pretrained: bool = False
    recommended_loss: Optional[str] = None
    recommended_threshold_opt: bool = False
    supports_meta_learning: bool = False

@dataclass
class TrainingConfig:
    """Complete training configuration with all components"""
    architecture: str
    loss_function: str
    use_threshold_optimization: bool
    use_meta_learning: bool
    symbolic_controller: Optional[str] = None
    replay_buffer: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

class ArchitectureSelector:
    """Comprehensive intelligent selector for all NeSy-CML architectures and components"""
    
    def __init__(self):
        self.architectures = self._initialize_all_architectures()
        self.loss_functions = self._initialize_loss_functions()
        self.symbolic_controllers = self._initialize_symbolic_controllers()
        self.replay_buffers = self._initialize_replay_buffers()
    
    def _initialize_all_architectures(self) -> Dict[str, ArchitectureInfo]:
        """Initialize all available architectures with their properties"""
        return {
            # === Standard Vision Architectures ===
            "resnet18": ArchitectureInfo(
                name="ResNet-18",
                description="Lightweight ResNet architecture, excellent for 4GB GPU constraints",
                feature_dim=512,
                best_for=["image_classification", "transfer_learning", "limited_memory"],
                use_cases=["chess_analysis", "general_vision", "resource_constrained"],
                advantages=["Fast training", "Memory efficient", "Proven architecture", "Pretrained available"],
                architecture_type="convolutional",
                memory_efficient=True,
                pretrained=True,
                recommended_loss="CombinedLoss",
                recommended_threshold_opt=True,
                supports_meta_learning=True
            ),
            "resnet50": ArchitectureInfo(
                name="ResNet-50",
                description="Medium-sized ResNet with better accuracy than ResNet-18",
                feature_dim=2048,
                best_for=["image_classification", "high_accuracy", "sufficient_memory"],
                use_cases=["detailed_analysis", "production_systems"],
                advantages=["Better accuracy", "Pretrained", "Well-tested"],
                architecture_type="convolutional",
                pretrained=True,
                recommended_loss="CombinedLoss",
                recommended_threshold_opt=True
            ),
            "resnet101": ArchitectureInfo(
                name="ResNet-101",
                description="Large ResNet architecture for maximum accuracy",
                feature_dim=2048,
                best_for=["high_accuracy", "research", "sufficient_resources"],
                use_cases=["benchmarking", "research_projects"],
                advantages=["Highest accuracy", "Pretrained", "State-of-the-art"],
                architecture_type="convolutional",
                pretrained=True,
                recommended_loss="CombinedLoss"
            ),
            "efficientnet_b0": ArchitectureInfo(
                name="EfficientNet-B0",
                description="Efficient architecture balancing accuracy and speed",
                feature_dim=1280,
                best_for=["efficiency", "mobile_deployment", "balanced_performance"],
                use_cases=["mobile_apps", "edge_devices", "efficient_inference"],
                advantages=["Efficient", "Good accuracy", "Low memory"],
                architecture_type="convolutional",
                pretrained=True,
                memory_efficient=True,
                recommended_loss="FocalLoss"
            ),
            "mobilenet_v2": ArchitectureInfo(
                name="MobileNet-V2",
                description="Ultra-lightweight architecture for mobile devices",
                feature_dim=1280,
                best_for=["mobile", "edge_computing", "ultra_low_memory"],
                use_cases=["mobile_apps", "embedded_systems", "real_time"],
                advantages=["Very fast", "Very efficient", "Mobile-optimized"],
                architecture_type="convolutional",
                pretrained=True,
                memory_efficient=True,
                recommended_loss="FocalLoss"
            ),
            # === Custom Architectures ===
            "custom_cnn": ArchitectureInfo(
                name="Custom CNN",
                description="Specialized CNN architecture optimized for image processing with NeSy-CML integration",
                feature_dim=512,
                best_for=["image_classification", "object_detection", "computer_vision"],
                use_cases=["secure_image_training", "medical_imaging", "surveillance_analysis"],
                advantages=["Fast inference", "Memory efficient", "Excellent for spatial features", "Customizable"],
                architecture_type="convolutional",
                recommended_loss="CombinedLoss",
                recommended_threshold_opt=True
            ),
            "custom_transformer": ArchitectureInfo(
                name="Custom Transformer",
                description="Advanced transformer architecture with attention mechanisms for multimodal NeSy-CML tasks",
                feature_dim=512,
                best_for=["multimodal_reasoning", "attention_based_tasks", "complex_patterns"],
                use_cases=["validation_testing", "multimodal_analysis", "attention_heavy_tasks"],
                advantages=["Attention mechanisms", "Parallel processing", "Excellent for sequences"],
                architecture_type="transformer",
                recommended_loss="CombinedLoss",
                recommended_threshold_opt=True,
                supports_meta_learning=True
            ),
            "custom_lstm": ArchitectureInfo(
                name="Custom LSTM",
                description="Specialized LSTM architecture for sequential data processing in NeSy-CML",
                feature_dim=512,
                best_for=["sequential_data", "code_analysis", "time_series"],
                use_cases=["code_training", "sequential_analysis", "pattern_recognition"],
                advantages=["Sequential memory", "Pattern understanding", "Code structure analysis"],
                architecture_type="recurrent",
                recommended_loss="FocalLoss",
                recommended_threshold_opt=True
            ),
            # === Specialized Architectures ===
            "newson": ArchitectureInfo(
                name="NewSon Transformer",
                description="Multimodal transformer architecture optimized for NLP and image processing",
                feature_dim=512,
                best_for=["multimodal", "nlp", "text_and_images", "chess_analysis"],
                use_cases=["chess_training", "multimodal_understanding", "text_image_fusion"],
                advantages=["Multimodal", "NLP optimized", "Image support", "Best for chess"],
                architecture_type="transformer",
                recommended_loss="CombinedLoss",
                recommended_threshold_opt=True,
                supports_meta_learning=True
            ),
            "gpt_style": ArchitectureInfo(
                name="GPT-Style Transformer",
                description="Decoder-only transformer for generation and code tasks",
                feature_dim=512,
                best_for=["code_generation", "text_generation", "autoregressive"],
                use_cases=["programming_analysis", "code_completion", "text_generation"],
                advantages=["Generation capability", "Code understanding", "Autoregressive"],
                architecture_type="transformer",
                recommended_loss="FocalLoss",
                supports_meta_learning=True
            ),
            "bert_style": ArchitectureInfo(
                name="BERT-Style Transformer",
                description="Encoder-only transformer for understanding tasks",
                feature_dim=512,
                best_for=["text_understanding", "classification", "encoding"],
                use_cases=["text_classification", "semantic_understanding", "feature_extraction"],
                advantages=["Bidirectional", "Understanding", "Pretrained available"],
                architecture_type="transformer",
                recommended_loss="CombinedLoss",
                recommended_threshold_opt=True
            ),
            "multimodal_transformer": ArchitectureInfo(
                name="Multimodal Transformer",
                description="Specialized transformer for combined text, code, and image processing",
                feature_dim=512,
                best_for=["multimodal", "cross_modal", "complex_reasoning"],
                use_cases=["multimodal_training", "cross_modal_understanding", "complex_tasks"],
                advantages=["True multimodal", "Cross-modal attention", "Flexible"],
                architecture_type="transformer",
                recommended_loss="CombinedLoss",
                recommended_threshold_opt=True,
                supports_meta_learning=True
            ),
            "code_transformer": ArchitectureInfo(
                name="Code Transformer",
                description="Specialized transformer for code analysis and generation",
                feature_dim=512,
                best_for=["code_analysis", "programming", "syntax_understanding"],
                use_cases=["code_training", "programming_analysis", "syntax_parsing"],
                advantages=["Code-aware", "Syntax understanding", "Programming optimized"],
                architecture_type="transformer",
                recommended_loss="FocalLoss",
                supports_meta_learning=True
            )
        }
    
    def _initialize_loss_functions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available loss functions with recommendations"""
        return {
            "CombinedLoss": {
                "description": "Combines Focal + Dice + F1 loss (best for multilabel, imbalanced data)",
                "best_for": ["multilabel", "imbalanced", "chess", "high_f1_target"],
                "components": ["FocalLoss (40%)", "DiceLoss (30%)", "F1Loss (30%)"],
                "recommended_for": ["chess_analysis", "multilabel_classification", "imbalanced_datasets"]
            },
            "FocalLoss": {
                "description": "Addresses class imbalance with focusing parameter",
                "best_for": ["imbalanced", "hard_examples", "classification"],
                "recommended_for": ["code_analysis", "sequential_data", "imbalanced_classes"]
            },
            "DiceLoss": {
                "description": "Optimized for multilabel classification",
                "best_for": ["multilabel", "segmentation", "overlap"],
                "recommended_for": ["multilabel_tasks", "overlapping_classes"]
            },
            "F1Loss": {
                "description": "Directly optimizes F1 score",
                "best_for": ["f1_optimization", "multilabel", "precision_recall"],
                "recommended_for": ["f1_critical_tasks", "multilabel_optimization"]
            },
            "AsymmetricLoss": {
                "description": "Different penalties for false positives vs false negatives",
                "best_for": ["asymmetric_costs", "precision_critical", "recall_critical"],
                "recommended_for": ["cost_sensitive", "asymmetric_requirements"]
            }
        }
    
    def _initialize_symbolic_controllers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available symbolic controllers"""
        return {
            "production_rule": {
                "description": "IF-THEN rules with forward chaining",
                "best_for": ["expert_systems", "rule_based", "interpretable"],
                "complexity": "Low"
            },
            "fuzzy_logic": {
                "description": "Fuzzy sets and membership functions",
                "best_for": ["uncertain_reasoning", "continuous_activation"],
                "complexity": "Medium"
            },
            "logic_based": {
                "description": "Propositional logic (AND, OR, NOT, IMPLIES)",
                "best_for": ["logical_inference", "rule_based"],
                "complexity": "Medium"
            },
            "neuro_symbolic": {
                "description": "Differentiable symbolic operations",
                "best_for": ["learning_symbols", "differentiable_reasoning"],
                "complexity": "High"
            },
            "graph_based": {
                "description": "Graph neural networks for symbolic reasoning",
                "best_for": ["relational_reasoning", "structured_knowledge"],
                "complexity": "High"
            }
        }
    
    def _initialize_replay_buffers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available replay buffers"""
        return {
            "text": {
                "description": "Text-based training experiences",
                "best_for": ["chess", "poetry", "programming", "nlp"],
                "recommended_size": 2000
            },
            "prioritized": {
                "description": "Priority-based sampling",
                "best_for": ["important_experiences", "focused_learning"],
                "recommended_size": 1500
            },
            "adaptive": {
                "description": "Performance-based adaptive sampling",
                "best_for": ["dynamic_environments", "adaptive_learning"],
                "recommended_size": 1000
            },
            "symbolic": {
                "description": "Symbolic reasoning experiences",
                "best_for": ["symbolic_ai", "reasoning_tasks"],
                "recommended_size": 800
            }
        }
    
    def select_architecture(self, task_type: str, data_type: str, 
                          requirements: List[str] = None,
                          memory_constraint: str = "medium") -> str:
        """
        Select the best architecture based on task requirements
        
        Args:
            task_type: Type of task (chess_analysis, code_analysis, image_classification, etc.)
            data_type: Type of data (text, images, multimodal, sequences)
            requirements: List of requirements (memory_efficient, high_accuracy, fast, etc.)
            memory_constraint: "low" (4GB), "medium" (8GB), "high" (16GB+)
        
        Returns:
            Architecture name
        """
        if requirements is None:
            requirements = []
        
        requirements_lower = [r.lower() for r in requirements]
        
        # Chess-specific selection
        if task_type == "chess_analysis":
            if memory_constraint == "low":
                return "newson"  # Best for chess with memory constraints
            else:
                return "newson"  # NewSon is optimized for chess
        
        # Code/programming selection
        if task_type in ["code_analysis", "programming", "code_generation"]:
            if "generation" in requirements_lower:
                return "gpt_style"
            else:
                return "code_transformer"
        
        # Image classification selection
        if task_type in ["image_classification", "object_detection", "computer_vision"]:
            if memory_constraint == "low":
                if "efficient" in requirements_lower:
                    return "efficientnet_b0"
                else:
                    return "resnet18"
            elif memory_constraint == "medium":
                return "resnet50"
            else:
                return "resnet101"
        
        # Multimodal selection
        if task_type in ["multimodal", "multimodal_reasoning", "text_and_images"]:
            return "multimodal_transformer"
        
        # Text/NLP selection
        if task_type in ["text_classification", "nlp", "text_understanding"]:
            return "bert_style"
        
        # Sequential data selection
        if task_type in ["sequential_data", "time_series"]:
            return "custom_lstm"
        
        # Data type-based selection
        if data_type == "images":
            if memory_constraint == "low":
                return "resnet18"
            return "resnet50"
        elif data_type == "multimodal":
            return "multimodal_transformer"
        elif data_type == "text":
            if "generation" in requirements_lower:
                return "gpt_style"
            return "bert_style"
        elif data_type == "sequences":
            return "custom_lstm"
        
        # Requirements-based selection
        if "memory_efficient" in requirements_lower or "low_memory" in requirements_lower:
            if data_type == "images":
                return "mobilenet_v2"
            return "resnet18"
        
        if "high_accuracy" in requirements_lower:
            if data_type == "images":
                return "resnet101"
            return "newson"
        
        if "attention" in requirements_lower or "parallel" in requirements_lower:
            return "custom_transformer"
        
        if "sequential" in requirements_lower or "memory" in requirements_lower:
            return "custom_lstm"
        
        if "spatial" in requirements_lower or "convolution" in requirements_lower:
            return "custom_cnn"
        
        # Default fallback - NewSon for general use
        return "newson"
    
    def get_architecture_info(self, architecture_name: str) -> ArchitectureInfo:
        """Get detailed information about a specific architecture"""
        if architecture_name not in self.architectures:
            raise ValueError(f"Unknown architecture: {architecture_name}. "
                           f"Available: {list(self.architectures.keys())}")
        return self.architectures[architecture_name]
    
    def list_architectures(self) -> List[str]:
        """List all available architectures"""
        return list(self.architectures.keys())
    
    def list_loss_functions(self) -> List[str]:
        """List all available loss functions"""
        return list(self.loss_functions.keys())
    
    def list_symbolic_controllers(self) -> List[str]:
        """List all available symbolic controllers"""
        return list(self.symbolic_controllers.keys())
    
    def list_replay_buffers(self) -> List[str]:
        """List all available replay buffers"""
        return list(self.replay_buffers.keys())
    
    def recommend_loss_function(self, task_type: str, data_type: str, 
                               is_multilabel: bool = True,
                               is_imbalanced: bool = True) -> str:
        """Recommend the best loss function for the task"""
        if is_multilabel and is_imbalanced:
            return "CombinedLoss"  # Best for chess and multilabel tasks
        elif is_imbalanced:
            return "FocalLoss"
        elif is_multilabel:
            return "DiceLoss"
        else:
            return "FocalLoss"  # Default
    
    def recommend_symbolic_controller(self, task_type: str, 
                                     complexity_preference: str = "medium") -> str:
        """Recommend symbolic controller for the task"""
        if task_type == "chess_analysis":
            return "production_rule"  # Good for chess rules
        elif complexity_preference == "low":
            return "production_rule"
        elif complexity_preference == "high":
            return "neuro_symbolic"
        else:
            return "fuzzy_logic"  # Balanced
    
    def recommend_replay_buffer(self, task_type: str) -> str:
        """Recommend replay buffer for the task"""
        if task_type in ["chess_analysis", "poetry", "programming", "nlp"]:
            return "text"
        elif task_type in ["image_classification", "computer_vision"]:
            return "prioritized"
        else:
            return "adaptive"
    
    def get_recommended_config(self, architecture_name: str, 
                              task_type: str = "chess_analysis",
                              task_specific: bool = True) -> Dict:
        """Get recommended configuration for a specific architecture with all new modules"""
        arch_info = self.get_architecture_info(architecture_name)
        
        # Base configuration
        base_config = {
            'neural_architecture': architecture_name,
            'hidden_sizes': [512, 256, 128] if arch_info.feature_dim >= 512 else [256, 128],
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
            'use_prior_state': True,
            # New module integrations
            'loss_function': arch_info.recommended_loss or self.recommend_loss_function(
                task_type, "text", is_multilabel=True, is_imbalanced=True
            ),
            'use_threshold_optimization': arch_info.recommended_threshold_opt,
            'use_meta_learning': arch_info.supports_meta_learning,
            'symbolic_controller': self.recommend_symbolic_controller(task_type),
            'replay_buffer': self.recommend_replay_buffer(task_type)
        }
        
        # Task-specific optimizations
        if task_specific:
            # Vision architectures
            if architecture_name in ["resnet18", "resnet50", "resnet101", "custom_cnn"]:
                base_config.update({
                    'batch_size': 8 if architecture_name == "resnet18" else 4,
                    'learning_rate': 0.001,
                    'mixed_precision': True,
                    'use_threshold_optimization': True
                })
            # Transformer architectures
            elif architecture_name in ["custom_transformer", "newson", "gpt_style", 
                                     "bert_style", "multimodal_transformer", "code_transformer"]:
                base_config.update({
                    'batch_size': 4,
                    'learning_rate': 0.0005,
                    'gradient_accumulation_steps': 2,
                    'use_threshold_optimization': True,
                    'use_meta_learning': True
                })
            # LSTM architectures
            elif architecture_name == "custom_lstm":
                base_config.update({
                    'batch_size': 6,
                    'learning_rate': 0.001,
                    'gradient_clipping': 1.0,
                    'use_threshold_optimization': True
                })
            # Efficient architectures
            elif architecture_name in ["efficientnet_b0", "mobilenet_v2"]:
                base_config.update({
                    'batch_size': 8,
                    'learning_rate': 0.001,
                    'mixed_precision': True,
                    'loss_function': 'FocalLoss'  # More efficient for these
                })
            
            # Chess-specific optimizations
            if task_type == "chess_analysis":
                base_config.update({
                    'loss_function': 'CombinedLoss',  # Best for chess multilabel
                    'use_threshold_optimization': True,  # Critical for F1 ≥ 0.92
                    'batch_size': 4,  # Chess text data
                    'learning_rate': 2e-4,
                    'num_classes': 9,
                    'evaluation_mode': 'multilabel'
                })
        
        return base_config
    
    def create_complete_training_config(self, task_type: str, data_type: str, 
                                       requirements: List[str] = None,
                                       memory_constraint: str = "medium") -> TrainingConfig:
        """
        Create a complete training configuration with architecture, loss, and all components
        
        Args:
            task_type: Type of task
            data_type: Type of data
            requirements: List of requirements
            memory_constraint: Memory constraint level
        
        Returns:
            Complete TrainingConfig with all recommendations
        """
        if requirements is None:
            requirements = []
        
        # Select best architecture
        architecture = self.select_architecture(task_type, data_type, requirements, memory_constraint)
        arch_info = self.get_architecture_info(architecture)
        
        # Get recommended config
        config = self.get_recommended_config(architecture, task_type, task_specific=True)
        
        # Determine if multilabel and imbalanced
        is_multilabel = task_type in ["chess_analysis", "multilabel_classification"]
        is_imbalanced = task_type in ["chess_analysis", "code_analysis"]
        
        # Create complete config
        training_config = TrainingConfig(
            architecture=architecture,
            loss_function=config.get('loss_function', self.recommend_loss_function(
                task_type, data_type, is_multilabel, is_imbalanced
            )),
            use_threshold_optimization=config.get('use_threshold_optimization', is_multilabel),
            use_meta_learning=config.get('use_meta_learning', False),
            symbolic_controller=config.get('symbolic_controller'),
            replay_buffer=config.get('replay_buffer'),
            config=config
        )
        
        return training_config
    
    def create_training_config(self, task_type: str, data_type: str, 
                             requirements: List[str] = None,
                             memory_constraint: str = "medium") -> Dict:
        """Create a training configuration dictionary (backward compatibility)"""
        complete_config = self.create_complete_training_config(
            task_type, data_type, requirements, memory_constraint
        )
        
        # Convert to dict format
        config_dict = complete_config.config.copy()
        config_dict.update({
            'task_type': task_type,
            'data_type': data_type,
            'requirements': requirements or [],
            'selected_architecture': complete_config.architecture,
            'loss_function': complete_config.loss_function,
            'use_threshold_optimization': complete_config.use_threshold_optimization,
            'use_meta_learning': complete_config.use_meta_learning,
            'symbolic_controller': complete_config.symbolic_controller,
            'replay_buffer': complete_config.replay_buffer
        })
        
        return config_dict

def demonstrate_architecture_selection():
    """Demonstrate the comprehensive architecture selection process"""
    selector = ArchitectureSelector()
    
    print("🧠 NeSy-CML Comprehensive Architecture Selector")
    print("=" * 70)
    print(f"Available Architectures: {len(selector.list_architectures())}")
    print(f"Available Loss Functions: {len(selector.list_loss_functions())}")
    print(f"Available Controllers: {len(selector.list_symbolic_controllers())}")
    print(f"Available Buffers: {len(selector.list_replay_buffers())}")
    print("=" * 70)
    
    # Example scenarios
    scenarios = [
        {
            'task_type': 'chess_analysis',
            'data_type': 'text',
            'requirements': ['high_f1', 'multilabel', 'imbalanced'],
            'memory_constraint': 'low'
        },
        {
            'task_type': 'code_analysis',
            'data_type': 'sequences',
            'requirements': ['sequential_memory', 'pattern_understanding'],
            'memory_constraint': 'medium'
        },
        {
            'task_type': 'image_classification',
            'data_type': 'images',
            'requirements': ['memory_efficient', 'fast'],
            'memory_constraint': 'low'
        },
        {
            'task_type': 'multimodal_reasoning',
            'data_type': 'multimodal',
            'requirements': ['attention', 'parallel_processing'],
            'memory_constraint': 'medium'
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['task_type']}")
        print(f"   Data: {scenario['data_type']} | Memory: {scenario['memory_constraint']}")
        print(f"   Requirements: {scenario['requirements']}")
        print("-" * 70)
        
        # Get complete training config
        complete_config = selector.create_complete_training_config(
            scenario['task_type'],
            scenario['data_type'],
            scenario['requirements'],
            scenario['memory_constraint']
        )
        
        # Get architecture info
        arch_info = selector.get_architecture_info(complete_config.architecture)
        
        print(f"   🎯 Architecture: {arch_info.name}")
        print(f"      Description: {arch_info.description}")
        print(f"      Feature Dim: {arch_info.feature_dim}")
        print(f"      Advantages: {', '.join(arch_info.advantages[:3])}")
        
        print(f"\n   📦 Recommended Components:")
        print(f"      Loss Function: {complete_config.loss_function}")
        print(f"      Threshold Optimization: {complete_config.use_threshold_optimization}")
        print(f"      Meta-Learning: {complete_config.use_meta_learning}")
        print(f"      Symbolic Controller: {complete_config.symbolic_controller}")
        print(f"      Replay Buffer: {complete_config.replay_buffer}")
        
        print(f"\n   ⚙️  Training Config:")
        config = complete_config.config
        print(f"      Batch Size: {config.get('batch_size', 'N/A')}")
        print(f"      Learning Rate: {config.get('learning_rate', 'N/A')}")
        print(f"      Mixed Precision: {config.get('mixed_precision', False)}")
        if config.get('use_threshold_optimization'):
            print(f"      ✅ Threshold Optimization Enabled (for F1 ≥ 0.92)")
    
    print("\n" + "=" * 70)
    print("✅ Comprehensive architecture selection complete!")
    print("=" * 70)

# Backward compatibility alias
CustomArchitectureSelector = ArchitectureSelector

if __name__ == "__main__":
    demonstrate_architecture_selection()
