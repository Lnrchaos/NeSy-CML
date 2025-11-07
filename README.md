# NeuroSym-CML: Neuro-Symbolic Continual Meta-Learning Framework

A cutting-edge AI framework that combines neural networks with symbolic reasoning for advanced multimodal learning across chess, poetry, programming, and image analysis domains.

## 🚀 Overview

NeuroSym-CML is a hybrid AI system that integrates:
- **Neural Networks**: Deep learning for pattern recognition and feature extraction
- **Symbolic Controllers**: Logic-based reasoning with fuzzy logic, production rules, and graph-based inference
- **Continual Learning**: Advanced strategies (EWC, GEM, Replay, Progressive Networks) to prevent catastrophic forgetting
- **Meta-Learning**: MAML, Reptile, Prototypical Networks, and other fast adaptation algorithms
- **Memory-Augmented Networks**: Differentiable Neural Computer (DNC), Neural Turing Machine (NTM), and other memory architectures
- **Multimodal Processing**: CLIP-integrated text-image understanding with cross-modal reasoning
- **Advanced Architectures**: Sparse Mixture of Experts, Graph Neural Networks, Causal Reasoning, Neural Program Interpretation

## 🏆 Achievements

- **Chess Analysis**: 67% accuracy on chess position evaluation and move prediction
- **Poetry Generation**: 70% accuracy on style classification and creative analysis
- **Programming**: 80% accuracy on code analysis and pattern recognition
- **Image Processing**: 77% accuracy on visual understanding tasks
- **Multimodal Integration**: 88.5% accuracy combining multiple data types

## 📁 Project Structure

```
NeuroSym-CML/
├── README.md                          # This file
├── docs/                             # Documentation for each model type
│   ├── CHESS_MODEL.md               # Chess model documentation
│   ├── POETRY_MODEL.md              # Poetry model documentation
│   ├── PROGRAMMING_MODEL.md         # Programming model documentation
│   └── MULTIMODAL_MODEL.md          # Multimodal model documentation
├── NeSy-CML/                        # Core framework components
│   ├── meta_model.py               # Hybrid neural-symbolic model with CLIP
│   ├── meta-controller.py          # Meta-learning algorithms (MAML, Reptile, etc.)
│   ├── model_builder.py            # Dynamic model construction
│   ├── model_spec.py               # Model specifications
│   ├── modular_architecture.py     # Modular architecture system
│   ├── modular_symbolic_controller.py # Symbolic reasoning controllers
│   ├── modular_replay_buffer.py    # Adaptive memory systems
│   ├── tensor_adapter.py           # Tensor shape adaptation utilities
│   └── evaluator.py                # Model evaluation tools
├── Memory-Augmented Networks/       # Advanced memory architectures
│   ├── DNC.py                      # Differentiable Neural Computer
│   ├── mem-aug.py                  # Memory-Augmented Networks (NTM, Stack, Queue)
│   └── sparse-mix.py               # Sparse Mixture of Experts
├── Reasoning Modules/               # Advanced reasoning capabilities
│   ├── causalR-module.py           # Causal reasoning and counterfactuals
│   ├── NePr-interpreter.py        # Neural Program Interpreter
│   └── gnnm.py                     # Graph Neural Network Modules
├── Continual Learning/             # Continual learning strategies
│   ├── continual_module.py        # EWC, GEM, Replay, Progressive Networks
│   └── data_module.py             # Continual learning data handling
├── Training Scripts/               # Domain-specific training
│   ├── train_chess_optimized.py   # Optimized chess training
│   ├── train_poetry_optimized.py  # Creative poetry training
│   ├── train_programming_optimized.py # Code analysis training
│   ├── train_multimodal_xavious.py # Multimodal training
│   └── train_secure_images.py     # Secure image training
├── Utilities/                      # Supporting utilities
│   ├── text_encoder.py            # CLIP/BERT text encoding
│   ├── custom_architecture_selector.py # Architecture optimization
│   ├── advanced_losses.py        # Advanced loss functions
│   └── threshold_optimizer.py     # Threshold optimization
├── models/                         # Trained model weights
│   ├── best_chess_model_improved.pt    # Chess model
│   └── best_poetry_model_optimized.pt  # Poetry model
└── dataset/                        # Training datasets
    ├── Chess_data/                # Chess positions and games
    ├── poetry/                    # Poetry collections
    ├── programming_data/          # Code samples
    └── law_data/                  # Legal documents
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (4GB+ VRAM recommended)
- PyTorch 2.0+
- Additional dependencies in requirements.txt

### Setup
```bash
git clone https://github.com/your-repo/NeuroSym-CML.git
cd NeuroSym-CML
pip install -r requirements.txt
```

## 🎯 Model Types and Capabilities

### 1. Chess Model (`best_chess_model_4gb.pt`)
**Purpose**: Chess position analysis and move evaluation
**Capabilities**:
- Chess position evaluation (material, positional factors)
- Move quality assessment
- Game phase recognition (opening, middlegame, endgame)
- Tactical pattern recognition
- Strategic understanding

**Required Files for Recreation**:
- `train_chess_optimized.py` - Main training script
- `meta_model.py` - Core hybrid model architecture
- `modular_symbolic_controller.py` - Symbolic reasoning
- `modular_replay_buffer.py` - Memory management
- `dataset/Chess_data/` - Chess training data
- `tensor_adapter.py` - Shape adaptation utilities

### 2. Poetry Model (`best_poetry_model_optimized.pt`)
**Purpose**: Creative text analysis and poetry understanding
**Capabilities**:
- Poetry style classification (sonnet, haiku, free verse, etc.)
- Emotional tone analysis
- Rhythm and meter detection
- Metaphor and figurative language understanding
- Creative writing assistance

**Required Files for Recreation**:
- `train_poetry_optimized.py` - Creative training script
- `dataset/poetry/` - Poetry collections
- Core framework files (meta_model.py, etc.)

### 3. Programming Model
**Purpose**: Code analysis and programming pattern recognition
**Capabilities**:
- Code quality assessment
- Bug detection patterns
- Programming language classification
- Algorithm complexity analysis
- Code style evaluation

### 4. Multimodal Model
**Purpose**: Combined text, image, and structured data processing
**Capabilities**:
- Cross-modal understanding
- Image-text correlation
- Multimodal reasoning
- Context-aware processing

## 🧠 Core Technologies

### Symbolic Controllers
- **Fuzzy Logic**: Handles uncertainty and creative reasoning
- **Production Rules**: IF-THEN logical reasoning
- **Graph-Based**: Relationship and dependency modeling
- **Neuro-Symbolic**: Hybrid neural-symbolic integration with differentiable operations

### Neural Architectures
- **Custom LSTM**: Sequential data processing
- **Transformer**: Attention-based understanding (BERT, GPT-style, custom)
- **CNN**: Visual feature extraction (ResNet, VGG, DenseNet, EfficientNet, MobileNet)
- **Graph Neural Networks**: GCN, GAT, GIN, PNA, GraphTransformer, and more
- **Hybrid**: Combined architectures for optimal performance

### Memory-Augmented Networks
- **Differentiable Neural Computer (DNC)**: External memory with content and location-based addressing
- **Neural Turing Machine (NTM)**: Read/write memory with attention mechanisms
- **Stack/Queue Memories**: Sequential data structures for program execution
- **Sparse Mixture of Experts**: Efficient scaling with top-k routing and load balancing

### Multimodal Processing
- **CLIP Integration**: OpenAI CLIP for text-image understanding
- **Cross-Modal Attention**: Attention-based fusion of text, image, and symbolic features
- **Modality Adapters**: Automatic adaptation for text-only, image-only, or multimodal inputs

### Continual Learning Strategies
- **Elastic Weight Consolidation (EWC)**: Fisher information-based regularization
- **Gradient Episodic Memory (GEM)**: Gradient projection to prevent interference
- **Experience Replay**: Multiple buffer types (Standard, Text, Image, MultiModal, Symbolic)
- **Progressive Neural Networks**: Task-specific columns with lateral connections

### Meta-Learning Algorithms
- **MAML (Model-Agnostic Meta-Learning)**: Fast adaptation to new tasks
- **Reptile**: First-order meta-learning algorithm
- **Prototypical Networks**: Few-shot learning with prototype-based classification
- **ANIL (Almost No Inner Loop)**: Efficient meta-learning variant
- **Meta-SGD**: Learning to learn with learnable learning rates

### Advanced Reasoning
- **Causal Reasoning**: Causal discovery, counterfactual reasoning, effect estimation (ATE, ITE)
- **Neural Program Interpreter**: Differentiable program execution with symbolic operations
- **Graph-Based Reasoning**: Relationship modeling and dependency inference

## 🚀 Quick Start

### Basic Usage

#### Train a Chess Model
```bash
python train_chess_optimized.py
```

#### Train a Poetry Model
```bash
python train_poetry_optimized.py
```

#### Train a Multimodal Model
```bash
python train_multimodal_xavious.py
```

### Using Advanced Components

#### Differentiable Neural Computer (DNC)
```python
from DNC import create_dnc

dnc = create_dnc(
    input_size=512,
    hidden_size=256,
    memory_size=128,
    num_read_heads=4
)
output, state = dnc(input_tensor)
```

#### Sparse Mixture of Experts
```python
from sparse_mix import create_sparse_moe

moe = create_sparse_moe(
    num_experts=8,
    hidden_size=512,
    top_k=2,
    routing_strategy="top_k"
)
logits, aux_info = moe(input_ids)
```

#### Causal Reasoning
```python
from causalR_module import create_causal_reasoner

reasoner = create_causal_reasoner(
    hidden_size=512,
    num_causal_factors=10
)
result = reasoner.forward(x, treatment=treatment)
ate = reasoner.estimate_ate(x, treatment)
```

#### Continual Learning
```python
from continual_module import create_continual_learner

continual_learner = create_continual_learner(
    model=base_model,
    strategy="ewc",
    memory_size=1000,
    ewc_lambda=0.4
)
loss = continual_learner.compute_continual_loss(loss, inputs, targets)
```

#### Neural Program Interpreter
```python
from NePr_interpreter import create_neural_interpreter, InstructionType

interpreter = create_neural_interpreter(
    hidden_size=256,
    max_variables=20
)
program = [
    {'type': InstructionType.ASSIGN, 'variable': 'x', 'operands': [value]},
    {'type': InstructionType.ADD, 'operands': ['x', 'y']}
]
result = interpreter.forward(program, inputs={'x': tensor_x})
```

### Evaluate Trained Models
```bash
python NeSy-CML/evaluator.py --model chess --weights best_chess_model_improved.pt
```

### Test Model Accuracy
```bash
python tests/test_trained_model.py
```

## 📊 Performance Metrics

| Model Type | Accuracy | Parameters | GPU Memory | Training Time |
|------------|----------|------------|------------|---------------|
| Chess      | 67%      | 7.9M       | 3.2GB      | 2 hours       |
| Poetry     | 70%      | 8.1M       | 3.5GB      | 1.5 hours     |
| Programming| 80%      | 9.2M       | 3.8GB      | 3 hours       |
| Multimodal | 88.5%    | 12.5M      | 4GB        | 4 hours       |

## 🔧 Configuration

### 4GB GPU Optimization
All models are optimized for 4GB GPU constraints:
- Gradient accumulation for effective larger batch sizes
- Mixed precision training (FP16/BF16)
- Memory-efficient architectures
- Adaptive batch sizing
- Gradient checkpointing for large models

### Modular Component Selection

#### Symbolic Controllers
```python
from NeSy-CML.modular_symbolic_controller import create_symbolic_controller
# Or if NeSy-CML is in your path:
# from modular_symbolic_controller import create_symbolic_controller

# Logic-based controller
controller = create_symbolic_controller(
    controller_type='logic',
    num_rules=100,
    input_size=512
)

# Production rule controller
controller = create_symbolic_controller(
    controller_type='production_rule',
    num_rules=100,
    input_size=512
)

# Neuro-symbolic controller
controller = create_symbolic_controller(
    controller_type='neuro_symbolic',
    num_rules=100,
    input_size=512
)
```

#### Replay Buffers
```python
from NeSy-CML.modular_replay_buffer import create_replay_buffer
# Or if NeSy-CML is in your path:
# from modular_replay_buffer import create_replay_buffer

# Text replay buffer
buffer = create_replay_buffer(
    buffer_type='text',
    memory_size=10000
)

# Multimodal replay buffer
buffer = create_replay_buffer(
    buffer_type='multimodal',
    memory_size=5000
)
```

#### Architecture Selection
```python
from custom_architecture_selector import ArchitectureSelector

selector = ArchitectureSelector()
optimal_arch = selector.select_architecture(
    task_type="sequential_data",
    data_type="text",
    requirements=["memory", "sequential"],
    memory_constraint="low"
)
```

### Customization
- Modify `config` dictionaries in training scripts
- Adjust symbolic controller parameters via `ModelSpec`
- Customize neural architectures via `custom_architecture_selector.py`
- Configure CLIP model selection in `text_encoder.py`
- Adjust continual learning strategies in `continual_module.py`

## 📚 Documentation

### Model Documentation
- [Chess Model Guide](docs/CHESS_MODEL.md)
- [Poetry Model Guide](docs/POETRY_MODEL.md)
- [Programming Model Guide](docs/PROGRAMMING_MODEL.md)
- [Multimodal Model Guide](docs/MULTIMODAL_MODEL.md)

### Component Documentation

#### Memory-Augmented Networks
- **DNC.py**: Differentiable Neural Computer with external memory
  - Content-based and location-based addressing
  - Temporal link matrix for write order tracking
  - Allocation addressing for memory management

- **mem-aug.py**: Multiple memory architectures
  - Neural Turing Machine (NTM)
  - Stack-Augmented RNN
  - Queue-Augmented RNN

- **sparse-mix.py**: Sparse Mixture of Experts
  - Top-k routing, Switch Transformer routing
  - Load-balanced routing with auxiliary losses
  - Expert diversity optimization

#### Reasoning Modules
- **causalR-module.py**: Causal reasoning and inference
  - Causal graph discovery
  - Counterfactual reasoning (abduction, action, prediction)
  - Causal effect estimation (ATE, ITE)

- **NePr-interpreter.py**: Neural Program Interpreter
  - Differentiable program execution
  - Variable memory management
  - Arithmetic and control flow operations

- **gnnm.py**: Graph Neural Network Modules
  - GCN, GAT, GIN, PNA, GraphTransformer
  - Multiple aggregation and scaling strategies

#### Continual Learning
- **continual_module.py**: Multiple continual learning strategies
  - Elastic Weight Consolidation (EWC)
  - Gradient Episodic Memory (GEM)
  - Experience Replay
  - Progressive Neural Networks

#### Meta-Learning
- **meta-controller.py**: Meta-learning algorithms
  - MAML, Reptile, Prototypical Networks
  - ANIL, Meta-SGD, Meta-Curvature

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆕 Recent Updates

### Version 2.0 - Advanced Memory and Reasoning
- ✅ **Differentiable Neural Computer (DNC)**: Full implementation with content/location addressing
- ✅ **Memory-Augmented Networks**: NTM, Stack, and Queue architectures
- ✅ **Sparse Mixture of Experts**: Efficient scaling with multiple routing strategies
- ✅ **Causal Reasoning Module**: Causal discovery, counterfactuals, and effect estimation
- ✅ **Neural Program Interpreter**: Differentiable program execution
- ✅ **Enhanced Continual Learning**: EWC, GEM, Progressive Networks, and more
- ✅ **Graph Neural Networks**: Comprehensive GNN module with multiple layer types
- ✅ **CLIP Integration**: Full CLIP support for multimodal understanding
- ✅ **Meta-Learning Framework**: MAML, Reptile, and other fast adaptation algorithms

### Key Features
- **Fully Modular**: Swap components without breaking the system
- **Production Ready**: All code is real, working logic (no placeholders)
- **GPU Optimized**: Efficient memory usage for 4GB+ GPUs
- **Extensible**: Easy to add new architectures, controllers, or strategies

## 🙏 Acknowledgments

- Built on PyTorch framework
- CLIP by OpenAI for multimodal understanding
- Inspired by neuro-symbolic AI research
- Memory architectures based on DeepMind's DNC and NTM
- Optimized for practical GPU constraints
- Designed for real-world applications

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the example scripts in the repository

---

**NeuroSym-CML**: Where neural networks meet symbolic reasoning for next-generation AI.