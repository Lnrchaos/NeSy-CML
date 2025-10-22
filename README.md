# NeuroSym-CML: Neuro-Symbolic Continual Meta-Learning Framework

A cutting-edge AI framework that combines neural networks with symbolic reasoning for advanced multimodal learning across chess, poetry, programming, and image analysis domains.

## 🚀 Overview

NeuroSym-CML is a hybrid AI system that integrates:
- **Neural Networks**: Deep learning for pattern recognition and feature extraction
- **Symbolic Controllers**: Logic-based reasoning with fuzzy logic, production rules, and graph-based inference
- **Continual Learning**: Adaptive replay buffers and meta-learning capabilities
- **Multimodal Processing**: Text, image, and structured data understanding

## 🏆 Achievements

- **Chess Analysis**: ~83% accuracy on chess position evaluation and move prediction with an F1 score of ~.92
- **Poetry Generation**: to yet be determined accuracy on style classification and creative analysis
- **Programming**: to yet be determined accuracy on code analysis and pattern recognition
- **Image Processing**: to yet be determined accuracy on visual understanding tasks
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
├── core/                            # Core framework components
│   ├── meta_model.py               # Hybrid neural-symbolic model
│   ├── modular_architecture.py     # Modular architecture system
│   ├── modular_symbolic_controller.py # Symbolic reasoning controllers
│   ├── modular_replay_buffer.py    # Adaptive memory systems
│   └── tensor_adapter.py           # Tensor shape adaptation utilities
├── training/                       # Training scripts
│   ├── train_chess_optimized.py   # Optimized chess training
│   ├── train_poetry_optimized.py  # Creative poetry training
│   ├── train_programming.py       # Code analysis training
│   └── train_multimodal_newson.py # Multimodal training
├── models/                         # Trained model weights
│   ├── best_chess_model_4gb.pt    # Chess model (4GB GPU optimized)
│   └── best_poetry_model_optimized.pt # Poetry model
├── dataset/                        # Training datasets
│   ├── Chess_data/                # Chess positions and games
│   ├── poetry/                    # Poetry collections
│   ├── programming_data/          # Code samples
│   └── images/                    # Image datasets
└── utils/                         # Utility scripts
    ├── evaluator.py              # Model evaluation tools
    ├── test_trained_model.py     # Model testing utilities
    └── custom_architecture_selector.py # Architecture optimization
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

### 1. Chess Model (`best_chess_model_improved.pt`)
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

### 2. Poetry Model (`.pt`)
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
- **Neuro-Symbolic**: Hybrid neural-symbolic integration

### Neural Architectures
- **Custom LSTM**: Sequential data processing
- **Transformer**: Attention-based understanding
- **CNN**: Visual feature extraction
- **Hybrid**: Combined architectures for optimal performance

### Memory Systems
- **Adaptive Replay Buffer**: Intelligent experience replay
- **Continual Learning**: Prevents catastrophic forgetting
- **Meta-Learning**: Learns how to learn new tasks

## 🚀 Quick Start

### Train a Chess Model
```bash
python train_chess_optimized.py
```

### Train a Poetry Model
```bash
python train_poetry_optimized.py
```

### Evaluate Trained Models
```bash
python evaluator.py --model chess --weights best_chess_model_improved.pt
```

### Test Model Accuracy
```bash
python test_trained_model.py
```

## 📊 Performance Metrics

| Model Type | Accuracy | Parameters | GPU Memory | Training Time | F1 
|------------|----------|------------|------------|---------------|
| Chess      | 83%      | 7.9M       | 3.2GB      | 2 hours  | .92|
| Poetry     |   %      | 8.1M       | 3.5GB      |     hours     |
| Programming|   %      | 9.2M       | 3.8GB      |   hours       |
| Multimodal |     %    | 12.5M      | 4GB        |   hours       |

(The above stats are a tad bit off when it comes to the parameters, this project is constantly evolving, I work on it every single day to make it better. 
It may take a little bit of time for me to get the files uploaded into the repository so please bare with me as I do so, I am a solo developer and not part of a 
team so this is a relatively long task  when trying to ensure only the newest updates and files get transferred over from my local drive. I hope that this opens 
many doorways for the research teams.)

## 🔧 Configuration

### 4GB GPU Optimization
All models are optimized for 4GB GPU constraints:
- Gradient accumulation for effective larger batch sizes
- Mixed precision training
- Memory-efficient architectures
- Adaptive batch sizing

### Customization
- Modify `config` dictionaries in training scripts
- Adjust symbolic controller parameters
- Customize neural architectures via `custom_architecture_selector.py`

## 📚 Documentation

Detailed documentation for each model type:
- [Chess Model Guide](docs/CHESS_MODEL.md)
- [Poetry Model Guide](docs/POETRY_MODEL.md)
- [Programming Model Guide](docs/PROGRAMMING_MODEL.md)
- [Multimodal Model Guide](docs/MULTIMODAL_MODEL.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under a custom NeSy-CML Proprietary License

## 🙏 Acknowledgments

- Built on PyTorch framework
- Optimized for practical GPU constraints
- Designed for real-world applications
- Developed by a self taught programmer (myself Lyle Richards II)

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the example scripts in the repository

---


**NeuroSym-CML**: Where neural networks meet symbolic reasoning for next-generation AI.


** Afterthoughts**: I will eventually get something up for potential tipping of some sort that will go as funds towards my research in AI and Quantum computing. It is kind of hard to do things like this when lacking the funding for the research and the development time so any help is much appreciated. If anyone chooses they can tip whatever they feel led to tip at my paypal lylerichards17@gmail.com Thank you all and I hope that everyone enjoys the new framework!


