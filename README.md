# 🧠 NeSy-CML: NeuroSymbolic Continual Meta-Learning Framework

A research framework combining neural networks with symbolic reasoning for interpretable AI. NeSy-CML demonstrates how hybrid architectures can achieve high performance on complex reasoning tasks while maintaining transparency and reliability.

---

## 🎯 Core Architecture: Neural + Symbolic Hybrid

NeSy-CML integrates two complementary AI approaches:

| Component | Purpose | Advantage |
|-----------|---------|-----------|
| **Neural Networks** | Pattern recognition, feature learning | Fast, adaptive, handles noisy data |
| **Symbolic Controllers** | Logic, rules, constraints | Interpretable, verifiable, reliable |
| **Hybrid Integration** | Weighted combination of both | Best of both worlds |

---

## �  Current Project: Chess Strategic Analysis

The primary demonstration is a chess analysis model currently achieving **F1 = 0.49** with the goal of reaching **F1 ≥ 0.92** through improved training techniques.

### Chess Model Performance

| Metric | Current Value | Target | Significance |
|--------|---------------|--------|--------------|
| **Macro F1 Score** | **0.49** | **≥ 0.92** | Primary metric for imbalanced data |
| **Accuracy** | ~46% | ~85% | Overall correctness |
| **Classes** | 9 concepts | 9 concepts | tactics, strategy, opening, endgame, etc. |
| **Data Source** | Chess books | Chess books | Real chess literature analysis |

### 🐛 Evaluation Bug Fixed
**Previous Issue**: The original evaluator had a critical bug that averaged component accuracies instead of using actual hybrid predictions, causing confusing and unreliable metrics that blocked user confidence.

**Solution**: Complete rewrite of `evaluator.py` with proper NeuroSymbolic evaluation:
- ✅ **True hybrid output calculation** - `alpha * neural + (1-alpha) * symbolic`
- ✅ **Comprehensive F1 metrics** - macro, weighted, micro with clear definitions
- ✅ **Component analysis** - shows neural vs symbolic vs hybrid performance
- ✅ **Research-grade reporting** - eliminates metric confusion

### Chess Concept Categories

1. **TACTICS** - pins, forks, skewers, combinations, sacrifices
2. **STRATEGY** - planning, positional play, initiative, outposts  
3. **OPENING** - development, castling, opening systems
4. **ENDGAME** - opposition, promotion, passed pawns
5. **PIECES** - piece values, exchanges, material
6. **NOTATION** - algebraic notation, move recording
7. **MIDDLEGAME** - attacks, calculations, tactics
8. **EVALUATION** - position assessment, advantages
9. **CHECKMATE** - mating patterns, forced sequences

---

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

## � Resetarch Contributions

### 1. **Fixed Evaluation Methodology**
Identified and fixed critical bugs in NeuroSymbolic evaluation that were causing metric confusion:
- **Bug**: Averaging component accuracies instead of using actual hybrid predictions
- **Fix**: Proper weighted combination with comprehensive F1 metrics  
- **Impact**: Trustworthy, research-grade evaluation that eliminates user confusion

### 2. **Comprehensive F1 Metrics**
Developed proper evaluation for imbalanced multi-label classification:
- **Macro F1**: Primary metric (unweighted average, best for imbalanced data)
- **Weighted F1**: Class-frequency weighted average
- **Micro F1**: Global F1 across all predictions
- **Component Analysis**: Neural vs Symbolic vs Hybrid performance breakdown

### 3. **Hybrid Architecture Validation**
Created framework to properly measure NeuroSymbolic benefits:
- **True Hybrid Output**: `alpha * neural + (1-alpha) * symbolic`
- **Component Comparison**: Shows when hybrid outperforms individual components
- **Transparent Reporting**: Clear explanations of what each metric means

---

## 🧪 Key Features

### 1. **Fixed Evaluation System**
- **Proper hybrid metrics** - eliminates averaging bugs that caused confusion
- **Comprehensive F1 scores** - macro, weighted, micro variants with clear definitions
- **Component analysis** - neural vs symbolic performance breakdown
- **Research-grade reporting** - trustworthy metrics for technical users

### 2. **Advanced Training Techniques** 
- **Multi-loss optimization** - focal + dice + F1 loss for imbalanced data
- **Smart data balancing** - handles severe class imbalance without overcorrection
- **Threshold optimization** - maximizes F1 scores per class
- **Early stopping** - stops at target performance to prevent overfitting

### 3. **Modular NeuroSymbolic Architecture**
- **Pluggable components** - easy to modify symbolic controllers and replay buffers
- **Multiple controller types** - production rules, logic networks, custom adapters
- **Tensor adapters** - handles shape mismatches between components
- **Experience replay** - adaptive memory for continual learning

---

## 🛠️ Installation and Accessibility

### Prerequisites (Designed for Global Accessibility)

* Python 3.8+
* **CUDA-capable GPU ($\text{4GB+}$ VRAM recommended):** All models are explicitly optimized for consumer-grade hardware.
* PyTorch 2.0+
* Additional dependencies in `requirements.txt`

### Quick Start
```bash
git clone https://github.com/yourusername/NeSy-CML.git
cd NeSy-CML
pip install -r requirements.txt
```


🎯 Model Types and Capabilities
1. Chess Model (best_chess_model_improved.pt)
Purpose: Deep Chess strategic and tactical analysis.
Capabilities:
Chess position evaluation (material, positional factors)
Move quality assessment with strategic depth
Tactical pattern recognition across 58 sub-labels


Required Files for Recreation:
train_chess_improved.py - Main training script
meta_model.py - Core hybrid model architecture
modular_symbolic_controller.py - Symbolic reasoning
modular_replay_buffer.py - Memory management
dataset/Chess_data/ - Chess training data
tensor_adapter.py - Shape adaptation utilities


2. Poetry Model (.pt) (Files Pending)
Purpose: Creative text analysis and poetry understanding
Capabilities:
Poetry style classification (sonnet, haiku, free verse, etc.)
Emotional tone analysis

4. Programming Model (Files Pending)
Purpose: Code analysis and programming pattern recognition
Capabilities:
Code quality assessment
Bug detection patterns
Algorithm complexity analysis

5. Multimodal Model (Files Pending)
Purpose: Combined text, image, and structured data processing
Capabilities:
Cross-modal understanding
Context-aware processing

📄 NeSy-CML Proprietary License (Attention Required)
This project is released under a Custom NeSy-CML Proprietary License designed to enforce scientific accountability on commercial entities.
Non-Commercial Use: Free for academic research, education, and non-profit projects.

Commercialization: Requires explicit written approval from the original creator, Lyle Richards II.
Approval will only be granted upon demonstrable, measurable, and verifiable technical improvement to the core framework, which must be submitted and approved before monetization.

Purpose: To ensure that profits derived from this AGI framework are directly tied to documented scientific advancement, not mere scaling or repackaging.

🤝 Contributing & Support
I am a solo developer, and I hope this framework opens many doorways for research teams.
Fork the repository and open a Pull Request for bug fixes or features.
Open a GitHub Issue for bug reports or detailed questions.

🙏 Acknowledgments: Built on the power of the PyTorch framework and designed by a self-taught programmer (Lyle Richards II) to accelerate the future of open science.

Afterthoughts: I will eventually get something up for potential tipping of some sort that will go as funds towards my research in AI and Quantum computing. 
It is kind of hard to do things like this when lacking the funding for the research and the development time so any help is much appreciated. 
If anyone chooses they can tip whatever they feel led to tip at my paypal lylerichards17@gmail.com. Thank you all and I hope that everyone enjoys the new framework!
