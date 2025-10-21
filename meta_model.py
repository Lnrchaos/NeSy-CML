from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD, Optimizer
from torch.utils.data import DataLoader
from transformers import CLIPModel

@dataclass
class ModelSpec:
    """Configuration class for the Hybrid Neuro-Symbolic Continual Meta Model"""
    
    # Neural Architecture Configuration
    neural_architecture: str = "resnet18"  # Base neural network architecture
    # Supported architectures: resnet18, resnet50, resnet101, vgg16, vgg19, 
    # densenet121, efficientnet_b0, efficientnet_b1, mobilenet_v2, 
    # custom_cnn, custom_transformer, custom_lstm
    num_classes: int = 10                  # Number of output classes
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])  # Hidden layer sizes
    activation: str = "relu"               # Activation function
    
    # Symbolic Reasoning Configuration
    use_symbolic_reasoning: bool = True    # Whether to use symbolic reasoning
    rule_set_size: int = 50                # Number of symbolic rules
    rule_embedding_dim: int = 64           # Dimension of rule embeddings
    
    # Continual Learning Configuration
    memory_size: int = 1000                # Size of experience replay memory
    memory_batch_size: int = 32            # Batch size for memory replay
    memory_sampling_strategy: str = "random"  # Sampling strategy for memory
    
    # Meta Learning Configuration
    meta_batch_size: int = 4               # Number of tasks per meta-update
    inner_lr: float = 0.01                 # Inner loop learning rate
    outer_lr: float = 0.001                # Outer loop learning rate
    
    # Training Configuration
    optimizer: str = "adam"                # Optimizer type
    learning_rate: float = 0.001           # Learning rate
    weight_decay: float = 0.0              # Weight decay
    epochs: int = 100                      # Number of training epochs
    batch_size: int = 32                   # Training batch size
    
    # Data Configuration
    input_shape: tuple = (3, 224, 224)     # Input data shape
    num_channels: int = 3                  # Number of input channels
    
    # Additional Parameters
    dropout_rate: float = 0.2              # Dropout rate
    use_batch_norm: bool = True            # Whether to use batch normalization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Device to use

class HybridModel(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.device = torch.device(self.spec.device)
        
        # Neural Architecture - Support multiple backbones
        self.backbone, self.feature_dim = self._build_backbone()
        
        # Text Encoder
        self.text_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # Fix: CLIP config uses text_config.hidden_size
        text_hidden_size = self.text_encoder.config.text_config.hidden_size
        
        # Create text projection that can handle different input sizes
        self.text_projection = None  # Will be created dynamically
        
        # Rule Embeddings
        self.rule_embeddings = nn.Embedding(
            num_embeddings=self.spec.rule_set_size,
            embedding_dim=self.spec.rule_embedding_dim
        )
        
        # Fusion Layer - will be dynamically sized based on actual input
        self.fusion = None  # Will be created in forward pass
        
        # Output Layer - will be created dynamically
        self.output = None
    
    def _build_backbone(self):
        """Build the neural backbone based on architecture specification"""
        architecture = self.spec.neural_architecture.lower()
        
        if architecture == "resnet18":
            return self._build_resnet18()
        elif architecture == "resnet50":
            return self._build_resnet50()
        elif architecture == "resnet101":
            return self._build_resnet101()
        elif architecture == "vgg16":
            return self._build_vgg16()
        elif architecture == "vgg19":
            return self._build_vgg19()
        elif architecture == "densenet121":
            return self._build_densenet121()
        elif architecture == "efficientnet_b0":
            return self._build_efficientnet_b0()
        elif architecture == "efficientnet_b1":
            return self._build_efficientnet_b1()
        elif architecture == "mobilenet_v2":
            return self._build_mobilenet_v2()
        elif architecture == "custom_cnn":
            return self._build_custom_cnn()
        elif architecture == "custom_transformer":
            return self._build_custom_transformer()
        elif architecture == "custom_lstm":
            return self._build_custom_lstm()
        elif architecture == "newson":
            return self._build_newson()
        elif architecture == "gpt_style":
            return self._build_gpt_style()
        elif architecture == "bert_style":
            return self._build_bert_style()
        elif architecture == "multimodal_transformer":
            return self._build_multimodal_transformer()
        elif architecture == "code_transformer":
            return self._build_code_transformer()
        else:
            raise ValueError(f"Unsupported architecture: {self.spec.neural_architecture}. "
                           f"Supported architectures: resnet18, resnet50, resnet101, vgg16, vgg19, "
                           f"densenet121, efficientnet_b0, efficientnet_b1, mobilenet_v2, "
                           f"custom_cnn, custom_transformer, custom_lstm, newson, gpt_style, "
                           f"bert_style, multimodal_transformer, code_transformer")
    
    def _build_resnet18(self):
        """Build ResNet18 backbone"""
        from torchvision.models import resnet18
        backbone = resnet18(pretrained=True)
        backbone.fc = nn.Identity()
        return backbone, 512
    
    def _build_resnet50(self):
        """Build ResNet50 backbone"""
        from torchvision.models import resnet50
        backbone = resnet50(pretrained=True)
        backbone.fc = nn.Identity()
        return backbone, 2048
    
    def _build_resnet101(self):
        """Build ResNet101 backbone"""
        from torchvision.models import resnet101
        backbone = resnet101(pretrained=True)
        backbone.fc = nn.Identity()
        return backbone, 2048
    
    def _build_vgg16(self):
        """Build VGG16 backbone"""
        from torchvision.models import vgg16
        backbone = vgg16(pretrained=True)
        backbone.classifier = nn.Identity()
        return backbone, 25088
    
    def _build_vgg19(self):
        """Build VGG19 backbone"""
        from torchvision.models import vgg19
        backbone = vgg19(pretrained=True)
        backbone.classifier = nn.Identity()
        return backbone, 25088
    
    def _build_densenet121(self):
        """Build DenseNet121 backbone"""
        from torchvision.models import densenet121
        backbone = densenet121(pretrained=True)
        backbone.classifier = nn.Identity()
        return backbone, 1024
    
    def _build_efficientnet_b0(self):
        """Build EfficientNet-B0 backbone"""
        try:
            from torchvision.models import efficientnet_b0
            backbone = efficientnet_b0(pretrained=True)
            backbone.classifier = nn.Identity()
            return backbone, 1280
        except ImportError:
            print("Warning: EfficientNet not available, falling back to ResNet18")
            return self._build_resnet18()
    
    def _build_efficientnet_b1(self):
        """Build EfficientNet-B1 backbone"""
        try:
            from torchvision.models import efficientnet_b1
            backbone = efficientnet_b1(pretrained=True)
            backbone.classifier = nn.Identity()
            return backbone, 1280
        except ImportError:
            print("Warning: EfficientNet not available, falling back to ResNet18")
            return self._build_resnet18()
    
    def _build_mobilenet_v2(self):
        """Build MobileNetV2 backbone"""
        from torchvision.models import mobilenet_v2
        backbone = mobilenet_v2(pretrained=True)
        backbone.classifier = nn.Identity()
        return backbone, 1280
    
    def _build_custom_cnn(self):
        """Build custom CNN architecture"""
        class CustomCNN(nn.Module):
            def __init__(self, input_channels=3, num_classes=1000):
                super().__init__()
                self.features = nn.Sequential(
                    # First conv block
                    nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    
                    # Second conv block
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    # Third conv block
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    # Fourth conv block
                    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
            def forward(self, x):
                features = self.features(x)
                # Flatten the output to 2D: (batch_size, features)
                return features.view(features.size(0), -1)
        
        backbone = CustomCNN()
        return backbone, 512
    
    def _build_custom_transformer(self):
        """Build custom Transformer architecture for image processing"""
        class CustomImageTransformer(nn.Module):
            def __init__(self, input_size=224, patch_size=16, embed_dim=512, num_heads=8, num_layers=6):
                super().__init__()
                self.patch_size = patch_size
                self.num_patches = (input_size // patch_size) ** 2
                self.embed_dim = embed_dim
                
                # Patch embedding
                self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
            def forward(self, x):
                B = x.shape[0]
                
                # Patch embedding
                x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
                x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
                
                # Add CLS token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                
                # Add positional embedding
                x = x + self.pos_embed
                
                # Transformer encoding
                x = self.transformer(x)
                
                # Return CLS token representation
                return x[:, 0]  # (B, embed_dim)
        
        backbone = CustomImageTransformer()
        return backbone, 512
    
    def _build_custom_lstm(self):
        """Build custom LSTM architecture for sequential data"""
        class CustomLSTM(nn.Module):
            def __init__(self, input_size=224*224*3, hidden_size=512, num_layers=3):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                # Adaptive input projection - will be created dynamically
                self.input_projection = None
                self.input_size = input_size
                
                # LSTM layers
                self.lstm = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.1 if num_layers > 1 else 0
                )
                
                # Output projection
                self.output_projection = nn.Linear(hidden_size, hidden_size)
                
            def _create_input_projection(self, actual_input_size):
                """Create input projection with the correct size"""
                if self.input_projection is None or self.input_projection.in_features != actual_input_size:
                    self.input_projection = nn.Linear(actual_input_size, self.hidden_size)
                    # Move to the same device as the model
                    device = next(self.parameters()).device
                    self.input_projection = self.input_projection.to(device)
                
            def forward(self, x):
                # Reshape input: (B, C, H, W) -> (B, H*W, C)
                B, C, H, W = x.shape
                x = x.view(B, C, H*W).transpose(1, 2)  # (B, H*W, C)
                x = x.reshape(B, H*W, C)  # (B, H*W, C)
                
                # Create input projection with correct size
                actual_input_size = x.size(-1)  # C dimension
                self._create_input_projection(actual_input_size)
                
                # Project to hidden size
                x = self.input_projection(x)
                
                # LSTM processing
                lstm_out, _ = self.lstm(x)
                
                # Use the last output
                output = self.output_projection(lstm_out[:, -1, :])
                
                return output
        
        backbone = CustomLSTM()
        return backbone, 512
        
    def forward(self, x: torch.Tensor, text_embeddings: torch.Tensor, rule_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model with multimodal fusion
        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)
            text_embeddings: Text embeddings tensor of shape (batch_size, text_dim)
            rule_indices: Rule indices for each sample in the batch
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Extract image features
        image_features = self.backbone(x)
        
        # Create text projection layer if not exists
        if self.text_projection is None:
            text_input_dim = text_embeddings.size(-1)
            self.text_projection = nn.Linear(text_input_dim, self.feature_dim).to(text_embeddings.device)
        
        # Project text embeddings to match image feature dimension
        text_features = self.text_projection(text_embeddings)
        
        # Get rule embeddings
        rule_embs = self.rule_embeddings(rule_indices)
        
        # Ensure all features have the same number of dimensions
        if rule_embs.dim() > image_features.dim():
            rule_embs = rule_embs.squeeze(1)  # Remove extra dimension if present
        
        # Concatenate all features
        fused = torch.cat([
            image_features,
            text_features,
            rule_embs
        ], dim=1)
        
        # Create fusion layer if not exists
        if self.fusion is None:
            input_dim = fused.size(1)
            self.fusion = nn.Sequential(
                nn.Linear(input_dim, self.spec.hidden_sizes[0]),
                nn.ReLU(),
                nn.Dropout(self.spec.dropout_rate)
            ).to(fused.device)
        
        # Pass through fusion layer
        fused = self.fusion(fused)
        
        # Create output layer if not exists
        if self.output is None:
            self.output = nn.Linear(fused.size(1), self.spec.num_classes).to(fused.device)
        
        # Output
        return self.output(fused)
    
    def _build_newson(self):
        """Build NewSon multimodal architecture for NLP and image processing"""
        class NewSon(nn.Module):
            def __init__(self, 
                         vocab_size=30522, 
                         max_seq_len=512, 
                         image_size=224, 
                         patch_size=16, 
                         embed_dim=512, 
                         num_heads=8, 
                         num_layers=6):
                super().__init__()
                
                self.embed_dim = embed_dim
                self.num_patches = (image_size // patch_size) ** 2
                
                # === NLP Embedding ===
                self.token_embed = nn.Embedding(vocab_size, embed_dim)
                self.text_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
                
                # === Image Embedding ===
                self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
                self.image_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                
                # === Shared Transformer Encoder ===
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # === Output Heads ===
                self.text_head = nn.Linear(embed_dim, vocab_size)
                self.image_head = nn.Linear(embed_dim, embed_dim)

            def forward_text(self, x):  # x: (B, seq_len)
                x = self.token_embed(x) + self.text_pos_embed[:, :x.size(1), :]
                x = self.transformer(x)
                return self.text_head(x[:, 0])  # CLS token output

            def forward_image(self, x):  # x: (B, 3, H, W)
                B = x.shape[0]
                x = self.patch_embed(x)  # (B, embed_dim, H//patch, W//patch)
                x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.image_pos_embed
                x = self.transformer(x)
                return self.image_head(x[:, 0])  # CLS token output
            
            def forward(self, x, text_embeddings=None, rule_indices=None):
                # Handle different input types
                if x.dim() == 4:  # Image input (B, C, H, W)
                    return self.forward_image(x)
                elif x.dim() == 2:  # Text input (B, seq_len)
                    return self.forward_text(x)
                else:
                    # Fallback to text processing
                    return self.forward_text(x)
        
        backbone = NewSon(
            vocab_size=30522,
            max_seq_len=512,
            embed_dim=self.spec.hidden_sizes[0] if self.spec.hidden_sizes else 512,
            num_heads=8,
            num_layers=6
        )
        feature_dim = self.spec.hidden_sizes[0] if self.spec.hidden_sizes else 512
        return backbone, feature_dim
    
    def _build_gpt_style(self):
        """Build GPT-style decoder-only transformer for code generation"""
        class GPTStyle(nn.Module):
            def __init__(self, vocab_size=30522, embed_dim=512, num_heads=8, num_layers=6, max_seq_len=512):
                super().__init__()
                self.embed_dim = embed_dim
                self.token_embed = nn.Embedding(vocab_size, embed_dim)
                self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
                
                # Decoder-only transformer
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
                self.output_head = nn.Linear(embed_dim, vocab_size)
                
            def forward(self, x, text_embeddings=None, rule_indices=None):
                # Create causal mask for autoregressive generation
                seq_len = x.size(1)
                mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                
                x = self.token_embed(x) + self.pos_embed[:, :seq_len, :]
                x = self.transformer(x, x, tgt_mask=mask)
                return self.output_head(x)
        
        backbone = GPTStyle(
            vocab_size=getattr(self.spec, 'vocab_size', 30522),
            embed_dim=self.spec.hidden_sizes[0] if self.spec.hidden_sizes else 512,
            num_heads=8,
            num_layers=6
        )
        feature_dim = self.spec.hidden_sizes[0] if self.spec.hidden_sizes else 512
        return backbone, feature_dim
    
    def _build_bert_style(self):
        """Build BERT-style encoder-only transformer for code understanding"""
        class BERTStyle(nn.Module):
            def __init__(self, vocab_size=30522, embed_dim=512, num_heads=8, num_layers=6, max_seq_len=512):
                super().__init__()
                self.embed_dim = embed_dim
                self.token_embed = nn.Embedding(vocab_size, embed_dim)
                self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
                self.segment_embed = nn.Embedding(2, embed_dim)  # For different segments
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_head = nn.Linear(embed_dim, vocab_size)
                
            def forward(self, x, text_embeddings=None, rule_indices=None):
                seq_len = x.size(1)
                x = self.token_embed(x) + self.pos_embed[:, :seq_len, :]
                x = self.transformer(x)
                return self.output_head(x)
        
        backbone = BERTStyle(
            vocab_size=getattr(self.spec, 'vocab_size', 30522),
            embed_dim=self.spec.hidden_sizes[0] if self.spec.hidden_sizes else 512,
            num_heads=8,
            num_layers=6
        )
        feature_dim = self.spec.hidden_sizes[0] if self.spec.hidden_sizes else 512
        return backbone, feature_dim
    
    def _build_multimodal_transformer(self):
        """Build multimodal transformer for code and text processing"""
        class MultimodalTransformer(nn.Module):
            def __init__(self, vocab_size=30522, embed_dim=512, num_heads=8, num_layers=6):
                super().__init__()
                self.embed_dim = embed_dim
                self.token_embed = nn.Embedding(vocab_size, embed_dim)
                self.modality_embed = nn.Embedding(3, embed_dim)  # text, code, image
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_head = nn.Linear(embed_dim, vocab_size)
                
            def forward(self, x, text_embeddings=None, rule_indices=None):
                # Add modality embedding based on input type
                if x.dim() == 4:  # Image
                    modality = 2
                elif hasattr(x, 'dtype') and 'int' in str(x.dtype):  # Token input
                    modality = 1  # Code tokens
                else:  # Text
                    modality = 0
                
                modality_tensor = torch.tensor(modality, device=x.device)
                x = self.token_embed(x) + self.modality_embed(modality_tensor)
                x = self.transformer(x)
                return self.output_head(x)
        
        backbone = MultimodalTransformer(
            vocab_size=getattr(self.spec, 'vocab_size', 30522),
            embed_dim=self.spec.hidden_sizes[0] if self.spec.hidden_sizes else 512,
            num_heads=8,
            num_layers=6
        )
        feature_dim = self.spec.hidden_sizes[0] if self.spec.hidden_sizes else 512
        return backbone, feature_dim
    
    def _build_code_transformer(self):
        """Build specialized transformer for code analysis and generation"""
        class CodeTransformer(nn.Module):
            def __init__(self, vocab_size=30522, embed_dim=512, num_heads=8, num_layers=6):
                super().__init__()
                self.embed_dim = embed_dim
                self.token_embed = nn.Embedding(vocab_size, embed_dim)
                self.syntax_embed = nn.Embedding(50, embed_dim)  # For syntax tokens
                
                # Multi-head attention with different attention patterns
                self.attention_layers = nn.ModuleList([
                    nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ])
                
                self.norm_layers = nn.ModuleList([
                    nn.LayerNorm(embed_dim) for _ in range(num_layers)
                ])
                
                self.feed_forward = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.ReLU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                )
                
                self.output_head = nn.Linear(embed_dim, vocab_size)
                
            def forward(self, x, text_embeddings=None, rule_indices=None):
                # Add syntax-aware embeddings
                syntax_tokens = torch.zeros_like(x)  # Simplified syntax detection
                x = self.token_embed(x) + self.syntax_embed(syntax_tokens)
                
                # Apply attention layers
                for attention, norm in zip(self.attention_layers, self.norm_layers):
                    attn_out, _ = attention(x, x, x)
                    x = norm(x + attn_out)
                    ff_out = self.feed_forward(x)
                    x = norm(x + ff_out)
                
                return self.output_head(x)
        
        backbone = CodeTransformer(
            vocab_size=getattr(self.spec, 'vocab_size', 30522),
            embed_dim=self.spec.hidden_sizes[0] if self.spec.hidden_sizes else 512,
            num_heads=8,
            num_layers=6
        )
        feature_dim = self.spec.hidden_sizes[0] if self.spec.hidden_sizes else 512
        return backbone, feature_dim

class ModelBuilder:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.model = None
        self.optimizer = None
        self.device = torch.device(self.spec.device)
    
    def build_optimizer(self, model: nn.Module) -> Optimizer:
        """Build the optimizer based on the specified configuration"""
        if self.spec.optimizer.lower() == "adam":
            return Adam(
                model.parameters(),
                lr=self.spec.learning_rate,
                weight_decay=self.spec.weight_decay
            )
        elif self.spec.optimizer.lower() == "sgd":
            return SGD(
                model.parameters(),
                lr=self.spec.learning_rate,
                weight_decay=self.spec.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.spec.optimizer}")
    
    def build(self) -> nn.Module:
        """Build the complete model"""
        self.model = HybridModel(self.spec)
        self.model.to(self.device)
        self.optimizer = self.build_optimizer(self.model)
        return self.model
    


# Example usage:
if __name__ == "__main__":
    # Create model specification
    spec = ModelSpec(
        neural_architecture="custom",
        num_classes=10,
        hidden_sizes=[256, 128],
        use_symbolic_reasoning=True,
        memory_size=1000,
        meta_batch_size=4,
        learning_rate=0.001
    )
    
    # Build model using ModelBuilder
    builder = ModelBuilder(spec)
    model = builder.build()
    
    print("Model architecture:", model)
    print("Device:", builder.device)

