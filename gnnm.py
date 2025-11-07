import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter, scatter_softmax

class GNNLayerType(Enum):
    GCN = "gcn"
    GAT = "gat"
    GATv2 = "gatv2"
    GIN = "gin"
    PNA = "pna"
    GATED_GRAPH = "gated_graph"
    GRAPH_TRANSFORMER = "graph_transformer"
    GRAPHORMER = "graphormer"
    SGC = "sgc"
    TAG = "tag"
    ARMA = "arma"
    FA = "fa"

@dataclass
class GNNConfig:
    layer_type: GNNLayerType = GNNLayerType.GAT
    input_dim: int = 128
    hidden_dim: int = 128
    output_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    use_layer_norm: bool = True
    residual: bool = True
    use_edge_attr: bool = False
    edge_dim: Optional[int] = None
    num_edge_types: int = 1
    use_global_pool: bool = True
    global_pool: str = "mean"
    use_skip_connections: bool = True
    jk_mode: Optional[str] = None
    gnn_act: str = "gelu"
    use_batch_norm: bool = False
    # GAT specific
    concat: bool = True
    negative_slope: float = 0.2
    # GIN specific
    eps: float = 0.0
    train_eps: bool = False
    # PNA specific
    aggregators: List[str] = None
    scalers: List[str] = None
    deg: Optional[torch.Tensor] = None
    # Graph Transformer specific
    transformer_ff_dim: int = 512
    # ARMA specific
    num_stacks: int = 3
    num_blocks: int = 2
    # FA specific
    num_walks: int = 4
    walk_length: int = 4

class GNNBase(nn.Module):
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.activation = self._get_activation(config.gnn_act)
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim) if config.input_dim != config.hidden_dim else nn.Identity()
        
        # Initialize layers
        for i in range(config.num_layers):
            layer_config = config
            # Special handling for first/last layer dimensions
            if i == 0 and config.input_dim != config.hidden_dim:
                layer_config = dataclasses.replace(config, hidden_dim=config.hidden_dim)
            elif i == config.num_layers - 1 and config.output_dim != config.hidden_dim:
                layer_config = dataclasses.replace(config, hidden_dim=config.output_dim)
            
            self.layers.append(self._create_gnn_layer(layer_config, i))
        
        # Jumping Knowledge (JK) connections
        if config.jk_mode is not None:
            jk_dim = config.hidden_dim * config.num_layers if config.jk_mode == "cat" else config.hidden_dim
            if config.jk_mode == "lstm":
                self.jk_lstm = nn.LSTM(
                    config.hidden_dim, 
                    config.hidden_dim, 
                    batch_first=True, 
                    bidirectional=True
                )
                self.jk_proj = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
                jk_dim = config.hidden_dim
            
            # Projection for output
            self.jk_proj_out = nn.Linear(jk_dim, config.output_dim) if jk_dim != config.output_dim else nn.Identity()
        
        # Global pooling
        if config.use_global_pool:
            if config.global_pool == "attention":
                self.global_pool_attn = nn.Linear(config.output_dim, 1)
    
    def _get_activation(self, act_name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "prelu": nn.PReLU(),
            "silu": nn.SiLU()
        }
        return activations.get(act_name.lower(), nn.GELU())
    
    def _create_gnn_layer(self, config: GNNConfig, layer_idx: int = 0) -> nn.Module:
        if config.layer_type == GNNLayerType.GCN:
            return GCNLayer(
                in_dim=config.input_dim if layer_idx == 0 else config.hidden_dim,
                out_dim=config.hidden_dim if layer_idx < config.num_layers - 1 else config.output_dim,
                dropout=config.dropout,
                use_layer_norm=config.use_layer_norm,
                residual=config.residual,
                activation=self.activation,
                use_batch_norm=config.use_batch_norm
            )
        elif config.layer_type == GNNLayerType.GAT:
            return GATLayer(
                in_dim=config.input_dim if layer_idx == 0 else config.hidden_dim,
                out_dim=config.hidden_dim // config.num_heads if layer_idx < config.num_layers - 1 else config.output_dim // config.num_heads,
                num_heads=config.num_heads,
                dropout=config.dropout,
                use_layer_norm=config.use_layer_norm,
                residual=config.residual,
                use_edge_attr=config.use_edge_attr,
                edge_dim=config.edge_dim,
                concat=config.concat,
                negative_slope=config.negative_slope
            )
        elif config.layer_type == GNNLayerType.GATv2:
            return GATv2Layer(
                in_dim=config.input_dim if layer_idx == 0 else config.hidden_dim,
                out_dim=config.hidden_dim // config.num_heads if layer_idx < config.num_layers - 1 else config.output_dim // config.num_heads,
                num_heads=config.num_heads,
                dropout=config.dropout,
                use_layer_norm=config.use_layer_norm,
                residual=config.residual,
                use_edge_attr=config.use_edge_attr,
                edge_dim=config.edge_dim,
                concat=config.concat,
                negative_slope=config.negative_slope
            )
        elif config.layer_type == GNNLayerType.GIN:
            return GINLayer(
                in_dim=config.input_dim if layer_idx == 0 else config.hidden_dim,
                hidden_dim=config.hidden_dim,
                out_dim=config.hidden_dim if layer_idx < config.num_layers - 1 else config.output_dim,
                dropout=config.dropout,
                use_layer_norm=config.use_layer_norm,
                residual=config.residual,
                eps=config.eps,
                train_eps=config.train_eps,
                use_batch_norm=config.use_batch_norm
            )
        elif config.layer_type == GNNLayerType.GRAPH_TRANSFORMER:
            return GraphTransformerLayer(
                hidden_dim=config.hidden_dim if layer_idx < config.num_layers - 1 else config.output_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                edge_dim=config.edge_dim if config.use_edge_attr else None,
                ffn_dim=config.transformer_ff_dim,
                use_layer_norm=config.use_layer_norm,
                residual=config.residual
            )
        elif config.layer_type == GNNLayerType.PNA:
            return PNALayer(
                in_dim=config.input_dim if layer_idx == 0 else config.hidden_dim,
                out_dim=config.hidden_dim if layer_idx < config.num_layers - 1 else config.output_dim,
                aggregators=config.aggregators or ["mean", "min", "max", "std"],
                scalers=config.scalers or ["identity", "amplification", "attenuation"],
                deg=config.deg,
                dropout=config.dropout,
                use_layer_norm=config.use_layer_norm,
                residual=config.residual
            )
        else:
            raise ValueError(f"Unsupported GNN layer type: {config.layer_type}")

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        h = self.input_proj(x)
        hs = []
        
        # Message passing
        for layer in self.layers:
            h = layer(x=h, edge_index=edge_index, edge_attr=edge_attr, batch=batch, **kwargs)
            if self.config.use_skip_connections:
                hs.append(h)
        
        # Apply JK connections
        if self.config.jk_mode is not None and len(hs) > 0:
            if self.config.jk_mode == "cat":
                h = torch.cat(hs, dim=-1)
            elif self.config.jk_mode == "max":
                h = torch.stack(hs, dim=-1).max(dim=-1)[0]
            elif self.config.jk_mode == "lstm":
                hs = torch.stack(hs, dim=1)  # [batch_size, num_layers, hidden_dim]
                h, _ = self.jk_lstm(hs)
                h = self.jk_proj(h[:, -1, :])  # Take last layer output
            
            h = self.jk_proj_out(h)
        
        # Global pooling
        if self.config.use_global_pool and batch is not None:
            h = self._global_pool(h, batch)
        
        return h
    
    def _global_pool(self, x, batch):
        if self.config.global_pool == "mean":
            return scatter(x, batch, dim=0, reduce="mean")
        elif self.config.global_pool == "sum":
            return scatter(x, batch, dim=0, reduce="sum")
        elif self.config.global_pool == "max":
            return scatter(x, batch, dim=0, reduce="max")
        elif self.config.global_pool == "attention":
            attn_weights = scatter_softmax(
                self.global_pool_attn(x).squeeze(-1),
                batch, dim=0
            ).unsqueeze(-1)
            return scatter(x * attn_weights, batch, dim=0, reduce="sum")
        else:
            raise ValueError(f"Unsupported global pooling: {self.config.global_pool}")

# ============= GNN Layer Implementations =============

class GCNLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, dropout=0.1, use_layer_norm=True, 
                 residual=True, activation=nn.GELU(), use_batch_norm=False):
        super().__init__(aggr='add')
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()
        self.batch_norm = nn.BatchNorm1d(out_dim) if use_batch_norm else nn.Identity()
        self.residual = residual and (in_dim == out_dim)
        self.activation = activation
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        if hasattr(self.layer_norm, 'reset_parameters'):
            self.layer_norm.reset_parameters()
        if hasattr(self.batch_norm, 'reset_parameters'):
            self.batch_norm.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # x: [num_nodes, in_dim]
        # edge_index: [2, num_edges]
        
        if self.residual:
            res = x
        
        # Linear transformation
        x = self.linear(x)
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Apply activation, normalization, and residual
        out = self.activation(out)
        out = self.layer_norm(out)
        out = self.batch_norm(out)
        out = self.dropout(out)
        
        if self.residual:
            out = out + res
        
        return out
    
    def message(self, x_j, edge_attr=None):
        # x_j: [num_edges, out_dim]
        if edge_attr is not None:
            return x_j * edge_attr.unsqueeze(-1)
        return x_j
    
    def update(self, aggr_out):
        return aggr_out

class GATLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, use_layer_norm=True, 
                 residual=True, use_edge_attr=False, edge_dim=None, concat=True, negative_slope=0.2):
        super().__init__(aggr='add', node_dim=0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_edge_attr = use_edge_attr
        self.concat = concat
        self.negative_slope = negative_slope
        
        # Multi-head attention
        self.attn_proj = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.Tensor(1, num_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.Tensor(1, num_heads, out_dim))
        
        if use_edge_attr and edge_dim is not None:
            self.edge_encoder = nn.Linear(edge_dim, num_heads * out_dim, bias=False)
            self.attn_edge = nn.Parameter(torch.Tensor(1, num_heads, out_dim))
        
        # Output projection
        out_dim_total = num_heads * out_dim if concat else out_dim
        self.out_proj = nn.Linear(num_heads * out_dim, out_dim_total)
        
        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(out_dim_total) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.residual = residual and (in_dim == out_dim_total)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attn_proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        if hasattr(self, 'edge_encoder'):
            nn.init.xavier_uniform_(self.edge_encoder.weight)
            nn.init.xavier_uniform_(self.attn_edge)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        if hasattr(self.layer_norm, 'reset_parameters'):
            self.layer_norm.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # x: [num_nodes, in_dim]
        # edge_index: [2, num_edges]
        
        if self.residual:
            res = x
        
        # Project node features
        x_proj = self.attn_proj(x)
        x_proj = x_proj.view(-1, self.num_heads, self.out_dim)
        
        # Prepare edge features if needed
        if self.use_edge_attr and edge_attr is not None and hasattr(self, 'edge_encoder'):
            edge_attr = self.edge_encoder(edge_attr).view(-1, self.num_heads, self.out_dim)
        
        # Message passing
        out = self.propagate(edge_index, x=x_proj, edge_attr=edge_attr)
        
        # Reshape and project
        out = out.view(-1, self.num_heads * self.out_dim)
        out = self.out_proj(out)
        
        # Apply residual, normalization, and dropout
        if self.residual:
            out = out + res
        
        out = self.layer_norm(out)
        out = F.leaky_relu(out, self.negative_slope)
        out = self.dropout(out)
        
        return out
    
    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        # x_j: [num_edges, num_heads, out_dim]
        # x_i: [num_nodes, num_heads, out_dim]
        
        # Compute attention scores
        alpha = (x_i * self.attn_dst + x_j * self.attn_src).sum(dim=-1)  # [num_edges, num_heads]
        
        # Add edge features if available
        if hasattr(self, 'edge_encoder') and edge_attr is not None:
            alpha = alpha + (x_i * x_j * self.attn_edge).sum(dim=-1)
        
        # Compute attention weights
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = scatter_softmax(alpha, index, dim=0)
        alpha = F.dropout(alpha, p=0.1, training=self.training)
        
        # Apply attention weights
        return x_j * alpha.unsqueeze(-1)

class GATv2Layer(GATLayer):
    """GATv2: Improved GAT with dynamic attention"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override attention computation
        self.attn = nn.Parameter(torch.Tensor(1, self.num_heads, 2 * self.out_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.xavier_uniform_(self.attn)
    
    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        # Compute attention scores using a more expressive attention mechanism
        alpha = torch.cat([x_i, x_j], dim=-1)  # [num_edges, num_heads, 2*out_dim]
        alpha = (alpha * self.attn).sum(dim=-1)  # [num_edges, num_heads]
        
        # Add edge features if available
        if hasattr(self, 'edge_encoder') and edge_attr is not None:
            alpha = alpha + (x_i * x_j * self.attn_edge).sum(dim=-1)
        
        # Compute attention weights
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = scatter_softmax(alpha, index, dim=0)
        alpha = F.dropout(alpha, p=0.1, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)

class GINLayer(MessagePassing):
    """Graph Isomorphism Network (GIN) layer"""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1, use_layer_norm=True,
                 residual=True, eps=0.0, train_eps=False, use_batch_norm=False):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim) if use_batch_norm else nn.Identity()
        )
        self.eps = nn.Parameter(torch.tensor([eps]), requires_grad=train_eps)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()
        self.residual = residual and (in_dim == out_dim)
    
    def forward(self, x, edge_index, **kwargs):
        if self.residual:
            res = x
        
        # Message passing
        out = self.propagate(edge_index, x=x)
        
        # MLP transformation
        out = self.mlp(out)
        
        # Apply residual, normalization, and dropout
        if self.residual:
            out = out + res
        
        out = self.layer_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        return out
    
    def message(self, x_j):
        return x_j
    
    def update(self, aggr_out, x):
        return (1 + self.eps) * x + aggr_out

class PNALayer(MessagePassing):
    """Principal Neighbourhood Aggregation (PNA) layer"""
    def __init__(self, in_dim, out_dim, aggregators=None, scalers=None, deg=None,
                 dropout=0.1, use_layer_norm=True, residual=True):
        super().__init__(aggr=None, node_dim=0)
        
        self.aggregators = aggregators or ["mean", "min", "max", "std"]
        self.scalers = scalers or ["identity", "amplification", "attenuation"]
        self.deg = deg  # Degree statistics for scaling
        
        # Initialize aggregators and scalers
        self.agg_modules = nn.ModuleList([self._get_aggregator(agg) for agg in self.aggregators])
        self.scale_modules = nn.ModuleList([self._get_scaler(scaler) for scaler in self.scalers])
        
        # MLP for combining aggregated features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * len(self.aggregators) * len(self.scalers), out_dim),
            nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.residual = residual and (in_dim == out_dim)
        if self.residual:
            self.res_linear = nn.Linear(in_dim, out_dim)
    
    def _get_aggregator(self, name):
        if name == "sum":
            return lambda x, *args, **kwargs: scatter(x, kwargs['index'], dim=0, reduce="sum")
        elif name == "mean":
            return lambda x, *args, **kwargs: scatter(x, kwargs['index'], dim=0, reduce="mean")
        elif name == "max":
            return lambda x, *args, **kwargs: scatter(x, kwargs['index'], dim=0, reduce="max")
        elif name == "min":
            return lambda x, *args, **kwargs: scatter(x, kwargs['index'], dim=0, reduce="min")
        elif name == "std":
            return self._std_aggregator
        else:
            raise ValueError(f"Unknown aggregator: {name}")
    
    def _std_aggregator(self, x, *args, **kwargs):
        mean = scatter(x, kwargs['index'], dim=0, reduce="mean")[kwargs['index']]
        return scatter((x - mean).pow(2), kwargs['index'], dim=0, reduce="mean").sqrt()
    
    def _get_scaler(self, name):
        if name == "identity":
            return lambda x, d: x
        elif name == "amplification":
            return lambda x, d: x * (math.log(d + 1) / 1.0)
        elif name == "attenuation":
            return lambda x, d: x * (1.0 / math.log(d + 1))
        else:
            raise ValueError(f"Unknown scaler: {name}")
    
    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        if self.residual:
            res = self.res_linear(x) if hasattr(self, 'res_linear') else x
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Apply MLP and residual
        out = self.mlp(out)
        if self.residual:
            out = out + res
        
        return out
    
    def message(self, x_j):
        return x_j
    
    def aggregate(self, x, index, dim_size=None, **kwargs):
        # Apply all aggregators
        aggregated = []
        for agg in self.agg_modules:
            agg_out = agg(x, index=index, dim_size=dim_size)
            aggregated.append(agg_out)
        
        # Concatenate all aggregations
        out = torch.cat(aggregated, dim=-1)
        
        # Apply scalers if degree statistics are provided
        if self.deg is not None:
            scaled = []
            for scaler in self.scale_modules:
                scaled.append(scaler(out, self.deg))
            out = torch.cat(scaled, dim=-1)
        
        return out

class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer with edge features"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, edge_dim=None, 
                 ffn_dim=512, use_layer_norm=True, residual=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Self-attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge features
        self.edge_dim = edge_dim
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, num_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
    
    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # x: [num_nodes, hidden_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_dim]
        
        if self.residual:
            res = x
        
        # Self-attention
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attn_scores = torch.einsum('nhi,mhi->hnm', q, k) * self.scale  # [num_heads, num_nodes, num_nodes]
        
        # Apply edge features if available
        if edge_attr is not None and hasattr(self, 'edge_proj'):
            edge_attn = self.edge_proj(edge_attr).t()  # [num_heads, num_edges]
            attn_scores[edge_index[0], edge_index[1]] += edge_attn
        
        # Apply attention mask (only attend to neighbors)
        attn_mask = torch.zeros_like(attn_scores, dtype=torch.bool)
        attn_mask[edge_index[0], edge_index[1]] = True
        attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        out = torch.einsum('hnm,mhd->nhd', attn_weights, v)
        out = out.reshape(-1, self.hidden_dim)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # Residual connection and layer norm
        if self.residual:
            out = out + res
        out = self.norm1(out)
        
        # Feed-forward network
        ffn_out = self.ffn(out)
        
        # Residual connection and layer norm
        if self.residual:
            ffn_out = ffn_out + out
        out = self.norm2(ffn_out)
        
        return out

# ============= Example Usage =============

if __name__ == "__main__":
    # Example configuration
    config = GNNConfig(
        layer_type=GNNLayerType.GRAPH_TRANSFORMER,
        input_dim=128,
        hidden_dim=256,
        output_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        use_edge_attr=True,
        edge_dim=32,
        jk_mode="lstm",
        global_pool="attention",
        transformer_ff_dim=512
    )
    
    # Create model
    model = GNNBase(config)
    
    # Example input
    num_nodes = 10
    num_edges = 30
    x = torch.randn(num_nodes, config.input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, config.edge_dim) if config.use_edge_attr else None
    batch = torch.zeros(num_nodes, dtype=torch.long)  # All nodes in same graph
    
    # Forward pass
    out = model(x, edge_index, edge_attr, batch=batch)
    print(f"Output shape: {out.shape}")  # Should be [1, config.output_dim] with global pooling