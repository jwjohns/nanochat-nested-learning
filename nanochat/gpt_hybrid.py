"""
GPT Hybrid Model - Enhanced Architecture for NanoChat
Inspired by LFM2, IBM Granite 4.0, and original nanochat

Key improvements over base nanochat:
1. SwiGLU activation (replacing ReLU²)
2. Gated Short Convolution blocks for local context (LFM2-style)
3. Minimal GQA attention blocks for global context
4. Optional Mamba-2 SSM blocks for hybrid mode
5. Higher RoPE theta (100K) for better long-context
6. Gradient checkpointing support for memory efficiency

This module is designed to be a drop-in replacement for gpt.py
while maintaining compatibility with the existing training infrastructure.
"""

import math
from functools import partial
from dataclasses import dataclass, field
from typing import Optional, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

# Try to import Mamba for hybrid_mamba architecture
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


@dataclass
class GPTHybridConfig:
    """Configuration for the hybrid model."""
    # Core dimensions
    sequence_len: int = 2048
    vocab_size: int = 50304  # Keep compatible with nanochat tokenizer
    n_layer: int = 16
    n_embd: int = 1024
    
    # Attention configuration
    n_head: int = 16         # Number of query heads
    n_kv_head: int = 8       # Number of key/value heads (GQA)
    n_attn_layers: int = 6   # Number of attention layers (rest are local context)
    
    # MLP configuration
    mlp_ratio: float = 4.0   # FFN expansion ratio
    activation: str = "swiglu"  # swiglu, geglu, relu2
    
    # Local context block configuration
    conv_kernel_size: int = 3
    
    # Architecture type
    architecture: str = "hybrid_conv"  # transformer, hybrid_conv, hybrid_mamba
    
    # Training options
    use_gradient_checkpointing: bool = False
    rope_theta: float = 100000.0  # Higher base for better long-context
    
    # Attention layer indices (which layers use attention vs local context)
    attn_layer_indices: Optional[List[int]] = None
    
    def __post_init__(self):
        """Set default attention layer indices if not provided."""
        if self.attn_layer_indices is None:
            if self.n_attn_layers >= self.n_layer:
                self.attn_layer_indices = list(range(self.n_layer))
            else:
                # Distribute attention layers evenly
                step = self.n_layer / self.n_attn_layers
                self.attn_layer_indices = [int(i * step) for i in range(self.n_attn_layers)]
                # Ensure last layer has attention
                if self.n_layer - 1 not in self.attn_layer_indices:
                    self.attn_layer_indices[-1] = self.n_layer - 1


def norm(x):
    """Purely functional RMSNorm with no learnable params."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """Apply rotary position embeddings."""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out


# =============================================================================
# Activation Functions
# =============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU activation: Swish-Gated Linear Unit
    From "GLU Variants Improve Transformer" (Shazeer, 2020)
    Used in LLaMA, PaLM, and many modern LLMs.
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)  # Gate
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)  # Up projection
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)  # Down projection
    
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GEGLU(nn.Module):
    """GEGLU activation: GELU-Gated Linear Unit."""
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
    
    def forward(self, x):
        return self.w3(F.gelu(self.w1(x)) * self.w2(x))


class ReLU2MLP(nn.Module):
    """Original nanochat MLP with ReLU² activation."""
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.c_fc = nn.Linear(in_features, hidden_features, bias=bias)
        self.c_proj = nn.Linear(hidden_features, out_features, bias=bias)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


def get_mlp(config: GPTHybridConfig) -> nn.Module:
    """Factory function to create the appropriate MLP."""
    hidden_dim = int(config.n_embd * config.mlp_ratio)
    # For GLU variants, use 2/3 of hidden_dim since they have an extra projection
    glu_hidden_dim = int(hidden_dim * 2 / 3)
    
    if config.activation == "swiglu":
        return SwiGLU(config.n_embd, glu_hidden_dim, config.n_embd, bias=False)
    elif config.activation == "geglu":
        return GEGLU(config.n_embd, glu_hidden_dim, config.n_embd, bias=False)
    elif config.activation == "relu2":
        return ReLU2MLP(config.n_embd, hidden_dim, config.n_embd, bias=False)
    else:
        raise ValueError(f"Unknown activation: {config.activation}")


# =============================================================================
# Attention Block
# =============================================================================

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with GQA support."""
    def __init__(self, config: GPTHybridConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
    
    def forward(self, x, cos_sin, kv_cache=None):
        B, T, C = x.size()
        
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK-Norm
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        
        Tq, Tk = q.size(2), k.size(2)
        enable_gqa = self.n_head != self.n_kv_head
        
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
        
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


# =============================================================================
# Gated Short Convolution Block (LFM2-style)
# =============================================================================

class GatedShortConvolution(nn.Module):
    """
    Gated Short Convolution block for local context modeling.
    Inspired by LFM2's architecture.
    
    Formula:
        (B, C, h̃) = Linear(h)
        y = SiLU(B) ⊙ h̃
        z = Conv_k(y)
        o = Linear_out(SiLU(C) ⊙ z)
    """
    def __init__(self, config: GPTHybridConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd
        self.kernel_size = config.conv_kernel_size
        self.expand_dim = config.n_embd * 2
        
        self.in_proj = nn.Linear(config.n_embd, self.expand_dim * 3, bias=False)
        self.conv = nn.Conv1d(
            self.expand_dim,
            self.expand_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size - 1,
            groups=self.expand_dim,
            bias=False
        )
        self.out_proj = nn.Linear(self.expand_dim, config.n_embd, bias=False)
    
    def forward(self, x, cos_sin=None, kv_cache=None):
        B, T, C = x.size()
        
        proj = self.in_proj(x)
        gate, value, hidden = proj.chunk(3, dim=-1)
        
        y = F.silu(gate) * hidden
        y = y.transpose(1, 2)
        y = self.conv(y)[:, :, :T]
        y = y.transpose(1, 2)
        y = F.silu(value) * y
        y = self.out_proj(y)
        return y


# =============================================================================
# Mamba SSM Block (Optional)
# =============================================================================

class MambaBlock(nn.Module):
    """Wrapper around Mamba SSM for hybrid architecture."""
    def __init__(self, config: GPTHybridConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm package required. Install with: pip install mamba-ssm")
        
        self.mamba = Mamba(
            d_model=config.n_embd,
            d_state=16,
            d_conv=4,
            expand=2,
        )
    
    def forward(self, x, cos_sin=None, kv_cache=None):
        return self.mamba(x)


# =============================================================================
# Transformer Block
# =============================================================================

class Block(nn.Module):
    """A single transformer block with either attention or local context mixing."""
    def __init__(self, config: GPTHybridConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_attention = layer_idx in config.attn_layer_indices
        
        if self.use_attention:
            self.mixer = CausalSelfAttention(config, layer_idx)
        elif config.architecture == "hybrid_mamba" and MAMBA_AVAILABLE:
            self.mixer = MambaBlock(config, layer_idx)
        else:
            self.mixer = GatedShortConvolution(config, layer_idx)
        
        self.mlp = get_mlp(config)
        self.use_checkpoint = config.use_gradient_checkpointing
    
    def _forward(self, x, cos_sin, kv_cache):
        x = x + self.mixer(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x
    
    def forward(self, x, cos_sin, kv_cache=None):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, cos_sin, kv_cache,
                use_reentrant=False
            )
        return self._forward(x, cos_sin, kv_cache)


# =============================================================================
# Main Model
# =============================================================================

class GPTHybrid(nn.Module):
    """
    Hybrid GPT model with Transformer + Local Context architecture.
    
    Features:
    - Hybrid Transformer + Gated Conv/Mamba architecture
    - SwiGLU/GEGLU activation functions
    - Grouped Query Attention (GQA)
    - QK-Norm for stability
    - High-theta rotary position embeddings
    """
    def __init__(self, config: GPTHybridConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size}")
        
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        self._print_architecture_summary()
    
    def _print_architecture_summary(self):
        """Print architecture summary."""
        attn_layers = [i for i in range(self.config.n_layer) if i in self.config.attn_layer_indices]
        local_layers = [i for i in range(self.config.n_layer) if i not in self.config.attn_layer_indices]
        
        print0(f"\n{'='*60}")
        print0(f"GPT Hybrid - Architecture Summary")
        print0(f"{'='*60}")
        print0(f"Architecture: {self.config.architecture}")
        print0(f"Total layers: {self.config.n_layer}")
        print0(f"Attention layers ({len(attn_layers)}): {attn_layers}")
        print0(f"Local context layers ({len(local_layers)}): {local_layers}")
        print0(f"Model dimension: {self.config.n_embd}")
        print0(f"Attention heads: {self.config.n_head} (KV heads: {self.config.n_kv_head})")
        print0(f"Activation: {self.config.activation}")
        print0(f"RoPE theta: {self.config.rope_theta}")
        print0(f"Gradient checkpointing: {self.config.use_gradient_checkpointing}")
        print0(f"{'='*60}\n")
    
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000.0, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin
    
    def init_weights(self):
        """Initialize model weights (compatible with nanochat)."""
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        
        for block in self.transformer.h:
            if hasattr(block.mixer, 'c_q'):  # Attention
                torch.nn.init.uniform_(block.mixer.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.mixer.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.mixer.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.mixer.c_proj.weight)
            elif hasattr(block.mixer, 'in_proj'):  # Gated conv
                torch.nn.init.uniform_(block.mixer.in_proj.weight, -s, s)
                torch.nn.init.zeros_(block.mixer.out_proj.weight)
            
            if hasattr(block.mlp, 'w1'):  # SwiGLU/GEGLU
                torch.nn.init.uniform_(block.mlp.w1.weight, -s, s)
                torch.nn.init.uniform_(block.mlp.w2.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.w3.weight)
            elif hasattr(block.mlp, 'c_fc'):  # ReLU²
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
        
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, self.config.rope_theta)
        self.cos, self.sin = cos, sin
        
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
    
    def get_device(self):
        return self.transformer.wte.weight.device
    
    def estimate_flops(self):
        """Estimate FLOPs per token."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        n_attn = len(self.config.attn_layer_indices)
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * n_attn * h * q * t
        return num_flops_per_token
    
    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        """Setup optimizers (compatible with nanochat)."""
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling LR ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers
    
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        
        assert T <= self.cos.size(1), f"Sequence too long: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device
        assert self.cos.dtype == torch.bfloat16
        
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)
        
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        return logits
    
    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Autoregressive generation."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()


# =============================================================================
# Model Factory Functions
# =============================================================================

def create_hybrid_small(architecture="hybrid_conv", activation="swiglu", use_checkpointing=False):
    """Create a small hybrid model (~350M params) for DGX Spark."""
    config = GPTHybridConfig(
        sequence_len=2048,
        vocab_size=50304,
        n_layer=16,
        n_embd=1024,
        n_head=16,
        n_kv_head=8,
        n_attn_layers=6,
        mlp_ratio=4.0,
        activation=activation,
        conv_kernel_size=3,
        architecture=architecture,
        use_gradient_checkpointing=use_checkpointing,
        rope_theta=100000.0,
    )
    return GPTHybrid(config)


def create_hybrid_medium(architecture="hybrid_conv", activation="swiglu", use_checkpointing=True):
    """Create a medium hybrid model (~700M params)."""
    config = GPTHybridConfig(
        sequence_len=2048,
        vocab_size=50304,
        n_layer=20,
        n_embd=1536,
        n_head=24,
        n_kv_head=8,
        n_attn_layers=6,
        mlp_ratio=4.0,
        activation=activation,
        conv_kernel_size=3,
        architecture=architecture,
        use_gradient_checkpointing=use_checkpointing,
        rope_theta=100000.0,
    )
    return GPTHybrid(config)


def create_hybrid_large(architecture="hybrid_conv", activation="swiglu", use_checkpointing=True):
    """Create a large hybrid model (~1.2B params)."""
    config = GPTHybridConfig(
        sequence_len=2048,
        vocab_size=50304,
        n_layer=24,
        n_embd=2048,
        n_head=32,
        n_kv_head=8,
        n_attn_layers=6,
        mlp_ratio=4.0,
        activation=activation,
        conv_kernel_size=3,
        architecture=architecture,
        use_gradient_checkpointing=use_checkpointing,
        rope_theta=100000.0,
    )
    return GPTHybrid(config)


# Alias for compatibility
GPT = GPTHybrid
GPTConfig = GPTHybridConfig
