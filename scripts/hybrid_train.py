#!/usr/bin/env python3
"""
Training script for GPT Hybrid model.

This script is designed to work with the existing nanochat infrastructure
while using the new hybrid architecture.

Usage:
    # Small model with hybrid_conv + swiglu (recommended for DGX Spark)
    python -m scripts.hybrid_train --model_size small --architecture hybrid_conv
    
    # With gradient checkpointing for larger models
    python -m scripts.hybrid_train --model_size medium --use_checkpointing
    
    # Pure transformer baseline (like original nanochat)
    python -m scripts.hybrid_train --model_size small --architecture transformer --activation relu2
"""

import os
import sys
import argparse
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.gpt_hybrid import (
    GPTHybridConfig, 
    GPTHybrid,
    create_hybrid_small,
    create_hybrid_medium,
    create_hybrid_large,
)


def get_model_config(args) -> GPTHybridConfig:
    """Create model configuration based on arguments."""
    
    # Base configurations for different sizes
    size_configs = {
        "small": {
            "n_layer": 16,
            "n_embd": 1024,
            "n_head": 16,
            "n_kv_head": 8,
            "n_attn_layers": 6,
        },
        "medium": {
            "n_layer": 20,
            "n_embd": 1536,
            "n_head": 24,
            "n_kv_head": 8,
            "n_attn_layers": 6,
        },
        "large": {
            "n_layer": 24,
            "n_embd": 2048,
            "n_head": 32,
            "n_kv_head": 8,
            "n_attn_layers": 6,
        },
        "xlarge": {
            "n_layer": 32,
            "n_embd": 2560,
            "n_head": 40,
            "n_kv_head": 8,
            "n_attn_layers": 8,
        },
    }
    
    if args.model_size not in size_configs:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    size_config = size_configs[args.model_size]
    
    config = GPTHybridConfig(
        sequence_len=args.sequence_len,
        vocab_size=50304,  # nanochat tokenizer vocab size
        n_layer=size_config["n_layer"],
        n_embd=size_config["n_embd"],
        n_head=size_config["n_head"],
        n_kv_head=size_config["n_kv_head"],
        n_attn_layers=args.n_attn_layers if args.n_attn_layers else size_config["n_attn_layers"],
        mlp_ratio=args.mlp_ratio,
        activation=args.activation,
        conv_kernel_size=args.conv_kernel_size,
        architecture=args.architecture,
        use_gradient_checkpointing=args.use_checkpointing,
        rope_theta=args.rope_theta,
    )
    
    return config


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description="Train GPT Hybrid Model")
    
    # Model configuration
    parser.add_argument("--model_size", type=str, default="small",
                       choices=["small", "medium", "large", "xlarge"],
                       help="Model size preset")
    parser.add_argument("--architecture", type=str, default="hybrid_conv",
                       choices=["transformer", "hybrid_conv", "hybrid_mamba"],
                       help="Architecture type")
    parser.add_argument("--activation", type=str, default="swiglu",
                       choices=["swiglu", "geglu", "relu2"],
                       help="Activation function")
    parser.add_argument("--sequence_len", type=int, default=2048,
                       help="Sequence length")
    parser.add_argument("--n_attn_layers", type=int, default=None,
                       help="Number of attention layers (default: from size preset)")
    parser.add_argument("--mlp_ratio", type=float, default=4.0,
                       help="MLP expansion ratio")
    parser.add_argument("--conv_kernel_size", type=int, default=3,
                       help="Convolution kernel size for local context blocks")
    parser.add_argument("--rope_theta", type=float, default=100000.0,
                       help="RoPE base theta")
    parser.add_argument("--use_checkpointing", action="store_true",
                       help="Use gradient checkpointing")
    
    # Just print config for now (actual training uses base_train.py)
    parser.add_argument("--print_config", action="store_true",
                       help="Print configuration and exit")
    
    args = parser.parse_args()
    
    # Create configuration
    config = get_model_config(args)
    
    # Create model
    model = GPTHybrid(config)
    
    # Count parameters
    total, trainable = count_parameters(model)
    
    print(f"\nModel Configuration:")
    print(f"  Size: {args.model_size}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Activation: {args.activation}")
    print(f"  Layers: {config.n_layer}")
    print(f"  Model dim: {config.n_embd}")
    print(f"  Heads: {config.n_head} (KV: {config.n_kv_head})")
    print(f"  Attention layers: {config.n_attn_layers}")
    print(f"  Sequence length: {config.sequence_len}")
    print(f"  RoPE theta: {config.rope_theta}")
    print(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
    print(f"\nParameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    
    if args.print_config:
        return
    
    print("\n" + "="*60)
    print("To train this model, modify scripts/base_train.py to use GPTHybrid")
    print("or run with the existing nanochat training infrastructure.")
    print("="*60)


if __name__ == "__main__":
    main()
