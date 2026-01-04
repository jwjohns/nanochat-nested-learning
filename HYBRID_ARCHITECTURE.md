# Hybrid Architecture for NanoChat

This branch introduces a **hybrid Transformer + Local Context** architecture inspired by [LFM2](https://arxiv.org/abs/2511.23404) and [IBM Granite 4.0](https://www.ibm.com/granite). The design is optimized for training on NVIDIA DGX Spark GB10.

## Key Improvements

| Feature | Original nanochat | Hybrid Architecture |
|---------|-------------------|---------------------|
| **Activation** | ReLU² | SwiGLU (default), GEGLU, ReLU² |
| **Architecture** | Pure Transformer | Hybrid Transformer + Local Context |
| **Local Context** | None | Gated Short Convolution / Mamba-2 SSM |
| **Attention Layers** | All layers | Minimal (6 layers by default) |
| **RoPE Base** | 10,000 | 100,000 |
| **Memory Efficiency** | Baseline | ~30-50% reduction |

## Architecture Overview

The hybrid architecture uses a **minimal attention** design:

```
Layer 0:  [Gated Conv] → [SwiGLU MLP]
Layer 1:  [Gated Conv] → [SwiGLU MLP]
Layer 2:  [GQA Attention] → [SwiGLU MLP]  ← Attention layer
Layer 3:  [Gated Conv] → [SwiGLU MLP]
...
Layer 15: [GQA Attention] → [SwiGLU MLP]  ← Attention layer
```

### Why Hybrid?

1. **Attention is expensive**: O(n²) memory and compute for sequence length n
2. **Local context is cheap**: O(n) for convolutions and SSMs
3. **Most information is local**: Only ~6 attention layers needed for global reasoning
4. **Better scaling**: Linear memory growth with sequence length

## Files Added

```
nanochat/
├── gpt_hybrid.py          # Hybrid model implementation
scripts/
├── hybrid_train.py        # Training configuration script
run_hybrid.sh              # Easy training launcher
HYBRID_ARCHITECTURE.md     # This documentation
```

## Quick Start

### 1. Check Model Configuration

```bash
# Print model config without training
python -m scripts.hybrid_train --model_size small --architecture hybrid_conv --print_config
```

### 2. Model Sizes

| Size | Params | Layers | d_model | Attn Layers | DGX Spark Time |
|------|--------|--------|---------|-------------|----------------|
| small | ~350M | 16 | 1024 | 6 | ~9-10 days |
| medium | ~700M | 20 | 1536 | 6 | ~18-20 days |
| large | ~1.2B | 24 | 2048 | 6 | ~30+ days |

### 3. Training on DGX Spark

For the **small** model (recommended first):
```bash
# Batch size 32, no gradient checkpointing
./run_hybrid.sh small hybrid_conv
```

For the **medium** model:
```bash
# Batch size 16, gradient checkpointing enabled
./run_hybrid.sh medium hybrid_conv
```

## Architecture Options

### `hybrid_conv` (Recommended)
- Uses **Gated Short Convolution** for local context
- No additional dependencies
- Best balance of speed and quality

```python
from nanochat.gpt_hybrid import create_hybrid_small

model = create_hybrid_small(architecture="hybrid_conv", activation="swiglu")
```

### `hybrid_mamba`
- Uses **Mamba-2 SSM** for local context
- Requires `pip install mamba-ssm`
- Better for very long sequences

```python
model = create_hybrid_small(architecture="hybrid_mamba", activation="swiglu")
```

### `transformer`
- Pure transformer (like original nanochat)
- All layers use attention
- Useful as baseline comparison

```python
model = create_hybrid_small(architecture="transformer", activation="swiglu")
```

## Activation Functions

### SwiGLU (Default)
```
SwiGLU(x) = (SiLU(xW₁) ⊙ xW₂) W₃
```
- Used in LLaMA, PaLM, Mistral
- ~10-15% better than ReLU² at same compute

### GEGLU
```
GEGLU(x) = (GELU(xW₁) ⊙ xW₂) W₃
```
- Similar to SwiGLU but uses GELU
- Slightly smoother gradients

### ReLU² (Original)
```
ReLU²(x) = ReLU(xW)² W₂
```
- Original nanochat activation
- Simpler but less expressive

## Integration with Existing Code

### Option 1: Direct Replacement

In `nanochat/engine.py`, replace:
```python
from nanochat.gpt import GPT, GPTConfig
```
with:
```python
from nanochat.gpt_hybrid import GPTHybrid as GPT, GPTHybridConfig as GPTConfig
```

### Option 2: Conditional Import

```python
import os
USE_HYBRID = os.environ.get("USE_HYBRID", "0") == "1"

if USE_HYBRID:
    from nanochat.gpt_hybrid import GPTHybrid as GPT, GPTHybridConfig as GPTConfig
else:
    from nanochat.gpt import GPT, GPTConfig
```

Then run with:
```bash
USE_HYBRID=1 python -m nanochat.engine ...
```

## DGX Spark GB10 Optimization

### Hardware Specs
- **Memory**: 128GB unified (96GB GPU-allocated)
- **Compute**: 1 PFLOP FP4, 6,144 CUDA cores
- **Bandwidth**: 273 GB/s

### Recommended Settings

| Model | Batch Size | Grad Accum | Checkpointing | Est. Memory |
|-------|------------|------------|---------------|-------------|
| small | 32 | 1 | No | ~60GB |
| small | 40 | 1 | No | ~75GB |
| medium | 16 | 2 | Yes | ~70GB |
| large | 8 | 4 | Yes | ~80GB |

### Performance Tips

1. **Use Docker**: NVIDIA's PyTorch container has optimized CUDA
2. **Enable torch.compile**: 10-20% speedup
3. **BF16 mixed precision**: Enabled by default
4. **Gradient checkpointing**: For medium/large models

## Comparison with Related Work

### vs. LFM2
- **Similar**: Minimal hybrid design, gated convolutions
- **Different**: Simpler implementation, no hardware-in-the-loop search
- **Trade-off**: Less optimized but more accessible

### vs. IBM Granite 4.0
- **Similar**: Hybrid Mamba-Transformer concept
- **Different**: Smaller scale, different layer ratios (6 attn vs 1/3)
- **Trade-off**: Trainable on consumer/prosumer hardware

### vs. Original nanochat
- **Better**: Modern activations, hybrid architecture, higher RoPE theta
- **Similar**: Training infrastructure, optimizer choices
- **Trade-off**: Slightly more complex architecture

## References

1. [nanochat](https://github.com/karpathy/nanochat) - Karpathy's minimal chat model
2. [LFM2 Technical Report](https://arxiv.org/abs/2511.23404) - Liquid Foundation Models
3. [IBM Granite 4.0](https://www.ibm.com/granite) - Hybrid Mamba-Transformer
4. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU activation
5. [Mamba](https://arxiv.org/abs/2312.00752) - State Space Models
