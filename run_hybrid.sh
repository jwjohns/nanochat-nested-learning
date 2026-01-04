#!/bin/bash
# =============================================================================
# run_hybrid.sh - Training script for GPT Hybrid model on DGX Spark GB10
# =============================================================================
#
# This script trains the hybrid Transformer + Local Context model.
# Optimized for NVIDIA DGX Spark with 128GB unified memory.
#
# Usage:
#   ./run_hybrid.sh [small|medium|large] [hybrid_conv|hybrid_mamba|transformer]
#
# Examples:
#   ./run_hybrid.sh small hybrid_conv    # Recommended for first run
#   ./run_hybrid.sh medium hybrid_conv   # Larger model with checkpointing
#   ./run_hybrid.sh small transformer    # Pure transformer baseline
#
# =============================================================================

set -e

# Default values
MODEL_SIZE=${1:-small}
ARCHITECTURE=${2:-hybrid_conv}
ACTIVATION=${3:-swiglu}

# Training hyperparameters (tuned for DGX Spark)
case $MODEL_SIZE in
    small)
        BATCH_SIZE=32
        GRAD_ACCUM=1
        USE_CHECKPOINTING=""
        ;;
    medium)
        BATCH_SIZE=16
        GRAD_ACCUM=2
        USE_CHECKPOINTING="--use_checkpointing"
        ;;
    large)
        BATCH_SIZE=8
        GRAD_ACCUM=4
        USE_CHECKPOINTING="--use_checkpointing"
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        echo "Usage: $0 [small|medium|large] [hybrid_conv|hybrid_mamba|transformer]"
        exit 1
        ;;
esac

echo "=============================================="
echo "GPT Hybrid Training Configuration"
echo "=============================================="
echo "Model size:        $MODEL_SIZE"
echo "Architecture:      $ARCHITECTURE"
echo "Activation:        $ACTIVATION"
echo "Batch size:        $BATCH_SIZE"
echo "Grad accumulation: $GRAD_ACCUM"
echo "Checkpointing:     ${USE_CHECKPOINTING:-disabled}"
echo "=============================================="

# Print model configuration
python -m scripts.hybrid_train \
    --model_size $MODEL_SIZE \
    --architecture $ARCHITECTURE \
    --activation $ACTIVATION \
    $USE_CHECKPOINTING \
    --print_config

echo ""
echo "=============================================="
echo "To start training, you need to:"
echo "1. Prepare your data in data/train.bin format"
echo "2. Modify engine.py to use GPTHybrid instead of GPT"
echo "   OR use the standalone training script below"
echo "=============================================="

# Uncomment below to run actual training with modified engine
# python -m nanochat.engine \
#     --model_type hybrid \
#     --model_size $MODEL_SIZE \
#     --architecture $ARCHITECTURE \
#     --activation $ACTIVATION \
#     --batch_size $BATCH_SIZE \
#     --gradient_accumulation_steps $GRAD_ACCUM \
#     $USE_CHECKPOINTING
