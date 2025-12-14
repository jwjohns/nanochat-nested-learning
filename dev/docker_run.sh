#!/bin/bash
# Helper script to launch the nanochat docker container on DGX
# Usage: bash dev/docker_run.sh

# Build the image
docker build -t nanochat:latest -f docker/Dockerfile .

# Run the container
# - Mount current directory to /workspace/nanochat
# - Enable GPUs
# - Share host network (optional, but good for DDP/wandb)
# - Run interactively
docker run --gpus all -it --rm \
    --net=host \
    -v $(pwd):/workspace/nanochat \
    nanochat:latest
