FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /workspace/nanochat

# Install dependencies that aren't in the base image
# We install these globally in the container
RUN pip install datasets wandb tiktoken files-to-prompt uvicorn fastapi psutil

# Set env var to tell scripts we are in docker
ENV NANOCHAT_DOCKER=1

# Default command
CMD ["bash"]
