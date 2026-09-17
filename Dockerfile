# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    GRADIO_ANALYTICS_ENABLED=False \
    TORCH_CUDA_ARCH_LIST=8.6

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libsndfile1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the locked application dependencies separately so source-only changes
# do not invalidate the large CUDA/PyTorch layer.
COPY pyproject.toml uv.lock README.md ./
RUN python3 -m pip install --no-cache-dir uv \
    && uv sync --frozen --extra api --no-dev --no-install-project

# Stable Audio 3 Medium requires Flash Attention 2. This wheel exactly matches
# this image's Python 3.10, CUDA 12.6 and locked PyTorch 2.7 installation.
RUN uv pip install --python /app/.venv/bin/python \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu126torch2.7-cp310-cp310-linux_x86_64.whl"

COPY . .
RUN uv sync --frozen --extra api --no-dev --inexact

RUN mkdir -p /app/outputs/jobs /cache/huggingface /cache/torch

WORKDIR /app/outputs
EXPOSE 5335

CMD ["/app/.venv/bin/python", "/app/run_api.py", "--model", "medium"]
