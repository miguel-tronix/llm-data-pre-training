# --- Build Stage ---
FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM python:3.10-slim-bullseye AS builder

# Install build dependencies if needed (e.g., for compiled extensions)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image
COPY --from=uv_bin /uv /usr/local/bin/uv

# Set working directory
WORKDIR /opt/llm-data-pretraining

# Enable bytecode compilation and use the uv link mode
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy only dependency files to leverage layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies without the project itself
# --no-install-project ensures we cache the environment layer
# --no-dev excludes developmental dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application
COPY . .

# Install the project
RUN uv sync --frozen --no-dev

# --- Runtime Stage ---
FROM python:3.10-slim-bullseye

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/llm-data-pretraining/.venv/bin:$PATH"
# Redirect HuggingFace cache to a directory owned by appuser.
# Without this, hf_hub defaults to ~/.cache/huggingface which does not exist
# for system users created with `useradd -r` (no home directory).
ENV HF_HOME="/opt/llm-data-pretraining/.cache/huggingface"
ENV HF_DATASETS_CACHE="/opt/llm-data-pretraining/.cache/huggingface/datasets"

# Set working directory
WORKDIR /opt/llm-data-pretraining

# Create a non-root user for security
# Use explicit UID/GID 999 so the Kubernetes Job fsGroup can be set accurately.
RUN groupadd -r -g 999 appuser && useradd -r -u 999 -g appuser appuser \
    && mkdir -p /opt/llm-data-pretraining/logs \
    && mkdir -p /opt/llm-data-pretraining/.cache/huggingface \
    && chown -R appuser:appuser /opt/llm-data-pretraining

# Copy the virtual environment and application from the builder
COPY --from=builder --chown=appuser:appuser /opt/llm-data-pretraining /opt/llm-data-pretraining

# Switch to non-root user
USER appuser

# Entrypoint using the virtual environment (via PATH)
ENTRYPOINT ["python", "main.py"]
