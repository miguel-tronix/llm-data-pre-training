# Build stage
FROM python:3.10-slim-bullseye as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Clone your project
RUN git clone https://github.com/miguel-tronix/llm-data-pretraining /opt/llm-data-pretraining

# Set working directory and sync dependencies
WORKDIR /opt/llm-data-pretraining
RUN uv sync --frozen

# Runtime stage
FROM python:3.10-slim-bullseye

# Copy only the necessary files from the builder stage
COPY --from=builder /opt/llm-data-pretraining /opt/llm-data-pretraining
COPY --from=builder /root/.cache/uv /root/.cache/uv

# Set working directory and entrypoint
WORKDIR /opt/llm-data-pretraining
ENTRYPOINT ["uv", "run", "src/main.py"]