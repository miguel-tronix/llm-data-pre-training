# Build stage
FROM python:3.10-slim-bullseye as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
#RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl -LsSf https://astral.sh/uv/install.sh -o install.sh
RUN sh install.sh 
RUN mv /root/.local/bin/uv /usr/local/bin/uv
#ENV PATH="/root/.cargo/bin:${PATH}"

# Clone your project
RUN git clone https://github.com/miguel-tronix/llm-data-pre-training.git /opt/llm-data-pretraining

# Set working directory and create a virtual environment
WORKDIR /opt/llm-data-pretraining
RUN uv venv
RUN uv lock
RUN uv sync --frozen

# Runtime stage
FROM python:3.10-slim-bullseye

# Set the working directory
WORKDIR /opt/llm-data-pretraining

# Copy the project with the virtual environment from the builder stage
COPY --from=builder /opt/llm-data-pretraining /opt/llm-data-pretraining

# Set the entrypoint to use the Python interpreter from the virtual environment
ENTRYPOINT ["/opt/llm-data-pretraining/.venv/bin/python", "src/main.py"]