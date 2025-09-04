# LLM Data Pretraining Pipeline

A high-performance pipeline for processing PubMed abstracts for LLM pretraining, featuring efficient data cleaning, deduplication, and preparation for tokenization.

## Features

- **Efficient Processing**: Uses Parquet files and parallel processing for large datasets
- **PII Removal**: Automated detection and removal of personally identifiable information
- **Deduplication**: Content-based duplicate removal
- **Modern Tooling**: Built with Pydantic V2, Ruff, Typer, and UV
- **Container Ready**: Full Docker support for reproducible execution

## Quick Start

### Prerequisites

- Docker or Podman
- Git
- (Optional) UV for local development

### Running with Docker

1. **Clone the repository**:
   ```bash
   git clone https://github.com/miguel-tronix/llm-data-pretraining.git
   cd llm-data-pretraining
   # Build optimized image
   docker build -t llm-data-pretraining:prod .

   # Run with resource limits
   docker run -it --rm \
   --memory=4g \
   --cpus=2 \
   -v $(pwd)/data:/opt/llm-data-pretraining/data \
   -v $(pwd)/logs:/opt/llm-data-pretraining/logs \
   llm-data-pretraining:prod
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.