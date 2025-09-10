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

1. **Fetch the Dockerfile, build and run the container**:
   ```bash
   curl -LsSf https://raw.githubusercontent.com/miguel-tronix/llm-data-pre-training/refs/heads/master/Dockerfile -o Dockerfile
   # Build optimized image
   docker build -t llm-data-pretraining:0.1.0 .

   # Run with resource limits
   docker run -it --rm \
   --memory=4g \
   --cpus=2 \
   -v $(pwd)/rawdata:/opt/llm-data-pretraining/rawdata \
   -v $(pwd)/preclean:/opt/llm-data-pretraining/precleandata \
   -v $(pwd)/cleandata:/opt/llm-data-pretraining/cleandata \
   -v $(pwd)/traindata:/opt/llm-data-pretraining/traindata \
   -v $(pwd)/logs:/opt/llm-data-pretraining/logs \
   llm-data-pretraining:0.1.0
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.