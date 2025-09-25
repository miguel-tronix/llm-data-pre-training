# LLM Data Pretraining Pipeline

A high-performance pipeline for processing HuggingFace Uncopyrighted Pile records for LLM pretraining, featuring efficient data cleaning, deduplication, and preparation for tokenization.

## Features

- **Efficient Processing**: Capable of using Parquet files and parallel processing for large datasets if running in a cluster
- **PII Removal**: Automated detection and removal of personally identifiable information
- **Deduplication**: Content-based duplicate removal
- **Modern Tooling**: Built with Pydantic V2, Ruff, Typer, and UV
- **Container Ready**: Full Docker support for reproducible execution

## Quick Start

### Prerequisites

- Docker or Podman
- (Optional) Git
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
   --cpus=4 \
   -v $(pwd)/rawdata:/opt/llm-data-pretraining/rawdata \
   -v $(pwd)/precleandata:/opt/llm-data-pretraining/precleandata \
   -v $(pwd)/cleandata:/opt/llm-data-pretraining/cleandata \
   -v $(pwd)/traindata:/opt/llm-data-pretraining/traindata \
   -v $(pwd)/logs:/opt/llm-data-pretraining/logs \
   llm-data-pretraining:0.1.0
    ```
2. **Environment Variables**:
By default the following environment variables are loaded from a .env file at /opt/llm-data-pretraining/.env
```
NUM_PROCESSES=4
PUBMED_JSONL_SIZE_MB=50
GITHUB_JSONL_SIZE_MB=50
WIKI_JSONL_SIZE_MB=20
WEB_JSONL_SIZE_MB=50
BASEDATA_PATH=/opt/llm-data-pre-training
RAWDATA_PATH=rawdata
PRECLEANDATA_PATH=precleandata
CLEANDATA_PATH=cleandata
TRAINDATA_PATH=traindata
BPE_CORPUS_FILE=training_corpus.txt
PUBMED_EXTRACT_FILE=pubmed_abstract_records.jsonl
GITHUB_EXTRACT_FILE=github_records.jsonl
WIKI_EXTRACT_FILE=wikipedia_articles.jsonl
WEB_EXTRACT_FILE=web_c4_records.jsonl
TOKENIZER_PARELLEISM=false
BPE_VOCAB_SIZE=50000
BPE_MERGE_SIZE=100000
BPE_MIN_FREQUENCY=2
BPE_THREADS=4
```

3. **Data Produced**:
The final data is available at traindata/ it consists of:
tokens binary files -  eg token_pubmed.bin
tokenizer metadata  -  eg tokenization_metadata_pubmed.json
tokenizer configs   -  eg traindata/tokenizer/tokenizer_pubmed.json

Intermediate data is produced at:
rawdata/train/ -  eg 00.jsonl.zst
precleandata/  -  eg pubmed_abstract_records.jsonl
cleandata/     -  eg final_abstracs_pubmed.jsonl


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.