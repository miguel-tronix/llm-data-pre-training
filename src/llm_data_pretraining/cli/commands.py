import asyncio
from pathlib import Path
from typing import Annotated

import typer

from llm_data_pretraining.cleaning.clean_and_tokenize import (
    DeduplicationMethod,
    JsonlDataCleanPipeline,
    PIIDetectionConfig,
    PipelineConfig,
)
from llm_data_pretraining.data_fetch.download_utils import (
    DownloadConfig,
    HFDatasetDownloader,
)
from llm_data_pretraining.extraction.allenai_extractor import WebRecordExtractor
from llm_data_pretraining.extraction.configs import PipelineType
from llm_data_pretraining.extraction.github_extractor import GitHubRecordExtractor
from llm_data_pretraining.extraction.pubmed_extractor import PubMedAbstractExtractor
from llm_data_pretraining.extraction.wikipedia_extractor import WikiArticleExtractor
from llm_data_pretraining.training.tokenization import BPETokenizer, TokenizerConfig

app = typer.Typer(help="LLM Data Pretraining Pipeline CLI")


@app.command()
def download(  # noqa: PLR0913
    repo_id: Annotated[
        str, typer.Option(help="HuggingFace repository ID")
    ] = "monology/pile-uncopyrighted",
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path(
        "rawdata"
    ),
    file_pattern: Annotated[
        str | None, typer.Option(help="File pattern filter")
    ] = None,
    max_retries: Annotated[int, typer.Option(help="Max retry attempts")] = 3,
    timeout: Annotated[int, typer.Option(help="Request timeout (seconds)")] = 30,
    chunk_size: Annotated[int, typer.Option(help="Download chunk size")] = 8192,
    max_files: Annotated[int | None, typer.Option(help="Max files to download")] = None,
    parallel_downloads: Annotated[
        int, typer.Option(help="Parallel download processes")
    ] = 4,
) -> None:
    config = DownloadConfig(
        repo_id=repo_id,
        raw_data_dir=output_dir,
        max_retries=max_retries,
        timeout=timeout,
        file_pattern=file_pattern,
        chunk_size=chunk_size,
        max_files=max_files,
        num_parallel_downloads=parallel_downloads,
    )

    async def run() -> None:
        async with HFDatasetDownloader(config) as downloader:
            dl_result = await downloader.download_dataset()
        typer.echo(f"Download completed: {dl_result.message}")

    asyncio.run(run())


@app.command()
def extract(
    input_path: Annotated[Path, typer.Argument(help="Input file path")],
    output_path: Annotated[Path, typer.Option(help="Output file path")],
    source_type: Annotated[
        PipelineType, typer.Option(help="Data source type")
    ] = PipelineType.PUBMED,
    file_size_mb: Annotated[int, typer.Option(help="Target output size (MB)")] = 50,
) -> None:
    async def run() -> None:
        result = None
        if source_type == PipelineType.PUBMED:
            pubmed_extractor = PubMedAbstractExtractor(
                use_parallel_zstd=True, num_processes=1, file_size_mb=file_size_mb
            )
            result = await pubmed_extractor.extract_abstracts_to_file(
                str(input_path), str(output_path)
            )
        elif source_type == PipelineType.GITHUB:
            github_extractor = GitHubRecordExtractor(
                use_parallel_zstd=True, num_processes=1, file_size_mb=file_size_mb
            )
            result = await github_extractor.extract_records_to_file(
                str(input_path), str(output_path)
            )
        elif source_type == PipelineType.WIKI:
            wiki_extractor = WikiArticleExtractor(
                use_parallel_zstd=True, num_processes=1, file_size_mb=file_size_mb
            )
            result = await wiki_extractor.extract_articles_to_file(
                str(input_path), str(output_path)
            )
        elif source_type == PipelineType.WEB:
            web_extractor = WebRecordExtractor(
                use_parallel_zstd=True, num_processes=1, file_size_mb=file_size_mb
            )
            result = await web_extractor.extract_articles_to_file(
                str(input_path), str(output_path)
            )
        typer.echo(f"Extraction completed: {result}")

    asyncio.run(run())


@app.command()
def clean(  # noqa: PLR0913
    input_path: Annotated[Path, typer.Argument(help="Input JSONL file")],
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path(
        "cleandata"
    ),
    pipeline_type: Annotated[
        PipelineType, typer.Option(help="Pipeline type")
    ] = PipelineType.PUBMED,
    min_length: Annotated[int, typer.Option(help="Minimum text length")] = 50,
    max_length: Annotated[int, typer.Option(help="Maximum text length")] = 1500,
    detect_pii: Annotated[bool, typer.Option(help="Enable PII detection")] = True,
) -> None:
    pii_config: PIIDetectionConfig | None = (
        PIIDetectionConfig(
            detect_emails=detect_pii,
            detect_phones=detect_pii,
            detect_ssn=detect_pii,
            detect_patient_ids=detect_pii,
            detect_demographics=detect_pii,
        )
        if detect_pii
        else None
    )
    config = PipelineConfig(
        input_path=input_path,
        output_dir=output_dir,
        min_abstract_length=min_length,
        max_abstract_length=max_length,
        deduplication_method=DeduplicationMethod.CONTENT_HASH,
        pii_config=pii_config if pii_config is not None else PIIDetectionConfig(),
        pipeline_type=pipeline_type,
    )

    pipeline = JsonlDataCleanPipeline(config)
    result = pipeline.run_pipeline()
    typer.echo(f"Cleaning completed: {result.model_dump_json(indent=2)}")


@app.command()
def tokenize(
    corpus_path: Annotated[Path, typer.Argument(help="Training corpus file")],
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path(
        "traindata"
    ),
    vocab_size: Annotated[int, typer.Option(help="Vocabulary size")] = 30000,
    min_frequency: Annotated[int, typer.Option(help="Minimum token frequency")] = 2,
    max_length: Annotated[int, typer.Option(help="Maximum sequence length")] = 512,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TokenizerConfig(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_length=max_length,
    )

    tokenizer = BPETokenizer(config)
    tokenizer.train(corpus_path)
    tokenizer.save_tokenizer(output_dir / "tokenizer")

    tokens_path = output_dir / "tokens.bin"
    total_tokens = tokenizer.tokenize_corpus(corpus_path, tokens_path)

    typer.echo(f"Tokenization completed. Total tokens: {total_tokens}")


if __name__ == "__main__":
    app()
