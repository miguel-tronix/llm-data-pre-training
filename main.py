import asyncio
import os
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv as setenvs

from llm_data_pretraining.cleaning.clean_and_tokenize import (
    DeduplicationMethod,
    JsonlDataCleanPipeline,
    PIIDetectionConfig,
    PipelineConfig,
    PipelineResult,
    TokenizationConfig,
    TokenizationPreparer,
)
from llm_data_pretraining.data_fetch.download_utils import (
    DownloadConfig,
    DownloadResult,
    HFDatasetDownloader,
)
from llm_data_pretraining.extraction.allenai_extractor import WebRecordExtractor
from llm_data_pretraining.extraction.configs import PipelineType, ProcessingStats
from llm_data_pretraining.extraction.github_extractor import GitHubRecordExtractor
from llm_data_pretraining.extraction.pubmed_extractor import PubMedAbstractExtractor
from llm_data_pretraining.extraction.wikipedia_extractor import WikiArticleExtractor
from llm_data_pretraining.training.tokenization import (
    BPETokenizer,
    TokenizationResult,
    TokenizerConfig,
)
from llm_data_pretraining.utils.pipeline_logger import get_pipeline_logger

ENV_FILE_PATH = ".env"
BASEDATA_PATH = "/opt/llm-data-pretraining"
DATASET_URL = "https://h"
RAWDATA_PATH = f"{BASEDATA_PATH}/rawdata"
PRECLEANDATA_PATH = f"{BASEDATA_PATH}/precleandata"
CLEANDATA_PATH = f"{BASEDATA_PATH}/cleandata"
TRAINDATA_PATH = f"{BASEDATA_PATH}/traindata"
BPE_CORPUS_FILE = "training_corpus.txt"
PUBMED_EXTRACT_FILE = "pubmed_abstracts.jsonl"
GITHUB_EXTRACT_FILE = "github_records.jsonl"
WIKI_EXTRACT_FILE = "wikipedia_articles.jsonl"
WEB_EXTRACT_FILE = "web_c4_records.jsonl"
PARALLEL_EXECS = 4
PUBMED_JSONL_SIZE_MB = 50
GITHUB_JSONL_SIZE_MB = 50
WIKI_JSONL_SIZE_MB = 20
WEB_JSONL_SIZE_MB = 50

logger = get_pipeline_logger()


async def download_pile_uncopyrighted_multiproc(
    repo_id: str = "monology/pile-uncopyrighted",
    raw_data_dir: str = "rawdata",
    file_pattern: str | None = None,
    chunk_size: int = 8192,
    **kwargs: Any,
) -> DownloadResult:
    """
    Main function to download files from a Hugging Face dataset repository.
    It automatically uses parallel processes for files larger than 500MB.
    """
    config = DownloadConfig(
        repo_id=repo_id,
        raw_data_dir=Path(raw_data_dir),
        max_retries=int(kwargs.get("max_retries", 3)),
        timeout=int(kwargs.get("timeout", 30)),
        file_pattern=file_pattern,
        chunk_size=chunk_size,
        max_files=kwargs.get("max_files", None),
        num_parallel_downloads=int(kwargs.get("num_parallel_downloads", 4)),
    )

    async with HFDatasetDownloader(config) as downloader:
        result: DownloadResult = await downloader.download_dataset()
        return result


async def download_pile_uncopyrighted_fast(
    repo_id: str = "monology/pile-uncopyrighted",
    raw_data_dir: str = "rawdata",
    file_pattern: str | None = None,
    max_retries: int = 3,
    timeout: int = 30,
    **kwargs: Any,
) -> DownloadResult:
    """
    Main function to download files from a Hugging Face dataset repository.
    It automatically uses parallel downloads for files larger than 500MB.

    Args:
        repo_id: Hugging Face dataset repository ID.
        raw_data_dir: Directory to store downloaded files.
        file_pattern: Regex pattern to filter files.
        max_retries: Maximum number of retry attempts per file.
        timeout: Request timeout in seconds.
        chunk_size: Chunk size for downloading.
        max_files: Maximum number of files to download (None for no limit).
        num_parallel_downloads: Number of parallel threads for large files.

    Returns:
        DownloadResult with download statistics.
    """
    # Create DownloadConfig with validation, including the parallel download setting
    config = DownloadConfig(
        repo_id=repo_id,
        raw_data_dir=Path(raw_data_dir),
        max_retries=max_retries,
        timeout=timeout,
        file_pattern=file_pattern,
        chunk_size=int(kwargs.get("chunk_size", 8192)),
        max_files=kwargs.get("max_files", 1),
        num_parallel_downloads=int(kwargs.get("num_parallel_downloads", 4)),
    )

    # Create downloader and execute download
    async with HFDatasetDownloader(config) as downloader:
        result: DownloadResult = await downloader.download_dataset()
        return result


# --- Main Download Function ---
async def download_pile_uncopyrighted(
    repo_id: str = "monology/pile-uncopyrighted",
    raw_data_dir: str = "rawdata",
    file_pattern: str | None = None,
    max_retries: int = 3,
    timeout: int = 30,
    **kwargs: Any,
) -> DownloadResult:
    """
    Main function to download files from the monology/pile-uncopyrighted dataset

    Args:
        repo_id: Hugging Face dataset repository ID
        raw_data_dir: Directory to store downloaded files
        file_pattern: Regex pattern to filter files
        max_retries: Maximum number of retry attempts per file
        timeout: Request timeout in seconds
        chunk_size: Chunk size for downloading
        max_files: Maximum number of files to download (None for no limit)

    Returns:
        DownloadResult with download statistics
    """
    # Create configuration with validation
    config = DownloadConfig(
        repo_id=repo_id,
        raw_data_dir=Path(raw_data_dir),
        max_retries=max_retries,
        timeout=timeout,
        file_pattern=file_pattern,
        chunk_size=kwargs.get("chunk_size", 8192),
        max_files=kwargs.get("max_files", 1),
    )

    # Create downloader and execute download
    downloader = HFDatasetDownloader(config)

    async with downloader:
        return await downloader.download_dataset()


# --- Generate PubMed Abstract JSONL File  ---
async def run_pubmed_extraction(
    input_path: str, output_path: str | None = None, return_objects: bool = False
) -> Any:
    """
    Main coroutine to extract PubMed abstracts from Pile-Uncopyrighted dataset

    Args:
        input_path: Path to input dataset file
        output_path: Path to output JSONL file (required if return_objects=False)
        return_objects: If True, returns list of objects instead of writing to file

    Returns:
        Either statistics dict (if writing to file) or list of PubMedAbstract objects
    """

    extractor = PubMedAbstractExtractor(
        use_parallel_zstd=True, num_processes=1, file_size_mb=PUBMED_JSONL_SIZE_MB
    )

    logger.info("Starting PubMed abstract extraction...")

    if return_objects:
        abstracts = await extractor.extract_abstracts_to_memory(input_path)
        logger.info(f"Extracted {len(abstracts)} abstracts to memory")
        return abstracts
    else:
        if not output_path:
            raise ValueError("output_path is required when return_objects=False")

        stats = await extractor.extract_abstracts_to_file(input_path, output_path)
        logger.info("Extraction completed successfully!")
        return stats


# -- Generate GitHub Records JSONL File  ---
async def run_github_extraction(
    input_path: str, output_path: str | None = None, return_objects: bool = False
) -> Any:
    """
    Main coroutine to extract GitHub records from Pile-Uncopyrighted dataset

    Args:
        input_path: Path to input dataset file
        output_path: Path to output JSONL file (required if return_objects=False)
        return_objects: If True, returns list of objects instead of writing to file

    Returns:
        Either statistics dict (if writing to file) or list of GitHubRecord objects
    """

    extractor = GitHubRecordExtractor(
        use_parallel_zstd=True, num_processes=1, file_size_mb=GITHUB_JSONL_SIZE_MB
    )

    logger.info("Starting GitHub record extraction...")

    if return_objects:
        records = await extractor.extract_records_to_memory(input_path)
        logger.info(f"Extracted {len(records)} abstracts to memory")
        return records
    else:
        if not output_path:
            raise ValueError("output_path is required when return_objects=False")

        stats = await extractor.extract_records_to_file(input_path, output_path)
        logger.info("Extraction completed successfully!")
        return stats


# -- Generate Wikipedia Records JSONL File  ---
async def run_wikipedia_extraction(
    input_path: str, output_path: str | None = None, return_objects: bool = False
) -> Any:
    """
    Main coroutine to extract Wikipedia articles from Pile-Uncopyrighted dataset

    Args:
        input_path: Path to input dataset file
        output_path: Path to output JSONL file (required if return_objects=False)
        return_objects: If True, returns list of objects instead of writing to file

    Returns:
        Either statistics dict (if writing to file) or list of GitHubRecord objects
    """

    extractor = WikiArticleExtractor(
        use_parallel_zstd=True, num_processes=1, file_size_mb=WIKI_JSONL_SIZE_MB
    )

    logger.info("Starting Wikipedia article extraction...")

    if return_objects:
        records = await extractor.extract_articles_to_memory(input_path)
        logger.info(f"Extracted {len(records)} abstracts to memory")
        return records
    else:
        if not output_path:
            raise ValueError("output_path is required when return_objects=False")

        stats = await extractor.extract_articles_to_file(input_path, output_path)
        logger.info("Extraction completed successfully!")
        return stats


# Generate Web Records JSONL File  ---
async def run_allenai_extraction(
    input_path: str, output_path: str | None = None, return_objects: bool = False
) -> Any:
    """
    Main coroutine to extract Wikipedia articles from Pile-Uncopyrighted dataset

    Args:
        input_path: Path to input dataset file
        output_path: Path to output JSONL file (required if return_objects=False)
        return_objects: If True, returns list of objects instead of writing to file

    Returns:
        Either statistics dict (if writing to file) or list of GitHubRecord objects
    """

    extractor = WebRecordExtractor(
        use_parallel_zstd=False,
        use_streaming=True,
        num_processes=1,
        file_size_mb=WEB_JSONL_SIZE_MB,
    )

    logger.info("Starting Wikipedia article extraction...")

    if return_objects:
        records = await extractor.extract_articles_to_memory(input_path)
        logger.info(f"Extracted {len(records)} abstracts to memory")
        return records
    else:
        if not output_path:
            raise ValueError("output_path is required when return_objects=False")

        stats = await extractor.extract_articles_to_file(input_path, output_path)
        logger.info("Extraction completed successfully!")
        return stats


# Generate PubMed Abstracts JSONL File from ZST Parallel Processing
async def generate_pubmed_abstracts_jsonl(
    input_zst_path: Path, output_jsonl_path: Path, max_abstracts: int | None = None
) -> ProcessingStats:
    """
    Generate pubmed_abstracts.jsonl from a ZST file using parallel processing

    Args:
        input_zst_path: Path to input ZST file
        output_jsonl_path: Path to output JSONL file
        max_abstracts: Maximum number of abstracts to extract

    Returns:
        Dictionary with extraction statistics
    """
    extractor = PubMedAbstractExtractor(num_processes=1)

    return await extractor.extract_abstracts_to_file(
        input_path=f"{input_zst_path.absolute}",
        output_path=f"{output_jsonl_path.absolute}",
    )


# Main function for transforming JSON recors into clean, deduplicated text files
def run_complete_clean_tokenize_pipeline(
    input_jsonl_path: str = "path/to/your/pubmed_abstracts.jsonl",
    output_clean_dir: str = "processed_data_pydantic",
    pipeline_record_type: str = PipelineType.PUBMED,
) -> PipelineResult:
    """Run the complete clean processing pipeline with Pydantic V2"""

    # Configuration with Pydantic validation
    config = PipelineConfig(
        input_path=Path(input_jsonl_path),
        output_dir=Path(output_clean_dir),
        min_abstract_length=50,
        max_abstract_length=1500,
        deduplication_method=DeduplicationMethod.CONTENT_HASH,
        pii_config=PIIDetectionConfig(
            detect_emails=True,
            detect_phones=True,
            detect_ssn=True,
            detect_patient_ids=True,
            detect_demographics=True,
        ),
        batch_size=1000,
        pipeline_type=pipeline_record_type
        if isinstance(pipeline_record_type, PipelineType)
        else PipelineType.PUBMED,
    )

    # Run the pipeline
    pipeline = JsonlDataCleanPipeline(config)
    result = pipeline.run_pipeline()

    # Log results
    logger.info(f"Pipeline completed in {result.processing_time:.2f} seconds")
    logger.info(f"Results: {result.model_dump_json(indent=2)}")

    # Prepare for tokenization
    token_config = TokenizationConfig(output_dir=config.output_dir)
    token_preparer = TokenizationPreparer(token_config)

    # Create training corpus
    corpus_file = (
        config.output_dir / f"training_corpus_{config.pipeline_type.value}.txt"
    )
    if result.final_file is not None:
        line_count = token_preparer.create_training_corpus(
            result.final_file, corpus_file
        )

        logger.info(f"Created training corpus with {line_count} lines")
        logger.info("Files ready for BPE tokenizer training:")
        logger.info(f"Final dataset: {result.final_file}")
        logger.info(f"Training corpus: {corpus_file}")
    else:
        logger.warning(
            "Pipeline did not produce a final file to tokenize - please investigate"
        )

    return result


# Main function for tokenization pipeline to
# produce BPE tokenizer and tokenized corpus outputs
def run_tokenization_pipeline(
    corpus_path: str | Path,
    output_dir: str | Path,
    pipeline_record_type: PipelineType = PipelineType.PUBMED,
    **kwargs: Any,
) -> TokenizationResult:
    """Complete tokenization pipeline with Pydantic V2"""
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer with validated config
    config = TokenizerConfig(
        vocab_size=kwargs.get("vocab_size", 30000),
        min_frequency=kwargs.get("min_frequency", 2),
        max_length=kwargs.get("max_length", 512),
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<|endoftext|>"],
    )

    tokenizer = BPETokenizer(config)

    # Train tokenizer
    tokenizer.train(corpus_path)

    # Save tokenizer
    tokenizer.save_tokenizer(output_dir / "tokenizer")

    # Tokenize corpus and save as binary
    tokens_path = output_dir / f"tokens_{pipeline_record_type.value}.bin"
    total_tokens = tokenizer.tokenize_corpus(corpus_path, tokens_path)

    # Create and return result
    result = TokenizationResult(
        success=True,
        output_dir=output_dir,
        vocab_size=tokenizer.tokenizer.get_vocab_size(),
        total_tokens=total_tokens,
        tokenizer_config=config,
    )

    logger.info(f"Tokenization pipeline complete. Files saved to {output_dir}")
    return result


# Integration with Typer for CLI
def add_tokenization_commands(app: typer.Typer) -> typer.Typer:
    """Add tokenization commands to Typer app"""

    @app.command()
    def tokenize(
        corpus_path: Path,
        output_dir: Path,
        vocab_size: int,
        min_frequency: int,
        max_length: int,
    ) -> None:
        """Tokenize training corpus using BPE"""
        corpus_path = typer.Argument(..., help="Path to training corpus")
        output_dir = typer.Option(Path("tokenized"), help="Output directory")
        vocab_size = typer.Option(30000, help="Vocabulary size")
        min_frequency = typer.Option(2, help="Minimum token frequency")
        max_length = typer.Option(512, help="Maximum sequence length")
        result = run_tokenization_pipeline(
            corpus_path=corpus_path,
            output_dir=output_dir,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            max_length=max_length,
        )

        # Output result as JSON
        typer.echo(result.model_dump_json(indent=2))

    return app


async def main() -> None:
    # Download Pile-Uncopyrighted dataset files
    download_result = await download_pile_uncopyrighted_multiproc(
        repo_id="monology/pile-uncopyrighted",
        raw_data_dir=RAWDATA_PATH,
        file_pattern=r".*\.jsonl\.zst",  # Only download compressed JSONL files
        max_retries=10,  # More retries for large files
        timeout=120,  # Longer timeout for large files
        chunk_size=32768,  # Larger chunk size for faster downloads
        max_files=1,
        num_parallel_downloads=PARALLEL_EXECS,  # Use parallel threads for the download
    )

    if download_result.success:
        # Extract records from downloaded files
        for file_path in download_result.downloaded_files:
            await zstd_extractor(file_path)
    else:
        logger.error(f"Download failed: {download_result.message}")

    web_extraction_stats = await run_allenai_extraction(
        input_path="allenai/c4",
        output_path=f"{PRECLEANDATA_PATH}/{WEB_EXTRACT_FILE}",
        return_objects=False,
    )
    if web_extraction_stats:
        stream_extractor(web_extraction_stats)


# Stream based extraction
def stream_extractor(web_extraction_stats: Any) -> None:
    BPE_CORPUS_FILE = f"training_corpus_{PipelineType.WEB.value}.txt"
    logger.info(f"Extracted {web_extraction_stats}")
    if (
        isinstance(web_extraction_stats, ProcessingStats)
        and int(f"{web_extraction_stats.output_size_mb}") > 0
    ):
        clean_tokenize_stats = run_complete_clean_tokenize_pipeline(
            input_jsonl_path=f"{PRECLEANDATA_PATH}/{WEB_EXTRACT_FILE}",
            output_clean_dir=CLEANDATA_PATH,
            pipeline_record_type=PipelineType.WEB,
        )
        if clean_tokenize_stats and clean_tokenize_stats.success:
            logger.info(
                f"Produced a training corpus at: {clean_tokenize_stats.final_file}"
            )
            bpe_tokenize_stats = run_tokenization_pipeline(
                corpus_path=f"{CLEANDATA_PATH}/{BPE_CORPUS_FILE}",
                output_dir=TRAINDATA_PATH,
                pipeline_record_type=PipelineType.WEB,
            )
            logger.info(f"{bpe_tokenize_stats.model_dump_json(indent=2)}")


# ZST File based extraction
async def zstd_extractor(file_path: str) -> None:
    pubmed_extraction_stats = None
    clean_tokenize_stats = None
    bpe_tokenize_stats = None

    # Extract Pubmed abstracts to JSONL
    pubmed_extraction_stats = await run_pubmed_extraction(
        input_path=f"{RAWDATA_PATH}/{file_path}",
        output_path=f"{PRECLEANDATA_PATH}/{PUBMED_EXTRACT_FILE}",
        return_objects=False,
    )
    # Clean Pubmed abstracts and prepare training corpus
    if pubmed_extraction_stats:
        BPE_CORPUS_FILE = f"training_corpus_{PipelineType.PUBMED.value}.txt"
        logger.info(f"Extracted {pubmed_extraction_stats}")
        if (
            isinstance(pubmed_extraction_stats, ProcessingStats)
            and int(f"{pubmed_extraction_stats.output_size_mb}") > 0
        ):
            clean_tokenize_stats = run_complete_clean_tokenize_pipeline(
                input_jsonl_path=f"{PRECLEANDATA_PATH}/{PUBMED_EXTRACT_FILE}",
                output_clean_dir=CLEANDATA_PATH,
                pipeline_record_type=PipelineType.PUBMED,
            )
        # Tokenize Pubmed training corpus with BPE tokenizer
        if clean_tokenize_stats and clean_tokenize_stats.success:
            logger.info(
                f"Produced a training corpus at: \
                        {clean_tokenize_stats.final_file}"
            )
            bpe_tokenize_stats = run_tokenization_pipeline(
                corpus_path=f"{CLEANDATA_PATH}/{BPE_CORPUS_FILE}",
                output_dir=TRAINDATA_PATH,
                pipeline_record_type=PipelineType.PUBMED,
            )
            logger.info(f"{bpe_tokenize_stats.model_dump_json(indent=2)}")
    # Extract GitHub records to JSONL
    github_extraction_stats = await run_github_extraction(
        input_path=f"{RAWDATA_PATH}/{file_path}",
        output_path=f"{PRECLEANDATA_PATH}/{GITHUB_EXTRACT_FILE}",
        return_objects=False,
    )
    # Clean GitHub records and prepare training corpus
    if github_extraction_stats:
        BPE_CORPUS_FILE = f"training_corpus_{PipelineType.GITHUB.value}.txt"
        logger.info(f"Extracted {github_extraction_stats}")
        if (
            isinstance(github_extraction_stats, ProcessingStats)
            and int(f"{github_extraction_stats.output_size_mb}") > 0
        ):
            clean_tokenize_stats = run_complete_clean_tokenize_pipeline(
                input_jsonl_path=f"{PRECLEANDATA_PATH}/{GITHUB_EXTRACT_FILE}",
                output_clean_dir=CLEANDATA_PATH,
                pipeline_record_type=PipelineType.GITHUB,
            )
        # Tokenize GitHub training corpus with BPE tokenizer
        if clean_tokenize_stats and clean_tokenize_stats.success:
            logger.info(
                f"Produced a training corpus at: \
                        {clean_tokenize_stats.final_file}"
            )
            bpe_tokenize_stats = run_tokenization_pipeline(
                corpus_path=f"{CLEANDATA_PATH}/{BPE_CORPUS_FILE}",
                output_dir=TRAINDATA_PATH,
                pipeline_record_type=PipelineType.GITHUB,
            )
            logger.info(f"{bpe_tokenize_stats.model_dump_json(indent=2)}")
    # Extract Wikipedia articles to JSONL
    wiki_extraction_stats = await run_wikipedia_extraction(
        input_path=f"{RAWDATA_PATH}/{file_path}",
        output_path=f"{PRECLEANDATA_PATH}/{WIKI_EXTRACT_FILE}",
        return_objects=False,
    )
    # Clean Wikipedia articles and prepare training corpus
    if wiki_extraction_stats:
        BPE_CORPUS_FILE = f"training_corpus_{PipelineType.WIKI.value}.txt"
        logger.info(f"Extracted {wiki_extraction_stats}")
        if (
            isinstance(wiki_extraction_stats, ProcessingStats)
            and int(f"{wiki_extraction_stats.output_size_mb}") > 0
        ):
            clean_tokenize_stats = run_complete_clean_tokenize_pipeline(
                input_jsonl_path=f"{PRECLEANDATA_PATH}/{WIKI_EXTRACT_FILE}",
                output_clean_dir=CLEANDATA_PATH,
                pipeline_record_type=PipelineType.WIKI,
            )
        # Tokenize Wikipedia training corpus with BPE tokenizer
        if clean_tokenize_stats and clean_tokenize_stats.success:
            logger.info(
                f"Produced a training corpus at: \
                        {clean_tokenize_stats.final_file}"
            )
            bpe_tokenize_stats = run_tokenization_pipeline(
                corpus_path=f"{CLEANDATA_PATH}/{BPE_CORPUS_FILE}",
                output_dir=TRAINDATA_PATH,
                pipeline_record_type=PipelineType.WIKI,
            )
            logger.info(f"{bpe_tokenize_stats.model_dump_json(indent=2)}")


if __name__ == "__main__":
    setenvs(f"{ENV_FILE_PATH}")
    BASEDATA_PATH = os.getenv("BASEDATA_PATH", BASEDATA_PATH)
    RAWDATA_PATH = f"{BASEDATA_PATH}/{os.getenv('RAWDATA_PATH', RAWDATA_PATH)}"
    PRECLEANDATA_PATH = (
        f"{BASEDATA_PATH}/{os.getenv('PRECLEANDATA_PATH', PRECLEANDATA_PATH)}"
    )
    CLEANDATA_PATH = f"{BASEDATA_PATH}/{os.getenv('CLEANDATA_PATH', CLEANDATA_PATH)}"
    TRAINDATA_PATH = f"{BASEDATA_PATH}/{os.getenv('TRAINDATA_PATH', TRAINDATA_PATH)}"
    PUBMED_EXTRACT_FILE = os.getenv("PUBMED_EXTRACT_FILE", PUBMED_EXTRACT_FILE)
    GITHUB_EXTRACT_FILE = os.getenv("GITHUB_EXTRACT_FILE", GITHUB_EXTRACT_FILE)
    WIKI_EXTRACT_FILE = os.getenv("WIKI_EXTRACT_FILE", WIKI_EXTRACT_FILE)
    WEB_EXTRACT_FILE = os.getenv("WEB_EXTRACT_FILE", WEB_EXTRACT_FILE)
    PUBMED_JSONL_SIZE_MB = int(
        f"{os.getenv('PUBMED_JSONL_SIZE_MB', PUBMED_JSONL_SIZE_MB)}"
    )
    GITHUB_JSONL_SIZE_MB = int(
        f"{os.getenv('GITHUB_JSONL_SIZE_MB', GITHUB_JSONL_SIZE_MB)}"
    )
    WIKI_JSONL_SIZE_MB = int(f"{os.getenv('WIKI_JSONL_SIZE_MB', WIKI_JSONL_SIZE_MB)}")
    WEB_JSONL_SIZE_MB = int(f"{os.getenv('WEB_JSONL_SIZE_MB', WEB_JSONL_SIZE_MB)}")
    PARALLEL_EXECS = int(f"{os.getenv('NUM_PROCESSES', PARALLEL_EXECS)}")
    BPE_CORPUS_FILE = f"{os.getenv('BPE_CORPUS_FILE', BPE_CORPUS_FILE)}"
    LOG_FILE = os.getenv("LOG_FILE", "logs/pretraining_pipeline.log")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMATTER = os.getenv(
        "LOG_FORMATTER", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.setLevel(LOG_LEVEL)
    for handler in logger.handlers:
        handler.setLevel(LOG_LEVEL)
    asyncio.run(main())
