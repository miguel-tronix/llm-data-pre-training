#
import asyncio
import logging
import os
import typer
from dotenv import load_dotenv as env
from typing import Dict, Any, Optional, Union
from pathlib import Path
from data_fetch.download_utils import DownloadResult, DownloadConfig, HFDatasetDownloader
from data_prep.pubmed_extractor import PubMedAbstractExtractor
from data_prep.github_extractor import GitHubRecordExtractor
from data_prep.wikipedia_extractor import WikiArticleExtractor
from data_clean.clean_and_tokenize import DeduplicationMethod,\
      PipelineConfig, TokenizationConfig, TokenizationPreparer, \
    PIIDetectionConfig, PubMedPipeline, PipelineResult
from data_train.tokenization import TokenizationResult, TokenizerConfig, BPETokenizer
ENV_FILE_PATH = ".venv/.env"
BASEDATA_PATH = "/home/migtronix/llm-data-pre-training"
DATASET_URL = "https://h"
RAWDATA_PATH = f"{BASEDATA_PATH}/rawdata"
PRECLEANDATA_PATH = f"{BASEDATA_PATH}/precleandata"
CLEANDATA_PATH = f"{BASEDATA_PATH}/cleandata"
TRAINDATA_PATH = f"{BASEDATA_PATH}/traindata"
BPE_CORPUS_FILE = "training_corpus.txt"
PUBMED_EXTRACT_FILE = "pubmed_abstracts.jsonl"
GITHUB_EXTRACT_FILE = "github_records.jsonl"
WIKI_EXTRACT_FILE = "wikipedia_articles.jsonl"
PARALLEL_EXECS = 4
PUBMED_JSONL_SIZE_MB = 50
GITHUB_JSONL_SIZE_MB = 50
WIKI_JSONL_SIZE_MB = 20

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from pathlib import Path
from typing import Optional

# Assuming the previously defined DownloadConfig, DownloadResult, 
# and HFDatasetDownloader classes are in the same file or imported.

async def download_pile_uncopyrighted_multiproc(
    repo_id: str = "monology/pile-uncopyrighted",
    raw_data_dir: str = "rawdata",
    file_pattern: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 30,
    chunk_size: int = 8192,
    max_files: Optional[int] = None,
    num_parallel_downloads: int = 4
) -> DownloadResult:
    """
    Main function to download files from a Hugging Face dataset repository.
    It automatically uses parallel processes for files larger than 500MB.
    """
    config = DownloadConfig(
        repo_id=repo_id,
        raw_data_dir=Path(raw_data_dir),
        max_retries=max_retries,
        timeout=timeout,
        file_pattern=file_pattern,
        chunk_size=chunk_size,
        max_files=max_files,
        num_parallel_downloads=num_parallel_downloads
    )
    
    async with HFDatasetDownloader(config) as downloader:
        return await downloader.download_dataset()


async def download_pile_uncopyrighted_fast(
    repo_id: str = "monology/pile-uncopyrighted",
    raw_data_dir: str = "rawdata",
    file_pattern: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 30,
    chunk_size: int = 8192,
    max_files: Optional[int] = None,
    num_parallel_downloads: int = 4
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
    # Create configuration with validation, including the parallel download setting
    config = DownloadConfig(
        repo_id=repo_id,
        raw_data_dir=Path(raw_data_dir),
        max_retries=max_retries,
        timeout=timeout,
        file_pattern=file_pattern,
        chunk_size=chunk_size,
        max_files=max_files,
        num_parallel_downloads=num_parallel_downloads
    )
    
    # Create downloader and execute download
    async with HFDatasetDownloader(config) as downloader:
        return await downloader.download_dataset()

# Example of how to run this function:
#
# import asyncio
#
# async def main():
#     result = await download_pile_uncopyrighted(
#         # To test, let's download just one large file from the dataset
#         file_pattern=r"train/00.jsonl.zst", 
#         max_files=1,
#         num_parallel_downloads=8 # Use 8 parallel threads for the download
#     )
#     print(f"Download successful: {result.success}")
#     print(f"Message: {result.message}")
#     print(f"Downloaded files: {result.downloaded_files}")
#
# if __name__ == "__main__":
#     asyncio.run(main())

# --- Main Download Function ---
async def download_pile_uncopyrighted(
    repo_id: str = "monology/pile-uncopyrighted",
    raw_data_dir: str = "rawdata",
    file_pattern: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 30,
    chunk_size: int = 8192,
    max_files: Optional[int] = None
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
        chunk_size=chunk_size,
        max_files=max_files
    )
    
    # Create downloader and execute download
    downloader = HFDatasetDownloader(config)
    
    async with downloader:
        return await downloader.download_dataset()

# --- Main Extraction Coroutine ---
async def run_pubmed_extraction(
    input_path: str, 
    output_path: Optional[str] = None,
    return_objects: bool = False
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
        use_parallel_zstd=True, 
        num_processes=1,
        file_size_mb=PUBMED_JSONL_SIZE_MB
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

async def run_github_extraction(
    input_path: str, 
    output_path: Optional[str] = None,
    return_objects: bool = False
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
        use_parallel_zstd=True, 
        num_processes=1,
        file_size_mb=GITHUB_JSONL_SIZE_MB
    )
    
    logger.info("Starting PubMed abstract extraction...")
    
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

async def run_wikipedia_extraction(
    input_path: str, 
    output_path: Optional[str] = None,
    return_objects: bool = False
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
        use_parallel_zstd=True, 
        num_processes=1,
        file_size_mb=WIKI_JSONL_SIZE_MB)
    
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

# New routine specifically for generating pubmed_abstracts.jsonl
async def generate_pubmed_abstracts_jsonl(
    input_zst_path: Path,
    output_jsonl_path: Path,
    max_abstracts: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate pubmed_abstracts.jsonl from a ZST file using parallel processing
    
    Args:
        input_zst_path: Path to input ZST file
        output_jsonl_path: Path to output JSONL file
        max_abstracts: Maximum number of abstracts to extract
        
    Returns:
        Dictionary with extraction statistics
    """
    extractor = PubMedAbstractExtractor(
        num_processes=1
    )
    
    return await extractor.extract_abstracts_to_file(
        input_path=f"{input_zst_path.absolute}",
        output_path=f"{output_jsonl_path.absolute}"
    )

# --- Example Usage with Pydantic V2 ---
def run_complete_clean_tokenize_pipeline(
        input_jsonl_path: str = "path/to/your/pubmed_abstracts.jsonl",
        output_clean_dir: str = "processed_data_pydantic"
    ) -> PipelineResult:
    """Run the complete PubMed processing pipeline with Pydantic V2"""
    
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
            detect_demographics=True
        ),
        batch_size=1000
    )
    
    # Run the pipeline
    pipeline = PubMedPipeline(config)
    result = pipeline.run_pipeline()
    
    # Log results
    logger.info(f"Pipeline completed in {result.processing_time:.2f} seconds")
    logger.info(f"Results: {result.model_dump_json(indent=2)}")
    
    # Prepare for tokenization
    token_config = TokenizationConfig(output_dir=config.output_dir)
    token_preparer = TokenizationPreparer(token_config)
    
    # Create training corpus
    corpus_file = config.output_dir / BPE_CORPUS_FILE
    if result.final_file is not None:
        line_count = token_preparer.create_training_corpus(result.final_file, corpus_file)
    
        logger.info(f"Created training corpus with {line_count} lines")
        logger.info(f"Files ready for BPE tokenizer training:")
        logger.info(f"Final dataset: {result.final_file}")
        logger.info(f"Training corpus: {corpus_file}")
    else:
        logger.warning(f"Pipeline did not produce a final file to tokenize - please investigate")
    
    return result

# Main function for tokenization pipeline
def run_tokenization_pipeline(
    corpus_path: Union[str, Path],
    output_dir: Union[str, Path],
    vocab_size: int = 30000,
    min_frequency: int = 2,
    max_length: int = 512
) -> TokenizationResult:
    """Complete tokenization pipeline with Pydantic V2"""
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer with validated config
    config = TokenizerConfig(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_length=max_length,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<|endoftext|>"]
    )
    
    tokenizer = BPETokenizer(config)
    
    # Train tokenizer
    tokenizer.train(corpus_path)
    
    # Save tokenizer
    tokenizer.save_tokenizer(output_dir / "tokenizer")
    
    # Tokenize corpus and save as binary
    tokens_path = output_dir / "tokens.bin"
    total_tokens = tokenizer.tokenize_corpus(corpus_path, tokens_path)
    
    # Create and return result
    result = TokenizationResult(
        success=True,
        output_dir=output_dir,
        vocab_size=tokenizer.tokenizer.get_vocab_size(),
        total_tokens=total_tokens,
        tokenizer_config=config
    )
    
    logger.info(f"Tokenization pipeline complete. Files saved to {output_dir}")
    return result

# Integration with your existing Typer CLI
def add_tokenization_commands(app):
    """Add tokenization commands to Typer app"""
    @app.command()
    def tokenize(
        corpus_path: Path = typer.Argument(..., help="Path to training corpus"),
        output_dir: Path = typer.Option(Path("tokenized"), help="Output directory"),
        vocab_size: int = typer.Option(30000, help="Vocabulary size"),
        min_frequency: int = typer.Option(2, help="Minimum token frequency"),
        max_length: int = typer.Option(512, help="Maximum sequence length"),
    ):
        """Tokenize training corpus using BPE"""
        
        result = run_tokenization_pipeline(
            corpus_path=corpus_path,
            output_dir=output_dir,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            max_length=max_length
        )
        
        # Output result as JSON
        typer.echo(result.model_dump_json(indent=2))
        
        return result
    
    return app

async def main():
    # Download the dataset
    #download_result = await download_pile_uncopyrighted(
    #    file_pattern=r".*\.jsonl\.zst"  # Only download gzipped JSONL files
    #)

    download_result = await download_pile_uncopyrighted_multiproc(
        repo_id="monology/pile-uncopyrighted",
        raw_data_dir=RAWDATA_PATH,
        file_pattern=r".*\.jsonl\.zst",  # Only download compressed JSONL files
        max_retries=10,                   # More retries for large files
        timeout=120,                     # Longer timeout for large files
        chunk_size=32768,                 # Larger chunk size for faster downloads
         max_files=1,
         num_parallel_downloads=PARALLEL_EXECS # Use 8 parallel threads for the download
     )

#    download_result = await download_pile_uncopyrighted(
#    repo_id="monology/pile-uncopyrighted",
#    raw_data_dir=RAWDATA_PATH,
#    file_pattern=r".*\.jsonl\.zst",  # Only download compressed JSONL files
#    max_retries=10,                   # More retries for large files
#    timeout=120,                     # Longer timeout for large files
#    chunk_size=32768,                 # Larger chunk size for faster downloads
#    max_files=1
#    )

    if download_result.success:
        # Extract PubMed abstracts from downloaded files
        for file_path in download_result.downloaded_files:
            pubmed_extraction_stats = None
            clean_tokenize_stats = None
            bpe_tokenize_stats = None
            pubmed_extraction_stats = await run_pubmed_extraction(
                input_path=f"{RAWDATA_PATH}/{file_path}",
                output_path=f"{PRECLEANDATA_PATH}/{PUBMED_EXTRACT_FILE}",
                return_objects=False
            )
            if pubmed_extraction_stats:
                logger.info(f"Extracted {pubmed_extraction_stats}")
                if isinstance(pubmed_extraction_stats, Dict) \
                and int(f'{pubmed_extraction_stats.get("output_size_mb","0")}') > 0:
                    clean_tokenize_stats = run_complete_clean_tokenize_pipeline(
                        input_jsonl_path=f"{PRECLEANDATA_PATH}/{PUBMED_EXTRACT_FILE}",
                        output_clean_dir=CLEANDATA_PATH
                    )
            if clean_tokenize_stats and clean_tokenize_stats.success:
                logger.info(f"Produced a training corpus at: {clean_tokenize_stats.final_file}")
                bpe_tokenize_stats = run_tokenization_pipeline(
                    corpus_path=f"{CLEANDATA_PATH}/{BPE_CORPUS_FILE}",
                    output_dir= TRAINDATA_PATH
                )
                logger.info(f"{bpe_tokenize_stats.model_dump_json(indent=2)}")
            
            github_extraction_stats = await run_github_extraction(
                input_path=f"{RAWDATA_PATH}/{file_path}",
                output_path=f"{PRECLEANDATA_PATH}/{PUBMED_EXTRACT_FILE}",
                return_objects=False
            )
            if github_extraction_stats:
                logger.info(f"Extracted {github_extraction_stats}")
                if isinstance(github_extraction_stats, Dict) \
                and int(f'{github_extraction_stats.get("output_size_mb","0")}') > 0:
                    clean_tokenize_stats = run_complete_clean_tokenize_pipeline(
                        input_jsonl_path=f"{PRECLEANDATA_PATH}/{GITHUB_EXTRACT_FILE}",
                        output_clean_dir=CLEANDATA_PATH
                    )
            if clean_tokenize_stats and clean_tokenize_stats.success:
                logger.info(f"Produced a training corpus at: {clean_tokenize_stats.final_file}")
                bpe_tokenize_stats = run_tokenization_pipeline(
                    corpus_path=f"{CLEANDATA_PATH}/{BPE_CORPUS_FILE}",
                    output_dir= TRAINDATA_PATH
                )
                logger.info(f"{bpe_tokenize_stats.model_dump_json(indent=2)}")
            
            wiki_extraction_stats = await run_wikipedia_extraction(
                input_path=f"{RAWDATA_PATH}/{file_path}",
                output_path=f"{PRECLEANDATA_PATH}/{WIKI_EXTRACT_FILE}",
                return_objects=False
            )
            if wiki_extraction_stats:
                logger.info(f"Extracted {wiki_extraction_stats}")
                if isinstance(wiki_extraction_stats, Dict) \
                and int(f'{wiki_extraction_stats.get("output_size_mb","0")}') > 0:
                    clean_tokenize_stats = run_complete_clean_tokenize_pipeline(
                        input_jsonl_path=f"{PRECLEANDATA_PATH}/{WIKI_EXTRACT_FILE}",
                        output_clean_dir=CLEANDATA_PATH
                    )
            if clean_tokenize_stats and clean_tokenize_stats.success:
                logger.info(f"Produced a training corpus at: {clean_tokenize_stats.final_file}")
                bpe_tokenize_stats = run_tokenization_pipeline(
                    corpus_path=f"{CLEANDATA_PATH}/{BPE_CORPUS_FILE}",
                    output_dir= TRAINDATA_PATH
                )
                logger.info(f"{bpe_tokenize_stats.model_dump_json(indent=2)}")
    else:
        logger.error(f"Download failed: {download_result.message}")
        return None
    
if __name__ == "__main__":
    env(ENV_FILE_PATH)
    BASEDATA_PATH = os.getenv("BASEDATA_PATH",BASEDATA_PATH)
    RAWDATA_PATH = f"{BASEDATA_PATH}/{os.getenv('RAWDATA_PATH',RAWDATA_PATH)}"
    PRECLEANDATA_PATH = f"{BASEDATA_PATH}/{os.getenv('PRECLEANDATA_PATH',PRECLEANDATA_PATH)}"
    CLEANDATA_PATH = f"{BASEDATA_PATH}/{os.getenv('CLEANDATA_PATH',CLEANDATA_PATH)}"
    TRAINDATA_PATH = f"{BASEDATA_PATH}/{os.getenv('TRAINDATA_PATH',TRAINDATA_PATH)}"
    PUBMED_EXTRACT_FILE = os.getenv("PUBMED_EXTRACT_FILE",PUBMED_EXTRACT_FILE)
    GITHUB_EXTRACT_FILE = os.getenv("GITHUB_EXTRACT_FILE",GITHUB_EXTRACT_FILE)
    WIKI_EXTRACT_FILE = os.getenv("WIKI_EXTRACT_FILE",WIKI_EXTRACT_FILE)
    PUBMED_JSONL_SIZE_MB = int(f"{os.getenv('PUBMED_JSONL_SIZE_MB',PUBMED_JSONL_SIZE_MB)}")
    GITHUB_JSONL_SIZE_MB = int(f"{os.getenv('GITHUB_JSONL_SIZE_MB',GITHUB_JSONL_SIZE_MB)}")
    WIKI_JSONL_SIZE_MB = int(f"{os.getenv('WIKI_JSONL_SIZE_MB',WIKI_JSONL_SIZE_MB)}")
    PARALLEL_EXECS = int(f"{os.getenv('NUM_PROCESSES',PARALLEL_EXECS)}")
    BPE_CORPUS_FILE = f"{os.getenv('BPE_CORPUS_FILE',BPE_CORPUS_FILE)}"
    asyncio.run(main())
    