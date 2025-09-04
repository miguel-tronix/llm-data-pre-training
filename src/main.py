#
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from data_fetch.fetch_raw_data import DownloadResult, DownloadConfig, HFDatasetDownloader
from data_prep.pubmed_extractor_fast import PubMedAbstractExtractor
BASEDATA_PATH = "/home/migtronix/llm-data-pre-training"
DATASET_URL = "https://h"
RAWDATA_PATH = f"{BASEDATA_PATH}/rawdata"
PRECLEANDATA_PATH = f"{BASEDATA_PATH}/precleandata"
CLEANDATA_PATH = f"{BASEDATA_PATH}/cleandata"
PUBMED_EXTRACT_FILE = "pubmed_abstracts.jsonl"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    extractor = PubMedAbstractExtractor()
    
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
        use_parallel_zstd=False
    )
    
    return await extractor.extract_abstracts_to_file(
        input_path=f"{input_zst_path.absolute}",
        output_path=f"{output_jsonl_path.absolute}"
    )



async def main():
    # Download the dataset
    #download_result = await download_pile_uncopyrighted(
    #    file_pattern=r".*\.jsonl\.zst"  # Only download gzipped JSONL files
    #)

    download_result = await download_pile_uncopyrighted(
    repo_id="monology/pile-uncopyrighted",
    raw_data_dir=RAWDATA_PATH,
    file_pattern=r".*\.jsonl\.zst",  # Only download compressed JSONL files
    max_retries=10,                   # More retries for large files
    timeout=120,                     # Longer timeout for large files
    chunk_size=32768,                 # Larger chunk size for faster downloads
    max_files=1
    )
    if download_result.success:
        # Extract PubMed abstracts from downloaded files
        for file_path in download_result.downloaded_files:
            extraction_stats = await run_pubmed_extraction(
                input_path=f"{RAWDATA_PATH}/{file_path}",
                output_path=f"{PRECLEANDATA_PATH}/{PUBMED_EXTRACT_FILE}",
                return_objects=False
            )
            logger.info(f"Extracted {extraction_stats}")
            
    else:
        logger.error(f"Download failed: {download_result.message}")
        return None

if __name__ == "__main__":
    asyncio.run(main())
    