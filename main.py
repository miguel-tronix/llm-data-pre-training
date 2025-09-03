#
import asyncio
import logging
from typing import Dict, Any
from fetch_training_data import download_pile_uncopyrighted
from extract_pubmed import run_pubmed_extraction

DATASET_URL = "https://h"
RAWDATA_PATH = "rawdata"
PRECLEANDATA_PATH = "precleandata"
CLEANDATA_PATH = "cleandata"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def pubmed_abstrac_gen(
        pubmed_input_path : str ,
        pubmed_jsonl : str
) -> None:
    """
        Extract PubMed abstracts from the path provided and create a JSONL file
        This routine can be run as a seperate process or thread
    """
    
    stats = await run_pubmed_extraction(pubmed_input_path, pubmed_jsonl)
    print(f"Extracted {stats['valid']} abstracts to {pubmed_jsonl}")
    
    # Option 2: Extract to memory
    #abstracts = await run_pubmed_extraction(input_path, return_objects=True)
    #print(f"Extracted {len(abstracts)} abstracts to memory")
    
    # You can then process the abstracts as needed
    #for abstract in abstracts[:5]:
    #    print(f"ID: {abstract.id}, Text length: {len(abstract.abstract_text)}")
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
            extraction_result = await run_pubmed_extraction(
                input_path=f"{RAWDATA_PATH}/{file_path}",
                output_path=f"{CLEANDATA_PATH}/pubmed_abstracts.jsonl"
            )
            logger.info(f"Extracted {extraction_result}")
    else:
        logger.error(f"Download failed: {download_result.message}")
        return None

if __name__ == "__main__":
    asyncio.run(main())
    