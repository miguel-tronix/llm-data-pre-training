#
import asyncio
import logging
from typing import Dict, Any
from fetch_training_data import download_pile_uncopyrighted
from extract_pubmed import run_pubmed_extraction

DATASET_URL = "https://h"
RAWDATA_PATH = "rawdata/"
PRECLEANDATA_PATH = "precleandata/"
CLEANDATA_PATH = "cleandata/"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    download_result : Dict[str,Any] = await download_pile_uncopyrighted(
        file_pattern=r".*\.jsonl\.gz"  # Only download gzipped JSONL files
    )
    try:
        if isinstance(download_result, Dict) and download_result['success']:
            # Extract PubMed abstracts from downloaded files
            extraction_result = await run_pubmed_extraction(
                input_path=str(download_result['download_dir']),
                output_path="pubmed_abstracts.jsonl"
            )
            return extraction_result
        else:
            logger.error(f"Download failed: {download_result['message']}")
            return None
    except KeyError as e:
        logger.error(f"The key {e} was not found in dictionary {download_result}")
        return None

if __name__ == "__main__":
    download_path = RAWDATA_PATH
    jsonl_path = PRECLEANDATA_PATH
    asyncio.run(main())
    