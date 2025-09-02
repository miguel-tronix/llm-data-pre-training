#
import asyncio
from extract_pubmed import run_pubmed_extraction
from fetch_training_data import download_pile_uncopyrighted

DATASET_URL = "https://h"
RAWDATA_PATH = "rawdata/"
PRECLEANDATA_PATH = "precleandata/"
CLEANDATA_PATH = "cleandata/"

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

if __name__ == "__main__":
    download_path = "~/llm-data-pre-trainings/rawdata/"
    jsonl_path = "~/llm-data-pre-trainings/pre-clean-data/"
    asyncio.run(download_pile_uncopyrighted())
    asyncio.run(pubmed_abstrac_gen(download_path,jsonl_path))