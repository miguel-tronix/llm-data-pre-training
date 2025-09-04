import zstandard as zstd
import json
import io
import mmap
from pathlib import Path
from typing import Iterator, Dict, Any, List, Tuple
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import itertools
from functools import partial

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParallelZstdJsonlReader:
    """Parallel reader for Zstandard-compressed JSONL files using memory mapping"""
    
    def __init__(self, file_path: Path, num_processes: int = 1, chunk_size: int = 1024 * 1024):
        self.file_path = file_path
        self.num_processes = num_processes or mp.cpu_count()
        self.chunk_size = chunk_size  # Size of chunks to process in bytes
    
    def get_file_size(self) -> int:
        """Get the size of the compressed file"""
        return self.file_path.stat().st_size
    
    def find_chunk_boundaries(self) -> List[Tuple[int, int]]:
        """
        Find appropriate boundaries for chunks to avoid splitting JSON objects
        
        Returns:
            List of (start_offset, end_offset) tuples for each chunk
        """
        file_size = self.get_file_size()
        chunks = []
        
        # Use memory mapping to find safe split points
        with open(self.file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Create chunks with approximate size
                for start in range(0, file_size, self.chunk_size):
                    end = min(start + self.chunk_size, file_size)
                    
                    # If not at the end, find the next newline to avoid splitting JSON objects
                    if end < file_size:
                        # Look for a newline character near the end of the chunk
                        newline_pos = mm.find(b'\n', end - min(1000, self.chunk_size // 10), end + 100)
                        if newline_pos != -1:
                            end = newline_pos + 1  # Include the newline
                    
                    chunks.append((start, end))
        
        return chunks
    
    def process_chunk(self, chunk_range: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Process a chunk of the compressed file
        
        Args:
            chunk_range: Tuple of (start_offset, end_offset) for the chunk
            
        Returns:
            List of parsed JSON objects from this chunk
        """
        start, end = chunk_range
        results = []
        dctx = zstd.ZstdDecompressor()
        
        with open(self.file_path, 'rb') as f:
            # Memory map the file for efficient reading
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Extract the chunk
                compressed_chunk = mm[start:end]
                #logger.debug(f"size of bytes read from zst {len(compressed_chunk)}")
                # Decompress the chunk
                try:
                    with dctx.stream_reader(source=compressed_chunk) as decompressed:
                        text = io.TextIOWrapper(decompressed, encoding='utf-8')
                        # Parse JSON lines
                        for line in text:
                            if line.strip():  # Skip empty lines
                                #logger.debug(f"decompressed {line} from the stream chunk")
                                try:
                                    data = json.loads(line)
                                    results.append(data)
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse JSON line in chunk: {line[:100]}...")
                                    continue
                except Exception as e:
                    logger.error(f"Error processing chunk {start}-{end}: {e}")
        
        return results
    
    def read_parallel(self) -> Iterator[Dict[str, Any]]:
        """
        Read and parse the file using multiple processes
        
        Yields:
            Parsed JSON objects as dictionaries
        """
        # Find chunk boundaries
        chunks = self.find_chunk_boundaries()
        logger.info(f"Processing {len(chunks)} chunks with {self.num_processes} processes")
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Process each chunk and collect results
            for result in executor.map(self.process_chunk, chunks):
                for item in result:
                    yield item
    
    def read_parallel_batches(self, batch_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """
        Read data in parallel and yield batches
        
        Args:
            batch_size: Number of JSON objects to yield at once
            
        Yields:
            Lists of parsed JSON objects
        """
        batch = []
        
        for item in self.read_parallel():
            batch.append(item)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield any remaining items
        if batch:
            yield batch

# --- Example usage ---
def process_large_zstd_file_parallel(file_path: Path, num_processes: int = 1):
    """
    Example function to process a large Zstandard-compressed JSONL file in parallel
    """
    reader = ParallelZstdJsonlReader(file_path, num_processes)
    
    # Process file in parallel
    total_records = 0
    for record in reader.read_parallel():
        yield record
        total_records += 1
        
        # Log progress every 10000 records
        if total_records % 10000 == 0:
            logger.info(f"Processed {total_records} records")
    
    logger.info(f"Finished processing. Total records: {total_records}")

def process_in_parallel_batches(file_path: Path, num_processes: int = 1, batch_size: int = 1000):
    """
    Process the file in parallel and process results in batches
    """
    reader = ParallelZstdJsonlReader(file_path, num_processes)
    total_records = 0
    
    for batch in reader.read_parallel_batches(batch_size):
        # Process the batch of records
        for record in batch:
            yield record
            total_records += 1
        
        # Log progress
        logger.info(f"Processed batch of {len(batch)} records. Total: {total_records}")
    
    logger.info(f"Finished processing. Total records: {total_records}")
