import io
import json
import multiprocessing as mp
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import zstandard as zstd

from llm_data_pretraining.utils.pipeline_logger import get_pipeline_logger

# Configure logging
logger = get_pipeline_logger()


def _parse_lines(lines: list[str]) -> list[dict[str, Any]]:
    """Helper function to parse a batch of JSON lines."""
    results = []
    for line in lines:
        if line.strip():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


class ParallelZstdJsonlReader:
    """Parallel reader for Zstandard-compressed JSONL files"""

    def __init__(
        self, file_path: Path, num_processes: int = 1, chunk_size: int = 10000
    ):
        self.file_path = file_path
        self.num_processes = num_processes or mp.cpu_count()
        self.chunk_size = chunk_size  # Number of lines per batch

    def read_parallel(self) -> Iterator[dict[str, Any]]:
        """
        Read and parse the file. Uses multiple processes for JSON parsing if requested.

        Yields:
            Parsed JSON objects as dictionaries
        """
        dctx = zstd.ZstdDecompressor()
        with open(self.file_path, "rb") as f:
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")

                if self.num_processes <= 1:
                    # Sequential processing
                    for line in text_stream:
                        if line.strip():
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                pass
                else:
                    # Parallel processing
                    with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                        def batch_generator() -> Iterator[list[str]]:
                            batch = []
                            for line in text_stream:
                                batch.append(line)
                                if len(batch) >= self.chunk_size:
                                    yield batch
                                    batch = []
                            if batch:
                                yield batch

                        # Process batches in parallel
                        for results in executor.map(_parse_lines, batch_generator()):
                            yield from results

    def read_parallel_batches(
        self, batch_size: int = 1000
    ) -> Iterator[list[dict[str, Any]]]:
        """
        Read data and yield batches

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
def process_large_zstd_file_parallel(
    file_path: Path, num_processes: int = 1
) -> Iterator[dict[str, Any]]:
    """
    Example function to process a large Zstandard-compressed JSONL file
    """
    reader = ParallelZstdJsonlReader(file_path, num_processes)

    # Process file
    total_records = 0
    for record in reader.read_parallel():
        yield record
        total_records += 1

        # Log progress every 100000 records
        if total_records % 100000 == 0:
            logger.info(f"Processed {total_records} records")

    logger.info(f"Finished processing. Total records: {total_records}")


def process_in_parallel_batches(
    file_path: Path, num_processes: int = 1, batch_size: int = 1000
) -> Iterator[dict[str, Any]]:
    """
    Process the file and process results in batches
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
