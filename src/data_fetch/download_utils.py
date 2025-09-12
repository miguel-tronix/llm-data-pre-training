import asyncio
import aiohttp
import aiofiles
import aiofiles.os
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from tqdm import tqdm
import time
import re
import os
import requests
from concurrent.futures import ProcessPoolExecutor
import functools
from utils.pipeline_logger import get_pipeline_logger

logger = get_pipeline_logger()

# --- Top-level worker function for multiprocessing ---
# NOTE: The signature has been reordered for compatibility with functools.partial
def _download_chunk_process(
    url: str,
    chunk_size: int,
    max_retries: int,
    timeout: int,
    part_path: Path,
    start_byte: int,
    end_byte: int
) -> bool:
    """
    Worker function to download a file chunk in a separate process.
    Uses synchronous `requests` for simplicity within the process.
    """
    for attempt in range(max_retries):
        try:
            start_offset = 0
            if part_path.exists():
                start_offset = part_path.stat().st_size

            expected_size = (end_byte - start_byte + 1)
            if start_offset >= expected_size:
                return True  # Chunk is already complete

            headers = {'Range': f'bytes={start_byte + start_offset}-{end_byte}'}
            
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                
                mode = 'ab' if start_offset > 0 else 'wb'
                with open(part_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                
                if part_path.stat().st_size == expected_size:
                    return True
                else:
                    logger.warning(f"Chunk {part_path.name} size mismatch. Retrying...")
                    continue

        except requests.exceptions.RequestException as e:
            logger.warning(f"Error downloading chunk {part_path.name} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"An unexpected error occurred in worker for {part_path.name}: {e}")
            break

    logger.error(f"Failed to download chunk {part_path.name} after {max_retries} attempts.")
    return False

# --- Pydantic V2 Models ---
class FileInfo(BaseModel):
    path: str = Field(..., description="File path in the repository")
    type: str = Field(..., description="File type (file or directory)")
    size: Optional[int] = Field(None, description="File size in bytes")
    oid: Optional[str] = Field(None, description="Git object ID")
    model_config = ConfigDict(extra='ignore')

class DownloadConfig(BaseModel):
    repo_id: str = Field(default="monology/pile-uncopyrighted", description="Hugging Face dataset repository ID")
    raw_data_dir: Path = Field(default=Path("rawdata"), description="Directory to store downloaded files")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum number of retry attempts per file")
    timeout: int = Field(default=30, ge=5, le=300, description="Request timeout in seconds")
    file_pattern: Optional[str] = Field(default=None, description="Regex pattern to filter files")
    chunk_size: int = Field(default=8192, ge=1024, le=65536, description="Chunk size for downloading")
    max_files: Optional[int] = Field(default=None, ge=1, description="Maximum number of files to download")
    num_parallel_downloads: int = Field(default=4, ge=1, description="Number of parallel processes per large file.")
    
    @field_validator('raw_data_dir', mode='before')
    @classmethod
    def validate_raw_data_dir(cls, v):
        return Path(v) if isinstance(v, str) else v

class DownloadResult(BaseModel):
    success: bool = Field(..., description="Whether the download was successful")
    total_files: int = Field(0, ge=0, description="Total number of files attempted")
    success_count: int = Field(0, ge=0, description="Number of successfully downloaded files")
    failed_count: int = Field(0, ge=0, description="Number of failed downloads")
    download_dir: str = Field(..., description="Directory where files were downloaded")
    downloaded_files: List[str] = Field(default_factory=list, description="List of successfully downloaded file paths")
    message: Optional[str] = Field(None, description="Additional message or error details")
    
    @model_validator(mode='after')
    def validate_counts(self):
        if self.total_files != self.success_count + self.failed_count:
            raise ValueError("Total files must equal success_count + failed_count")
        if len(self.downloaded_files) != self.success_count:
            raise ValueError("Number of downloaded files must equal success_count")
        return self

# --- Hugging Face Dataset Downloader ---
class HFDatasetDownloader:
    MB_100 = 100 * 1024 * 1024
    MB_500 = 500 * 1024 * 1024
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.file_extensions = [".jsonl", ".jsonl.zst", ".jsonl.gz", ".txt", ".zst"]
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session: await self.session.close()
    
    async def get_dataset_files(self) -> List[FileInfo]:
        api_url = f"https://huggingface.co/api/datasets/{self.config.repo_id}/tree/main/train"
        if self.session is None: raise Exception("No session available")
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(api_url) as response:
                    response.raise_for_status()
                    files_data = await response.json()
                    return [FileInfo(**fd) for fd in files_data if fd.get('type') == 'file' and any(fd.get('path', '').endswith(ext) for ext in self.file_extensions)]
            except Exception as e:
                logger.warning(f"Error fetching file list (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1: await asyncio.sleep(2 ** attempt)
        raise Exception(f"Failed to get file list after {self.config.max_retries} attempts")
    
    async def check_resume_support(self, url: str) -> bool:
        if self.session is None: raise Exception("No session available")
        try:
            async with self.session.head(url) as response:
                return response.headers.get('Accept-Ranges') == 'bytes'
        except Exception: return False

    async def _merge_parts(self, final_path: Path, num_parts: int):
        logger.info(f"Merging {num_parts} parts for {final_path.name}")
        try:
            async with aiofiles.open(final_path, 'wb') as final_file:
                for i in range(num_parts):
                    part_path = final_path.with_name(f"{final_path.name}.part{i}")
                    async with aiofiles.open(part_path, 'rb') as part_file:
                        while chunk := await part_file.read(self.config.chunk_size):
                            await final_file.write(chunk)
            for i in range(num_parts):
                part_path = final_path.with_name(f"{final_path.name}.part{i}")
                await aiofiles.os.remove(part_path)
            logger.info(f"Successfully merged and cleaned up parts for {final_path.name}")
        except Exception as e:
            logger.error(f"Failed to merge parts for {final_path.name}: {e}")
            raise

    async def download_file_parallel(self, file_info: FileInfo) -> bool:
        file_url = f"https://huggingface.co/datasets/{self.config.repo_id}/resolve/main/{file_info.path}"
        local_path = self.config.raw_data_dir / file_info.path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        num_parts = self.config.num_parallel_downloads
        file_size = file_info.size if file_info.size else 0
        if file_size < self.MB_100:
            logger.warning(f"File {file_info.path} is smaller than 100MB, skipping parallel download.")
            return await self._download_file_single_stream(file_info)
        
        ranges: List[Tuple[int, int]] = []
        even_split_size = file_size // num_parts
        thread_over_size = even_split_size % self.MB_100
        part_size = even_split_size - thread_over_size
        
        current_pos = 0
        for i in range(num_parts - 1):
            ranges.append((current_pos, current_pos + part_size - 1))
            current_pos += part_size
        ranges.append((current_pos, file_size - 1))

        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=num_parts) as executor:
            # NOTE: Corrected functools.partial call with positional arguments
            worker_func = functools.partial(
                _download_chunk_process,
                file_url,
                self.config.chunk_size,
                self.config.max_retries,
                self.config.timeout
            )
            
            # NOTE: Corrected run_in_executor call with positional arguments
            tasks = [
                loop.run_in_executor(
                    executor,
                    worker_func,
                    local_path.with_name(f"{local_path.name}.part{i}"), # part_path
                    start,                                             # start_byte
                    end                                                # end_byte
                ) for i, (start, end) in enumerate(ranges)
            ]
            
            results = await asyncio.gather(*tasks)

        if all(results):
            await self._merge_parts(local_path, num_parts)
            return True
        else:
            logger.error(f"Parallel download failed for {file_info.path}, cleaning up parts.")
            for i in range(num_parts):
                part_path = local_path.with_name(f"{local_path.name}.part{i}")
                if part_path.exists(): os.remove(part_path)
            return False

    async def _download_file_single_stream(self, file_info: FileInfo) -> bool:
        file_url = f"https://huggingface.co/datasets/{self.config.repo_id}/resolve/main/{file_info.path}"
        local_path = self.config.raw_data_dir / file_info.path
        if self.session is None: raise Exception("No session available")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(self.config.max_retries):
            resume_supported = await self.check_resume_support(file_url)
            start_byte = 0
            if local_path.exists():
                local_size = local_path.stat().st_size
                if resume_supported and local_size > 0: start_byte = local_size
                else: local_path.unlink()
            
            try:
                headers = {'Range': f'bytes={start_byte}-'} if start_byte > 0 and resume_supported else {}
                async with self.session.get(file_url, headers=headers) as response:
                    if response.status in (200, 206):
                        mode = 'ab' if start_byte > 0 else 'wb'
                        async with aiofiles.open(local_path, mode) as f:
                            async for chunk in response.content.iter_chunked(self.config.chunk_size):
                                await f.write(chunk)
                        logger.info(f"Downloaded {file_info.path}")
                        return True
                    else:
                        logger.warning(f"Failed download {file_info.path}, status {response.status}, attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Error downloading {file_info.path} (attempt {attempt + 1}): {e}")
            
            if attempt < self.config.max_retries - 1: await asyncio.sleep(2 ** attempt)
        
        logger.error(f"Failed to download {file_info.path} after {self.config.max_retries} attempts")
        return False

    async def download_file(self, file_info: FileInfo) -> bool:
        local_path = self.config.raw_data_dir / file_info.path
        if file_info.size and local_path.exists() and local_path.stat().st_size == file_info.size:
            logger.info(f"Skipping {file_info.path} (already exists and complete)")
            return True

        if file_info.size and file_info.size >= self.MB_500 and self.config.num_parallel_downloads > 1:
            logger.info(f"Using parallel download for {file_info.path} ({self.config.num_parallel_downloads} processes)")
            return await self.download_file_parallel(file_info)
        else:
            logger.info(f"Using single stream download for {file_info.path}")
            return await self._download_file_single_stream(file_info)

    async def download_dataset(self) -> DownloadResult:
        self.config.raw_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Fetching file list for {self.config.repo_id}")
        try:
            files = await self.get_dataset_files()
        except Exception as e:
            return DownloadResult(
                success=False,
                total_files=0,
                success_count=0,
                failed_count=0,
                download_dir=str(self.config.raw_data_dir),
                downloaded_files=[],
                message=f"Failed to get file list: {e}"
            )
        
        if self.config.file_pattern:
            try:
                pattern = re.compile(self.config.file_pattern)
                files = [f for f in files if pattern.search(f.path)]
            except re.error as e:
                return DownloadResult(
                    success=False,
                    total_files=0,
                    success_count=0,
                    failed_count=0,
                    download_dir=str(self.config.raw_data_dir),
                    downloaded_files=[],
                    message=f"Invalid file pattern: {e}"
                )
        
        if not files:
            return DownloadResult(
                success=False,
                total_files=0,
                success_count=0,
                failed_count=0,
                download_dir=str(self.config.raw_data_dir),
                downloaded_files=[],
                message="No files found matching the criteria"
            )
        
        if self.config.max_files is not None:
            files = files[:self.config.max_files]
        
        logger.info(f"Found {len(files)} files to download")
        total_size = sum(f.size or 0 for f in files)
        success_count, downloaded_files = 0, []
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for file_info in files:
                success = await self.download_file(file_info)
                if success:
                    success_count += 1
                    downloaded_files.append(file_info.path)
                    pbar.update(file_info.size or 0)
        
        failed_count = len(files) - success_count
        return DownloadResult(
            success=failed_count == 0,
            total_files=len(files),
            success_count=success_count,
            failed_count=failed_count,
            download_dir=str(self.config.raw_data_dir),
            downloaded_files=downloaded_files,
            message="Download process completed." if failed_count == 0 else f"{failed_count} files failed to download."
        )

