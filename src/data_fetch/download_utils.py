import asyncio
import aiohttp
import aiofiles
import aiofiles.os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, ClassVar, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from tqdm import tqdm
import time
import re
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic V2 Models ---
class FileInfo(BaseModel):
    """Model for file information from Hugging Face API"""
    path: str = Field(..., description="File path in the repository")
    type: str = Field(..., description="File type (file or directory)")
    size: Optional[int] = Field(None, description="File size in bytes")
    oid: Optional[str] = Field(None, description="Git object ID")
    
    model_config = ConfigDict(extra='ignore')  # Ignore extra fields from API

class DownloadConfig(BaseModel):
    """Configuration for dataset downloading"""
    repo_id: str = Field(default="monology/pile-uncopyrighted", description="Hugging Face dataset repository ID")
    raw_data_dir: Path = Field(default=Path("rawdata"), description="Directory to store downloaded files")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum number of retry attempts per file")
    timeout: int = Field(default=30, ge=5, le=300, description="Request timeout in seconds")
    file_pattern: Optional[str] = Field(default=None, description="Regex pattern to filter files")
    chunk_size: int = Field(default=8192, ge=1024, le=65536, description="Chunk size for downloading")
    max_files: Optional[int] = Field(default=None, ge=1, description="Maximum number of files to download")
    num_parallel_downloads: int = Field(default=4, ge=1, description="Number of parallel downloads per file.")
    
    @field_validator('raw_data_dir')
    @classmethod
    def validate_raw_data_dir(cls, v):
        """Ensure raw data directory is a Path object"""
        if isinstance(v, str):
            return Path(v)
        return v

class DownloadResult(BaseModel):
    """Result of a download operation"""
    success: bool = Field(..., description="Whether the download was successful")
    total_files: int = Field(0, ge=0, description="Total number of files attempted")
    success_count: int = Field(0, ge=0, description="Number of successfully downloaded files")
    failed_count: int = Field(0, ge=0, description="Number of failed downloads")
    download_dir: str = Field(..., description="Directory where files were downloaded")
    downloaded_files: List[str] = Field(default_factory=list, description="List of successfully downloaded file paths")
    message: Optional[str] = Field(None, description="Additional message or error details")
    
    @model_validator(mode='after')
    def validate_counts(self):
        """Ensure counts are consistent"""
        if self.total_files != self.success_count + self.failed_count:
            raise ValueError("Total files must equal success_count + failed_count")
        
        # Also validate that the downloaded_files list matches success_count
        if len(self.downloaded_files) != self.success_count:
            raise ValueError("Number of downloaded files must equal success_count")
            
        return self

# --- Hugging Face Dataset Downloader ---
class HFDatasetDownloader:
    """Download files from Hugging Face datasets using Pydantic V2 models"""
    
    MB_100 = 100 * 1024 * 1024
    MB_500 = 500 * 1024 * 1024
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.file_extensions = [".jsonl", ".jsonl.zst", ".jsonl.gz", ".txt", ".zst"]
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_dataset_files(self) -> List[FileInfo]:
        """Get list of files in the dataset from Hugging Face API"""
        api_url = f"https://huggingface.co/api/datasets/{self.config.repo_id}/tree/main/train"
        
        if self.session is None:
            raise Exception("No session is available to proceed with downloads")

        logger.debug(f"Getting metadata from {api_url} for datasets")

        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(api_url) as response:
                    if response.status == 200:
                        files_data = await response.json()
                        files = []
                        
                        for file_data in files_data:
                            try:
                                file_info = FileInfo(**file_data)
                                if (file_info.type == "file" and 
                                    any(file_info.path.endswith(ext) for ext in self.file_extensions)):
                                    files.append(file_info)
                            except Exception as e:
                                logger.debug(f"Skipping invalid file data: {e}")
                                continue
                        
                        return files
                    else:
                        logger.warning(f"API returned status {response.status}, attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Error fetching file list (attempt {attempt + 1}): {e}")
            
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        
        raise Exception(f"Failed to get file list after {self.config.max_retries} attempts")
    
    async def check_resume_support(self, url: str) -> bool:
        """Check if the server supports resume (Range requests)"""
        if self.session is None:
            raise Exception("No session is available to proceed with downloads")
        
        logger.debug(f"checking if we can resume download of {url}")
        
        try:
            async with self.session.head(url) as response:
                return response.headers.get('Accept-Ranges') == 'bytes'
        except Exception:
            return False

    async def _download_chunk(
        self, url: str,
        part_path: Path, 
        start_byte: int,
        end_byte: int,
        pbar: Optional[tqdm] = None
    ) -> bool:
        """Downloads a specific byte range of a file."""
        if self.session is None:
            raise Exception("No session is available to proceed with downloads")

        for attempt in range(self.config.max_retries):
            start_offset = 0
            if part_path.exists():
                start_offset = part_path.stat().st_size
            
            # If part is already complete, skip
            if start_offset >= (end_byte - start_byte + 1):
                if pbar:
                    pbar.update(end_byte - start_byte + 1 - start_offset)
                return True

            headers = {'Range': f'bytes={start_byte + start_offset}-{end_byte}'}
            
            try:
                async with self.session.get(url, headers=headers) as response:
                    if response.status == 206:  # Partial Content
                        mode = 'ab' if start_offset > 0 else 'wb'
                        async with aiofiles.open(part_path, mode) as f:
                            async for chunk in response.content.iter_chunked(self.config.chunk_size):
                                await f.write(chunk)
                                if pbar:
                                    pbar.update(len(chunk))
                        return True
                    else:
                        logger.warning(f"Failed chunk {part_path.name}, status {response.status}, attempt {attempt + 1}")

            except Exception as e:
                logger.warning(f"Error downloading chunk {part_path.name} (attempt {attempt + 1}): {e}")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        
        return False

    async def _merge_parts(self, final_path: Path, num_parts: int):
        """Merges downloaded parts into a single file."""
        logger.info(f"Merging {num_parts} parts for {final_path.name}")
        try:
            async with aiofiles.open(final_path, 'wb') as final_file:
                for i in range(num_parts):
                    part_path = final_path.with_name(f"{final_path.name}.part{i}")
                    async with aiofiles.open(part_path, 'rb') as part_file:
                        while chunk := await part_file.read(self.config.chunk_size):
                            await final_file.write(chunk)
            
            # Cleanup part files
            for i in range(num_parts):
                part_path = final_path.with_name(f"{final_path.name}.part{i}")
                await aiofiles.os.remove(part_path)
            logger.info(f"Successfully merged and cleaned up parts for {final_path.name}")
        except Exception as e:
            logger.error(f"Failed to merge parts for {final_path.name}: {e}")
            raise

    async def download_file_parallel(self, file_info: FileInfo, pbar: Optional[tqdm] = None ) -> bool:
        """Downloads a file in parallel chunks."""
        file_url = f"https://huggingface.co/datasets/{self.config.repo_id}/resolve/main/{file_info.path}"
        local_path = self.config.raw_data_dir / file_info.path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        num_parts = self.config.num_parallel_downloads
        file_size = file_info.size
        if file_size is None:
            logger.warning(f"File size unknown for {file_info.path}, cannot perform parallel download.")
            return False
        # Calculate byte ranges for each part
        ranges: List[Tuple[int, int]] = []
        even_split_size = file_size // num_parts
        thread_over_size = even_split_size % self.MB_100
        part_size = even_split_size - thread_over_size
        
        current_pos = 0
        for i in range(num_parts - 1):
            start = current_pos
            end = start + part_size - 1
            ranges.append((start, end))
            current_pos = end + 1
        ranges.append((current_pos, file_size - 1)) # Last part takes the remainder

        # Create download tasks
        tasks = []
        for i, (start, end) in enumerate(ranges):
            part_path = local_path.with_name(f"{local_path.name}.part{i}")
            tasks.append(self._download_chunk(file_url, part_path, start, end, pbar))

        results = await asyncio.gather(*tasks)

        if all(results):
            await self._merge_parts(local_path, num_parts)
            return True
        else:
            logger.error(f"Parallel download failed for {file_info.path}, some chunks failed.")
            # Cleanup failed parts
            for i in range(num_parts):
                part_path = local_path.with_name(f"{local_path.name}.part{i}")
                if part_path.exists():
                    os.remove(part_path)
            return False

    async def download_file(self, file_info: FileInfo, progress_bar: Optional[tqdm] = None) -> bool:
        """
        Dispatcher for downloading a single file.
        Uses parallel download for files > 500MB, otherwise single stream.
        """
        local_path = self.config.raw_data_dir / file_info.path
        
        # If file exists and is complete, skip
        if file_info.size and local_path.exists() and local_path.stat().st_size == file_info.size:
            if progress_bar:
                progress_bar.update(file_info.size)
            logger.info(f"Skipping {file_info.path} (already exists and complete)")
            return True

        # Decide download strategy
        if file_info.size and file_info.size >= self.MB_500 and self.config.num_parallel_downloads > 1:
            logger.info(f"Using parallel download for {file_info.path} ({self.config.num_parallel_downloads} parts)")
            return await self.download_file_parallel(file_info, progress_bar)
        else:
            logger.info(f"Using single stream download for {file_info.path}")
            return await self._download_file_single_stream(file_info, progress_bar)

    async def _download_file_single_stream(self, file_info: FileInfo, progress_bar: Optional[tqdm] = None) -> bool:
        """Download a single file from the dataset with resume support"""
        file_url = f"https://huggingface.co/datasets/{self.config.repo_id}/resolve/main/{file_info.path}"
        local_path = self.config.raw_data_dir / file_info.path
        
        if self.session is None:
            raise Exception("No session is available to proceed with downloads")
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(self.config.max_retries):
            resume_supported = await self.check_resume_support(file_url)
            start_byte = 0
            
            if local_path.exists():
                local_size = local_path.stat().st_size
                if resume_supported and local_size > 0:
                    start_byte = local_size
                else:
                    local_path.unlink()
            
            try:
                headers = {}
                if start_byte > 0 and resume_supported:
                    headers['Range'] = f'bytes={start_byte}-'
                
                async with self.session.get(file_url, headers=headers) as response:
                    if response.status in (200, 206):
                        mode = 'ab' if start_byte > 0 else 'wb'
                        async with aiofiles.open(local_path, mode) as f:
                            async for chunk in response.content.iter_chunked(self.config.chunk_size):
                                await f.write(chunk)
                                if progress_bar:
                                    progress_bar.update(len(chunk))
                        
                        logger.info(f"Downloaded {file_info.path}")
                        return True
                    else:
                        logger.warning(f"Failed to download {file_info.path}, status {response.status}, attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Error downloading {file_info.path} (attempt {attempt + 1}): {e}")
            
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        
        logger.error(f"Failed to download {file_info.path} after {self.config.max_retries} attempts")
        return False
    
    async def download_dataset(self) -> DownloadResult:
        """
        Download all files from the dataset matching the pattern
        
        Returns:
            DownloadResult with download statistics
        """
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
            logger.info(f"Limiting download to {self.config.max_files} files")
        
        logger.info(f"Found {len(files)} files to download")
        
        total_size = sum(f.size or 0 for f in files)
        success_count = 0
        downloaded_files = []
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for file_info in files:
                success = await self.download_file(file_info, pbar)
                if success:
                    success_count += 1
                    downloaded_files.append(file_info.path)
        
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