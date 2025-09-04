import asyncio
import aiohttp
import aiofiles
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, ClassVar
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from tqdm import tqdm
import time
import re
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
            raise Exception(f"No session is available to proceed with downloads")

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
                                # Filter for data files with appropriate extensions
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
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"Failed to get file list after {self.config.max_retries} attempts")
    
    async def check_resume_support(self, url: str) -> bool:
        """Check if the server supports resume (Range requests)"""
        if self.session is None:
            raise Exception(f"No session is available to proceed with downloads")
        
        logger.debug(f"checking if we can resume download of {url}")
        
        try:
            async with self.session.head(url) as response:
                return response.headers.get('Accept-Ranges') == 'bytes'
        except Exception:
            return False
    
    async def download_file(self, file_info: FileInfo, progress_bar: Optional[tqdm] = None) -> bool:
        """Download a single file from the dataset with resume support"""
        file_url = f"https://huggingface.co/datasets/{self.config.repo_id}/resolve/main/{file_info.path}"
        local_path = self.config.raw_data_dir / file_info.path
        
        if self.session is None:
            raise Exception(f"No session is available to proceed with downloads")
        
        logger.debug(f"Downloading {file_url} to {local_path}")

        # Create directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(self.config.max_retries):
            # Check if we can resume the download
            resume_supported = await self.check_resume_support(file_url)
            start_byte = 0
            
            # If file exists, check if we can resume
            if local_path.exists():
                local_size = local_path.stat().st_size
                
                # If we have the complete file, skip
                if file_info.size and local_size == file_info.size:
                    if progress_bar:
                        progress_bar.update(file_info.size)
                    logger.info(f"Skipping {file_info.path} (already exists and complete)")
                    return True
                
                # If resume is supported and we have a partial file, resume
                if resume_supported and local_size > 0:
                    start_byte = local_size
                    logger.info(f"Resuming download of {file_info.path} from byte {start_byte}")
                else:
                    # Can't resume, so remove the partial file and start over
                    local_path.unlink()
                    logger.info(f"Starting fresh download of {file_info.path}")
            
            try:
                headers = {}
                if start_byte > 0 and resume_supported:
                    headers['Range'] = f'bytes={start_byte}-'
                
                async with self.session.get(file_url, headers=headers) as response:
                    if response.status in (200, 206):  # 200 OK or 206 Partial Content
                        # Get the total size for progress tracking
                        if 'Content-Range' in response.headers:
                            # Parse Content-Range header to get total size
                            content_range = response.headers['Content-Range']
                            total_size = int(content_range.split('/')[-1])
                        else:
                            total_size = int(response.headers.get('content-length', 0))
                        
                        # Open file in append mode if resuming, otherwise write mode
                        mode = 'ab' if start_byte > 0 else 'wb'
                        async with aiofiles.open(local_path, mode) as f:
                            downloaded = start_byte
                            async for chunk in response.content.iter_chunked(self.config.chunk_size):
                                await f.write(chunk)
                                downloaded += len(chunk)
                                if progress_bar and total_size > 0:
                                    progress_bar.update(len(chunk))
                        
                        # Verify download size if we have the expected size
                        if file_info.size and downloaded != file_info.size:
                            logger.warning(f"Size mismatch for {file_info.path}: expected {file_info.size}, got {downloaded}")
                            # Check if we should retry or continue
                            if attempt < self.config.max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                        
                        logger.info(f"Downloaded {file_info.path} ({downloaded} bytes)")
                        return True
                    else:
                        logger.warning(f"Failed to download {file_info.path}, status {response.status}, attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Error downloading {file_info.path} (attempt {attempt + 1}): {e}")
            
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(1 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to download {file_info.path} after {self.config.max_retries} attempts")
        return False
    
    async def download_dataset(self) -> DownloadResult:
        """
        Download all files from the dataset matching the pattern
        
        Returns:
            DownloadResult with download statistics
        """
        # Ensure rawdata directory exists
        self.config.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of files
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
        
        # Filter files if pattern provided
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
        
        # Apply max_files limit if specified
        if self.config.max_files is not None:
            files = files[:self.config.max_files]
            logger.info(f"Limiting download to {self.config.max_files} files")
        
        logger.info(f"Found {len(files)} files to download")
        
        # Calculate total size for progress bar
        total_size = sum(f.size or 0 for f in files)
        
        # Download files one by one
        success_count = 0
        failed_count = 0
        downloaded_files = []
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for file_info in files:
                success = await self.download_file(file_info, pbar)
                if success:
                    success_count += 1
                    downloaded_files.append(file_info.path)
                else:
                    failed_count += 1
        
        return DownloadResult(
            success=failed_count == 0,
            total_files=len(files),
            success_count=success_count,
            failed_count=failed_count,
            download_dir=str(self.config.raw_data_dir),
            downloaded_files=downloaded_files,
            message="Successfully acquired the dataset"
        )
