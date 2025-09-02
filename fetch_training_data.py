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
    xetHash: Optional[str] = Field(None, description="Xet Hash")
    lfs: Optional[dict] = Field(None, description="LFS Object")
    
    model_config = ConfigDict(extra='ignore')  # Ignore extra fields from API

class DownloadConfig(BaseModel):
    """Configuration for dataset downloading"""
    repo_id: str = Field(default="monology/pile-uncopyrighted", description="Hugging Face dataset repository ID")
    raw_data_dir: Path = Field(default=Path("rawdata"), description="Directory to store downloaded files")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum number of retry attempts per file")
    timeout: int = Field(default=30, ge=5, le=300, description="Request timeout in seconds")
    file_pattern: Optional[str] = Field(default=None, description="Regex pattern to filter files")
    
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
    message: Optional[str] = Field(None, description="Additional message or error details")
    
    @model_validator(mode='after')
    def validate_counts(self):
        """Ensure counts are consistent"""
        if self.total_files != self.success_count + self.failed_count:
            raise ValueError("Total files must equal success_count + failed_count")
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
            raise Exception(f"Cannot fetch file information from {api_url}")
        
        logger.debug(f"Fetching file information from {api_url}")

        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(api_url) as response:
                    if response.status == 200:
                        logger.debug(f"Connected to {api_url}")
                        files_data = await response.json()
                        logger.debug(f"File information retrieved {files_data}")
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
    
    async def download_file(self, file_info: FileInfo, progress_bar: Optional[tqdm] = None) -> bool:
        """Download a single file from the dataset"""
        file_url = f"https://huggingface.co/datasets/{self.config.repo_id}/resolve/main/{file_info.path}"
        local_path = self.config.raw_data_dir / file_info.path
        
        if self.session is None:
            raise Exception(f"Cannot proceed with file download as session is not available")

        # Create directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if file already exists and size matches
        if local_path.exists() and file_info.size:
            local_size = local_path.stat().st_size
            if local_size == file_info.size:
                if progress_bar:
                    progress_bar.update(1)
                logger.info(f"Skipping {file_info.path} (already exists)")
                return True
        
        #for attempt in range(self.config.max_retries):
        attempt = 0
        while(True):
            try:
                async with self.session.get(file_url) as response:
                    if response.status == 200:
                        # Stream the file content
                        total_size = int(response.headers.get('content-length', 0))
                        chunk_size = 8192
                        
                        async with aiofiles.open(local_path, 'wb') as f:
                            downloaded = 0
                            async for chunk in response.content.iter_chunked(chunk_size):
                                await f.write(chunk)
                                downloaded += len(chunk)
                                if progress_bar and total_size > 0:
                                    progress_bar.update(len(chunk))
                        
                        # Verify download size if we have the expected size
                        if file_info.size and downloaded != file_info.size:
                            logger.warning(f"Size mismatch for {file_info.path}: expected {file_info.size}, got {downloaded}")
                            # We'll still consider it a success for now
                        
                        logger.info(f"Downloaded {file_info.path} ({downloaded} bytes)")
                        break
                        #return True
                    else:
                        logger.warning(f"Failed to download {file_info.path}, status {response.status}, attempt {attempt + 1}")
            except Exception as e:
                attempt = attempt + 1
                logger.warning(f"Error downloading {file_info.path} (attempt { attempt }): {e}")
            
            #if attempt < self.config.max_retries - 1:
            #    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        #logger.error(f"Failed to download {file_info.path} after {self.config.max_retries} attempts")
        return True
    
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
                    message=f"Invalid file pattern: {e}"
                )
        
        if not files:
            return DownloadResult(
                success=False,
                total_files=0,
                success_count=0,
                failed_count=0,
                download_dir=str(self.config.raw_data_dir),
                message="No files found matching the criteria"
            )
        
        logger.info(f"Found {len(files)} files to download")
        
        # Calculate total size for progress bar
        total_size = sum(f.size or 0 for f in files)
        
        # Download files one by one
        success_count = 0
        failed_count = 0
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for file_info in files:
                success = await self.download_file(file_info, pbar)
                if success:
                    success_count += 1
                    break
                else:
                    failed_count += 1
        
        return DownloadResult(
            success=failed_count == 0,
            total_files=len(files),
            success_count=success_count,
            failed_count=failed_count,
            download_dir=str(self.config.raw_data_dir),
            message="Successfully downloaded dataset"
        )

# --- Main Download Function ---
async def download_pile_uncopyrighted(
    repo_id: str = "monology/pile-uncopyrighted",
    raw_data_dir: str = "rawdata",
    file_pattern: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 30
) -> DownloadResult:
    """
    Main function to download files from the monology/pile-uncopyrighted dataset
    
    Args:
        repo_id: Hugging Face dataset repository ID
        raw_data_dir: Directory to store downloaded files
        file_pattern: Regex pattern to filter files
        max_retries: Maximum number of retry attempts per file
        timeout: Request timeout in seconds
        
    Returns:
        DownloadResult with download statistics
    """
    # Create configuration with validation
    config = DownloadConfig(
        repo_id=repo_id,
        raw_data_dir=Path(raw_data_dir),
        max_retries=max_retries,
        timeout=timeout,
        file_pattern=file_pattern
    )
    
    # Create downloader and execute download
    downloader = HFDatasetDownloader(config)
    
    async with downloader:
        return await downloader.download_dataset()
