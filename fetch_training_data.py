import asyncio
import aiohttp
import aiofiles
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HFDatasetDownloader:
    """Download files from Hugging Face datasets"""
    repo_id: str = "monology/pile-uncopyrighted"
    raw_data_dir: Path = Path("rawdata")
    max_retries: int = 3
    timeout: int = 30
    session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_dataset_files(self) -> List[Dict[str, Any]]:
        """Get list of files in the dataset from Hugging Face API"""
        api_url = f"https://huggingface.co/api/datasets/{self.repo_id}/tree/main"
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(api_url) as response:
                    if response.status == 200:
                        files = await response.json()
                        # Filter for data files (assuming they have common extensions)
                        data_files = [
                            f for f in files 
                            if isinstance(f, dict) and 
                            f.get("type") == "file" and
                            any(f["path"].endswith(ext) for ext in [".jsonl", ".jsonl.zst", ".jsonl.gz", ".txt", ".zst"])
                        ]
                        return data_files
                    else:
                        logger.warning(f"API returned status {response.status}, attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Error fetching file list (attempt {attempt + 1}): {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"Failed to get file list after {self.max_retries} attempts")
    
    async def download_file(self, file_info: Dict[str, Any], progress_bar: Optional[tqdm] = None) -> bool:
        """Download a single file from the dataset"""
        file_path = file_info["path"]
        file_url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{file_path}"
        local_path = self.raw_data_dir / file_path
        
        # Create directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if file already exists and size matches
        if local_path.exists():
            local_size = local_path.stat().st_size
            if local_size == file_info.get("size", 0):
                if progress_bar:
                    progress_bar.update(1)
                logger.info(f"Skipping {file_path} (already exists)")
                return True
        
        for attempt in range(self.max_retries):
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
                        
                        logger.info(f"Downloaded {file_path} ({downloaded} bytes)")
                        return True
                    else:
                        logger.warning(f"Failed to download {file_path}, status {response.status}, attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Error downloading {file_path} (attempt {attempt + 1}): {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to download {file_path} after {self.max_retries} attempts")
        return False
    
    async def download_dataset(self, file_pattern: Optional[str] = None) -> Dict[str, Any]:
        """
        Download all files from the dataset matching the pattern
        
        Args:
            file_pattern: Regex pattern to filter files (e.g., r".*train.*\.jsonl\.zst")
            
        Returns:
            Dictionary with download statistics
        """
        # Ensure rawdata directory exists
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of files
        logger.info(f"Fetching file list for {self.repo_id}")
        files = await self.get_dataset_files()
        
        # Filter files if pattern provided
        if file_pattern:
            pattern = re.compile(file_pattern)
            files = [f for f in files if pattern.search(f["path"])]
        
        if not files:
            logger.warning("No files found matching the criteria")
            return {"success": False, "message": "No files found"}
        
        logger.info(f"Found {len(files)} files to download")
        
        # Calculate total size for progress bar
        total_size = sum(f.get("size", 0) for f in files)
        
        # Download files one by one
        success_count = 0
        failed_count = 0
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for file_info in files:
                success = await self.download_file(file_info, pbar)
                if success:
                    success_count += 1
                else:
                    failed_count += 1
        
        return {
            "success": True,
            "total_files": len(files),
            "success_count": success_count,
            "failed_count": failed_count,
            "download_dir": str(self.raw_data_dir)
        }

async def download_pile_uncopyrighted(
    repo_id: str = "monology/pile-uncopyrighted",
    raw_data_dir: str = "rawdata",
    file_pattern: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Main function to download files from the monology/pile-uncopyrighted dataset
    
    Args:
        repo_id: Hugging Face dataset repository ID
        raw_data_dir: Directory to store downloaded files
        file_pattern: Regex pattern to filter files
        max_retries: Maximum number of retry attempts per file
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with download statistics
    """
    downloader = HFDatasetDownloader(
        repo_id=repo_id,
        raw_data_dir=Path(raw_data_dir),
        max_retries=max_retries,
        timeout=timeout
    )
    
    async with downloader:
        return await downloader.download_dataset(file_pattern)
