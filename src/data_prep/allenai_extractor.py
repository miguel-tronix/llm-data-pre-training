import aiofiles
import json
import re
import os
import asyncio
from datasets import load_dataset, DownloadConfig
from datetime import datetime
from typing import List, Optional, Dict, Any, Pattern
from pathlib import Path
from .configs import ProcessingStats, WebRecord, SourceFormat
import hashlib
from utils.pipeline_logger import get_pipeline_logger

# Configure logging
logger = get_pipeline_logger()
# Try to import ParallelZstdJsonlReader
try:
    from .fast_zst_reader import ParallelZstdJsonlReader, process_large_zstd_file_parallel as zstreader
    HAS_ZSTD_READER = True
except ImportError:
    HAS_ZSTD_READER = False
    logger.warning("ParallelZstdJsonlReader not available. Falling back to standard processing.")

# --- Wikipedia Abstract Extractor ---
class WebRecordExtractor:
    """Extract Web scrape records from AllenAI/C4 dataset using Pydantic V2"""
    
    def __init__(
            self, 
            use_parallel_zstd: bool = True, 
            use_streaming: bool = False,
            num_processes: int = 1,
            file_size_mb: int = 20
    ):
        self.WebRecord_pattern: Pattern[str] = re.compile(
            r'C4- (\d+)\nAB  - (.*?)(?=\n[A-Z]{2,4}  -|\n\n|\Z)', 
            re.DOTALL
        )
        self.target_size: int = file_size_mb * 1024 * 1024 * num_processes # 50MB in bytes
        self.processed_count: int = 0
        self.valid_count: int = 0
        self.invalid_count: int = 0
        self.num_processes: int = num_processes
        self.use_parallel_zstd = use_parallel_zstd and HAS_ZSTD_READER
        self.use_streaming = use_streaming
    
    async def extract_articles_to_file(
            self, 
            input_path: str, 
            output_path: str
    ) -> ProcessingStats:
        """
        Extract Web records and save to JSONL file
        
        Args:
            input_path: Path to input file
            output_path: Path to output JSONL file
            
        Returns:
            Dictionary with processing statistics
        """
        current_size = 0
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use ParallelZstdJsonlReader if available and requested
        if self.use_parallel_zstd and input_path.endswith('.jsonl.zst'):
            return await self._process_zstd_with_parallel_reader(
                input_path,
                output_path,
                num_processes=self.num_processes
            )
        
        if self.use_streaming:
            return await self._process_dataset_streaming(
                input_path, 
                output_path
            ) 

        # Open input file
        input_file = await self._open_input_file(input_path)
        
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as output_file:
            try:
                # Process based on file format
                if input_path.endswith('.jsonl') or input_path.endswith('.jsonl.gz'):
                    await self._process_jsonl(input_file, output_file, current_size)
                else:
                    # Assume it's a text file with Wikipedia format
                    await self._process_text(input_file, output_file, current_size)
                    
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                raise
            finally:
                if not (self.use_parallel_zstd and input_path.endswith('.jsonl.zst')):
                    await input_file.close()
        
        logger.info(f"Processing complete: {self.processed_count} processed, "
                   f"{self.valid_count} valid, {self.invalid_count} invalid")
        
        return ProcessingStats(
            processed=self.processed_count,
            valid=self.valid_count,
            invalid=self.invalid_count,
            output_size_mb=current_size // 1024 // 1024
        )
    
    async def extract_articles_to_memory(
            self, 
            input_path: str, 
            max_size: Optional[int] = None
    ) -> List[WebRecord]:
        """
        Extract Wikipedia articles and return as list of Pydantic objects
        
        Args:
            input_path: Path to input file
            max_size: Maximum size in bytes (defaults to target_size)
            
        Returns:
            List of WebRecord objects
        """
        max_size = max_size or self.target_size
        current_size = 0
        records = []
        
        # Use ParallelZstdJsonlReader if available and requested
        if self.use_parallel_zstd and input_path.endswith('.jsonl.zst'):
            return await self._process_zstd_to_memory_with_parallel_reader(input_path, max_size)
        
        # Open input file
        input_file = await self._open_input_file(input_path)
        
        try:
            # Process based on file format
            if input_path.endswith('.jsonl') or input_path.endswith('.jsonl.gz'):
                async for line in input_file:
                    self.processed_count += 1
                    
                    try:
                        data = json.loads(line)
                        
                        if self._is_WebRecord_entry(data):
                            abstract = self._extract_articles_from_json(data)
                            if abstract:
                                entry_id = data.get('id', self._generate_id(data))
                                
                                Web_records = WebRecord(
                                    id=entry_id,
                                    web_text=abstract,
                                    metadata={"original_data_keys": list(data.keys())},
                                    source_format=SourceFormat.JSONL
                                )
                                
                                # Use model_dump_json() - Pydantic V2 handles Unicode properly
                                json_size = len(Web_records.model_dump_json().encode('utf-8'))
                                
                                if current_size + json_size > max_size:
                                    break
                                    
                                records.append(Web_records)
                                current_size += json_size
                                self.valid_count += 1
                    
                    except Exception as e:
                        self.invalid_count += 1
                        continue
            else:
                content = await input_file.read()
                matches = self.WebRecord_pattern.findall(content)
                
                for wiki_id, abstract in matches:
                    self.processed_count += 1
                    
                    try:
                        Web_records = WebRecord(
                            id=wiki_id,
                            article_text=abstract,
                            metadata={"wiki_id": wiki_id},
                            source_format=SourceFormat.TEXT
                        )
                        
                        # Use model_dump_json() - Pydantic V2 handles Unicode properly
                        json_size = len(Web_records.model_dump_json().encode('utf-8'))
                        
                        if current_size + json_size > max_size:
                            break
                            
                        records.append(Web_records)
                        current_size += json_size
                        self.valid_count += 1
                    
                    except Exception as e:
                        self.invalid_count += 1
                        continue
        
        finally:
            await input_file.close()
        
        return records
    
    async def _process_zstd_with_parallel_reader(
            self, 
            input_path: str, 
            output_path: str, 
            num_processes: int = 1
    ) -> ProcessingStats:
        """Process .jsonl.zst files using ParallelZstdJsonlReader"""
        current_size = 0
        # Use ParallelZstdJsonlReader for efficient processing
        for data in zstreader(file_path=Path(input_path), num_processes=num_processes):
            #logger.debug(f"reading {data} from zst file - current output size is {current_size // 1024 // 1024}")
            if current_size >= self.target_size :
                break
            self.processed_count += 1
            
            try:
                if self._is_WebRecord_entry(data):
                    abstract = self._extract_articles_from_json(data)
                    if abstract:
                        entry_id = data.get('id', self._generate_id(data))
                        
                        Web_records = WebRecord(
                            id=entry_id,
                            article_text=abstract,
                            metadata={"original_data_keys": list(data.keys())},
                            source_format=SourceFormat.JSONL
                        )
                        # Use model_dump_json() - Pydantic V2 handles Unicode properly
                        json_line = Web_records.model_dump_json() + '\n'
                        line_size = len(json_line.encode('utf-8'))
                        
                                
                        async with aiofiles.open(output_path, 'a', encoding='utf-8') as output_file:
                            try:
                                #logger.debug(f"writing {line_size} to file")
                                current_size = os.fstat(output_file.fileno()).st_size + line_size
                                if current_size <= self.target_size:
                                    await output_file.write(json_line)
                                    self.valid_count += 1
                            except Exception as e:
                                logger.error(f"Error processing file with ParallelZstdJsonlReader: {e}")
                                raise
                        
                        if self.valid_count % 1000 == 0:
                            logger.info(f"Extracted {self.valid_count} records, "
                                        f"current size: {current_size//1024//1024}MB")
            
            except Exception as e:
                self.invalid_count += 1
                logger.debug(f"Invalid abstract: {e}")
                continue
        
        logger.info(f"Completed extraction of {self.valid_count} records, "
                    f"total size: {current_size/1024/1024:.2f}MB")
        
        
        return ProcessingStats(
            processed=self.processed_count,
            valid=self.valid_count,
            invalid=self.invalid_count,
            output_size_mb=current_size // 1024 // 1024
        )
    
    async def _process_dataset_streaming(
            self, 
            input_path: str, 
            output_path: str
    ) -> ProcessingStats:
         
    
        """Process dataset in streaming mode to minimize memory usage"""
        current_size = 0
        
        dataset = load_dataset(
            input_path, 
            "en",
            download_config=DownloadConfig(
                cache_dir="./.cache",
                max_retries=3,
                force_download=False,
                resume_download=True,
                extract_compressed_file=True
            ),
            download_mode="force_redownload", 
            split="train", 
            streaming=True)

        target_size_bytes = self.target_size * 1024 * 1024
        current_size = 0
        record_count = 0
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        for line in dataset:
            if current_size  >= self.target_size:
                break
            data = json.loads(line) if isinstance(line, str) else line
            logger.debug(f"reading {data} from dataset stream - current output size is {current_size // 1024 // 1024}")
            try:
                if self._is_WebRecord_entry(data):
                    abstract = self._extract_articles_from_json(data)
                    if abstract:
                        entry_id = data.get('id', self._generate_id(data))
                        
                        Web_records = WebRecord(
                            id=entry_id,
                            web_text=abstract,
                            timestamp=data.get("timestamp", ""),
                            url=data.get("url", ""),
                            metadata={"original_data_keys": list(data.keys())},
                            source_format=SourceFormat.JSONL
                        )
                        # Use model_dump_json() - Pydantic V2 handles Unicode properly
                        json_line = Web_records.model_dump_json() + '\n'
                        line_size = len(json_line.encode('utf-8'))
                        
                                
                        async with aiofiles.open(output_path, 'a', encoding='utf-8') as output_file:
                            try:
                                #logger.debug(f"writing {line_size} to file")
                                current_size = os.fstat(output_file.fileno()).st_size + line_size
                                if current_size <= self.target_size:
                                    await output_file.write(json_line)
                                    self.valid_count += 1
                            except Exception as e:
                                logger.error(f"Error processing file with ParallelZstdJsonlReader: {e}")
                                raise
                        
                        if self.valid_count % 1000 == 0:
                            logger.info(f"Extracted {self.valid_count} records, "
                                        f"current size: {current_size//1024//1024}MB")
            
            except Exception as e:
                self.invalid_count += 1
                logger.debug(f"Invalid abstract: {e}")
                continue
            record_count += 1
            self.processed_count += 1
            
        print(f"completed web records extraction.")
        
        return ProcessingStats(
            processed=self.processed_count,
            valid=self.valid_count,
            invalid=self.invalid_count,
            output_size_mb=current_size // 1024 // 1024
        )

    async def _process_zstd_to_memory_with_parallel_reader(
            self, 
            input_path: str, 
            max_size: int
    ) -> List[WebRecord]:
        """Process .jsonl.zst files to memory using ParallelZstdJsonlReader"""
        current_size = 0
        records = []
        
        try:
            # Use ParallelZstdJsonlReader for efficient processing
            for data in zstreader(file_path=Path(input_path), num_processes=4):
                #logger.debug(f"read {data} from zst")
                self.processed_count += 1
                try:
                    if self._is_WebRecord_entry(data):
                        abstract = self._extract_articles_from_json(data)
                        if abstract:
                            entry_id = data.get('id', self._generate_id(data))
                            
                            Web_records = WebRecord(
                                id=entry_id,
                                article_text=abstract,
                                metadata={"original_data_keys": list(data.keys())},
                                source_format=SourceFormat.JSONL
                            )
                            
                            # Use model_dump_json() - Pydantic V2 handles Unicode properly
                            json_size = len(Web_records.model_dump_json().encode('utf-8'))
                            
                            if current_size + json_size > max_size:
                                break
                                
                            records.append(Web_records)
                            current_size += json_size
                            self.valid_count += 1
                
                except Exception as e:
                    self.invalid_count += 1
                    continue
            
        except Exception as e:
            logger.error(f"Error processing file with ParallelZstdJsonlReader: {e}")
            raise
        
        return records
    
    async def _open_input_file(self, input_path: str):
        """Open input file with appropriate handler based on extension"""
        if input_path.endswith('.zst'):
            return await aiofiles.open(input_path, 'rt', encoding='utf-8')
        else:
            return await aiofiles.open(input_path, 'r', encoding='utf-8')
    
    async def _process_text(self, input_file, output_file, current_size):
        """Process text format files with Wikipedia content"""
        content = await input_file.read()
        matches = self.WebRecord_pattern.findall(content)
        
        for wiki_id, abstract in matches:
            self.processed_count += 1
            
            try:
                Web_records = WebRecord(
                    id=wiki_id,
                    article_text=abstract,
                    metadata={"wikipediaid": wiki_id},
                    source_format=SourceFormat.TEXT
                )
                
                # Pydantic V2's model_dump_json() handles Unicode properly by default
                json_line = Web_records.model_dump_json() + '\n'
                line_size = len(json_line.encode('utf-8'))
                
                if current_size + line_size > self.target_size:
                    break
                    
                await output_file.write(json_line)
                current_size += line_size
                self.valid_count += 1
                
                if self.valid_count % 1000 == 0:
                    logger.info(f"Extracted {self.valid_count} records, "
                               f"current size: {current_size/1024/1024:.2f}MB")
                
            except Exception as e:
                self.invalid_count += 1
                logger.debug(f"Invalid abstract (Wikipedia: {wiki_id}: {e}")
                continue
        
        logger.info(f"Completed extraction of {self.valid_count} records, "
                   f"total size: {current_size/1024/1024:.2f}MB")
    
    async def _process_jsonl(self, input_file, output_file, current_size):
        """Process JSONL format files"""
        async for line in input_file:
            self.processed_count += 1
            
            try:
                data = json.loads(line)
                
                if self._is_WebRecord_entry(data):
                    abstract = self._extract_articles_from_json(data)
                    if abstract:
                        entry_id = data.get('id', self._generate_id(data))
                        
                        Web_records = WebRecord(
                            id=entry_id,
                            web_text=abstract,
                            metadata={"original_data_keys": list(data.keys())},
                            source_format=SourceFormat.JSONL
                        )
                        
                        # Pydantic V2's model_dump_json() handles Unicode properly by default
                        json_line = Web_records.model_dump_json() + '\n'
                        line_size = len(json_line.encode('utf-8'))
                        
                        if current_size + line_size > self.target_size:
                            break
                            
                        await output_file.write(json_line)
                        current_size += line_size
                        self.valid_count += 1
                        
                        if self.valid_count % 1000 == 0:
                            logger.info(f"Extracted {self.valid_count} records, "
                                       f"current size: {current_size//1024//1024}MB")
                
            except json.JSONDecodeError:
                self.invalid_count += 1
                logger.warning("Skipping invalid JSON line")
                continue
            except Exception as e:
                self.invalid_count += 1
                logger.debug(f"Invalid abstract: {e}")
                continue
        
        logger.info(f"Completed extraction of {self.valid_count} records, "
                   f"total size: {current_size/1024/1024:.2f}MB")
    
    def _is_WebRecord_entry(self, data: Dict[str, Any]) -> bool:
        """Check if the entry is a web scrape """
        text = ','.join(data.keys()).lower()
        if any(keyword in text for keyword in ['text', 'url', 'timestamp']):
            return True

        return False
    
    def _extract_articles_from_json(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract abstract text from JSON data"""
        possible_fields = ['web_text', 'text', 'content', 'body', 'url']
        
        for field in possible_fields:
            if field in data and data[field]:
                return self._clean_articles(str(data[field]))
        
        return None
    
    def _clean_articles(self, abstract: str) -> str:
        """Clean and normalize abstract text"""
        abstract = re.sub(r'\s+', ' ', abstract)
        abstract = re.sub(r'^\s*AB\s*-\s*', '', abstract)
        abstract = re.sub(r'\s*\[[^\]]*\]\s*', ' ', abstract)
        return abstract.strip()
    
    def _generate_id(self, data: Dict[str, Any]) -> str:
        """Generate a unique ID for the abstract"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()