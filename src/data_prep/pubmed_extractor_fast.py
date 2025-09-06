import asyncio
import aiofiles
import json
import gzip
import re
import os
from typing import List, Optional, Dict, Any, Pattern
import logging
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
import hashlib
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import ParallelZstdJsonlReader
try:
    from .fast_zst_reader import ParallelZstdJsonlReader, process_large_zstd_file_parallel as zstreader
    HAS_ZSTD_READER = True
except ImportError:
    HAS_ZSTD_READER = False
    logger.warning("ParallelZstdJsonlReader not available. Falling back to standard processing.")

# --- Pydantic V2 Models ---
class SourceFormat(str, Enum):
    TEXT = "text"
    JSONL = "jsonl"
    UNKNOWN = "unknown"

class PubMedAbstract(BaseModel):
    """Pydantic V2 model for PubMed abstracts"""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        frozen=False,
        extra='forbid'
    )
    
    id: str = Field(..., description="Unique identifier for the abstract", min_length=1)
    abstract_text: str = Field(..., description="The abstract content", min_length=10)
    source: str = Field(default="pile-uncopyrighted", description="Source dataset")
    extracted_at: datetime = Field(default_factory=datetime.now, description="Extraction timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source_format: SourceFormat = Field(default=SourceFormat.UNKNOWN, description="Format of source data")
    
    @field_validator('abstract_text')
    @classmethod
    def validate_abstract_text(cls, v: str) -> str:
        """Validate and clean abstract text"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Abstract text cannot be empty")
        
        # Clean the text
        v = re.sub(r'\s+', ' ', v).strip()
        
        if len(v) < 10:
            raise ValueError("Abstract text is too short")
            
        return v
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate ID format"""
        v = v.strip()
        if not v:
            raise ValueError("ID cannot be empty")
        return v

# --- PubMed Abstract Extractor ---
class PubMedAbstractExtractor:
    """Extract PubMed abstracts from Pile-Uncopyrighted dataset using Pydantic V2"""
    
    def __init__(self, use_parallel_zstd: bool = True, num_processes: int = 1):
        self.pubmed_pattern: Pattern[str] = re.compile(
            r'PMID- (\d+)\nAB  - (.*?)(?=\n[A-Z]{2,4}  -|\n\n|\Z)', 
            re.DOTALL
        )
        self.target_size: int = 55 * 1024 * 1024 * num_processes # 50MB in bytes
        self.processed_count: int = 0
        self.valid_count: int = 0
        self.invalid_count: int = 0
        self.num_processes: int = num_processes
        self.use_parallel_zstd = use_parallel_zstd and HAS_ZSTD_READER
    
    async def extract_abstracts_to_file(self, input_path: str, output_path: str) -> Dict[str, int]:
        """
        Extract PubMed abstracts and save to JSONL file
        
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
        
        # Open input file
        input_file = await self._open_input_file(input_path)
        
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as output_file:
            try:
                # Process based on file format
                if input_path.endswith('.jsonl') or input_path.endswith('.jsonl.gz'):
                    await self._process_jsonl(input_file, output_file, current_size)
                else:
                    # Assume it's a text file with PubMed format
                    await self._process_text(input_file, output_file, current_size)
                    
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                raise
            finally:
                if not (self.use_parallel_zstd and input_path.endswith('.jsonl.zst')):
                    await input_file.close()
        
        logger.info(f"Processing complete: {self.processed_count} processed, "
                   f"{self.valid_count} valid, {self.invalid_count} invalid")
        
        return {
            "processed": self.processed_count,
            "valid": self.valid_count,
            "invalid": self.invalid_count,
            "output_size_mb": current_size // 1024 // 1024
        }
    
    async def extract_abstracts_to_memory(self, input_path: str, max_size: Optional[int] = None) -> List[PubMedAbstract]:
        """
        Extract PubMed abstracts and return as list of Pydantic objects
        
        Args:
            input_path: Path to input file
            max_size: Maximum size in bytes (defaults to target_size)
            
        Returns:
            List of PubMedAbstract objects
        """
        max_size = max_size or self.target_size
        current_size = 0
        abstracts = []
        
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
                        
                        if self._is_pubmed_entry(data):
                            abstract = self._extract_abstract_from_json(data)
                            if abstract:
                                entry_id = data.get('id', self._generate_id(data))
                                
                                pubmed_abstract = PubMedAbstract(
                                    id=entry_id,
                                    abstract_text=abstract,
                                    metadata={"original_data_keys": list(data.keys())},
                                    source_format=SourceFormat.JSONL
                                )
                                
                                # Use model_dump_json() - Pydantic V2 handles Unicode properly
                                json_size = len(pubmed_abstract.model_dump_json().encode('utf-8'))
                                
                                if current_size + json_size > max_size:
                                    break
                                    
                                abstracts.append(pubmed_abstract)
                                current_size += json_size
                                self.valid_count += 1
                    
                    except Exception as e:
                        self.invalid_count += 1
                        continue
            else:
                content = await input_file.read()
                matches = self.pubmed_pattern.findall(content)
                
                for pmid, abstract in matches:
                    self.processed_count += 1
                    
                    try:
                        pubmed_abstract = PubMedAbstract(
                            id=pmid,
                            abstract_text=abstract,
                            metadata={"pmid": pmid},
                            source_format=SourceFormat.TEXT
                        )
                        
                        # Use model_dump_json() - Pydantic V2 handles Unicode properly
                        json_size = len(pubmed_abstract.model_dump_json().encode('utf-8'))
                        
                        if current_size + json_size > max_size:
                            break
                            
                        abstracts.append(pubmed_abstract)
                        current_size += json_size
                        self.valid_count += 1
                    
                    except Exception as e:
                        self.invalid_count += 1
                        continue
        
        finally:
            await input_file.close()
        
        return abstracts
    
    async def _process_zstd_with_parallel_reader(self, input_path: str, output_path: str, num_processes: int = 1) -> Dict[str, int]:
        """Process .jsonl.zst files using ParallelZstdJsonlReader"""
        current_size = 0
        # Use ParallelZstdJsonlReader for efficient processing
        for data in zstreader(file_path=Path(input_path), num_processes=num_processes):
            #logger.debug(f"reading {data} from zst file - current output size is {current_size // 1024 // 1024}")
            if current_size >= self.target_size :
                break
            self.processed_count += 1
            
            try:
                if self._is_pubmed_entry(data):
                    abstract = self._extract_abstract_from_json(data)
                    if abstract:
                        entry_id = data.get('id', self._generate_id(data))
                        
                        pubmed_abstract = PubMedAbstract(
                            id=entry_id,
                            abstract_text=abstract,
                            metadata={"original_data_keys": list(data.keys())},
                            source_format=SourceFormat.JSONL
                        )
                        # Use model_dump_json() - Pydantic V2 handles Unicode properly
                        json_line = pubmed_abstract.model_dump_json() + '\n'
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
                            logger.info(f"Extracted {self.valid_count} abstracts, "
                                        f"current size: {current_size//1024//1024}MB")
            
            except Exception as e:
                self.invalid_count += 1
                logger.debug(f"Invalid abstract: {e}")
                continue
        
        logger.info(f"Completed extraction of {self.valid_count} abstracts, "
                    f"total size: {current_size/1024/1024:.2f}MB")
        
        
        return {
            "processed": self.processed_count,
            "valid": self.valid_count,
            "invalid": self.invalid_count,
            "output_size_mb": current_size // 1024 // 1024
        }
    
    async def _process_zstd_to_memory_with_parallel_reader(self, input_path: str, max_size: int) -> List[PubMedAbstract]:
        """Process .jsonl.zst files to memory using ParallelZstdJsonlReader"""
        current_size = 0
        abstracts = []
        
        try:
            # Use ParallelZstdJsonlReader for efficient processing
            for data in zstreader(file_path=Path(input_path), num_processes=4):
                #logger.debug(f"read {data} from zst")
                self.processed_count += 1
                try:
                    if self._is_pubmed_entry(data):
                        abstract = self._extract_abstract_from_json(data)
                        if abstract:
                            entry_id = data.get('id', self._generate_id(data))
                            
                            pubmed_abstract = PubMedAbstract(
                                id=entry_id,
                                abstract_text=abstract,
                                metadata={"original_data_keys": list(data.keys())},
                                source_format=SourceFormat.JSONL
                            )
                            
                            # Use model_dump_json() - Pydantic V2 handles Unicode properly
                            json_size = len(pubmed_abstract.model_dump_json().encode('utf-8'))
                            
                            if current_size + json_size > max_size:
                                break
                                
                            abstracts.append(pubmed_abstract)
                            current_size += json_size
                            self.valid_count += 1
                
                except Exception as e:
                    self.invalid_count += 1
                    continue
            
        except Exception as e:
            logger.error(f"Error processing file with ParallelZstdJsonlReader: {e}")
            raise
        
        return abstracts
    
    async def _open_input_file(self, input_path: str):
        """Open input file with appropriate handler based on extension"""
        if input_path.endswith('.zst'):
            return await aiofiles.open(input_path, 'rt', encoding='utf-8')
        else:
            return await aiofiles.open(input_path, 'r', encoding='utf-8')
    
    async def _process_text(self, input_file, output_file, current_size):
        """Process text format files with PubMed content"""
        content = await input_file.read()
        matches = self.pubmed_pattern.findall(content)
        
        for pmid, abstract in matches:
            self.processed_count += 1
            
            try:
                pubmed_abstract = PubMedAbstract(
                    id=pmid,
                    abstract_text=abstract,
                    metadata={"pmid": pmid},
                    source_format=SourceFormat.TEXT
                )
                
                # Pydantic V2's model_dump_json() handles Unicode properly by default
                json_line = pubmed_abstract.model_dump_json() + '\n'
                line_size = len(json_line.encode('utf-8'))
                
                if current_size + line_size > self.target_size:
                    break
                    
                await output_file.write(json_line)
                current_size += line_size
                self.valid_count += 1
                
                if self.valid_count % 1000 == 0:
                    logger.info(f"Extracted {self.valid_count} abstracts, "
                               f"current size: {current_size/1024/1024:.2f}MB")
                
            except Exception as e:
                self.invalid_count += 1
                logger.debug(f"Invalid abstract (PMID: {pmid}): {e}")
                continue
        
        logger.info(f"Completed extraction of {self.valid_count} abstracts, "
                   f"total size: {current_size/1024/1024:.2f}MB")
    
    async def _process_jsonl(self, input_file, output_file, current_size):
        """Process JSONL format files"""
        async for line in input_file:
            self.processed_count += 1
            
            try:
                data = json.loads(line)
                
                if self._is_pubmed_entry(data):
                    abstract = self._extract_abstract_from_json(data)
                    if abstract:
                        entry_id = data.get('id', self._generate_id(data))
                        
                        pubmed_abstract = PubMedAbstract(
                            id=entry_id,
                            abstract_text=abstract,
                            metadata={"original_data_keys": list(data.keys())},
                            source_format=SourceFormat.JSONL
                        )
                        
                        # Pydantic V2's model_dump_json() handles Unicode properly by default
                        json_line = pubmed_abstract.model_dump_json() + '\n'
                        line_size = len(json_line.encode('utf-8'))
                        
                        if current_size + line_size > self.target_size:
                            break
                            
                        await output_file.write(json_line)
                        current_size += line_size
                        self.valid_count += 1
                        
                        if self.valid_count % 1000 == 0:
                            logger.info(f"Extracted {self.valid_count} abstracts, "
                                       f"current size: {current_size//1024//1024}MB")
                
            except json.JSONDecodeError:
                self.invalid_count += 1
                logger.warning("Skipping invalid JSON line")
                continue
            except Exception as e:
                self.invalid_count += 1
                logger.debug(f"Invalid abstract: {e}")
                continue
        
        logger.info(f"Completed extraction of {self.valid_count} abstracts, "
                   f"total size: {current_size/1024/1024:.2f}MB")
    
    def _is_pubmed_entry(self, data: Dict[str, Any]) -> bool:
        """Check if the entry is a PubMed abstract"""
        if any(key in data for key in ['pmid', 'pubmed_id', 'pubmed', 'abstract']):
            return True
        
        text = str(data.get('meta', '').get('pile_set_name')).lower()
        if any(keyword in text for keyword in ['pubmed']):
            return True

        return False
    
    def _extract_abstract_from_json(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract abstract text from JSON data"""
        possible_fields = ['abstract', 'text', 'content', 'body', 'abstract_text']
        
        for field in possible_fields:
            if field in data and data[field]:
                return self._clean_abstract(str(data[field]))
        
        return None
    
    def _clean_abstract(self, abstract: str) -> str:
        """Clean and normalize abstract text"""
        abstract = re.sub(r'\s+', ' ', abstract)
        abstract = re.sub(r'^\s*AB\s*-\s*', '', abstract)
        abstract = re.sub(r'\s*\[[^\]]*\]\s*', ' ', abstract)
        return abstract.strip()
    
    def _generate_id(self, data: Dict[str, Any]) -> str:
        """Generate a unique ID for the abstract"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()