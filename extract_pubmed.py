import asyncio
import aiofiles
import json
import gzip
import re
from typing import List, Optional, Dict, Any, Pattern
import logging
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
import hashlib
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    def __init__(self):
        self.pubmed_pattern: Pattern[str] = re.compile(
            r'PMID- (\d+)\nAB  - (.*?)(?=\n[A-Z]{2,4}  -|\n\n|\Z)', 
            re.DOTALL
        )
        self.target_size: int = 50 * 1024 * 1024  # 50MB in bytes
        self.processed_count: int = 0
        self.valid_count: int = 0
        self.invalid_count: int = 0
    
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
    
    async def _open_input_file(self, input_path: str):
        """Open input file with appropriate handler based on extension"""
        if input_path.endswith('.gz'):
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
        
        text = str(data.get('text', '')).lower()
        if any(keyword in text for keyword in ['pubmed', 'pmid', 'abstract', 'introduction', 'method', 'result', 'conclusion']):
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

# --- Main Extraction Coroutine ---
async def run_pubmed_extraction(
    input_path: str, 
    output_path: Optional[str] = None,
    return_objects: bool = False
) -> Any:
    """
    Main coroutine to extract PubMed abstracts from Pile-Uncopyrighted dataset
    
    Args:
        input_path: Path to input dataset file
        output_path: Path to output JSONL file (required if return_objects=False)
        return_objects: If True, returns list of objects instead of writing to file
        
    Returns:
        Either statistics dict (if writing to file) or list of PubMedAbstract objects
    """
    extractor = PubMedAbstractExtractor()
    
    logger.info("Starting PubMed abstract extraction...")
    
    if return_objects:
        abstracts = await extractor.extract_abstracts_to_memory(input_path)
        logger.info(f"Extracted {len(abstracts)} abstracts to memory")
        return abstracts
    else:
        if not output_path:
            raise ValueError("output_path is required when return_objects=False")
            
        stats = await extractor.extract_abstracts_to_file(input_path, output_path)
        logger.info("Extraction completed successfully!")
        return stats