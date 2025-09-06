import json
import jsonlines
import zstandard as zstd
from pathlib import Path
from typing import Iterator, Dict, Any, List, Tuple, Optional, Set, ClassVar
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import logging
import re
import hashlib
from tqdm import tqdm
import mmap
from enum import Enum
from contextlib import contextmanager
from data_prep.fast_zst_reader import ParallelZstdJsonlReader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic V2 Models ---
class DeduplicationMethod(str, Enum):
    CONTENT_HASH = "content_hash"
    EXACT_MATCH = "exact_match"
    NONE = "none"

class PIIDetectionConfig(BaseModel):
    """Configuration for PII detection"""
    detect_emails: bool = Field(default=True, description="Detect email addresses")
    detect_phones: bool = Field(default=True, description="Detect phone numbers")
    detect_ssn: bool = Field(default=True, description="Detect Social Security Numbers")
    detect_patient_ids: bool = Field(default=True, description="Detect patient identifiers")
    detect_demographics: bool = Field(default=True, description="Detect demographic information")
    
    model_config = ConfigDict(extra='forbid')

class PipelineConfig(BaseModel):
    """Configuration for the PubMed processing pipeline"""
    input_path: Path = Field(..., description="Path to input JSONL file")
    output_dir: Path = Field(..., description="Output directory for processed files")
    min_abstract_length: int = Field(default=100, ge=10, le=1000, description="Minimum abstract length in characters")
    max_abstract_length: int = Field(default=2000, ge=100, le=10000, description="Maximum abstract length in characters")
    deduplication_method: DeduplicationMethod = Field(default=DeduplicationMethod.CONTENT_HASH, description="Deduplication method")
    pii_config: PIIDetectionConfig = Field(default_factory=PIIDetectionConfig, description="PII detection configuration")
    batch_size: int = Field(default=1000, ge=100, le=10000, description="Batch size for processing")
    
    @field_validator('input_path')
    @classmethod
    def validate_input_path(cls, v: Path) -> Path:
        """Validate that input path exists"""
        if not v.exists():
            raise ValueError(f"Input path does not exist: {v}")
        return v
    
    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Ensure output directory is a Path object"""
        if isinstance(v, str):
            return Path(v)
        return v

class ProcessedRecord(BaseModel):
    """Model for processed PubMed abstract records"""
    id: str = Field(..., description="Unique identifier")
    text: str = Field(..., description="Cleaned abstract text", min_length=1)
    source: str = Field(default="pubmed", description="Data source format")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
#    extracted_at: Optional[str] = Field(None,description="Abstract extracted date")
#    title: Optional[str] = Field(None, description="Article title")
#    journal: Optional[str] = Field(None, description="Journal name")
#    extracted_at: Optional[str] = Field(None, description="Publication date")
#    authors: Optional[List[str]] = Field(None, description="List of authors")
#    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    
    model_config = ConfigDict(extra='ignore', frozen=False)
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate and clean text"""
        v = re.sub(r'\s+', ' ', v.strip())
        if not v:
            raise ValueError("Text cannot be empty")
        return v

class PipelineResult(BaseModel):
    """Result of pipeline execution"""
    success: bool = Field(..., description="Whether the pipeline completed successfully")
    input_records: int = Field(0, ge=0, description="Number of input records processed")
    output_records: int = Field(0, ge=0, description="Number of output records after processing")
    duplicates_removed: int = Field(0, ge=0, description="Number of duplicates removed")
    pii_records_removed: int = Field(0, ge=0, description="Number of records removed due to PII")
    short_records_removed: int = Field(0, ge=0, description="Number of records removed due to short length")
    final_file: Optional[Path] = Field(None, description="Path to final output file")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    @model_validator(mode='after')
    def validate_counts(self) -> 'PipelineResult':
        """Validate that counts are consistent"""
        expected_output = (self.input_records - self.duplicates_removed - 
                          self.pii_records_removed - self.short_records_removed)
        
        if self.output_records != expected_output:
            raise ValueError(f"Output records count mismatch: expected {expected_output}, got {self.output_records}")
        
        return self

# --- Parallel Zstd Reader (Pydantic V2 compatible) ---
class ParallelZstdReaderConfig(BaseModel):
    """Configuration for parallel Zstd reader"""
    file_path: Path = Field(..., description="Path to Zstandard compressed file")
    num_processes: Optional[int] = Field(None, ge=1, description="Number of processes to use")
    chunk_size: int = Field(default=1024 * 1024, ge=1024, le=10 * 1024 * 1024, description="Chunk size in bytes")
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: Path) -> Path:
        """Validate that file exists and has .zst extension"""
        if not v.exists():
            raise ValueError(f"File does not exist: {v}")
        if v.suffix != '.zst':
            raise ValueError(f"File must have .zst extension: {v}")
        return v

# --- PubMed Processing Pipeline with Pydantic V2 ---
class PubMedPipeline:
    """Complete pipeline for processing PubMed abstracts using Pydantic V2"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compile PII patterns based on configuration
        self.pii_patterns = self._compile_pii_patterns()
    
    def _compile_pii_patterns(self) -> Dict[str, re.Pattern]:
        """Compile PII detection patterns based on configuration"""
        patterns = {}
        
        if self.config.pii_config.detect_emails:
            patterns['email'] = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        if self.config.pii_config.detect_phones:
            patterns['phone'] = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        
        if self.config.pii_config.detect_ssn:
            patterns['ssn'] = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        
        if self.config.pii_config.detect_patient_ids:
            patterns['patient_id'] = re.compile(r'\b(patient|subject|participant)\s+#?\d+', re.IGNORECASE)
        
        if self.config.pii_config.detect_demographics:
            patterns['demographics'] = re.compile(r'\b(age|gender|sex|race|ethnicity)\s*[:=]', re.IGNORECASE)
        
        return patterns
    
    @contextmanager
    def read_jsonl_file(self) -> Iterator[Iterator[Dict[str, Any]]]:
        """Context manager for reading JSONL files"""
        if self.config.input_path.suffix == '.zst':
            reader = ParallelZstdJsonlReader(
                file_path=self.config.input_path,
                num_processes=4,
                chunk_size=1024 * 1024
            )
            yield reader.read_parallel()
        else:
            with jsonlines.open(self.config.input_path) as reader:
                yield reader
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text using Pydantic validation"""
        try:
            # Use Pydantic validation for text cleaning
            return ProcessedRecord.model_validate({'text': text, 'id': 'temp', 'source': 'temp'}).text
        except Exception:
            # Fallback to manual cleaning if validation fails
            text = re.sub(r'\s+', ' ', text.strip())
            text = re.sub(r'^\s*(ABSTRACT|ABSTRAKT|RESUMEN)\s*[:-\s]*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\(.*?\)', '', text)
            return text.strip()
    
    def contains_pii(self, text: str) -> bool:
        """Check if text contains personally identifiable information"""
        text_lower = text.lower()
        
        for pattern_name, pattern in self.pii_patterns.items():
            if pattern.search(text):
                logger.debug(f"Found {pattern_name} PII in text")
                return True
        
        return False
    
    def generate_content_hash(self, text: str) -> str:
        """Generate hash for content deduplication"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def process_record(self, record: Dict[str, Any]) -> Optional[ProcessedRecord]:
        """Process a single PubMed abstract record using Pydantic model"""
        
        # Extract abstract text
        abstract_text = record.get('abstract_text', '') or record.get('text', '') or record.get('abstract', '')
        if not abstract_text:
            return None
        
        # Clean the text
        cleaned_text = self.clean_text(abstract_text)
        
        # Check length constraints
        if len(cleaned_text) < self.config.min_abstract_length:
            return None
        
        if len(cleaned_text) > self.config.max_abstract_length:
            cleaned_text = cleaned_text[:self.config.max_abstract_length] + "..."
        
        # Check for PII
        if self.contains_pii(cleaned_text):
            return None
        
        try:
            # Create processed record using Pydantic model
            processed_data = {
                'id': record.get('id', ''),
                'text': cleaned_text,
                'source': record.get('source', 'pubmed'),
                'metadata': {
                    'original_length': len(abstract_text),
                    'cleaned_length': len(cleaned_text),
                    'content_hash': self.generate_content_hash(cleaned_text)
                }
            }
            
            # Add optional fields if they exist
            for field in ['title', 'journal', 'publication_date', 'authors', 'doi']:
                if field in record:
                    processed_data[field] = record[field]
            
            return ProcessedRecord(**processed_data)
            
        except Exception as e:
            logger.warning(f"Failed to create processed record: {e}")
            return None
    
    def run_deduplication(self, input_file: Path, output_file: Path) -> Tuple[int, int, int]:
        """Remove duplicate abstracts using Pydantic models"""
        seen_hashes: Set[str] = set()
        duplicates_removed = 0
        total_records = 0
        
        with jsonlines.open(input_file, 'r') as reader:
            with jsonlines.open(output_file, 'w') as writer:
                for record_data in tqdm(reader, desc="Deduplicating"):
                    total_records += 1
                    
                    try:
                        record = ProcessedRecord(**record_data)
                        content_hash = record.metadata.get('content_hash', '')
                        if content_hash and content_hash not in seen_hashes:
                            seen_hashes.add(content_hash)
                            writer.write(record.model_dump())
                        else:
                            duplicates_removed += 1
                    except Exception as e:
                        logger.warning(f"Invalid record during deduplication: {e}")
                        continue
        
        return len(seen_hashes), duplicates_removed, total_records
    
    def run_pipeline(self) -> PipelineResult:
        """Run the complete processing pipeline with Pydantic V2"""
        import time
        start_time = time.time()
        
        logger.info("Starting PubMed abstract processing pipeline with Pydantic V2")
        
        # Initialize counters
        stats = {
            'input_records': 0,
            'output_records': 0,
            'pii_removed': 0,
            'short_removed': 0
        }
        
        # Step 1: Initial processing and cleaning
        processed_file = self.output_dir / "processed_abstracts.jsonl"
        
        with self.read_jsonl_file() as records, \
             jsonlines.open(processed_file, 'w') as writer:
            
            for record in tqdm(records, desc="Processing abstracts"):
                stats['input_records'] += 1
                processed = self.process_record(record)
                
                if processed:
                    writer.write(processed.model_dump())
                    stats['output_records'] += 1
                else:
                    # Track reasons for removal
                    if record.get('abstract_text') and len(record['abstract_text']) < self.config.min_abstract_length:
                        stats['short_removed'] += 1
                    elif self.contains_pii(record.get('abstract_text', '')):
                        stats['pii_removed'] += 1
        
        # Step 2: Deduplication
        dedup_file = self.output_dir / "deduplicated_abstracts.jsonl"
        unique_count, duplicates_removed, total_dedup = self.run_deduplication(processed_file, dedup_file)
        
        # Step 3: Prepare for tokenization
        final_file = self.output_dir / "final_abstracts.jsonl"
        self.prepare_for_tokenization(dedup_file, final_file)
        
        processing_time = time.time() - start_time
        
        # Create and return pipeline result
        return PipelineResult(
            success=True,
            input_records=stats['input_records'],
            output_records=unique_count,
            duplicates_removed=duplicates_removed,
            pii_records_removed=stats['pii_removed'],
            short_records_removed=stats['short_removed'],
            final_file=final_file,
            processing_time=processing_time
        )
    
    def prepare_for_tokenization(self, input_file: Path, output_file: Path):
        """Prepare the final dataset for tokenization using Pydantic"""
        with jsonlines.open(input_file, 'r') as reader:
            with jsonlines.open(output_file, 'w') as writer:
                for record_data in tqdm(reader, desc="Preparing for tokenization"):
                    try:
                        record = ProcessedRecord(**record_data)
                        tokenization_record = {
                            'text': record.text,
                            'id': record.id,
                            'source': record.source
                        }
                        writer.write(tokenization_record)
                    except Exception as e:
                        logger.warning(f"Invalid record during tokenization prep: {e}")
                        continue

# --- Tokenization Preparation with Pydantic V2 ---
class TokenizationConfig(BaseModel):
    """Configuration for tokenization preparation"""
    output_dir: Path = Field(..., description="Output directory for tokenization files")
    train_ratio: float = Field(default=0.9, ge=0.5, le=1.0, description="Ratio of data for training")
    shuffle: bool = Field(default=True, description="Whether to shuffle data before splitting")
    
    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Ensure output directory exists"""
        v.mkdir(parents=True, exist_ok=True)
        return v

class TokenizationPreparer:
    """Prepare data for BPE tokenizer training using Pydantic V2"""
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
    
    def create_training_corpus(self, jsonl_file: Path, output_file: Path) -> int:
        """Create a text corpus from JSONL file for tokenizer training"""
        line_count = 0
        
        with jsonlines.open(jsonl_file, 'r') as reader:
            with open(output_file, 'w', encoding='utf-8') as writer:
                for record_data in tqdm(reader, desc="Creating training corpus"):
                    try:
                        record = ProcessedRecord(**record_data)
                        normalized = re.sub(r'\s+', ' ', record.text.strip())
                        writer.write(normalized + '\n')
                        line_count += 1
                    except Exception as e:
                        logger.warning(f"Invalid record in corpus creation: {e}")
                        continue
        
        return line_count
