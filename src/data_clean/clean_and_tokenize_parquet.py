import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional
import logging
import re
import hashlib
from dataclasses import dataclass
from tqdm import tqdm
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Configure logging
from src.main import logger

@dataclass
class ParquetPipelineConfig:
    """Configuration for Parquet-based processing pipeline"""
    input_path: Path
    output_dir: Path
    parquet_chunk_size: int = 100000  # Records per Parquet file
    min_abstract_length: int = 100
    max_abstract_length: int = 2000

class ParquetPubMedProcessor:
    """PubMed abstract processing using Parquet files"""
    
    def __init__(self, config: ParquetPipelineConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def jsonl_to_parquet(self, jsonl_path: Path, parquet_path: Path):
        """Convert JSONL file to Parquet format"""
        # Read JSONL and convert to PyArrow Table
        table = pq.read_table(
            jsonl_path,
            schema=self._get_schema()
        )
        
        # Write to Parquet with compression
        pq.write_table(
            table,
            parquet_path,
            compression='snappy',  # or 'zstd' for better compression
            chunk_size=self.config.parquet_chunk_size
        )
    
    def _get_schema(self) -> pa.Schema:
        """Define schema for PubMed abstracts"""
        return pa.schema([
            ('id', pa.string()),
            ('abstract_text', pa.string()),
            ('title', pa.string()),
            ('journal', pa.string()),
            ('publication_date', pa.string()),
            ('authors', pa.list_(pa.string())),
            ('doi', pa.string()),
            ('source', pa.string())
        ])
    
    def process_parquet_file(self, parquet_path: Path) -> Path:
        """Process Parquet file: clean, deduplicate, filter"""
        # Use PyArrow's memory mapping for efficient reading
        parquet_file = pq.ParquetFile(parquet_path)
        
        processed_data = []
        
        # Process in batches
        for batch in parquet_file.iter_batches(batch_size=10000):
            df = batch.to_pandas()
            
            # Clean and process data
            df_processed = self._process_dataframe(df)
            processed_data.append(df_processed)
        
        # Combine and save processed data
        final_df = pd.concat(processed_data, ignore_index=True)
        output_path = self.output_dir / "processed_abstracts.parquet"
        
        final_df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='zstd'
        )
        
        return output_path
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of data"""
        # Clean text
        df['cleaned_text'] = df['abstract_text'].apply(self._clean_text)
        
        # Filter by length
        text_lengths = df['cleaned_text'].str.len()
        df = df[(text_lengths >= self.config.min_abstract_length) & 
                (text_lengths <= self.config.max_abstract_length)]
        
        # Remove duplicates
        df = self._deduplicate_dataframe(df)
        
        # Remove PII (pseudocode - implement your PII detection)
        df = self._remove_pii(df)
        
        return df[['id', 'cleaned_text', 'title', 'journal', 'publication_date', 'doi']]
    
    def _clean_text(self, text: str) -> str:
        """Clean abstract text"""
        if pd.isna(text):
            return ""
        
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        return text
    
    def _deduplicate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate abstracts"""
        # Create content hash for deduplication
        df['content_hash'] = df['cleaned_text'].apply(
            lambda x: hashlib.md5(x.encode()).hexdigest()
        )
        
        return df.drop_duplicates(subset=['content_hash'])
    
    def _remove_pii(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records with PII"""
        # Implement your PII detection logic
        # This is a placeholder implementation
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        def contains_pii(text):
            return bool(email_pattern.search(text)) if isinstance(text, str) else False
        
        df['contains_pii'] = df['cleaned_text'].apply(contains_pii)
        return df[~df['contains_pii']]
    
    def create_huggingface_dataset(self, parquet_path: Path) -> Dataset:
        """Create Hugging Face Dataset from Parquet"""
        parquet_dataset = Dataset.from_parquet(str(parquet_path))
        if isinstance(parquet_dataset, Dataset):
            return parquet_dataset
        else:
            raise Exception(f"Could not create parquet huggingface dataset")  
        
    def prepare_for_tokenization(self, dataset: Dataset, tokenizer_name: str = "bert-base-uncased"):
        """Prepare dataset for tokenization using Hugging Face"""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["cleaned_text"],
                truncation=True,
                padding="max_length",
                max_length=512
            )
        
        return dataset.map(tokenize_function, batched=True)
    
    def create_torch_dataloader(self, tokenized_dataset, batch_size: int = 32):
        """Create PyTorch DataLoader for training"""
        tokenized_dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'token_type_ids']
        )
        
        return DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4  # Parallel data loading
        )

# --- Example Usage ---
def run_parquet_pipeline():
    """Run the complete Parquet-based pipeline"""
    config = ParquetPipelineConfig(
        input_path=Path("pubmed_abstracts.jsonl"),
        output_dir=Path("parquet_processed_data"),
        min_abstract_length=50,
        max_abstract_length=1500
    )
    
    processor = ParquetPubMedProcessor(config)
    
    # Step 1: Convert JSONL to Parquet
    parquet_path = config.output_dir / "raw_abstracts.parquet"
    processor.jsonl_to_parquet(config.input_path, parquet_path)
    
    # Step 2: Process the Parquet file
    processed_path = processor.process_parquet_file(parquet_path)
    
    # Step 3: Create Hugging Face dataset
    
    dataset = processor.create_huggingface_dataset(processed_path)
    
    # Step 4: Prepare for tokenization
    tokenized_dataset = processor.prepare_for_tokenization(dataset)
    
    # Step 5: Create PyTorch DataLoader
    dataloader = processor.create_torch_dataloader(tokenized_dataset)
    
    logger.info("Parquet pipeline completed successfully!")
    return dataloader, processed_path

# --- Benchmark Comparison ---
def benchmark_formats():
    """Compare performance of JSONL vs Parquet"""
    import time
    
    # Test with sample data
    test_file = Path("sample_data.jsonl")
    parquet_file = Path("sample_data.parquet")
    
    # Time JSONL reading
    start = time.time()
    with open(test_file, 'r') as f:
        for line in f:
            json.loads(line)
    jsonl_time = time.time() - start
    
    # Time Parquet reading
    start = time.time()
    table = pq.read_table(parquet_file)
    parquet_time = time.time() - start
    
    # Compare file sizes
    jsonl_size = test_file.stat().st_size
    parquet_size = parquet_file.stat().st_size
    
    logger.info(f"JSONL: {jsonl_time:.2f}s, {jsonl_size/1024/1024:.1f}MB")
    logger.info(f"Parquet: {parquet_time:.2f}s, {parquet_size/1024/1024:.1f}MB")
    logger.info(f"Parquet is {jsonl_time/parquet_time:.1f}x faster")
    logger.info(f"Parquet uses {parquet_size/jsonl_size:.1%} of JSONL storage")

#if __name__ == "__main__":
    # Run the pipeline
#    dataloader, processed_path = run_parquet_pipeline()
    
    # Show benchmark results
#    benchmark_formats()