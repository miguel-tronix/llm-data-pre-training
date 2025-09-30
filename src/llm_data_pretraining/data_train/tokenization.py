from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from data_clean.clean_and_tokenize import PipelineType
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing, Sequence, ByteLevel
import numpy as np
import logging
from tqdm import tqdm
import json
from utils.pipeline_logger import get_pipeline_logger

#set up logger
logger = get_pipeline_logger()

class TokenizerConfig(BaseModel):
    """Configuration for BPE tokenizer training using Pydantic V2"""
    vocab_size: int = Field(default=30000, ge=1000, le=1000000, description="Vocabulary size")
    min_frequency: int = Field(default=2, ge=1, description="Minimum token frequency")
    special_tokens: List[str] = Field(
        default_factory=lambda: ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<|endoftext|>"],
        description="Special tokens for the tokenizer"
    )
    max_length: int = Field(default=512, ge=64, le=4096, description="Maximum sequence length")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "vocab_size": 30000,
                "min_frequency": 2,
                "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<|endoftext|>"],
                "max_length": 512
            }
        }
    }
    
    @field_validator('special_tokens')
    @classmethod
    def validate_special_tokens(cls, v):
        """Ensure special tokens contain required tokens"""
        required_tokens = {"[UNK]"}
        if not required_tokens.issubset(set(v)):
            missing = required_tokens - set(v)
            raise ValueError(f"Special tokens must include {missing}")
        return v

class TokenizationResult(BaseModel):
    """Result of tokenization process"""
    success: bool = Field(..., description="Whether tokenization was successful")
    output_dir: Path = Field(..., description="Output directory for tokenization files")
    vocab_size: int = Field(..., description="Final vocabulary size")
    total_tokens: int = Field(..., description="Total tokens in the corpus")
    tokenizer_config: TokenizerConfig = Field(..., description="Tokenizer configuration used")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "output_dir": "/path/to/tokenized_data",
                "vocab_size": 30000,
                "total_tokens": 15000000,
                "tokenizer_config": {
                    "vocab_size": 30000,
                    "min_frequency": 2,
                    "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<|endoftext|>"],
                    "max_length": 512
                }
            }
        }
    }

class BPETokenizer:
    """BPE Tokenizer for converting text to tokens using Pydantic V2"""
    
    def __init__(
            self, 
            config: Optional[TokenizerConfig] = None,
            pipeline_type: PipelineType = PipelineType.PUBMED
    ):
        self.config = config or TokenizerConfig()
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.is_trained = False
        self.pipeline_type = pipeline_type
    
    def train(self, corpus_path: Union[str, Path]) -> None:
        """Train BPE tokenizer on corpus"""
        logger.info(f"Training BPE tokenizer on {corpus_path}")
        
        # Initialize trainer
        trainer = BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            show_progress=True,
            special_tokens=self.config.special_tokens
        )
        
        # Set up pre-tokenizer
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Train tokenizer
        self.tokenizer.train(files=[str(corpus_path)], trainer=trainer)
        
        template_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[
                ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
                ("[PAD]", self.tokenizer.token_to_id("[PAD]")),
                ("[MASK]", self.tokenizer.token_to_id("[MASK]")),
                ("UNK]", self.tokenizer.token_to_id("[UNK]")),
                ("<|endoftext|>", self.tokenizer.token_to_id("<|endoftext|>"))
            ]
        )
        # Set up post-processing
        self.tokenizer.post_processor = template_processor
        self.is_trained = True
        logger.info(f"Tokenizer trained with vocabulary size: {self.tokenizer.get_vocab_size()}")
    
    def save_tokenizer(self, output_dir: Union[str, Path]) -> None:
        """Save tokenizer to directory"""
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before saving")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save(str(output_dir / f"tokenizer_{self.pipeline_type.value}.json"))
        
        # Save config
        config_path = output_dir / f"tokenizer_config_{self.pipeline_type.value}.json"
        with open(config_path, "w") as f:
            f.write(self.config.model_dump_json(indent=2))
        
        logger.info(f"Tokenizer saved to {output_dir}")
    
    def load_tokenizer(self, tokenizer_dir: Union[str, Path]) -> None:
        """Load tokenizer from directory"""
        tokenizer_dir = Path(tokenizer_dir)
        
        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_dir / f"tokenizer_{self.pipeline_type.value}.json"))
        
        # Load config
        config_path = tokenizer_dir / f"tokenizer_config_{self.pipeline_type.value}.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = json.load(f)
            self.config = TokenizerConfig(**config_data)
        
        self.is_trained = True
        logger.info(f"Tokenizer loaded from {tokenizer_dir}")
    
    def tokenize_corpus(self, corpus_path: Union[str, Path], output_path: Union[str, Path], 
                       batch_size: int = 1000) -> int:
        """Tokenize entire corpus and save as binary file"""
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before tokenizing")
        
        corpus_path = Path(corpus_path)
        output_path = Path(output_path)
        
        logger.info(f"Tokenizing corpus: {corpus_path}")
        
        # Read and tokenize the corpus
        all_tokens = []
        total_tokens = 0
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            # First pass: count lines for progress bar
            total_lines = sum(1 for _ in f)
            f.seek(0)
            
            with tqdm(total=total_lines, desc="Tokenizing") as pbar:
                batch = []
                for line in f:
                    text = line.strip()
                    if text:
                        # Tokenize text
                        batch.extend(self.tokenize_text(text))
                        
                        # Process in batches to manage memory
                        if len(batch) >= batch_size:
                            all_tokens.extend(batch)
                            total_tokens += len(batch)
                            batch = []
                    
                    pbar.update(1)
                
                # Add remaining tokens
                if batch:
                    all_tokens.extend(batch)
                    total_tokens += len(batch)
        
        # Convert to numpy array and save as binary file
        tokens_array = np.array(all_tokens, dtype=np.uint16)  # Using uint16 for efficiency
        
        # Save tokens
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'ab') as f:
            f.write(tokens_array.tobytes())
        
        # Save metadata
        metadata = {
            "total_tokens": total_tokens,
            "vocab_size": self.tokenizer.get_vocab_size(),
            "corpus_path": str(corpus_path),
            "tokenizer_config": self.config.model_dump()
        }
        
        with open(output_path.parent / f"tokenization_metadata_{self.pipeline_type.value}.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Tokenization complete. Saved {total_tokens} tokens to {output_path}")
        return total_tokens
    
    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize a single text string"""
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before tokenizing")
        
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens back to text"""
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before decoding")
        
        return self.tokenizer.decode(tokens)

