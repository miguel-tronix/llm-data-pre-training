import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

MIN_CHAR_LENGTH = 50  # Minimum number of characters for abstracts


class ProcessingStats(BaseModel):
    """Pydantic V2 model for GitHub records"""

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}, frozen=False, extra="forbid"
    )
    processed: int = Field(
        ge=0, description="Total number of records seen - minimum is zero"
    )
    valid: int = Field(
        ge=0,
        description="Number of valid records counted - cannot be more than processed",
    )
    invalid: int = Field(
        ge=0,
        description="Number of invalid records counted - cannot be more than processed",
    )
    output_size_mb: int = Field(
        ge=10,
        le=1000,
        description="Output size in megabytes - must be between 10 and 1000 MB",
    )

    @field_validator("valid", "invalid")
    @classmethod
    def validate_counts(cls, v: int, info: ValidationInfo) -> int:
        # Get the other field values
        values = info.data

        if "processed" in values:
            if v > values["processed"]:
                field_name = info.field_name
                raise ValueError(f"{field_name} cannot exceed processed count")
        return v

    @field_validator("valid", "invalid", mode="after")
    @classmethod
    def validate_sum(cls, v: int, info: ValidationInfo) -> int:
        # Get all field values
        values = info.data

        if "processed" in values and "valid" in values and "invalid" in values:
            total = values["valid"] + values["invalid"]
            if total > values["processed"]:
                raise ValueError(
                    "Sum of valid and invalid records cannot exceed processed count"
                )
        return v


# --- Pydantic V2 Models ---
class SourceFormat(str, Enum):
    TEXT = "text"
    JSONL = "jsonl"
    UNKNOWN = "unknown"


# --- Pydantic V2 Models ---
class PipelineType(str, Enum):
    PUBMED = "pubmed"
    GITHUB = "github"
    WIKI = "wikipedia"
    WEB = "web_c4"


class GitHubRecord(BaseModel):
    """Pydantic V2 model for GitHub records"""

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}, frozen=False, extra="forbid"
    )

    id: str = Field(..., description="Unique identifier for the abstract", min_length=1)
    code_text: str = Field(..., description="The abstract content", min_length=10)
    source: str = Field(default="pile-uncopyrighted", description="Source dataset")
    extracted_at: datetime = Field(
        default_factory=datetime.now, description="Extraction timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    source_format: SourceFormat = Field(
        default=SourceFormat.UNKNOWN, description="Format of source data"
    )

    @field_validator("code_text")
    @classmethod
    def validate_records_text(cls, v: str) -> str:
        """Validate and clean abstract text"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Abstract text cannot be empty")

        # Clean the text
        v = re.sub(r"\s+", " ", v).strip()

        if len(v) < MIN_CHAR_LENGTH:
            raise ValueError("Abstract text is too short")

        return v

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate ID format"""
        v = v.strip()
        if not v:
            raise ValueError("ID cannot be empty")
        return v


class PubMedAbstract(BaseModel):
    """Pydantic V2 model for PubMed abstracts"""

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}, frozen=False, extra="forbid"
    )

    id: str = Field(..., description="Unique identifier for the abstract", min_length=1)
    abstract_text: str = Field(..., description="The abstract content", min_length=10)
    source: str = Field(default="pile-uncopyrighted", description="Source dataset")
    extracted_at: datetime = Field(
        default_factory=datetime.now, description="Extraction timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    source_format: SourceFormat = Field(
        default=SourceFormat.UNKNOWN, description="Format of source data"
    )

    @field_validator("abstract_text")
    @classmethod
    def validate_abstract_text(cls, v: str) -> str:
        """Validate and clean abstract text"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Abstract text cannot be empty")

        # Clean the text
        v = re.sub(r"\s+", " ", v).strip()

        if len(v) < MIN_CHAR_LENGTH:
            raise ValueError("Abstract text is too short")

        return v

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate ID format"""
        v = v.strip()
        if not v:
            raise ValueError("ID cannot be empty")
        return v


class WikiArticle(BaseModel):
    """Pydantic V2 model for Wikipedia records"""

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}, frozen=False, extra="forbid"
    )

    id: str = Field(..., description="Unique identifier for the abstract", min_length=1)
    article_text: str = Field(
        ..., description="The wiki article content", min_length=10
    )
    source: str = Field(default="pile-uncopyrighted", description="Source dataset")
    extracted_at: datetime = Field(
        default_factory=datetime.now, description="Extraction timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    source_format: SourceFormat = Field(
        default=SourceFormat.UNKNOWN, description="Format of source data"
    )

    @field_validator("article_text")
    @classmethod
    def validate_records_text(cls, v: str) -> str:
        """Validate and clean abstract text"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Abstract text cannot be empty")

        # Clean the text
        v = re.sub(r"\s+", " ", v).strip()

        if len(v) < MIN_CHAR_LENGTH:
            raise ValueError("Article text is too short")

        return v

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate ID format"""
        v = v.strip()
        if not v:
            raise ValueError("ID cannot be empty")
        return v


class WebRecord(BaseModel):
    """Pydantic V2 model for Wikipedia records"""

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}, frozen=False, extra="forbid"
    )

    id: str = Field(..., description="Unique identifier for the abstract", min_length=1)
    web_text: str = Field(..., description="Main content of the web page")
    timestamp: datetime | None = Field(None, description="Timestamp of the web page")
    url: str | None = Field(None, description="URL of the web page")
    extracted_at: datetime = Field(
        default_factory=datetime.now, description="Extraction timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    source_format: SourceFormat = Field(
        default=SourceFormat.UNKNOWN, description="Format of source data"
    )

    @field_validator("web_text")
    @classmethod
    def validate_records_text(cls, v: str) -> str:
        """Validate and clean abstract text"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Web scrape text cannot be empty")

        # Clean the text
        v = re.sub(r"\s+", " ", v).strip()

        if len(v) < MIN_CHAR_LENGTH:
            raise ValueError("Web scrape text is too short")

        return v

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate ID format"""
        v = v.strip()
        if not v:
            raise ValueError("ID cannot be empty")
        return v
