from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class IndexingRequest(BaseModel):
    codebase_path: str = Field(..., description="Absolute path to codebase root")
    languages: list[str] | None = Field(default=None, description="Languages to index (auto-detect if None)")
    exclude_patterns: list[str] | None = Field(default=None, description="Glob patterns to exclude")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Semantic search query")
    language: str | None = Field(default=None, description="Filter by programming language")
    max_results: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class ChunkType(Enum):
    """Semantic types for code chunks."""

    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    IMPORT_BLOCK = "import_block"
    VARIABLE = "variable"
    CODE_BLOCK = "code_block"


class CodeChunk(BaseModel):
    file_path: str
    content: str
    start_line: int
    end_line: int
    language: str
    semantic_type: str  # function, class, variable, etc.
    embedding: list[float] | None = None

    # Enhanced scalability fields
    chunk_type: ChunkType | None = None
    parent_chunk_id: str | None = None
    dependencies: list[str] | None = None
    dependents: list[str] | None = None
    complexity_score: float | None = None
    last_modified: datetime | None = None
    file_hash: str | None = None

    # Rich metadata fields
    function_signature: str | None = None
    class_name: str | None = None
    function_name: str | None = None
    parameter_types: list[str] | None = None
    return_type: str | None = None
    inheritance_chain: list[str] | None = None
    import_statements: list[str] | None = None
    docstring: str | None = None
    interfaces: list[str] | None = None
    decorators: list[str] | None = None


class SearchResponse(BaseModel):
    results: list[CodeChunk]
    total_matches: int
    query_time_ms: float


class IndexingResponse(BaseModel):
    indexed_files: int
    total_chunks: int
    processing_time_ms: float
    languages_detected: list[str]
    status: str = "success"


class FileUpdateRecord(BaseModel):
    """Track file state for incremental updates."""

    file_path: str
    content_hash: str
    last_indexed: datetime
    chunk_ids: list[str]
    dependencies: list[str]


class IndexUpdateResult(BaseModel):
    """Result of incremental update operation."""

    updated_chunks: list[str]
    deleted_chunks: list[str]
    affected_files: list[str]
    processing_time: float


class IndexingStatus(BaseModel):
    is_indexed: bool
    last_indexed: str | None = None
    total_files: int
    total_chunks: int
    index_size_mb: float
