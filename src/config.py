import os
from pathlib import Path
from typing import ClassVar


class Config:
    # Index storage configuration
    INDEX_CACHE_DIR: Path = Path(os.getenv("INDEX_CACHE_DIR", ".index_cache"))

    # Embedding model configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-code")
    _embedding_enabled_env = os.getenv("EMBEDDING_ENABLED", "true").lower()
    EMBEDDING_ENABLED: bool = _embedding_enabled_env == "true"
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    
    # Device configuration (cpu or cuda)
    # Default to CPU for better compatibility
    DEVICE: str = os.getenv("DEVICE", "cpu").lower()

    # Search performance configuration
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    SEARCH_TIMEOUT_SECONDS: int = int(os.getenv("SEARCH_TIMEOUT_SECONDS", "30"))
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "2000"))

    # Tree-sitter configuration
    SUPPORTED_LANGUAGES: ClassVar[list[str]] = [
        "python",
        "javascript",
        "typescript",
        "go",
        "rust",
        "java",
        "c",
        "cpp",
        "csharp",
        "php",
        "ruby",
        "html",
        "css",
        "json",
        "yaml",
    ]

    # Vector database configuration
    VECTOR_DB_PATH: Path = INDEX_CACHE_DIR / "embeddings_vec.db"
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

    # SQLite-vec configuration
    VEC_DIMENSION: int = 768  # for jina-embeddings-v2-base-code
    VEC_INDEX_TYPE: str = os.getenv("VEC_INDEX_TYPE", "flat")  # flat, ivf, hnsw

    # Performance optimization
    _quantization_env = os.getenv("ENABLE_QUANTIZATION", "true").lower()
    ENABLE_QUANTIZATION: bool = _quantization_env == "true"
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))

    # Incremental update configuration
    ENABLE_INCREMENTAL_UPDATES: bool = os.getenv("ENABLE_INCREMENTAL_UPDATES", "true").lower() == "true"
    AUTO_SYNC_ON_SEARCH: bool = os.getenv("AUTO_SYNC_ON_SEARCH", "true").lower() == "true"
    MAX_FILES_PER_BATCH: int = int(os.getenv("MAX_FILES_PER_BATCH", "100"))
    DEPENDENCY_GRAPH_ENABLED: bool = os.getenv("DEPENDENCY_GRAPH_ENABLED", "true").lower() == "true"

    # Memory management
    MAX_MEMORY_MB: int = int(os.getenv("MAX_MEMORY_MB", "4096"))
    CHUNK_CACHE_SIZE: int = int(os.getenv("CHUNK_CACHE_SIZE", "10000"))

    # Exclude patterns for indexing
    DEFAULT_EXCLUDE_PATTERNS: ClassVar[list[str]] = [
        "**/.git/**",
        "**/node_modules/**",
        "**/__pycache__/**",
        "**/venv/**",
        "**/env/**",
        "**/.env",
        "**/build/**",
        "**/dist/**",
        "**/*.log",
        "**/*.tmp",
        "**/coverage/**",
    ]


def get_settings() -> Config:
    """Get application configuration with environment variable resolution."""
    config = Config()

    # Ensure cache directory exists
    config.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return config


def validate_codebase_path(path: str) -> Path:
    """Validate and resolve codebase path to absolute path."""
    resolved_path = Path(path).resolve()

    if not resolved_path.exists():
        msg = f"Codebase path does not exist: {resolved_path}"
        raise ValueError(msg)

    if not resolved_path.is_dir():
        msg = f"Codebase path is not a directory: {resolved_path}"
        raise ValueError(msg)

    return resolved_path
