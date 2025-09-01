import os
from pathlib import Path
from typing import ClassVar


class Config:
    # Index storage configuration
    INDEX_CACHE_DIR: Path = Path(os.getenv("INDEX_CACHE_DIR", ".index_cache"))

    # Embedding model configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Search performance configuration
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    SEARCH_TIMEOUT_SECONDS: int = int(os.getenv("SEARCH_TIMEOUT_SECONDS", "30"))
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "2000"))

    # Tree-sitter configuration
    SUPPORTED_LANGUAGES: ClassVar[list[str]] = [
        "python", "javascript", "typescript", "go", "rust", "java",
        "c", "cpp", "csharp", "php", "ruby", "html", "css", "json", "yaml"
    ]

    # Vector database configuration
    VECTOR_DB_PATH: Path = INDEX_CACHE_DIR / "embeddings.db"
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

    # Performance optimization
    _quantization_env = os.getenv("ENABLE_QUANTIZATION", "true").lower()
    ENABLE_QUANTIZATION: bool = _quantization_env == "true"
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))

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
        "**/coverage/**"
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
