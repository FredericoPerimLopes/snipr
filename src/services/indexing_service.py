import hashlib
import json
import logging
import os
import time
from pathlib import Path

try:
    import tree_sitter_go as tsgo
except ImportError:
    tsgo = None

try:
    import tree_sitter_java as tsjava
except ImportError:
    tsjava = None

try:
    import tree_sitter_javascript as tsjs
except ImportError:
    tsjs = None

try:
    import tree_sitter_python as tspython
except ImportError:
    tspython = None

try:
    import tree_sitter_rust as tsrust
except ImportError:
    tsrust = None

try:
    import tree_sitter_typescript as tsts
except ImportError:
    tsts = None

from tree_sitter import Language, Node, Parser

from ..config import get_settings, validate_codebase_path
from ..models.indexing_models import (
    CodeChunk,
    IndexingRequest,
    IndexingResponse,
    IndexingStatus,
)

logger = logging.getLogger(__name__)


class IndexingService:
    def __init__(self):
        self.config = get_settings()
        self.parsers: dict[str, Parser] = {}
        self.languages: dict[str, Language] = {}
        self._init_parsers()

    def _init_parsers(self) -> None:
        """Initialize Tree-sitter parsers for supported languages."""
        language_mappings = {
            "python": (tspython, "python"),
            "javascript": (tsjs, "javascript"),
            "typescript": (tsts, "typescript"),
            "go": (tsgo, "go"),
            "rust": (tsrust, "rust"),
            "java": (tsjava, "java"),
        }

        for lang_name, (module, tree_name) in language_mappings.items():
            if module is None:
                logger.debug(f"Skipping {lang_name} parser - module not available")
                continue
                
            try:
                language = Language(module.language())
                parser = Parser()
                parser.language = language

                self.languages[lang_name] = language
                self.parsers[lang_name] = parser
                logger.debug(f"Initialized parser for {lang_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize {lang_name} parser: {e}")

    async def index_codebase(self, request: IndexingRequest) -> IndexingResponse:
        """Index entire codebase with Tree-sitter parsing."""
        start_time = time.time()

        # Validate codebase path
        codebase_path = validate_codebase_path(request.codebase_path)

        # Get all source files
        source_files = await self._discover_source_files(
            codebase_path,
            request.languages,
            request.exclude_patterns or self.config.DEFAULT_EXCLUDE_PATTERNS
        )

        # Parse files and create chunks
        all_chunks: list[CodeChunk] = []
        languages_detected: set[str] = set()

        for file_path in source_files:
            try:
                chunks = await self._parse_file(file_path)
                all_chunks.extend(chunks)
                languages_detected.update(chunk.language for chunk in chunks)
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")
                continue

        # Store index metadata
        await self._store_index_metadata(codebase_path, all_chunks)

        processing_time = (time.time() - start_time) * 1000

        return IndexingResponse(
            indexed_files=len(source_files),
            total_chunks=len(all_chunks),
            processing_time_ms=processing_time,
            languages_detected=list(languages_detected),
            status="success"
        )

    async def _discover_source_files(
        self,
        codebase_path: Path,
        languages: list[str] | None,
        exclude_patterns: list[str]
    ) -> list[Path]:
        """Discover source files to index."""

        source_files: list[Path] = []

        # File extensions mapping
        extension_map = {
            "python": [".py"],
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "go": [".go"],
            "rust": [".rs"],
            "java": [".java"],
            "c": [".c", ".h"],
            "cpp": [".cpp", ".cc", ".cxx", ".hpp"],
            "csharp": [".cs"],
            "php": [".php"],
            "ruby": [".rb"],
            "html": [".html", ".htm"],
            "css": [".css"],
            "json": [".json"],
            "yaml": [".yaml", ".yml"]
        }

        # Get target extensions
        target_extensions = set()
        if languages:
            for lang in languages:
                if lang in extension_map:
                    target_extensions.update(extension_map[lang])
        else:
            # Include all supported extensions
            for exts in extension_map.values():
                target_extensions.update(exts)

        # Walk directory tree
        for root, dirs, files in os.walk(codebase_path):
            root_path = Path(root)

            # Check if directory should be excluded
            if self._should_exclude_path(root_path, exclude_patterns):
                dirs.clear()  # Don't recurse into excluded directories
                continue

            for file in files:
                file_path = root_path / file

                # Check file extension and size
                max_size = self.config.MAX_FILE_SIZE_MB * 1024 * 1024
                if (file_path.suffix in target_extensions and
                    not self._should_exclude_path(file_path, exclude_patterns) and
                    file_path.stat().st_size < max_size):
                    source_files.append(file_path)

        return source_files

    def _should_exclude_path(self, path: Path, exclude_patterns: list[str]) -> bool:
        """Check if path should be excluded based on patterns."""
        import fnmatch

        path_str = str(path)
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
        return False

    async def _parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse single file with Tree-sitter and create semantic chunks."""
        # Detect language from file extension
        language = self._detect_language(file_path)

        if language not in self.parsers:
            logger.debug(f"No parser available for {language}")
            return []

        try:
            # Read file content
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Parse with Tree-sitter
            parser = self.parsers[language]
            tree = parser.parse(content.encode())

            # Extract semantic chunks from AST
            chunks = []
            await self._extract_chunks_from_node(
                tree.root_node,
                content,
                str(file_path),
                language,
                chunks
            )

            return chunks

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return []

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".html": "html",
            ".css": "css",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml"
        }

        return extension_map.get(file_path.suffix.lower(), "text")

    async def _extract_chunks_from_node(
        self,
        node: Node,
        content: str,
        file_path: str,
        language: str,
        chunks: list[CodeChunk]
    ) -> None:
        """Extract semantic code chunks from Tree-sitter AST node."""
        lines = content.split("\n")

        # Define semantic node types to extract
        semantic_types = {
            "function_definition", "async_function_definition", "function_declaration",
            "class_definition", "class_declaration", "interface_declaration",
            "method_definition", "constructor_definition",
            "variable_declaration", "const_declaration", "let_declaration",
            "struct_declaration", "enum_declaration", "type_alias_declaration",
            "import_statement", "import_declaration", "from_import_statement"
        }

        if node.type in semantic_types:
            # Extract chunk content
            start_line = node.start_point[0]
            end_line = node.end_point[0]

            # Get content with some context
            context_start = max(0, start_line - 1)
            context_end = min(len(lines), end_line + 2)

            chunk_content = "\n".join(lines[context_start:context_end])

            # Create code chunk
            chunk = CodeChunk(
                file_path=file_path,
                content=chunk_content,
                start_line=context_start + 1,  # Convert to 1-based line numbers
                end_line=context_end,
                language=language,
                semantic_type=node.type,
                embedding=None  # Will be populated by SearchService
            )

            chunks.append(chunk)

        # Recursively process child nodes
        for child in node.children:
            await self._extract_chunks_from_node(
                child, content, file_path, language, chunks
            )

    async def _store_index_metadata(
        self, codebase_path: Path, chunks: list[CodeChunk]
    ) -> None:
        """Store indexing metadata for incremental updates."""
        metadata = {
            "codebase_path": str(codebase_path),
            "total_chunks": len(chunks),
            "last_indexed": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "file_hashes": {}
        }

        # Calculate file hashes for incremental updates
        for chunk in chunks:
            file_path = Path(chunk.file_path)
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                file_hash = hashlib.sha256(content.encode()).hexdigest()
                metadata["file_hashes"][str(file_path)] = file_hash

        # Store metadata
        metadata_path = self.config.INDEX_CACHE_DIR / "index_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    async def get_indexing_status(self, codebase_path: str) -> IndexingStatus:
        """Get current indexing status for a codebase."""
        validate_codebase_path(codebase_path)
        metadata_path = self.config.INDEX_CACHE_DIR / "index_metadata.json"

        if not metadata_path.exists():
            return IndexingStatus(
                is_indexed=False,
                total_files=0,
                total_chunks=0,
                index_size_mb=0.0
            )

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Calculate index size
            index_size = sum(
                f.stat().st_size
                for f in self.config.INDEX_CACHE_DIR.iterdir()
                if f.is_file()
            ) / (1024 * 1024)  # Convert to MB

            return IndexingStatus(
                is_indexed=True,
                last_indexed=metadata.get("last_indexed"),
                total_files=len(metadata.get("file_hashes", {})),
                total_chunks=metadata.get("total_chunks", 0),
                index_size_mb=round(index_size, 2)
            )

        except Exception as e:
            logger.error(f"Error reading index metadata: {e}")
            return IndexingStatus(
                is_indexed=False,
                total_files=0,
                total_chunks=0,
                index_size_mb=0.0
            )

    async def needs_reindexing(self, codebase_path: str) -> bool:
        """Check if codebase needs reindexing based on file changes."""
        metadata_path = self.config.INDEX_CACHE_DIR / "index_metadata.json"

        if not metadata_path.exists():
            return True

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            stored_hashes = metadata.get("file_hashes", {})

            # Check if any tracked files have changed
            for file_path_str, stored_hash in stored_hashes.items():
                file_path = Path(file_path_str)

                if not file_path.exists():
                    return True  # File was deleted

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    current_hash = hashlib.sha256(content.encode()).hexdigest()

                    if current_hash != stored_hash:
                        return True  # File was modified
                except Exception:
                    return True  # Error reading file

            return False

        except Exception as e:
            logger.error(f"Error checking reindexing needs: {e}")
            return True
