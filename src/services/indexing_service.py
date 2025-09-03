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

try:
    from llama_index.core.node_parser import CodeSplitter
except ImportError:
    CodeSplitter = None
    logging.warning("llama-index-core not available, using basic chunking")

from ..config import get_settings, validate_codebase_path
from ..models.indexing_models import (
    CodeChunk,
    IndexingRequest,
    IndexingResponse,
    IndexingStatus,
)
from .metadata_extractor import MetadataExtractor
from .syntactic_chunker import SyntacticChunker
from .update_service import IncrementalUpdateService

logger = logging.getLogger(__name__)


class IndexingService:
    def __init__(self):
        self.config = get_settings()
        self.parsers: dict[str, Parser] = {}
        self.languages: dict[str, Language] = {}
        self.code_splitters: dict[str, CodeSplitter] = {}
        self.metadata_extractor = MetadataExtractor()
        self.syntactic_chunker = SyntacticChunker()
        self.update_service = IncrementalUpdateService()
        self._init_parsers()
        self._init_code_splitters()

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

        for lang_name, (module, _tree_name) in language_mappings.items():
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

    def _init_code_splitters(self) -> None:
        """Initialize LlamaIndex CodeSplitters for supported languages."""
        if CodeSplitter is None:
            logger.debug("LlamaIndex CodeSplitter not available")
            return

        # Initialize splitters for available parsers
        for lang_name in self.parsers.keys():
            try:
                splitter = CodeSplitter(
                    language=lang_name,
                    chunk_lines=15,  # Optimal chunk size for semantic coherence
                    chunk_lines_overlap=3,  # Context overlap for continuity
                    max_chars=600,  # Focused chunks for better embeddings
                )
                self.code_splitters[lang_name] = splitter
                logger.debug(f"Initialized CodeSplitter for {lang_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize {lang_name} CodeSplitter: {e}")

    async def index_codebase(self, request: IndexingRequest) -> IndexingResponse:
        """Index codebase with smart incremental updates using git-aware change detection."""
        start_time = time.time()

        # Validate codebase path
        codebase_path = validate_codebase_path(request.codebase_path)

        # Use enhanced change detection
        modified_files, new_files, deleted_files = await self.update_service.detect_changes(str(codebase_path))

        files_to_process = modified_files + new_files
        total_changed_files = len(files_to_process) + len(deleted_files)

        if total_changed_files == 0:
            logger.info("No file changes detected, skipping indexing")

            # Get current status for response
            status = await self.get_indexing_status(str(codebase_path))
            return IndexingResponse(
                indexed_files=0,
                total_chunks=status.total_chunks,
                processing_time_ms=0.0,
                languages_detected=[],
                status="up_to_date",
            )

        logger.info(
            f"Incremental indexing: {len(modified_files)} modified, {len(new_files)} new, "
            f"{len(deleted_files)} deleted files"
        )

        # Import SearchService for database cleanup
        from .search_service import SearchService

        search_service = SearchService()

        # Clean up deleted and modified files from database
        files_to_remove = [str(f) for f in deleted_files + modified_files]
        if files_to_remove:
            await search_service.remove_file_embeddings(files_to_remove)

        # Parse only changed files
        all_chunks: list[CodeChunk] = []
        languages_detected: set[str] = set()

        for file_path in files_to_process:
            try:
                chunks = await self._parse_file(file_path)
                all_chunks.extend(chunks)
                languages_detected.update(chunk.language for chunk in chunks)
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")
                continue

        # Generate and store embeddings for new chunks
        if all_chunks:
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
            embedded_chunks = await search_service.embed_code_chunks(all_chunks)
            logger.info(f"Successfully generated embeddings for {len(embedded_chunks)} chunks")

            # Update file records for processed files
            for file_path in files_to_process:
                file_chunks = [c for c in embedded_chunks if c.file_path == file_path]
                chunk_ids = [f"{c.file_path}:{c.start_line}" for c in file_chunks]
                content_hash = self.update_service.calculate_file_hash(file_path)
                dependencies = []

                # Extract dependencies from chunks
                for chunk in file_chunks:
                    if chunk.dependencies:
                        dependencies.extend(chunk.dependencies)

                await self.update_service.update_file_record(
                    file_path, content_hash, chunk_ids, list(set(dependencies))
                )

        # Build dependency graph for future incremental updates
        await self.update_service.build_dependency_graph(str(codebase_path))

        # Update index metadata with new file hashes
        await self._update_index_metadata(codebase_path, files_to_process, deleted_files)

        processing_time = (time.time() - start_time) * 1000

        return IndexingResponse(
            indexed_files=len(files_to_process),
            total_chunks=len(all_chunks),
            processing_time_ms=processing_time,
            languages_detected=list(languages_detected),
            status="success",
        )

    async def _discover_source_files(
        self, codebase_path: Path, languages: list[str] | None, exclude_patterns: list[str]
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
            "yaml": [".yaml", ".yml"],
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
                if (
                    file_path.suffix in target_extensions
                    and not self._should_exclude_path(file_path, exclude_patterns)
                    and file_path.stat().st_size < max_size
                ):
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
        """Parse single file with hybrid LlamaIndex + Tree-sitter approach."""
        # Detect language from file extension
        language = self._detect_language(file_path)

        if language not in self.parsers:
            logger.debug(f"No parser available for {language}")
            return []

        try:
            # Read file content
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Use syntactic chunking for better code integrity
            if language in self.parsers:
                parser = self.parsers[language]
                return await self.syntactic_chunker.chunk_with_integrity(str(file_path), content, language, parser)
            else:
                # Fallback for unsupported languages
                return await self._basic_chunking(file_path, content, language)

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return []

    async def _basic_chunking(self, file_path: Path, content: str, language: str) -> list[CodeChunk]:
        """Basic line-based chunking for unsupported languages."""
        lines = content.split("\n")
        chunks = []

        chunk_size = 15  # lines per chunk
        overlap = 3  # overlapping lines

        for i in range(0, len(lines), chunk_size - overlap):
            end_idx = min(i + chunk_size, len(lines))
            chunk_lines = lines[i:end_idx]
            chunk_content = "\n".join(chunk_lines)

            if chunk_content.strip():  # Skip empty chunks
                chunk = CodeChunk(
                    file_path=str(file_path),
                    content=chunk_content,
                    start_line=i + 1,
                    end_line=end_idx,
                    language=language,
                    semantic_type="code_block",
                )
                chunks.append(chunk)

        return chunks

    async def _hybrid_chunking(self, file_path: Path, content: str, language: str) -> list[CodeChunk]:
        """Hybrid chunking using LlamaIndex + Tree-sitter for optimal context preservation."""
        from llama_index.core.schema import Document

        try:
            # Step 1: Use LlamaIndex CodeSplitter for intelligent chunking
            splitter = self.code_splitters[language]
            document = Document(text=content)
            nodes = splitter.get_nodes_from_documents([document])

            chunks = []
            parser = self.parsers[language]

            for node in nodes:
                # Step 2: Extract semantic chunks from LlamaIndex nodes using Tree-sitter
                semantic_chunks = await self._extract_semantic_chunks_from_text(
                    node.text, str(file_path), language, parser
                )
                chunks.extend(semantic_chunks)

            logger.debug(f"Hybrid chunking created {len(chunks)} chunks for {file_path}")
            return chunks

        except Exception as e:
            logger.warning(f"Hybrid chunking failed for {file_path}: {e}, falling back to Tree-sitter")
            return await self._tree_sitter_chunking(file_path, content, language)

    async def _tree_sitter_chunking(self, file_path: Path, content: str, language: str) -> list[CodeChunk]:
        """Fallback Tree-sitter only chunking approach."""
        parser = self.parsers[language]
        tree = parser.parse(content.encode())

        # Extract semantic chunks from AST
        chunks = []
        await self._extract_chunks_from_node(tree.root_node, content, str(file_path), language, chunks)
        return chunks

    async def _get_semantic_type(self, chunk_content: str, parser: Parser, language: str) -> str:
        """Determine semantic type of a code chunk using Tree-sitter."""
        try:
            tree = parser.parse(chunk_content.encode())
            root_node = tree.root_node

            # Priority order for semantic types (higher priority = more specific)
            semantic_priority = {
                "class_definition": 10,
                "class_declaration": 10,
                "interface_declaration": 10,
                "function_definition": 8,
                "async_function_definition": 8,
                "function_declaration": 8,
                "method_definition": 7,
                "constructor_definition": 7,
                "struct_declaration": 6,
                "enum_declaration": 6,
                "type_alias_declaration": 6,
                "variable_declaration": 4,
                "const_declaration": 4,
                "let_declaration": 4,
                "import_statement": 2,
                "import_declaration": 2,
                "from_import_statement": 2,
            }

            def find_highest_priority_type(node: Node) -> tuple[str, int]:
                highest_type = "code_block"
                highest_priority = 0

                if node.type in semantic_priority:
                    priority = semantic_priority[node.type]
                    if priority > highest_priority:
                        highest_type = node.type
                        highest_priority = priority

                for child in node.children:
                    child_type, child_priority = find_highest_priority_type(child)
                    if child_priority > highest_priority:
                        highest_type = child_type
                        highest_priority = child_priority

                return highest_type, highest_priority

            semantic_type, _ = find_highest_priority_type(root_node)
            return semantic_type

        except Exception as e:
            logger.debug(f"Error determining semantic type: {e}")
            return "code_block"

    async def _extract_semantic_chunks_from_text(
        self, text: str, file_path: str, language: str, parser: Parser
    ) -> list[CodeChunk]:
        """Extract individual semantic chunks from LlamaIndex text using Tree-sitter."""
        try:
            tree = parser.parse(text.encode())
            chunks = []
            lines = text.split("\n")

            # Extract individual semantic constructs
            await self._extract_chunks_from_node(tree.root_node, text, file_path, language, chunks)

            # If no semantic chunks found, create a general code chunk
            if not chunks:
                # Extract file-level metadata for fallback chunk
                file_metadata = await self.metadata_extractor.extract_file_level_metadata(file_path, text)

                chunk = CodeChunk(
                    file_path=file_path,
                    content=text,
                    start_line=1,
                    end_line=len(lines),
                    language=language,
                    semantic_type="code_block",
                    embedding=None,
                    import_statements=file_metadata.get("imports"),
                    dependencies=file_metadata.get("dependencies"),
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.debug(f"Error extracting semantic chunks: {e}")
            # Fallback: create single chunk with basic metadata
            try:
                file_metadata = await self.metadata_extractor.extract_file_level_metadata(file_path, text)
                return [
                    CodeChunk(
                        file_path=file_path,
                        content=text,
                        start_line=1,
                        end_line=len(text.split("\n")),
                        language=language,
                        semantic_type="code_block",
                        embedding=None,
                        import_statements=file_metadata.get("imports"),
                        dependencies=file_metadata.get("dependencies"),
                    )
                ]
            except Exception:
                return [
                    CodeChunk(
                        file_path=file_path,
                        content=text,
                        start_line=1,
                        end_line=len(text.split("\n")),
                        language=language,
                        semantic_type="code_block",
                        embedding=None,
                    )
                ]

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
            ".yml": "yaml",
        }

        return extension_map.get(file_path.suffix.lower(), "text")

    async def _extract_chunks_from_node(
        self, node: Node, content: str, file_path: str, language: str, chunks: list[CodeChunk]
    ) -> None:
        """Extract semantic code chunks from Tree-sitter AST node."""
        lines = content.split("\n")

        # Define semantic node types to extract
        semantic_types = {
            "function_definition",
            "async_function_definition",
            "function_declaration",
            "class_definition",
            "class_declaration",
            "interface_declaration",
            "method_definition",
            "constructor_definition",
            "variable_declaration",
            "const_declaration",
            "let_declaration",
            "struct_declaration",
            "enum_declaration",
            "type_alias_declaration",
            "import_statement",
            "import_declaration",
            "from_import_statement",
        }

        if node.type in semantic_types:
            # Extract chunk content
            start_line = node.start_point[0]
            end_line = node.end_point[0]

            # Get content with some context
            context_start = max(0, start_line - 1)
            context_end = min(len(lines), end_line + 2)

            chunk_content = "\n".join(lines[context_start:context_end])

            # Extract rich metadata
            metadata = await self.metadata_extractor.extract_all_metadata(node, content, language)

            # Create enhanced code chunk
            chunk = CodeChunk(
                file_path=file_path,
                content=chunk_content,
                start_line=context_start + 1,  # Convert to 1-based line numbers
                end_line=context_end,
                language=language,
                semantic_type=metadata.semantic_type,
                embedding=None,  # Will be populated by SearchService
                function_signature=metadata.function_signature,
                class_name=metadata.class_name,
                function_name=metadata.function_name,
                parameter_types=metadata.parameter_types,
                return_type=metadata.return_type,
                inheritance_chain=metadata.inheritance_chain,
                import_statements=metadata.import_statements,
                docstring=metadata.docstring,
                complexity_score=metadata.complexity_score,
                dependencies=metadata.dependencies,
                interfaces=metadata.interfaces,
                decorators=metadata.decorators,
            )

            chunks.append(chunk)

        # Recursively process child nodes
        for child in node.children:
            await self._extract_chunks_from_node(child, content, file_path, language, chunks)

    async def _store_index_metadata(self, codebase_path: Path, chunks: list[CodeChunk]) -> None:
        """Store indexing metadata for incremental updates."""
        metadata = {
            "codebase_path": str(codebase_path),
            "total_chunks": len(chunks),
            "last_indexed": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "file_hashes": {},
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

    async def _update_index_metadata(
        self, codebase_path: Path, updated_files: list[Path], deleted_files: list[Path]
    ) -> None:
        """Update index metadata for incremental indexing."""
        metadata_path = self.config.INDEX_CACHE_DIR / "index_metadata.json"

        # Load existing metadata or create new
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {"file_hashes": {}}
        else:
            metadata = {"codebase_path": str(codebase_path), "file_hashes": {}}

        # Remove deleted files from metadata
        for deleted_file in deleted_files:
            metadata["file_hashes"].pop(str(deleted_file), None)

        # Update hashes for modified/new files
        for file_path in updated_files:
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    file_hash = hashlib.sha256(content.encode()).hexdigest()
                    metadata["file_hashes"][str(file_path)] = file_hash
                except Exception as e:
                    logger.warning(f"Error hashing file {file_path}: {e}")

        # Update metadata timestamp and totals
        metadata["last_indexed"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        metadata["total_chunks"] = len(metadata["file_hashes"])  # Approximate

        # Save updated metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    async def get_indexing_status(self, codebase_path: str) -> IndexingStatus:
        """Get current indexing status for a codebase."""
        validate_codebase_path(codebase_path)
        metadata_path = self.config.INDEX_CACHE_DIR / "index_metadata.json"

        if not metadata_path.exists():
            return IndexingStatus(is_indexed=False, total_files=0, total_chunks=0, index_size_mb=0.0)

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Calculate index size
            index_size = sum(f.stat().st_size for f in self.config.INDEX_CACHE_DIR.iterdir() if f.is_file()) / (
                1024 * 1024
            )  # Convert to MB

            return IndexingStatus(
                is_indexed=True,
                last_indexed=metadata.get("last_indexed"),
                total_files=len(metadata.get("file_hashes", {})),
                total_chunks=metadata.get("total_chunks", 0),
                index_size_mb=round(index_size, 2),
            )

        except Exception as e:
            logger.error(f"Error reading index metadata: {e}")
            return IndexingStatus(is_indexed=False, total_files=0, total_chunks=0, index_size_mb=0.0)

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

    async def get_changed_files(self, codebase_path: str) -> tuple[list[Path], list[Path], list[Path]]:
        """Detect which files have changed, been added, or deleted.

        Delegates to update_service for enhanced change detection.

        Returns:
            Tuple of (modified_files, new_files, deleted_files)
        """
        # Delegate to enhanced update service
        modified_strs, new_strs, deleted_strs = await self.update_service.detect_changes(codebase_path)

        # Convert strings back to Path objects for backward compatibility
        modified_files = [Path(f) for f in modified_strs]
        new_files = [Path(f) for f in new_strs]
        deleted_files = [Path(f) for f in deleted_strs]

        return modified_files, new_files, deleted_files
