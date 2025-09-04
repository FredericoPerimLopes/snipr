import asyncio
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
from .logging_service import create_indexing_logger
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

        # Initialize indexing logger
        indexing_logger = create_indexing_logger()
        indexing_logger.log_phase_start("INDEXING_START", codebase_path=request.codebase_path)

        # Validate codebase path
        codebase_path = validate_codebase_path(request.codebase_path)
        indexing_logger.log_info("Codebase path validated", path=str(codebase_path))

        # Check if this is initial indexing or incremental update
        indexing_logger.log_phase_start("DATABASE_CHECK")
        import sqlite3

        conn = sqlite3.connect(str(self.config.VECTOR_DB_PATH))
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings_vec_metadata")
            existing_chunks = cursor.fetchone()[0] or 0
        except sqlite3.OperationalError:
            # Table doesn't exist yet, this is definitely initial indexing
            existing_chunks = 0
        finally:
            conn.close()
            
        is_initial_indexing = existing_chunks == 0
        indexing_logger.log_phase_end(
            "DATABASE_CHECK", 0.0, existing_chunks=existing_chunks, initial_indexing=is_initial_indexing
        )

        if is_initial_indexing:
            # Initial indexing: process all source files
            indexing_logger.log_phase_start("FILE_DISCOVERY", indexing_type="initial")
            logger.info("Performing initial full codebase indexing")
            all_source_files = await self._discover_source_files(
                codebase_path, request.languages, request.exclude_patterns or []
            )
            files_to_process = all_source_files
            modified_files, new_files, deleted_files = [], [str(f) for f in all_source_files], []
            indexing_logger.update_metric("files_discovered", len(all_source_files))
            indexing_logger.log_phase_end("FILE_DISCOVERY", 0.0, files_found=len(all_source_files))
        else:
            # Incremental indexing: use enhanced change detection
            indexing_logger.log_phase_start("CHANGE_DETECTION", indexing_type="incremental")
            modified_files, new_files, deleted_files = await self.update_service.detect_changes(str(codebase_path))
            files_to_process = [Path(f) for f in modified_files + new_files]
            indexing_logger.update_metric("files_discovered", len(files_to_process))
            indexing_logger.log_phase_end(
                "CHANGE_DETECTION", 0.0, modified=len(modified_files), new=len(new_files), deleted=len(deleted_files)
            )

        total_changed_files = len(files_to_process) + len(deleted_files)

        if total_changed_files == 0:
            indexing_logger.log_info("No file changes detected, skipping indexing")
            logger.info("No file changes detected, skipping indexing")

            # Get current status for response
            status = await self.get_indexing_status(str(codebase_path))
            indexing_logger.log_session_summary(result="up_to_date", total_chunks=status.total_chunks)
            indexing_logger.close()
            return IndexingResponse(
                indexed_files=0,
                total_chunks=status.total_chunks,
                processing_time_ms=0.0,
                languages_detected=[],
                status="up_to_date",
            )

        indexing_logger.log_info(
            "Starting file processing", modified=len(modified_files), new=len(new_files), deleted=len(deleted_files)
        )
        logger.info(
            f"Incremental indexing: {len(modified_files)} modified, {len(new_files)} new, "
            f"{len(deleted_files)} deleted files"
        )

        # Import SearchService for database cleanup
        from .search_service import SearchService

        search_service = SearchService()

        # Clean up deleted and modified files from database
        files_to_remove = deleted_files + modified_files
        if files_to_remove:
            indexing_logger.log_phase_start("DATABASE_CLEANUP")
            cleanup_start = time.time()
            await search_service.remove_file_embeddings(files_to_remove)
            cleanup_time = (time.time() - cleanup_start) * 1000
            indexing_logger.log_phase_end("DATABASE_CLEANUP", cleanup_time, files_removed=len(files_to_remove))

        # Parse only changed files
        indexing_logger.log_phase_start("FILE_PARSING", total_files=len(files_to_process))
        all_chunks: list[CodeChunk] = []
        languages_detected: set[str] = set()

        for i, file_path in enumerate(files_to_process, 1):
            file_start_time = time.time()
            try:
                chunks = await self._parse_file(file_path)
                file_processing_time = (time.time() - file_start_time) * 1000

                all_chunks.extend(chunks)
                languages_detected.update(chunk.language for chunk in chunks)

                # Log file processing success
                language = self._detect_language(file_path)
                indexing_logger.log_file_processed(
                    str(file_path), len(chunks), file_processing_time, language, progress=f"{i}/{len(files_to_process)}"
                )
            except Exception as e:
                file_processing_time = (time.time() - file_start_time) * 1000
                indexing_logger.log_file_failed(str(file_path), e, processing_time_ms=file_processing_time)
                logger.warning(f"Failed to parse {file_path}: {e}")
                continue

        indexing_logger.log_phase_end(
            "FILE_PARSING", 0.0, total_chunks=len(all_chunks), languages=list(languages_detected)
        )

        # Generate and store embeddings for new chunks
        if all_chunks:
            indexing_logger.log_phase_start("EMBEDDING_GENERATION", chunk_count=len(all_chunks))
            embedding_start = time.time()
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
            embedded_chunks = await search_service.embed_code_chunks(all_chunks, indexing_logger)
            embedding_time = (time.time() - embedding_start) * 1000
            indexing_logger.log_phase_end("EMBEDDING_GENERATION", embedding_time, chunks_embedded=len(embedded_chunks))
            logger.info(f"Successfully generated embeddings for {len(embedded_chunks)} chunks")

            # Update file records for processed files
            for file_path in files_to_process:
                file_path_str = str(file_path)
                file_chunks = [c for c in embedded_chunks if c.file_path == file_path_str]
                chunk_ids = [f"{c.file_path}:{c.start_line}" for c in file_chunks]
                content_hash = self.update_service.calculate_file_hash(file_path_str)
                dependencies = []

                # Extract dependencies from chunks
                for chunk in file_chunks:
                    if chunk.dependencies:
                        dependencies.extend(chunk.dependencies)

                await self.update_service.update_file_record(
                    file_path_str, content_hash, chunk_ids, list(set(dependencies))
                )

        # Build dependency graph for future incremental updates
        indexing_logger.log_phase_start("DEPENDENCY_GRAPH")
        dep_start = time.time()
        await self.update_service.build_dependency_graph(str(codebase_path))
        dep_time = (time.time() - dep_start) * 1000
        indexing_logger.log_phase_end("DEPENDENCY_GRAPH", dep_time)

        processing_time = (time.time() - start_time) * 1000

        # Log final session summary
        indexing_logger.log_session_summary(
            result="success",
            indexed_files=len(files_to_process),
            total_chunks=len(all_chunks),
            total_time_ms=processing_time,
            languages=list(languages_detected),
        )
        indexing_logger.close()

        return IndexingResponse(
            indexed_files=len(files_to_process),
            total_chunks=len(all_chunks),
            processing_time_ms=processing_time,
            languages_detected=list(languages_detected),
            status="success",
        )

    async def index_codebase_with_progress(self, request: IndexingRequest, task) -> IndexingResponse:
        """Index codebase with progress tracking for async operations."""
        from ..services.logging_service import create_indexing_logger
        from .task_registry import task_registry

        start_time = time.time()

        # Create indexing logger (will be no-op if logging disabled)
        indexing_logger = create_indexing_logger(f"progress_{task.task_id}")

        try:
            indexing_logger.log_info("=== PROGRESS-BASED INDEXING SESSION STARTED ===")
            indexing_logger.log_info("Validating codebase path", codebase_path=request.codebase_path)

            # Validate codebase path
            from ..config import validate_codebase_path

            codebase_path = validate_codebase_path(request.codebase_path)

            # Phase 1: Database Status Check
            indexing_logger.log_info("Phase 1: Checking database status")

            # Check if this is initial indexing or incremental update
            # Use vector database to determine if initial indexing is needed
            import sqlite3

            conn = sqlite3.connect(str(self.config.VECTOR_DB_PATH))
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings_vec_metadata")
                existing_chunks = cursor.fetchone()[0] or 0
            except sqlite3.OperationalError:
                # Table doesn't exist yet, this is definitely initial indexing
                existing_chunks = 0
            finally:
                conn.close()
            is_initial_indexing = existing_chunks == 0

            indexing_logger.log_info(
                "Database check complete", existing_chunks=existing_chunks, is_initial_indexing=is_initial_indexing
            )

            if is_initial_indexing:
                # Phase 2: File Discovery (Initial)
                indexing_logger.log_info("Phase 2: Discovering source files (initial indexing)")
                logger.info("Performing initial full codebase indexing")

                all_source_files = await self._discover_source_files(
                    codebase_path, request.languages, request.exclude_patterns or []
                )
                files_to_process = all_source_files
                modified_files, new_files, deleted_files = [], [str(f) for f in all_source_files], []

                indexing_logger.log_info("File discovery complete", total_discovered_files=len(all_source_files))
            else:
                # Phase 2: Change Detection (Incremental)
                indexing_logger.log_info("Phase 2: Detecting file changes (incremental indexing)")

                # Incremental indexing: use enhanced change detection
                modified_files, new_files, deleted_files = await self.update_service.detect_changes(str(codebase_path))
                files_to_process = [Path(f) for f in modified_files + new_files]

                indexing_logger.log_info(
                    "Change detection complete",
                    modified_files=len(modified_files),
                    new_files=len(new_files),
                    deleted_files=len(deleted_files),
                )

            # Update task progress with total files
            task.progress.total_files = len(files_to_process)
            task_registry.update_task(task)

            # Phase 3: Processing Validation
            indexing_logger.log_info("Phase 3: Validating files to process")

            total_changed_files = len(files_to_process) + len(deleted_files)

            if total_changed_files == 0:
                indexing_logger.log_info("No changes detected, skipping indexing")
                logger.info("No file changes detected, skipping indexing")
                status = await self.get_indexing_status(str(codebase_path))
                return IndexingResponse(
                    indexed_files=0,
                    total_chunks=status.total_chunks,
                    processing_time_ms=0.0,
                    languages_detected=[],
                    status="up_to_date",
                )

            indexing_logger.log_info(
                "Files validated for processing",
                files_to_process=len(files_to_process),
                files_to_remove=len(deleted_files + modified_files),
            )

            logger.info(f"Processing {len(files_to_process)} files with progress tracking")

            # Phase 4: Database Cleanup
            indexing_logger.log_info("Phase 4: Cleaning up database for modified/deleted files")

            # Import SearchService for database cleanup
            from .search_service import SearchService

            search_service = SearchService()

            # Clean up deleted and modified files from database
            files_to_remove = deleted_files + modified_files
            if files_to_remove:
                cleanup_start = time.time()
                await search_service.remove_file_embeddings(files_to_remove)
                cleanup_time = (time.time() - cleanup_start) * 1000
                indexing_logger.log_info(
                    "Database cleanup complete", files_removed=len(files_to_remove), cleanup_time_ms=cleanup_time
                )

            # Phase 5: File Processing Setup
            indexing_logger.log_info("Phase 5: Setting up file processing", total_files=len(files_to_process))

            # Process files with concurrency control and progress tracking
            all_chunks = []
            languages_detected = set()
            failed_files = []

            # Create semaphore for concurrent file processing
            max_concurrent = min(self.config.MAX_FILES_PER_BATCH // 4, 10)  # Conservative concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_file_with_progress(file_path: Path, index: int) -> list:
                """Process a single file with progress tracking."""
                async with semaphore:
                    file_start_time = time.time()
                    try:
                        # Update progress
                        task.progress.files_processed = index
                        task.progress.current_file = str(file_path)
                        task_registry.update_task(task)

                        chunks = await self._parse_file(file_path)
                        file_time = (time.time() - file_start_time) * 1000

                        if index % 10 == 0:  # Log every 10 files
                            logger.info(f"Processed {index}/{len(files_to_process)} files")
                            indexing_logger.log_info(
                                f"Progress checkpoint: {index}/{len(files_to_process)} files processed"
                            )

                        indexing_logger.log_info(
                            "File processed successfully",
                            file_path=str(file_path),
                            chunks_created=len(chunks),
                            processing_time_ms=file_time,
                        )
                        return chunks
                    except Exception as e:
                        file_time = (time.time() - file_start_time) * 1000
                        logger.warning(f"Failed to parse {file_path}: {e}")
                        indexing_logger.log_warning(
                            "File processing failed",
                            file_path=str(file_path),
                            error=str(e),
                            processing_time_ms=file_time,
                        )
                        failed_files.append(str(file_path))
                        return []

            # Phase 6: Concurrent File Processing
            indexing_logger.log_info("Phase 6: Processing files concurrently", max_concurrent=max_concurrent)

            processing_start = time.time()

            # Process files concurrently
            tasks_list = [process_file_with_progress(file_path, i) for i, file_path in enumerate(files_to_process, 1)]

            file_chunks_results = await asyncio.gather(*tasks_list, return_exceptions=True)

            # Collect results
            for chunks_result in file_chunks_results:
                if isinstance(chunks_result, list):
                    all_chunks.extend(chunks_result)
                    languages_detected.update(chunk.language for chunk in chunks_result)
                    task.progress.chunks_created = len(all_chunks)
                    task_registry.update_task(task)

            processing_time = (time.time() - processing_start) * 1000
            indexing_logger.log_info(
                "File processing complete",
                successful_files=len(files_to_process) - len(failed_files),
                failed_files=len(failed_files),
                total_chunks=len(all_chunks),
                processing_time_ms=processing_time,
            )

            # Phase 7: Embedding Generation
            if all_chunks:
                indexing_logger.log_info("Phase 7: Generating embeddings", chunks_to_embed=len(all_chunks))
                embedding_start = time.time()

                logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
                indexing_logger.log_info(f"Starting batch embedding generation for {len(all_chunks)} chunks")

                embedded_chunks = await search_service.embed_code_chunks(all_chunks, indexing_logger)
                embedding_time = (time.time() - embedding_start) * 1000

                logger.info(f"Successfully generated embeddings for {len(embedded_chunks)} chunks")
                indexing_logger.log_info(
                    "Embedding generation complete",
                    embedded_chunks=len(embedded_chunks),
                    embedding_time_ms=embedding_time,
                )
            else:
                embedded_chunks = []
                indexing_logger.log_info("No chunks to embed, skipping embedding generation")

            # Phase 8: Database Storage
            indexing_logger.log_info("Phase 8: Storing embeddings in database")

            # Update file records for processed files
            for file_path in files_to_process:
                file_path_str = str(file_path)
                file_chunks = [c for c in embedded_chunks if c.file_path == file_path_str]
                chunk_ids = [f"{c.file_path}:{c.start_line}" for c in file_chunks]
                content_hash = self.update_service.calculate_file_hash(file_path_str)
                dependencies = []

                # Extract dependencies from chunks
                for chunk in file_chunks:
                    if chunk.dependencies:
                        dependencies.extend(chunk.dependencies)

                await self.update_service.update_file_record(
                    file_path_str, content_hash, chunk_ids, list(set(dependencies))
                )

            # Phase 9: Dependency Graph Building
            indexing_logger.log_info("Phase 9: Building dependency graph")
            dependency_start = time.time()

            # Build dependency graph for future incremental updates
            await self.update_service.build_dependency_graph(str(codebase_path))

            dependency_time = (time.time() - dependency_start) * 1000
            indexing_logger.log_info("Dependency graph building complete", dependency_build_time_ms=dependency_time)

            # Final Summary
            processing_time = (time.time() - start_time) * 1000

            indexing_logger.log_info(
                "=== PROGRESS-BASED INDEXING SESSION SUMMARY ===",
                total_processing_time_ms=processing_time,
                files_processed=len(files_to_process),
                chunks_created=len(all_chunks),
                failed_files=len(failed_files),
                languages_detected=list(languages_detected),
            )

            return IndexingResponse(
                indexed_files=len(files_to_process),
                total_chunks=len(all_chunks),
                processing_time_ms=processing_time,
                languages_detected=list(languages_detected),
                status="success",
            )

        except Exception as e:
            indexing_logger.log_error("Critical progress-based indexing failure", error=e)
            raise
        finally:
            indexing_logger.close()

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
        relative_path = str(path.relative_to(path.anchor))

        # Check user-provided exclude patterns
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(relative_path, pattern):
                return True

        # Check against common patterns to exclude
        common_excludes = [
            "*.venv*",
            "*/.venv/*",
            "*/venv/*",
            "*/env/*",
            "*/__pycache__/*",
            "*/.pytest_cache/*",
            "*/.mypy_cache/*",
            "*/.git/*",
            "*/node_modules/*",
            "*/.tox/*",
            "*/.nox/*",
            "*/build/*",
            "*/dist/*",
            "*/.coverage/*",
            "*/htmlcov/*",
            "*/.idea/*",
            "*/.vscode/*",
            "*/logs/*",
        ]

        for pattern in common_excludes:
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(relative_path, pattern):
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

    async def get_indexing_status(self, codebase_path: str) -> IndexingStatus:
        """Get current indexing status for a codebase."""
        validate_codebase_path(codebase_path)

        try:
            # Query vector database directly for status
            from .search_service import SearchService

            search_service = SearchService()
            stats = await search_service.get_embeddings_stats()

            # Calculate index size from actual database files
            index_size = sum(f.stat().st_size for f in self.config.INDEX_CACHE_DIR.iterdir() if f.is_file()) / (
                1024 * 1024
            )  # Convert to MB

            # Get unique file count from vector database
            import sqlite3

            conn = sqlite3.connect(str(self.config.VECTOR_DB_PATH))
            try:
                cursor = conn.execute("SELECT COUNT(DISTINCT file_path) FROM embeddings_vec_metadata")
                total_files = cursor.fetchone()[0] or 0

                # Get last indexed timestamp from newest record
                cursor = conn.execute("SELECT MAX(created_at) FROM embeddings_vec_metadata")
                last_indexed_result = cursor.fetchone()
                last_indexed = last_indexed_result[0] if last_indexed_result and last_indexed_result[0] else None
            except sqlite3.OperationalError:
                # Table doesn't exist yet
                total_files = 0
                last_indexed = None
            finally:
                conn.close()

            is_indexed = stats["total_embeddings"] > 0

            return IndexingStatus(
                is_indexed=is_indexed,
                last_indexed=last_indexed,
                total_files=total_files,
                total_chunks=stats["total_embeddings"],
                index_size_mb=round(index_size, 2),
            )

        except Exception as e:
            logger.error(f"Error getting indexing status: {e}")
            return IndexingStatus(is_indexed=False, total_files=0, total_chunks=0, index_size_mb=0.0)

    async def needs_reindexing(self, codebase_path: str) -> bool:
        """Check if codebase needs reindexing based on file changes or new files."""
        try:
            # Check if vector database has any data
            import sqlite3

            conn = sqlite3.connect(str(self.config.VECTOR_DB_PATH))
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings_vec_metadata")
                total_chunks = cursor.fetchone()[0] or 0
            except sqlite3.OperationalError:
                # Table doesn't exist yet, need initial indexing
                total_chunks = 0
            finally:
                conn.close()

            # If no chunks exist, need initial indexing
            if total_chunks == 0:
                return True

            # Use update service to detect changes
            modified_files, new_files, deleted_files = await self.update_service.detect_changes(codebase_path)

            # Need reindexing if any files changed
            return len(modified_files) + len(new_files) + len(deleted_files) > 0

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
