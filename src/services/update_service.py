"""Incremental update service for MCP-aware code indexing."""

import hashlib
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

from ..config import get_settings
from ..models.indexing_models import (
    FileUpdateRecord,
    IndexUpdateResult,
)

logger = logging.getLogger(__name__)


class IncrementalUpdateService:
    """Service for detecting and processing incremental codebase changes."""

    def __init__(self):
        self.config = get_settings()
        self.file_records: dict[str, FileUpdateRecord] = {}
        self.dependency_graph: dict[str, list[str]] = {}

    async def detect_changes(self, codebase_path: str) -> tuple[list[str], list[str], list[str]]:
        """Detect changed, new, and deleted files using git-aware approach.
        Returns:
            Tuple of (modified_files, new_files, deleted_files)
        """
        codebase_path_obj = Path(codebase_path)

        # Try git-based detection first
        if self._is_git_repository(codebase_path_obj):
            return await self._detect_git_changes(codebase_path_obj)
        else:
            # Fallback to filesystem-based detection
            return await self._detect_filesystem_changes(codebase_path_obj)

    def _is_git_repository(self, path: Path) -> bool:
        """Check if path is a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    async def _detect_git_changes(self, codebase_path: Path) -> tuple[list[str], list[str], list[str]]:
        """Use git to detect changes efficiently."""
        try:
            # Get modified and new files from git status
            result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=codebase_path,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.warning(f"Git status failed: {result.stderr}")
                return await self._detect_filesystem_changes(codebase_path)

            modified_files = []
            new_files = []
            deleted_files = []

            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                status = line[:2]
                file_path = line[3:].strip()

                # Skip non-source files
                if not self._is_source_file(file_path):
                    continue

                full_path = str(codebase_path / file_path)

                if status.startswith('M') or status.startswith(' M'):
                    modified_files.append(full_path)
                elif status.startswith('A') or status.startswith('??'):
                    new_files.append(full_path)
                elif status.startswith('D'):
                    deleted_files.append(full_path)
                elif status.startswith('R'):  # Renamed
                    # For renames, treat as delete old + add new
                    parts = file_path.split(' -> ')
                    if len(parts) == 2:
                        deleted_files.append(str(codebase_path / parts[0]))
                        new_files.append(str(codebase_path / parts[1]))

            return modified_files, new_files, deleted_files

        except Exception as e:
            logger.warning(f"Git change detection failed: {e}")
            return await self._detect_filesystem_changes(codebase_path)

    async def _detect_filesystem_changes(self, codebase_path: Path) -> tuple[list[str], list[str], list[str]]:
        """Fallback filesystem-based change detection."""
        # Load stored file records
        await self._load_file_records(codebase_path)

        # Discover current files
        current_files = await self._discover_source_files(codebase_path)
        current_file_set = {str(f) for f in current_files}
        stored_file_set = set(self.file_records.keys())

        modified_files = []
        new_files = []
        deleted_files = list(stored_file_set - current_file_set)

        # Check for modifications and new files
        for file_path in current_files:
            file_path_str = str(file_path)

            if file_path_str in self.file_records:
                # Check if file was modified
                if await self._file_was_modified(file_path, self.file_records[file_path_str]):
                    modified_files.append(file_path_str)
            else:
                # New file
                new_files.append(file_path_str)

        return modified_files, new_files, deleted_files

    async def _file_was_modified(self, file_path: Path, record: FileUpdateRecord) -> bool:
        """Check if file was modified since last indexing."""
        try:
            # Quick mtime check first
            current_mtime = file_path.stat().st_mtime
            record_time = record.last_indexed.timestamp()

            if current_mtime <= record_time:
                return False

            # If mtime is newer, verify with content hash
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            current_hash = hashlib.sha256(content.encode()).hexdigest()

            return current_hash != record.content_hash

        except Exception as e:
            logger.warning(f"Error checking modification for {file_path}: {e}")
            return True  # Assume modified on error

    async def update_affected_chunks(self, changed_files: list[str]) -> IndexUpdateResult:
        """Update chunks affected by file changes, including dependencies."""
        start_time = time.time()

        # Find all files affected by changes (including dependencies)
        all_affected_files = set(changed_files)

        for changed_file in changed_files:
            # Add files that depend on the changed file
            dependents = self.dependency_graph.get(changed_file, [])
            all_affected_files.update(dependents)

        affected_files_list = list(all_affected_files)

        # For now, return structure - actual reindexing will be handled by IndexingService
        processing_time = (time.time() - start_time) * 1000

        return IndexUpdateResult(
            updated_chunks=[],  # Will be populated by IndexingService
            deleted_chunks=[],  # Will be populated by IndexingService
            affected_files=affected_files_list,
            processing_time=processing_time,
        )

    async def build_dependency_graph(self, codebase_path: str) -> None:
        """Build dependency graph from import/include statements."""
        codebase_path_obj = Path(codebase_path)
        source_files = await self._discover_source_files(codebase_path_obj)

        self.dependency_graph.clear()

        for file_path in source_files:
            try:
                dependencies = await self._extract_file_dependencies(file_path)

                # Convert relative imports to absolute paths
                resolved_deps = []
                for dep in dependencies:
                    resolved_path = self._resolve_dependency_path(dep, file_path, codebase_path_obj)
                    if resolved_path:
                        resolved_deps.append(resolved_path)

                # Build reverse dependency graph
                file_path_str = str(file_path)
                for dep_path in resolved_deps:
                    if dep_path not in self.dependency_graph:
                        self.dependency_graph[dep_path] = []
                    if file_path_str not in self.dependency_graph[dep_path]:
                        self.dependency_graph[dep_path].append(file_path_str)

            except Exception as e:
                logger.warning(f"Error building dependencies for {file_path}: {e}")

    async def _extract_file_dependencies(self, file_path: Path) -> list[str]:
        """Extract import/include dependencies from a file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            language = self._detect_language(file_path)

            dependencies = []

            if language == "python":
                dependencies.extend(self._extract_python_imports(content))
            elif language in ["javascript", "typescript"]:
                dependencies.extend(self._extract_js_imports(content))
            elif language == "go":
                dependencies.extend(self._extract_go_imports(content))
            elif language in ["c", "cpp"]:
                dependencies.extend(self._extract_c_includes(content))

            return dependencies

        except Exception as e:
            logger.debug(f"Error extracting dependencies from {file_path}: {e}")
            return []

    def _extract_python_imports(self, content: str) -> list[str]:
        """Extract Python import statements."""
        import re

        imports = []

        # Match: import module, from module import ...
        import_patterns = [
            r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)',
            r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import',
        ]

        for line in content.split('\n'):
            line = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module = match.group(1)
                    if not module.startswith('.'):  # Skip relative imports for now
                        imports.append(module)

        return imports

    def _extract_js_imports(self, content: str) -> list[str]:
        """Extract JavaScript/TypeScript import statements."""
        import re

        imports = []

        # Match: import ... from 'module', require('module')
        patterns = [
            r'''import.*?from\s+['"`]([^'"`]+)['"`]''',
            r'''require\s*\(\s*['"`]([^'"`]+)['"`]\s*\)''',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            imports.extend(matches)

        return imports

    def _extract_go_imports(self, content: str) -> list[str]:
        """Extract Go import statements."""
        import re

        imports = []

        # Match Go import blocks and single imports
        import_block_pattern = r'import\s*\(\s*(.*?)\s*\)'
        single_import_pattern = r'import\s+"([^"]+)"'

        # Find import blocks
        for match in re.finditer(import_block_pattern, content, re.DOTALL):
            block_content = match.group(1)
            # Extract individual imports from block
            for line in block_content.split('\n'):
                line = line.strip()
                if line and line.startswith('"') and line.endswith('"'):
                    imports.append(line[1:-1])

        # Find single imports
        for match in re.finditer(single_import_pattern, content):
            imports.append(match.group(1))

        return imports

    def _extract_c_includes(self, content: str) -> list[str]:
        """Extract C/C++ include statements."""
        import re

        includes = []

        # Match #include "file.h" and #include <file.h>
        patterns = [
            r'#include\s*"([^"]+)"',
            r'#include\s*<([^>]+)>',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            includes.extend(matches)

        return includes

    def _resolve_dependency_path(self, dependency: str, source_file: Path, codebase_path: Path) -> str | None:
        """Resolve dependency to actual file path."""
        # For now, simplified resolution - can be enhanced later
        if dependency.startswith('.'):
            # Relative import
            return None  # Skip for now

        # Try to find the dependency as a file in the codebase
        # This is a simplified approach - real implementation would need language-specific logic
        possible_paths = [
            codebase_path / f"{dependency.replace('.', '/')}.py",  # Python
            codebase_path / f"{dependency}.py",
            codebase_path / f"{dependency}.js",  # JavaScript
            codebase_path / f"{dependency}.ts",  # TypeScript
            codebase_path / f"{dependency}.go",  # Go
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        return None

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
        }

        return extension_map.get(file_path.suffix.lower(), "text")

    def _is_source_file(self, file_path: str) -> bool:
        """Check if file is a source code file."""
        path = Path(file_path)
        language = self._detect_language(path)
        return language != "text"

    async def _discover_source_files(self, codebase_path: Path) -> list[Path]:
        """Discover source files in codebase."""
        source_files = []

        # Use config exclude patterns
        exclude_patterns = self.config.DEFAULT_EXCLUDE_PATTERNS

        import os

        for root, dirs, files in os.walk(codebase_path):
            root_path = Path(root)

            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude_path(root_path / d, exclude_patterns)]

            for file in files:
                file_path = root_path / file

                if (
                    self._is_source_file(str(file_path))
                    and not self._should_exclude_path(file_path, exclude_patterns)
                    and file_path.stat().st_size < self.config.MAX_FILE_SIZE_MB * 1024 * 1024
                ):
                    source_files.append(file_path)

        return source_files

    def _should_exclude_path(self, path: Path, exclude_patterns: list[str]) -> bool:
        """Check if path should be excluded."""
        import fnmatch

        path_str = str(path)
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
        return False

    async def _load_file_records(self, codebase_path: Path) -> None:
        """Load file records from metadata storage."""
        metadata_file = self.config.INDEX_CACHE_DIR / "file_records.json"

        if not metadata_file.exists():
            self.file_records = {}
            return

        try:
            import json

            with open(metadata_file) as f:
                data = json.load(f)

            self.file_records = {}
            for file_path, record_data in data.items():
                # Convert datetime string back to datetime object
                record_data["last_indexed"] = datetime.fromisoformat(record_data["last_indexed"])
                self.file_records[file_path] = FileUpdateRecord.model_validate(record_data)

        except Exception as e:
            logger.warning(f"Error loading file records: {e}")
            self.file_records = {}

    async def _save_file_records(self, codebase_path: Path) -> None:
        """Save file records to metadata storage."""
        metadata_file = self.config.INDEX_CACHE_DIR / "file_records.json"

        try:
            import json

            # Convert records to JSON-serializable format
            data = {}
            for file_path, record in self.file_records.items():
                record_dict = record.model_dump()
                # Convert datetime to ISO string
                record_dict["last_indexed"] = record.last_indexed.isoformat()
                data[file_path] = record_dict

            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving file records: {e}")

    async def update_file_record(
        self,
        file_path: str,
        content_hash: str,
        chunk_ids: list[str],
        dependencies: list[str]
    ) -> None:
        """Update record for a processed file."""
        record = FileUpdateRecord(
            file_path=file_path,
            content_hash=content_hash,
            last_indexed=datetime.now(),
            chunk_ids=chunk_ids,
            dependencies=dependencies,
        )

        self.file_records[file_path] = record

    def get_affected_files(self, changed_file: str) -> list[str]:
        """Get files that depend on the changed file."""
        return self.dependency_graph.get(changed_file, [])

    async def should_update_index(self, codebase_path: str) -> bool:
        """Check if index needs updating based on recent changes."""
        try:
            modified, new, deleted = await self.detect_changes(codebase_path)
            return len(modified) + len(new) + len(deleted) > 0
        except Exception as e:
            logger.warning(f"Error checking if update needed: {e}")
            return True  # Assume update needed on error

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file content."""
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            return hashlib.sha256(content.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Error calculating hash for {file_path}: {e}")
            return ""
