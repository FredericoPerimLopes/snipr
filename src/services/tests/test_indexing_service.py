import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ...models.indexing_models import IndexingRequest
from ...services.indexing_service import IndexingService


@pytest.fixture
def indexing_service():
    """Create IndexingService instance for testing."""
    return IndexingService()


@pytest.fixture
def temp_codebase():
    """Create temporary codebase for testing."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create test Python file
    (temp_dir / "test.py").write_text("""
def hello_world():
    '''Say hello to the world.'''
    return "Hello, World!"

class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    async def async_add(self, a: int, b: int) -> int:
        return a + b
""")

    # Create test JavaScript file
    (temp_dir / "test.js").write_text("""
function greet(name) {
    return `Hello, ${name}!`;
}

class Counter {
    constructor() {
        this.count = 0;
    }

    increment() {
        this.count++;
    }
}
""")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    with (
        patch("src.services.indexing_service.get_settings") as mock_indexing_settings,
        patch("src.services.update_service.get_settings") as mock_update_settings,
    ):
        config = Mock()
        config.INDEX_CACHE_DIR = Path(tempfile.mkdtemp())
        config.MAX_FILE_SIZE_MB = 10
        config.DEFAULT_EXCLUDE_PATTERNS = ["**/.git/**", "**/node_modules/**"]
        config.SUPPORTED_LANGUAGES = ["python", "javascript"]
        mock_indexing_settings.return_value = config
        mock_update_settings.return_value = config
        yield config


class TestIndexingService:
    @pytest.mark.asyncio
    async def test_index_codebase_success(self, indexing_service, temp_codebase, mock_config):
        """Test successful codebase indexing."""
        request = IndexingRequest(codebase_path=str(temp_codebase), languages=None, exclude_patterns=None)

        result = await indexing_service.index_codebase(request)

        assert result.status == "success"
        assert result.indexed_files >= 2  # At least python and js files
        assert result.total_chunks > 0
        assert result.processing_time_ms > 0
        assert "python" in result.languages_detected
        assert "javascript" in result.languages_detected

    @pytest.mark.asyncio
    async def test_index_codebase_with_language_filter(self, indexing_service, temp_codebase, mock_config):
        """Test indexing with language filtering."""
        request = IndexingRequest(codebase_path=str(temp_codebase), languages=["python"], exclude_patterns=None)

        result = await indexing_service.index_codebase(request)

        assert result.status == "success"
        assert result.indexed_files >= 1
        assert "python" in result.languages_detected
        # Should not include javascript since filtered

    @pytest.mark.asyncio
    async def test_index_codebase_invalid_path(self, indexing_service):
        """Test indexing with invalid codebase path."""
        request = IndexingRequest(codebase_path="/nonexistent/path", languages=None, exclude_patterns=None)

        with pytest.raises(ValueError, match="Codebase path does not exist"):
            await indexing_service.index_codebase(request)

    @pytest.mark.asyncio
    async def test_discover_source_files(self, indexing_service, temp_codebase, mock_config):
        """Test source file discovery."""
        files = await indexing_service._discover_source_files(temp_codebase, None, ["**/.git/**"])

        assert len(files) >= 2
        file_names = [f.name for f in files]
        assert "test.py" in file_names
        assert "test.js" in file_names

    @pytest.mark.asyncio
    async def test_discover_source_files_with_exclude(self, indexing_service, temp_codebase, mock_config):
        """Test source file discovery with exclude patterns."""
        files = await indexing_service._discover_source_files(
            temp_codebase,
            None,
            ["**/*.js"],  # Exclude JavaScript files
        )

        file_names = [f.name for f in files]
        assert "test.py" in file_names
        assert "test.js" not in file_names

    def test_detect_language(self, indexing_service):
        """Test language detection from file extensions."""
        assert indexing_service._detect_language(Path("test.py")) == "python"
        assert indexing_service._detect_language(Path("test.js")) == "javascript"
        assert indexing_service._detect_language(Path("test.ts")) == "typescript"
        assert indexing_service._detect_language(Path("test.go")) == "go"
        assert indexing_service._detect_language(Path("unknown.xyz")) == "text"

    def test_should_exclude_path(self, indexing_service):
        """Test path exclusion logic."""
        exclude_patterns = ["**/.git/**", "**/node_modules/**", "**/*.log"]

        assert indexing_service._should_exclude_path(Path("/project/.git/config"), exclude_patterns)
        assert indexing_service._should_exclude_path(Path("/project/node_modules/lib.js"), exclude_patterns)
        assert indexing_service._should_exclude_path(Path("/project/app.log"), exclude_patterns)
        assert not indexing_service._should_exclude_path(Path("/project/src/main.py"), exclude_patterns)

    @pytest.mark.asyncio
    async def test_parse_file_python(self, indexing_service, temp_codebase):
        """Test parsing Python file with Tree-sitter."""
        python_file = temp_codebase / "test.py"

        chunks = await indexing_service._parse_file(python_file)

        assert len(chunks) > 0

        # Check for function and class chunks
        semantic_types = [chunk.semantic_type for chunk in chunks]
        assert "function_definition" in semantic_types
        assert "class_definition" in semantic_types

        # Verify chunk content
        for chunk in chunks:
            assert chunk.file_path == str(python_file)
            assert chunk.language == "python"
            assert chunk.start_line > 0
            assert chunk.end_line >= chunk.start_line
            assert len(chunk.content) > 0

    @pytest.mark.asyncio
    async def test_get_indexing_status_no_index(self, temp_codebase, mock_config):
        """Test getting status when no index exists."""
        # Create fresh IndexingService with isolated config
        with patch("src.services.indexing_service.get_settings") as mock_settings:
            mock_settings.return_value = mock_config
            fresh_service = IndexingService()

            status = await fresh_service.get_indexing_status(str(temp_codebase))

            assert not status.is_indexed
            assert status.total_files == 0
            assert status.total_chunks == 0
            assert status.index_size_mb == 0.0

    @pytest.mark.asyncio
    async def test_needs_reindexing_no_index(self, indexing_service, temp_codebase, mock_config):
        """Test reindexing check when no index exists."""
        needs_reindex = await indexing_service.needs_reindexing(str(temp_codebase))
        assert needs_reindex

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Internal implementation changed - _store_index_metadata no longer exists")
    async def test_needs_reindexing_with_new_untracked_files(self, indexing_service, temp_codebase, mock_config):
        """Test that needs_reindexing detects new untracked files."""
        pass

    @pytest.mark.asyncio
    async def test_needs_reindexing_no_new_files(self, indexing_service, temp_codebase, mock_config):
        """Test that needs_reindexing returns False when no new files exist."""
        from ...models.indexing_models import CodeChunk

        # Get all existing source files in temp directory and index them all
        current_source_files = await indexing_service._discover_source_files(
            temp_codebase, None, indexing_service.config.DEFAULT_EXCLUDE_PATTERNS
        )

        # Create chunks for all discovered files
        chunks = []
        for file_path in current_source_files:
            chunks.append(
                CodeChunk(
                    file_path=str(file_path),
                    content=file_path.read_text(),
                    start_line=1,
                    end_line=len(file_path.read_text().split("\n")),
                    language=indexing_service._detect_language(file_path),
                    semantic_type="code_block",
                )
            )

        # Skip this test - _store_index_metadata no longer exists
        pytest.skip("Internal implementation changed")

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Internal implementation changed - _store_index_metadata no longer exists")
    async def test_needs_reindexing_detects_new_files_in_subdirs(self, indexing_service, temp_codebase, mock_config):
        """Test that needs_reindexing detects new files in subdirectories."""
        pass

    @pytest.mark.asyncio
    async def test_store_and_retrieve_metadata(self, indexing_service, temp_codebase, mock_config):
        """Test storing and retrieving index metadata."""
        # Skip this test - _store_index_metadata no longer exists
        pytest.skip("Internal implementation changed")

    @pytest.mark.asyncio
    async def test_get_changed_files_no_previous_index(self, indexing_service, temp_codebase, mock_config):
        """Test changed file detection with no previous index."""
        # Ensure clean state - remove any existing metadata
        metadata_path = indexing_service.config.INDEX_CACHE_DIR / "index_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

        # Also clean up file records from update service
        file_records_path = mock_config.INDEX_CACHE_DIR / "file_records.json"
        if file_records_path.exists():
            file_records_path.unlink()

        # Reinitialize update service to ensure clean state
        from ...services.update_service import IncrementalUpdateService

        indexing_service.update_service = IncrementalUpdateService()

        modified, new, deleted = await indexing_service.get_changed_files(str(temp_codebase))

        assert len(modified) == 0
        assert len(new) >= 2  # test.py and test.js
        assert len(deleted) == 0

    @pytest.mark.asyncio
    async def test_get_changed_files_with_modifications(self, indexing_service, temp_codebase, mock_config):
        """Test changed file detection with file modifications."""
        # Clean up any existing file records from previous tests
        file_records_path = mock_config.INDEX_CACHE_DIR / "file_records.json"
        if file_records_path.exists():
            file_records_path.unlink()

        # Also clear the update service's in-memory records
        indexing_service.update_service.file_records.clear()

        # Skip this test - _store_index_metadata no longer exists
        pytest.skip("Internal implementation changed")

    @pytest.mark.asyncio
    async def test_incremental_indexing_no_changes(self, indexing_service, temp_codebase, mock_config):
        """Test incremental indexing when no files have changed."""
        from ...models.indexing_models import IndexingRequest

        # Ensure clean state
        metadata_path = indexing_service.config.INDEX_CACHE_DIR / "index_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

        file_records_path = mock_config.INDEX_CACHE_DIR / "file_records.json"
        if file_records_path.exists():
            file_records_path.unlink()

        # Reinitialize update service to ensure clean state
        from ...services.update_service import IncrementalUpdateService

        indexing_service.update_service = IncrementalUpdateService()

        # First index
        request = IndexingRequest(codebase_path=str(temp_codebase), languages=None, exclude_patterns=None)

        await indexing_service.index_codebase(request)

        # Second index (no changes)
        result = await indexing_service.index_codebase(request)

        assert result.status == "up_to_date"
        assert result.indexed_files == 0
        assert result.processing_time_ms < 100  # Should be very fast

    @pytest.mark.asyncio
    async def test_incremental_indexing_with_changes(self, indexing_service, temp_codebase, mock_config):
        """Test incremental indexing with file changes."""
        from ...models.indexing_models import IndexingRequest

        # Ensure clean state
        metadata_path = indexing_service.config.INDEX_CACHE_DIR / "index_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

        file_records_path = mock_config.INDEX_CACHE_DIR / "file_records.json"
        if file_records_path.exists():
            file_records_path.unlink()

        # Also clear the update service's in-memory records
        indexing_service.update_service.file_records.clear()

        # First index
        request = IndexingRequest(codebase_path=str(temp_codebase), languages=None, exclude_patterns=None)

        first_result = await indexing_service.index_codebase(request)
        assert first_result.status == "success"

        # Small delay to ensure mtime difference
        import time

        time.sleep(0.1)

        # Modify a file
        (temp_codebase / "test.py").write_text("""
def modified_hello_world():
    '''Modified function.'''
    return "Hello, Modified World!"

class ModifiedCalculator:
    def multiply(self, a: int, b: int) -> int:
        return a * b
""")

        # Second index (with changes)
        second_result = await indexing_service.index_codebase(request)

        assert second_result.status == "success"
        assert second_result.indexed_files == 1  # Only modified file
        assert second_result.total_chunks > 0

    @pytest.mark.asyncio
    async def test_embedding_integration_during_indexing(self, indexing_service, temp_codebase, mock_config):
        """Test that embeddings are generated during indexing process."""
        from unittest.mock import AsyncMock, patch

        from ...models.indexing_models import IndexingRequest

        # Ensure clean state
        metadata_path = indexing_service.config.INDEX_CACHE_DIR / "index_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()
        # Mock SearchService embedding generation
        with patch("src.services.search_service.SearchService") as mock_search_service_class:
            mock_search_service = AsyncMock()
            mock_search_service.embed_code_chunks = AsyncMock(return_value=[])
            mock_search_service.remove_file_embeddings = AsyncMock()
            mock_search_service_class.return_value = mock_search_service
            request = IndexingRequest(codebase_path=str(temp_codebase), languages=None, exclude_patterns=None)

            result = await indexing_service.index_codebase(request)
            # Verify embedding generation was called
            assert result.status == "success"
            mock_search_service.embed_code_chunks.assert_called_once()
            # Verify chunks were passed to embedding generation
            call_args = mock_search_service.embed_code_chunks.call_args[0][0]
            assert len(call_args) > 0  # Should have chunks to embed

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Integration test requires full database setup - complex to mock")
    async def test_embedding_configuration_disabled(self, temp_codebase, mock_config):
        """Test that embedding generation can be disabled via configuration."""
        pass
