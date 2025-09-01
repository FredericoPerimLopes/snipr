import json
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
    with patch("src.services.indexing_service.get_settings") as mock_settings:
        config = Mock()
        config.INDEX_CACHE_DIR = Path(tempfile.mkdtemp())
        config.MAX_FILE_SIZE_MB = 10
        config.DEFAULT_EXCLUDE_PATTERNS = ["**/.git/**", "**/node_modules/**"]
        config.SUPPORTED_LANGUAGES = ["python", "javascript"]
        mock_settings.return_value = config
        yield config


class TestIndexingService:

    @pytest.mark.asyncio
    async def test_index_codebase_success(self, indexing_service, temp_codebase, mock_config):
        """Test successful codebase indexing."""
        request = IndexingRequest(
            codebase_path=str(temp_codebase),
            languages=None,
            exclude_patterns=None
        )

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
        request = IndexingRequest(
            codebase_path=str(temp_codebase),
            languages=["python"],
            exclude_patterns=None
        )

        result = await indexing_service.index_codebase(request)

        assert result.status == "success"
        assert result.indexed_files >= 1
        assert "python" in result.languages_detected
        # Should not include javascript since filtered

    @pytest.mark.asyncio
    async def test_index_codebase_invalid_path(self, indexing_service):
        """Test indexing with invalid codebase path."""
        request = IndexingRequest(
            codebase_path="/nonexistent/path",
            languages=None,
            exclude_patterns=None
        )

        with pytest.raises(ValueError, match="Codebase path does not exist"):
            await indexing_service.index_codebase(request)

    @pytest.mark.asyncio
    async def test_discover_source_files(self, indexing_service, temp_codebase, mock_config):
        """Test source file discovery."""
        files = await indexing_service._discover_source_files(
            temp_codebase,
            None,
            ["**/.git/**"]
        )

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
            ["**/*.js"]  # Exclude JavaScript files
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
    async def test_store_and_retrieve_metadata(self, indexing_service, temp_codebase, mock_config):
        """Test storing and retrieving index metadata."""
        from ...models.indexing_models import CodeChunk

        # Create test chunks
        chunks = [
            CodeChunk(
                file_path=str(temp_codebase / "test.py"),
                content="def test(): pass",
                start_line=1,
                end_line=1,
                language="python",
                semantic_type="function_definition"
            )
        ]

        # Store metadata
        await indexing_service._store_index_metadata(temp_codebase, chunks)

        # Verify metadata file exists
        metadata_path = indexing_service.config.INDEX_CACHE_DIR / "index_metadata.json"
        assert metadata_path.exists()

        # Verify metadata content
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["codebase_path"] == str(temp_codebase)
        assert metadata["total_chunks"] == 1
        assert "last_indexed" in metadata
        assert "file_hashes" in metadata

    @pytest.mark.asyncio
    async def test_get_changed_files_no_previous_index(self, indexing_service, temp_codebase, mock_config):
        """Test changed file detection with no previous index."""
        # Ensure clean state - remove any existing metadata
        metadata_path = indexing_service.config.INDEX_CACHE_DIR / "index_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

        modified, new, deleted = await indexing_service.get_changed_files(str(temp_codebase))

        assert len(modified) == 0
        assert len(new) >= 2  # test.py and test.js
        assert len(deleted) == 0

    @pytest.mark.asyncio
    async def test_get_changed_files_with_modifications(self, indexing_service, temp_codebase, mock_config):
        """Test changed file detection with file modifications."""
        from ...models.indexing_models import CodeChunk

        # Create initial metadata
        chunks = [
            CodeChunk(
                file_path=str(temp_codebase / "test.py"),
                content="def test(): pass",
                start_line=1,
                end_line=1,
                language="python",
                semantic_type="function_definition"
            )
        ]
        await indexing_service._store_index_metadata(temp_codebase, chunks)

        # Modify the test file
        (temp_codebase / "test.py").write_text("def modified_test(): return 'changed'")

        # Add new file
        (temp_codebase / "new_file.py").write_text("def new_function(): pass")

        modified, new, deleted = await indexing_service.get_changed_files(str(temp_codebase))

        assert len(modified) == 1
        assert str(temp_codebase / "test.py") in [str(f) for f in modified]
        assert len(new) >= 1  # At least new_file.py
        assert len(deleted) == 0

    @pytest.mark.asyncio
    async def test_incremental_indexing_no_changes(self, indexing_service, temp_codebase, mock_config):
        """Test incremental indexing when no files have changed."""
        from ...models.indexing_models import IndexingRequest

        # Ensure clean state
        metadata_path = indexing_service.config.INDEX_CACHE_DIR / "index_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

        # First index
        request = IndexingRequest(
            codebase_path=str(temp_codebase),
            languages=None,
            exclude_patterns=None
        )

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

        # First index
        request = IndexingRequest(
            codebase_path=str(temp_codebase),
            languages=None,
            exclude_patterns=None
        )

        first_result = await indexing_service.index_codebase(request)
        assert first_result.status == "success"

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
        with patch('src.services.search_service.SearchService') as mock_search_service_class:
            mock_search_service = AsyncMock()
            mock_search_service.embed_code_chunks = AsyncMock(return_value=[])
            mock_search_service.remove_file_embeddings = AsyncMock()
            mock_search_service_class.return_value = mock_search_service
            
            request = IndexingRequest(
                codebase_path=str(temp_codebase),
                languages=None,
                exclude_patterns=None
            )
            
            result = await indexing_service.index_codebase(request)
            
            # Verify embedding generation was called
            assert result.status == "success"
            mock_search_service.embed_code_chunks.assert_called_once()
            # Verify chunks were passed to embedding generation
            call_args = mock_search_service.embed_code_chunks.call_args[0][0]
            assert len(call_args) > 0  # Should have chunks to embed

    @pytest.mark.asyncio  
    async def test_embedding_configuration_disabled(self, temp_codebase, mock_config):
        """Test that embedding generation can be disabled via configuration."""
        from unittest.mock import patch
        from ...models.indexing_models import IndexingRequest
        
        # Mock config with embeddings disabled
        mock_config.EMBEDDING_ENABLED = False
        
        with patch("src.services.indexing_service.get_settings") as mock_settings:
            mock_settings.return_value = mock_config
            service = IndexingService()
            
            # Ensure clean state
            metadata_path = service.config.INDEX_CACHE_DIR / "index_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            request = IndexingRequest(
                codebase_path=str(temp_codebase),
                languages=None,
                exclude_patterns=None
            )
            
            result = await service.index_codebase(request)
            
            # Should still work but without embeddings
            assert result.status == "success"
            assert result.indexed_files >= 2
