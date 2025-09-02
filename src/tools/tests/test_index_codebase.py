import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ...models.indexing_models import IndexingResponse, IndexingStatus
from ...tools.index_codebase import clear_index, get_indexing_status, index_codebase


@pytest.fixture
def temp_codebase():
    """Create temporary codebase for testing."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create test files
    (temp_dir / "test.py").write_text("def hello(): return 'world'")
    (temp_dir / "test.js").write_text("function greet() { return 'hello'; }")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_indexing_service():
    """Mock IndexingService for testing."""
    with patch("src.tools.index_codebase.indexing_service") as mock:
        mock.index_codebase = AsyncMock()
        mock.get_indexing_status = AsyncMock()
        mock.needs_reindexing = AsyncMock()
        yield mock


@pytest.fixture
def mock_search_service():
    """Mock SearchService for testing."""
    with patch("src.tools.index_codebase.search_service") as mock:
        mock.get_embeddings_stats = AsyncMock()
        yield mock


class TestIndexCodebaseTool:
    @pytest.mark.asyncio
    async def test_index_codebase_success(self, temp_codebase, mock_indexing_service, mock_search_service):
        """Test successful codebase indexing."""
        # Setup mocks
        mock_indexing_service.needs_reindexing.return_value = True
        mock_indexing_service.index_codebase.return_value = IndexingResponse(
            indexed_files=2,
            total_chunks=4,
            processing_time_ms=150.5,
            languages_detected=["python", "javascript"],
            status="success",
        )
        mock_indexing_service.get_indexing_status.return_value = IndexingStatus(
            is_indexed=True, last_indexed="2025-01-01T00:00:00", total_files=2, total_chunks=4, index_size_mb=1.2
        )
        mock_search_service.get_embeddings_stats.return_value = {
            "total_embeddings": 4,
            "languages": {"python": 2, "javascript": 2},
        }

        # Execute tool
        result_json = await index_codebase(str(temp_codebase))
        result = json.loads(result_json)

        # Verify response
        assert result["status"] == "success"
        assert "indexing_result" in result
        assert "indexing_status" in result
        assert "embeddings_stats" in result
        assert result["indexing_result"]["indexed_files"] == 2
        assert result["indexing_result"]["total_chunks"] == 4

    @pytest.mark.asyncio
    async def test_index_codebase_already_indexed(self, temp_codebase, mock_indexing_service, mock_search_service):
        """Test indexing when codebase is already up to date."""
        # Setup mocks
        mock_indexing_service.needs_reindexing.return_value = False
        mock_indexing_service.get_indexing_status.return_value = IndexingStatus(
            is_indexed=True, last_indexed="2025-01-01T00:00:00", total_files=2, total_chunks=4, index_size_mb=1.2
        )

        # Execute tool
        result_json = await index_codebase(str(temp_codebase))
        result = json.loads(result_json)

        # Verify response
        assert result["status"] == "already_indexed"
        assert "indexing_status" in result
        assert "already up to date" in result["message"]

    @pytest.mark.asyncio
    async def test_index_codebase_with_language_filter(self, temp_codebase, mock_indexing_service, mock_search_service):
        """Test indexing with language filtering."""
        # Setup mocks
        mock_indexing_service.needs_reindexing.return_value = True
        mock_indexing_service.index_codebase.return_value = IndexingResponse(
            indexed_files=1, total_chunks=2, processing_time_ms=100.0, languages_detected=["python"], status="success"
        )
        mock_indexing_service.get_indexing_status.return_value = IndexingStatus(
            is_indexed=True, total_files=1, total_chunks=2, index_size_mb=0.5
        )
        mock_search_service.get_embeddings_stats.return_value = {"total_embeddings": 2}

        # Execute tool with language filter
        result_json = await index_codebase(str(temp_codebase), languages="python", exclude_patterns="**/*.js")
        result = json.loads(result_json)

        # Verify response
        assert result["status"] == "success"
        assert result["indexing_result"]["languages_detected"] == ["python"]

    @pytest.mark.asyncio
    async def test_index_codebase_invalid_path(self, mock_indexing_service, mock_search_service):
        """Test indexing with invalid path."""
        # Setup mock to raise exception
        mock_indexing_service.needs_reindexing.side_effect = ValueError("Codebase path does not exist")

        # Execute tool
        result_json = await index_codebase("/nonexistent/path")
        result = json.loads(result_json)

        # Verify error response
        assert result["status"] == "error"
        assert "does not exist" in result["message"]
        assert result["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_get_indexing_status_success(self, temp_codebase, mock_indexing_service, mock_search_service):
        """Test getting indexing status."""
        # Setup mocks
        mock_indexing_service.get_indexing_status.return_value = IndexingStatus(
            is_indexed=True, last_indexed="2025-01-01T00:00:00", total_files=5, total_chunks=20, index_size_mb=2.5
        )
        mock_search_service.get_embeddings_stats.return_value = {
            "total_embeddings": 20,
            "languages": {"python": 15, "javascript": 5},
        }

        # Execute tool
        result_json = await get_indexing_status(str(temp_codebase))
        result = json.loads(result_json)

        # Verify response
        assert result["status"] == "success"
        assert result["indexing_status"]["is_indexed"] is True
        assert result["indexing_status"]["total_files"] == 5
        assert result["embeddings_stats"]["total_embeddings"] == 20

    @pytest.mark.asyncio
    async def test_clear_index_success(self, mock_indexing_service):
        """Test clearing index successfully."""
        # Mock config
        mock_config = Mock()
        mock_config.INDEX_CACHE_DIR = Path(tempfile.mkdtemp())
        mock_indexing_service.config = mock_config

        # Create some test files in cache directory
        test_file = mock_config.INDEX_CACHE_DIR / "test.db"
        test_file.write_text("test data")

        # Execute tool
        result_json = await clear_index()
        result = json.loads(result_json)

        # Verify response
        assert result["status"] == "success"
        assert "cleared" in result["message"]

        # Verify cache directory was recreated empty
        assert mock_config.INDEX_CACHE_DIR.exists()
        assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_index_codebase_json_response_format(self, temp_codebase, mock_indexing_service, mock_search_service):
        """Test that tool returns valid JSON string."""
        # Setup mocks
        mock_indexing_service.needs_reindexing.return_value = True
        mock_indexing_service.index_codebase.return_value = IndexingResponse(
            indexed_files=1, total_chunks=1, processing_time_ms=50.0, languages_detected=["python"], status="success"
        )
        mock_indexing_service.get_indexing_status.return_value = IndexingStatus(
            is_indexed=True, total_files=1, total_chunks=1, index_size_mb=0.1
        )
        mock_search_service.get_embeddings_stats.return_value = {"total_embeddings": 1}

        # Execute tool
        result_json = await index_codebase(str(temp_codebase))

        # Verify it's valid JSON
        result = json.loads(result_json)  # Should not raise exception
        assert isinstance(result, dict)
        assert "status" in result

        # Verify it's a string (MCP requirement)
        assert isinstance(result_json, str)
