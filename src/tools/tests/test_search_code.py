import json
from unittest.mock import AsyncMock, patch

import pytest

from ...models.indexing_models import CodeChunk, SearchResponse
from ...tools.search_code import (
    get_search_stats,
    search_by_type,
    search_code,
    search_in_file,
)


@pytest.fixture
def mock_search_service():
    """Mock SearchService for testing."""
    with patch("src.tools.search_code.search_service") as mock:
        mock.search_code = AsyncMock()
        mock.search_by_keywords = AsyncMock()
        mock.get_embeddings_stats = AsyncMock()
        yield mock


@pytest.fixture
def sample_search_response():
    """Create sample search response for testing."""
    return SearchResponse(
        results=[
            CodeChunk(
                file_path="/test/file.py",
                content="def hello_world():\n    return 'Hello, World!'",
                start_line=1,
                end_line=2,
                language="python",
                semantic_type="function_definition",
                embedding=[0.1, 0.2, 0.3]
            ),
            CodeChunk(
                file_path="/test/utils.py",
                content=(
                    "class Logger:\n    def log(self, message):\n        print(message)"
                ),
                start_line=5,
                end_line=7,
                language="python",
                semantic_type="class_definition",
                embedding=[0.4, 0.5, 0.6]
            )
        ],
        total_matches=2,
        query_time_ms=125.5
    )


class TestSearchCodeTool:

    @pytest.mark.asyncio
    async def test_search_code_success(
        self, mock_search_service, sample_search_response
    ):
        """Test successful code search."""
        # Setup mock
        mock_search_service.search_code.return_value = sample_search_response

        # Execute tool
        result_json = await search_code("hello world function", "python", 10, 0.7)
        result = json.loads(result_json)

        # Verify response format
        assert result["status"] == "success"
        assert "search_result" in result
        assert result["query"] == "hello world function"
        assert result["filters"]["language"] == "python"
        assert result["filters"]["max_results"] == 10
        assert result["filters"]["similarity_threshold"] == 0.7

        # Verify search result data
        search_result = result["search_result"]
        assert search_result["total_matches"] == 2
        assert search_result["query_time_ms"] == 125.5
        assert len(search_result["results"]) == 2

    @pytest.mark.asyncio
    async def test_search_code_parameter_validation(
        self, mock_search_service, sample_search_response
    ):
        """Test parameter validation and clamping."""
        mock_search_service.search_code.return_value = sample_search_response

        # Test with out-of-range parameters
        # max_results > 100, threshold > 1.0
        result_json = await search_code("test", "python", 150, 1.5)
        result = json.loads(result_json)

        # Verify parameters were clamped
        assert result["filters"]["max_results"] == 100  # Clamped to max
        assert result["filters"]["similarity_threshold"] == 1.0  # Clamped to max

        # Test with negative parameters
        result_json = await search_code("test", "python", -5, -0.5)
        result = json.loads(result_json)

        # Verify parameters were clamped
        assert result["filters"]["max_results"] == 1  # Clamped to min
        assert result["filters"]["similarity_threshold"] == 0.0  # Clamped to min

    @pytest.mark.asyncio
    async def test_search_code_error_handling(self, mock_search_service):
        """Test error handling in search tool."""
        # Setup mock to raise exception
        mock_search_service.search_code.side_effect = ValueError("Search failed")

        # Execute tool
        result_json = await search_code("test query")
        result = json.loads(result_json)

        # Verify error response
        assert result["status"] == "error"
        assert "Search failed" in result["message"]
        assert result["error_type"] == "ValueError"
        assert result["query"] == "test query"

    @pytest.mark.asyncio
    async def test_search_by_type_success(self, mock_search_service):
        """Test searching by semantic type."""
        # Setup mock
        mock_search_service.search_by_keywords.return_value = [
            CodeChunk(
                file_path="/test/file.py",
                content="def test_func(): pass",
                start_line=1,
                end_line=1,
                language="python",
                semantic_type="function_definition",
                embedding=None
            ),
            CodeChunk(
                file_path="/test/file2.py",
                content="def another_func(): pass",
                start_line=1,
                end_line=1,
                language="python",
                semantic_type="function_definition",
                embedding=None
            )
        ]

        # Execute tool
        result_json = await search_by_type("function_definition", "python", 10)
        result = json.loads(result_json)

        # Verify response
        assert result["status"] == "success"
        assert result["semantic_type"] == "function_definition"
        assert result["language_filter"] == "python"
        assert result["total_matches"] == 2
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_search_in_file_success(
        self, mock_search_service, sample_search_response
    ):
        """Test searching within specific file."""
        # Setup mock
        mock_search_service.search_code.return_value = sample_search_response

        # Execute tool
        result_json = await search_in_file("/test/file.py", "hello function", 5)
        result = json.loads(result_json)

        # Verify response
        assert result["status"] == "success"
        assert result["file_path"] == "/test/file.py"
        assert result["query"] == "hello function"
        assert "results" in result

    @pytest.mark.asyncio
    async def test_get_search_stats_success(self, mock_search_service):
        """Test getting search statistics."""
        # Setup mock
        mock_search_service.get_embeddings_stats.return_value = {
            "total_embeddings": 100,
            "languages": {"python": 60, "javascript": 40},
            "semantic_types": {"function_definition": 50, "class_definition": 30},
            "database_size_mb": 5.2
        }

        # Mock config access
        with patch("src.tools.search_code.search_service.config") as mock_config:
            mock_config.SUPPORTED_LANGUAGES = ["python", "javascript", "go"]
            mock_config.ENABLE_QUANTIZATION = True
            mock_config.MAX_FILE_SIZE_MB = 10

            # Execute tool
            result_json = await get_search_stats()
            result = json.loads(result_json)

        # Verify response
        assert result["status"] == "success"
        assert result["stats"]["total_embeddings"] == 100
        expected_langs = ["python", "javascript", "go"]
        assert result["capabilities"]["supported_languages"] == expected_langs
        assert result["capabilities"]["quantization_enabled"] is True

    @pytest.mark.asyncio
    async def test_json_response_format(
        self, mock_search_service, sample_search_response
    ):
        """Test that all tools return valid JSON strings."""
        # Setup mock
        mock_search_service.search_code.return_value = sample_search_response
        mock_search_service.get_embeddings_stats.return_value = {"total_embeddings": 0}

        # Test all tool functions
        tools_to_test = [
            (search_code, ("test query",)),
            (search_by_type, ("function_definition",)),
            (search_in_file, ("/test/file.py", "test")),
            (get_search_stats, ())
        ]

        for tool_func, args in tools_to_test:
            result_json = await tool_func(*args)

            # Verify it's a valid JSON string
            result = json.loads(result_json)  # Should not raise exception
            assert isinstance(result, dict)
            assert "status" in result

            # Verify it's a string (MCP requirement)
            assert isinstance(result_json, str)

    @pytest.mark.asyncio
    async def test_search_tools_error_handling(self, mock_search_service):
        """Test error handling across all search tools."""
        # Setup mocks to raise exceptions
        mock_search_service.search_code.side_effect = Exception("Search service error")
        mock_search_service.search_by_keywords.side_effect = Exception(
            "Keyword search error"
        )
        mock_search_service.get_embeddings_stats.side_effect = Exception("Stats error")

        # Test each tool handles errors gracefully
        tools_to_test = [
            (search_code, ("test",), "Search service error"),
            (search_by_type, ("function",), "Keyword search error"),
            (search_in_file, ("/test/file.py", "test"), "Search service error"),
            (get_search_stats, (), "Stats error")
        ]

        for tool_func, args, expected_error in tools_to_test:
            result_json = await tool_func(*args)
            result = json.loads(result_json)

            assert result["status"] == "error"
            assert expected_error in result["message"]
            assert "error_type" in result
