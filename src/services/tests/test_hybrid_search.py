from unittest.mock import AsyncMock, Mock

import pytest

from ...models.indexing_models import CodeChunk, SearchRequest, SearchResponse
from ..hybrid_search import HybridSearchConfig, HybridSearchService


class TestHybridSearchService:
    @pytest.fixture
    def mock_search_service(self):
        service = Mock()
        service.search_code = AsyncMock()
        service.search_by_bm25 = AsyncMock()
        return service

    @pytest.fixture
    def mock_metadata_service(self):
        service = Mock()
        service.search = AsyncMock()
        return service

    @pytest.fixture
    def hybrid_service(self, mock_search_service, mock_metadata_service):
        return HybridSearchService(mock_search_service, mock_metadata_service)

    @pytest.fixture
    def sample_chunks(self):
        return [
            CodeChunk(
                file_path="auth.py",
                content="def authenticate_user(username: str) -> bool:",
                start_line=1,
                end_line=2,
                language="python",
                semantic_type="function_definition",
                function_name="authenticate_user"
            ),
            CodeChunk(
                file_path="user.py",
                content="class UserManager:\n    def get_user(self, id: int):",
                start_line=5,
                end_line=7,
                language="python",
                semantic_type="class_definition",
                class_name="UserManager"
            )
        ]

    @pytest.mark.asyncio
    async def test_reciprocal_rank_fusion(self, hybrid_service, sample_chunks):
        """Test RRF algorithm implementation."""
        # Create different result lists (simulating different search methods)
        bm25_results = [sample_chunks[0], sample_chunks[1]]  # BM25 order
        semantic_results = [sample_chunks[1], sample_chunks[0]]  # Semantic order (reversed)

        # Apply RRF
        fused_results = hybrid_service._reciprocal_rank_fusion([bm25_results, semantic_results], k=60)

        assert len(fused_results) == 2
        # Both chunks should be present
        chunk_paths = [chunk.file_path for chunk in fused_results]
        assert "auth.py" in chunk_paths
        assert "user.py" in chunk_paths

    @pytest.mark.asyncio
    async def test_hybrid_search_integration(
        self, hybrid_service, mock_search_service, mock_metadata_service, sample_chunks
    ):
        """Test full hybrid search integration."""
        # Setup mock responses
        mock_search_service.search_by_bm25.return_value = [sample_chunks[0]]

        semantic_response = SearchResponse(
            results=[sample_chunks[1]],
            total_matches=1,
            query_time_ms=10.0
        )
        mock_search_service.search_code.return_value = semantic_response
        mock_metadata_service.search.return_value = [sample_chunks[0]]

        # Create search request
        request = SearchRequest(
            query="user authentication",
            max_results=10
        )

        # Execute hybrid search
        response = await hybrid_service.search(request)

        assert response.total_matches > 0
        assert len(response.results) > 0
        assert response.query_time_ms > 0

    def test_detect_query_type(self, hybrid_service):
        """Test query type detection."""
        assert hybrid_service._detect_query_type("find authentication function") == "function_search"
        assert hybrid_service._detect_query_type("class that inherits from Base") == "class_search"
        assert hybrid_service._detect_query_type("import pandas library") == "import_search"
        assert hybrid_service._detect_query_type("function return type") == "type_search"
        assert hybrid_service._detect_query_type("general code search") == "general_search"

    def test_should_use_metadata_search(self, hybrid_service):
        """Test metadata search activation logic."""
        assert hybrid_service._should_use_metadata_search("find function authenticate") is True
        assert hybrid_service._should_use_metadata_search("class UserManager") is True
        assert hybrid_service._should_use_metadata_search("return type string") is True
        assert hybrid_service._should_use_metadata_search("complexity score") is True
        assert hybrid_service._should_use_metadata_search("general text search") is False

    def test_apply_diversity_filter(self, hybrid_service, sample_chunks):
        """Test diversity filtering."""
        # Create chunks from same file
        same_file_chunks = [
            CodeChunk(
                file_path="same.py", content="func1", start_line=1, end_line=2,
                language="python", semantic_type="function"
            ),
            CodeChunk(
                file_path="same.py", content="func2", start_line=3, end_line=4,
                language="python", semantic_type="function"
            ),
            CodeChunk(
                file_path="same.py", content="func3", start_line=5, end_line=6,
                language="python", semantic_type="function"
            ),
            CodeChunk(
                file_path="same.py", content="func4", start_line=7, end_line=8,
                language="python", semantic_type="function"
            ),
        ]

        diverse_results = hybrid_service._apply_diversity_filter(same_file_chunks, max_per_file=2)

        assert len(diverse_results) == 2  # Limited to 2 per file

    @pytest.mark.asyncio
    async def test_query_expansion(self, hybrid_service):
        """Test query expansion functionality."""
        expanded = await hybrid_service.expand_query("authentication")

        assert len(expanded) > 1
        assert "authentication" in expanded
        # Should contain expanded terms
        expanded_text = " ".join(expanded)
        assert any(term in expanded_text for term in ["login", "credential", "token"])

    @pytest.mark.asyncio
    async def test_search_failure_handling(self, hybrid_service, mock_search_service, mock_metadata_service):
        """Test handling of search method failures."""
        # Setup one method to fail
        mock_search_service.search_by_bm25.side_effect = Exception("BM25 failed")
        mock_search_service.search_code.return_value = SearchResponse(results=[], total_matches=0, query_time_ms=0)
        mock_metadata_service.search.return_value = []

        request = SearchRequest(query="test", max_results=10)
        response = await hybrid_service.search(request)

        # Should handle failure gracefully
        assert response.total_matches == 0
        assert response.query_time_ms >= 0

    def test_hybrid_search_config(self):
        """Test hybrid search configuration."""
        config = HybridSearchConfig()

        assert config.bm25_weight == 0.4
        assert config.semantic_weight == 0.4
        assert config.metadata_weight == 0.2
        assert config.rrf_k_parameter == 60
        assert config.enable_query_expansion is True

        # Test custom configuration
        custom_config = HybridSearchConfig(
            bm25_weight=0.5,
            semantic_weight=0.3,
            metadata_weight=0.2
        )

        assert custom_config.bm25_weight == 0.5
        assert custom_config.semantic_weight == 0.3
