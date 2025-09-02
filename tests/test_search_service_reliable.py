"""Reliable SearchService tests that work consistently."""

from unittest.mock import Mock, patch

import numpy as np

from src.models.indexing_models import CodeChunk, SearchRequest, SearchResponse
from src.services.search_service import SearchService


class TestSearchServiceReliable:
    """Reliable tests focused on core functionality."""

    def test_cosine_similarity_basic(self):
        """Test cosine similarity calculation."""
        service = SearchService.__new__(SearchService)  # Create without __init__

        # Test identical vectors
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        similarity = service._cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 1e-6

        # Test orthogonal vectors
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        similarity = service._cosine_similarity(a, b)
        assert abs(similarity - 0.0) < 1e-6

    def test_cosine_similarity_error_handling(self):
        """Test cosine similarity with error."""
        service = SearchService.__new__(SearchService)  # Create without __init__

        # Test with invalid input
        a = np.array([])
        b = np.array([1, 0, 0])
        similarity = service._cosine_similarity(a, b)
        assert similarity == 0.0

    def test_quantize_embeddings_basic(self):
        """Test embedding quantization."""
        service = SearchService.__new__(SearchService)  # Create without __init__

        embeddings = np.array([[2.0, -1.5], [0.5, 3.0]])
        quantized = service._quantize_embeddings(embeddings)

        assert np.all(quantized >= -1.0)
        assert np.all(quantized <= 1.0)
        assert quantized.shape == embeddings.shape

    def test_quantize_embeddings_edge_cases(self):
        """Test quantization edge cases."""
        service = SearchService.__new__(SearchService)  # Create without __init__

        # Test with values at boundaries
        embeddings = np.array([[-1.0, 1.0], [0.0, 0.5]])
        quantized = service._quantize_embeddings(embeddings)

        assert np.all(quantized >= -1.0)
        assert np.all(quantized <= 1.0)

    @patch("src.services.search_service.get_settings")
    async def test_search_code_no_model(self, mock_get_settings):
        """Test search when no model available."""
        mock_config = Mock()
        mock_config.EMBEDDING_ENABLED = False
        mock_get_settings.return_value = mock_config

        with patch.object(SearchService, "_init_vector_db"):
            with patch.object(SearchService, "_init_embedding_model"):
                service = SearchService()
                service.model = None

                request = SearchRequest(query="test")
                response = await service.search_code(request)

                assert isinstance(response, SearchResponse)
                assert response.results == []
                assert response.total_matches == 0

    @patch("src.services.search_service.get_settings")
    async def test_embed_code_chunks_no_model(self, mock_get_settings):
        """Test embedding when no model available."""
        mock_config = Mock()
        mock_config.EMBEDDING_ENABLED = False
        mock_get_settings.return_value = mock_config

        chunks = [
            CodeChunk(
                file_path="/test.py",
                content="def test(): pass",
                start_line=1,
                end_line=1,
                language="python",
                semantic_type="function_definition",
            )
        ]

        with patch.object(SearchService, "_init_vector_db"):
            with patch.object(SearchService, "_init_embedding_model"):
                service = SearchService()
                service.model = None

                result = await service.embed_code_chunks(chunks)
                assert result == chunks  # Should return unchanged
