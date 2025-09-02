"""Simple tests to boost coverage on easily testable functionality."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import numpy as np

from src.services.bm25_search import BM25SearchEngine
from src.services.hybrid_search import HybridSearchService
from src.services.search_service import SearchService
from src.models.indexing_models import SearchRequest


class TestCoverageBoost:
    """Simple tests to improve coverage percentages."""
    
    @patch('src.services.bm25_search.get_settings')
    @patch('sqlite3.connect')
    def test_bm25_init_only(self, mock_connect, mock_get_settings):
        """Test BM25 initialization for coverage."""
        mock_config = Mock()
        mock_config.INDEX_CACHE_DIR = Path("/tmp")
        mock_get_settings.return_value = mock_config
        
        mock_conn = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        engine = BM25SearchEngine()
        assert engine.k1 == 1.2
        assert engine.b == 0.75
    
    def test_hybrid_search_init_only(self):
        """Test hybrid search initialization for coverage."""
        mock_search_service = Mock()
        mock_metadata_search = Mock()
        
        service = HybridSearchService(mock_search_service, mock_metadata_search)
        assert service.search_service == mock_search_service
        assert service.metadata_search_service == mock_metadata_search
    
    def test_hybrid_search_utility_methods(self):
        """Test hybrid search utility methods for coverage."""
        service = HybridSearchService(Mock(), Mock())
        
        # Test query type detection
        assert service._detect_query_type("function test") == "function_search"
        assert service._detect_query_type("class Test") == "class_search"
        
        # Test metadata search decision
        assert service._should_use_metadata_search("function") is True
        assert service._should_use_metadata_search("random") is False
        
        # Test diversity filter
        chunk1 = Mock()
        chunk1.file_path = "/same.py"
        chunk2 = Mock()
        chunk2.file_path = "/same.py"
        
        filtered = service._apply_diversity_filter([chunk1, chunk2], max_per_file=1)
        assert len(filtered) == 1
    
    @patch('src.services.search_service.get_settings')
    def test_search_service_utilities(self, mock_get_settings):
        """Test search service utility methods for coverage."""
        mock_config = Mock()
        mock_config.EMBEDDING_ENABLED = False
        mock_get_settings.return_value = mock_config
        
        service = SearchService()
        
        # Test cosine similarity
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        similarity = service._cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 1e-6
        
        # Test quantization
        embeddings = np.array([[2.0, -1.5]])
        quantized = service._quantize_embeddings(embeddings)
        assert np.all(quantized >= -1.0)
        assert np.all(quantized <= 1.0)
    
    async def test_hybrid_search_query_expansion(self):
        """Test query expansion for coverage."""
        service = HybridSearchService(Mock(), Mock())
        
        # Test with expansion enabled
        expanded = await service.expand_query("auth")
        assert "auth" in expanded
        assert len(expanded) > 1
        
        # Test with expansion disabled
        service.config.enable_query_expansion = False
        expanded = await service.expand_query("auth")
        assert expanded == ["auth"]
    
    async def test_hybrid_search_post_processing(self):
        """Test post-processing methods for coverage."""
        service = HybridSearchService(Mock(), Mock())
        
        # Test contextual scoring (currently pass-through)
        chunk1 = Mock()
        chunk1.file_path = "/test.py"
        results = await service._apply_contextual_scoring([chunk1], "test query")
        assert len(results) == 1