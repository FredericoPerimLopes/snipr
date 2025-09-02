import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from src.services.search_service import SearchService
from src.services.bm25_search import BM25SearchEngine
from src.services.hybrid_search import HybridSearchService


class TestSearchSimple:
    """Simple search tests that should work reliably."""
    
    @patch('src.services.search_service.get_settings')
    def test_search_service_init_disabled(self, mock_get_settings):
        """Test search service with embedding disabled."""
        mock_config = Mock()
        mock_config.EMBEDDING_ENABLED = False
        mock_config.VECTOR_DB_PATH = Path("/tmp/test.db")
        mock_get_settings.return_value = mock_config
        
        service = SearchService()
        assert service.model is None
        assert service.config == mock_config
    
    def test_search_service_cosine_similarity(self):
        """Test cosine similarity calculation."""
        service = SearchService()
        
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
    
    def test_search_service_quantization(self):
        """Test embedding quantization."""
        service = SearchService()
        
        embeddings = np.array([[2.0, -1.5], [0.5, 3.0]])
        quantized = service._quantize_embeddings(embeddings)
        
        assert np.all(quantized >= -1.0)
        assert np.all(quantized <= 1.0)
    
    @patch('src.services.bm25_search.get_settings')
    @patch('sqlite3.connect')
    def test_bm25_init_basic(self, mock_connect, mock_get_settings):
        """Test BM25 basic initialization."""
        mock_config = Mock()
        mock_config.INDEX_CACHE_DIR = Path("/tmp")
        mock_get_settings.return_value = mock_config
        
        mock_conn = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        engine = BM25SearchEngine()
        assert engine.k1 == 1.2
        assert engine.b == 0.75
    
    def test_hybrid_search_init_basic(self):
        """Test hybrid search basic initialization."""
        mock_search = Mock()
        mock_metadata = Mock()
        
        service = HybridSearchService(mock_search, mock_metadata)
        assert service.search_service == mock_search
        assert service.metadata_search_service == mock_metadata
        assert hasattr(service, 'config')
    
    def test_hybrid_search_query_detection(self):
        """Test query type detection."""
        service = HybridSearchService(Mock(), Mock())
        
        assert service._detect_query_type("function test") == "function_search"
        assert service._detect_query_type("class Test") == "class_search"
        assert service._detect_query_type("return type") == "type_search"
        assert service._detect_query_type("random text") == "general_search"
    
    def test_hybrid_search_metadata_decision(self):
        """Test metadata search decision."""
        service = HybridSearchService(Mock(), Mock())
        
        assert service._should_use_metadata_search("function call") is True
        assert service._should_use_metadata_search("class definition") is True
        assert service._should_use_metadata_search("random text") is False
    
    def test_hybrid_search_diversity_filter(self):
        """Test result diversity filtering."""
        service = HybridSearchService(Mock(), Mock())
        
        chunk1 = Mock()
        chunk1.file_path = "/same.py"
        chunk2 = Mock()
        chunk2.file_path = "/same.py"
        chunk3 = Mock()
        chunk3.file_path = "/different.py"
        
        results = [chunk1, chunk2, chunk3]
        filtered = service._apply_diversity_filter(results, max_per_file=1)
        
        file_paths = [chunk.file_path for chunk in filtered]
        assert file_paths.count("/same.py") == 1
        assert "/different.py" in file_paths
    
    async def test_hybrid_search_query_expansion(self):
        """Test query expansion functionality."""
        service = HybridSearchService(Mock(), Mock())
        
        # Test expansion enabled
        expanded = await service.expand_query("auth function")
        assert "auth function" in expanded
        assert len(expanded) > 1
        
        # Test expansion disabled
        service.config.enable_query_expansion = False
        expanded = await service.expand_query("auth function")
        assert expanded == ["auth function"]