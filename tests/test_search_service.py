import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.models.indexing_models import CodeChunk, SearchRequest, SearchResponse
from src.services.search_service import SearchService


@pytest.fixture
def mock_config():
    """Mock configuration for tests."""
    config = Mock()
    config.EMBEDDING_ENABLED = True
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.VECTOR_DB_PATH = Path("/tmp/test.db")
    config.INDEX_CACHE_DIR = Path("/tmp")
    config.EMBEDDING_BATCH_SIZE = 32
    config.ENABLE_QUANTIZATION = False
    return config


@pytest.fixture
def sample_chunks():
    """Sample code chunks for testing."""
    return [
        CodeChunk(
            file_path="/test/auth.py",
            content="def authenticate_user(username, password):\n    return validate_credentials(username, password)",
            start_line=1,
            end_line=2,
            language="python",
            semantic_type="function_definition",
            function_name="authenticate_user",
        ),
        CodeChunk(
            file_path="/test/utils.py",
            content="class Logger:\n    def log(self, message):\n        print(message)",
            start_line=5,
            end_line=7,
            language="python",
            semantic_type="class_definition",
            class_name="Logger",
        ),
    ]


class TestSearchService:
    @patch("src.services.search_service.get_settings")
    def test_init_without_sentence_transformers(self, mock_get_settings, mock_config):
        """Test initialization when sentence-transformers is not available."""
        mock_get_settings.return_value = mock_config

        with patch("src.services.search_service.SentenceTransformer", None):
            service = SearchService()
            assert service.model is None

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.SentenceTransformer")
    def test_init_with_embedding_disabled(self, mock_st, mock_get_settings, mock_config):
        """Test initialization with embedding disabled."""
        mock_config.EMBEDDING_ENABLED = False
        mock_get_settings.return_value = mock_config

        service = SearchService()
        assert service.model is None
        mock_st.assert_not_called()

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.SentenceTransformer")
    def test_init_embedding_model_failure(self, mock_st, mock_get_settings, mock_config):
        """Test initialization when embedding model fails to load."""
        mock_get_settings.return_value = mock_config
        mock_st.side_effect = Exception("Model loading failed")

        service = SearchService()
        assert service.model is None

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.sqlite3")
    def test_init_vector_db_success(self, mock_sqlite3, mock_get_settings, mock_config):
        """Test successful vector database initialization."""
        mock_get_settings.return_value = mock_config
        mock_conn = Mock()
        mock_sqlite3.connect.return_value = mock_conn

        service = SearchService()

        # Verify database setup calls
        mock_sqlite3.connect.assert_called()
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()
        mock_conn.close.assert_called()

    @patch("src.services.search_service.get_settings")
    def test_init_vector_db_failure(self, mock_get_settings, mock_config, caplog):
        """Test vector database initialization failure."""
        mock_get_settings.return_value = mock_config

        with patch("src.services.search_service.sqlite3") as mock_sqlite3:
            mock_sqlite3.connect.side_effect = Exception("DB connection failed")

            service = SearchService()

            # Should log error but not crash
            assert "Failed to initialize vector database" in caplog.text

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.SentenceTransformer")
    @patch("src.services.search_service.sqlite3")
    async def test_embed_code_chunks_success(
        self, mock_sqlite3, mock_st, mock_get_settings, mock_config, sample_chunks
    ):
        """Test successful embedding generation."""
        mock_get_settings.return_value = mock_config
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_st.return_value = mock_model

        mock_conn = Mock()
        mock_sqlite3.connect.return_value = mock_conn

        service = SearchService()
        service.model = mock_model

        result = await service.embed_code_chunks(sample_chunks)

        assert len(result) == 2
        assert all(chunk.embedding is not None for chunk in result)
        mock_model.encode.assert_called()

    @patch("src.services.search_service.get_settings")
    async def test_embed_code_chunks_no_model(self, mock_get_settings, mock_config, sample_chunks):
        """Test embedding when no model is available."""
        mock_get_settings.return_value = mock_config

        service = SearchService()
        service.model = None

        result = await service.embed_code_chunks(sample_chunks)

        assert result == sample_chunks  # Should return original chunks

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.SentenceTransformer")
    @patch("src.services.search_service.sqlite3")
    async def test_embed_code_chunks_with_quantization(
        self, mock_sqlite3, mock_st, mock_get_settings, mock_config, sample_chunks
    ):
        """Test embedding with quantization enabled."""
        mock_config.ENABLE_QUANTIZATION = True
        mock_get_settings.return_value = mock_config

        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1.5, -0.8], [0.9, 2.1]])
        mock_st.return_value = mock_model

        mock_conn = Mock()
        mock_sqlite3.connect.return_value = mock_conn

        service = SearchService()
        service.model = mock_model

        result = await service.embed_code_chunks(sample_chunks)

        assert len(result) == 2
        # Check that quantization was applied (values should be in [-1, 1] range)
        for chunk in result:
            assert all(-1 <= val <= 1 for val in chunk.embedding)

    def test_quantize_embeddings(self):
        """Test embedding quantization."""
        service = SearchService()

        # Test with values outside [-1, 1] range
        embeddings = np.array([[2.0, -1.5], [0.5, 3.0]])
        quantized = service._quantize_embeddings(embeddings)

        # Should be clipped to [-1, 1] range
        assert np.all(quantized >= -1.0)
        assert np.all(quantized <= 1.0)

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.sqlite3")
    async def test_store_embeddings_batch_success(self, mock_sqlite3, mock_get_settings, mock_config, sample_chunks):
        """Test successful embedding storage."""
        mock_get_settings.return_value = mock_config
        mock_conn = Mock()
        mock_sqlite3.connect.return_value = mock_conn

        service = SearchService()

        # Add embeddings to chunks
        for chunk in sample_chunks:
            chunk.embedding = [0.1, 0.2, 0.3]

        await service._store_embeddings_batch(sample_chunks)

        # Should insert embeddings for each chunk
        assert mock_conn.execute.call_count == len(sample_chunks)
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.sqlite3")
    async def test_store_embeddings_batch_error(
        self, mock_sqlite3, mock_get_settings, mock_config, sample_chunks, caplog
    ):
        """Test embedding storage with database error."""
        mock_get_settings.return_value = mock_config
        mock_sqlite3.connect.side_effect = Exception("DB error")

        service = SearchService()

        for chunk in sample_chunks:
            chunk.embedding = [0.1, 0.2, 0.3]

        await service._store_embeddings_batch(sample_chunks)

        assert "Error storing embeddings" in caplog.text

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.SentenceTransformer")
    @patch("src.services.search_service.sqlite3")
    async def test_search_code_success(self, mock_sqlite3, mock_st, mock_get_settings, mock_config):
        """Test successful semantic search."""
        mock_get_settings.return_value = mock_config
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model

        # Mock database response
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (
                "/test/file.py",
                "test content",
                1,
                2,
                "python",
                "function_definition",
                json.dumps([0.1, 0.2, 0.3]).encode(),
                "test_func",
                "TestClass",
                "test_func",
                None,
                "str",
                None,
                None,
                None,
                1,
                None,
                None,
                None,
            )
        ]
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        service = SearchService()
        service.model = mock_model

        request = SearchRequest(query="test function", max_results=10)
        result = await service.search_code(request)

        assert isinstance(result, SearchResponse)
        assert len(result.results) == 1
        assert result.total_matches == 1
        assert result.query_time_ms > 0

    @patch("src.services.search_service.get_settings")
    async def test_search_code_no_model(self, mock_get_settings, mock_config):
        """Test search when no embedding model is available."""
        mock_get_settings.return_value = mock_config

        service = SearchService()
        service.model = None

        request = SearchRequest(query="test", max_results=10)
        result = await service.search_code(request)

        assert isinstance(result, SearchResponse)
        assert len(result.results) == 0
        assert result.total_matches == 0

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.SentenceTransformer")
    async def test_search_code_error(self, mock_st, mock_get_settings, mock_config, caplog):
        """Test search with error during processing."""
        mock_get_settings.return_value = mock_config
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_st.return_value = mock_model

        service = SearchService()
        service.model = mock_model

        request = SearchRequest(query="test", max_results=10)
        result = await service.search_code(request)

        assert isinstance(result, SearchResponse)
        assert len(result.results) == 0
        assert "Error during semantic search" in caplog.text

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        service = SearchService()

        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        similarity = service._cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 1e-6  # Should be 1.0 for identical vectors

        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        similarity = service._cosine_similarity(a, b)
        assert abs(similarity - 0.0) < 1e-6  # Should be 0.0 for orthogonal vectors

    def test_cosine_similarity_error(self):
        """Test cosine similarity with error handling."""
        service = SearchService()

        # Test with problematic vectors
        a = np.array([0, 0, 0])  # Zero vector
        b = np.array([1, 0, 0])
        similarity = service._cosine_similarity(a, b)
        assert similarity == 0.0  # Should handle gracefully

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.sqlite3")
    async def test_get_embeddings_stats_success(self, mock_sqlite3, mock_get_settings, mock_config):
        """Test successful stats retrieval."""
        mock_get_settings.return_value = mock_config
        mock_config.VECTOR_DB_PATH.exists.return_value = True
        mock_config.VECTOR_DB_PATH.stat.return_value.st_size = 1024 * 1024  # 1MB

        mock_conn = Mock()

        # Mock different execute calls
        def mock_execute_side_effect(query, *args):
            mock_cursor = Mock()
            if "COUNT(*)" in query:
                mock_cursor.fetchone.return_value = (100,)
            elif "language" in query:
                mock_cursor.fetchall.return_value = [("python", 60), ("javascript", 40)]
            elif "semantic_type" in query:
                mock_cursor.fetchall.return_value = [("function_definition", 70), ("class_definition", 30)]
            return mock_cursor

        mock_conn.execute.side_effect = mock_execute_side_effect
        mock_sqlite3.connect.return_value = mock_conn

        service = SearchService()
        stats = await service.get_embeddings_stats()

        assert stats["total_embeddings"] == 100
        assert stats["languages"]["python"] == 60
        assert stats["semantic_types"]["function"] == 70
        assert stats["database_size_mb"] == 1.0

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.sqlite3")
    async def test_get_embeddings_stats_error(self, mock_sqlite3, mock_get_settings, mock_config, caplog):
        """Test stats retrieval with database error."""
        mock_get_settings.return_value = mock_config
        mock_sqlite3.connect.side_effect = Exception("DB error")

        service = SearchService()
        stats = await service.get_embeddings_stats()

        assert stats["total_embeddings"] == 0
        assert "Error getting embeddings stats" in caplog.text

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.sqlite3")
    async def test_remove_file_embeddings_success(self, mock_sqlite3, mock_get_settings, mock_config):
        """Test successful file embedding removal."""
        mock_get_settings.return_value = mock_config

        # Mock separate connections for init and the actual operation
        init_conn = Mock()
        remove_conn = Mock()
        mock_sqlite3.connect.side_effect = [init_conn, remove_conn]

        service = SearchService()
        await service.remove_file_embeddings(["/test/file1.py", "/test/file2.py"])

        # Verify the DELETE operation was called on the remove connection
        expected_call = remove_conn.execute.call_args_list[-1]
        assert "DELETE FROM embeddings WHERE file_path IN" in expected_call[0][0]
        remove_conn.commit.assert_called_once()
        remove_conn.close.assert_called_once()

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.sqlite3")
    async def test_remove_file_embeddings_error(self, mock_sqlite3, mock_get_settings, mock_config, caplog):
        """Test file embedding removal with error."""
        mock_get_settings.return_value = mock_config
        mock_sqlite3.connect.side_effect = Exception("DB error")

        service = SearchService()
        await service.remove_file_embeddings(["/test/file.py"])

        assert "Error removing file embeddings" in caplog.text

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.sqlite3")
    async def test_clear_all_embeddings_success(self, mock_sqlite3, mock_get_settings, mock_config):
        """Test successful embedding clearing."""
        mock_get_settings.return_value = mock_config
        mock_conn = Mock()
        mock_sqlite3.connect.return_value = mock_conn

        service = SearchService()
        await service.clear_all_embeddings()

        mock_conn.execute.assert_called_with("DELETE FROM embeddings")
        assert mock_conn.commit.call_count >= 1  # At least one commit for the clear operation
        mock_conn.close.assert_called_once()

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.sqlite3")
    async def test_clear_all_embeddings_error(self, mock_sqlite3, mock_get_settings, mock_config, caplog):
        """Test embedding clearing with error."""
        mock_get_settings.return_value = mock_config
        mock_sqlite3.connect.side_effect = Exception("DB error")

        service = SearchService()
        await service.clear_all_embeddings()

        assert "Error clearing embeddings" in caplog.text

    @patch("src.services.search_service.get_settings")
    async def test_search_by_bm25_success(self, mock_get_settings, mock_config):
        """Test BM25 search functionality."""
        mock_get_settings.return_value = mock_config

        with patch("src.services.search_service.BM25SearchEngine") as mock_bm25:
            with patch("src.services.search_service.sqlite3") as mock_sqlite3:
                mock_engine = Mock()
                # Mock BM25 to return doc IDs and scores, not CodeChunks
                mock_engine.search.return_value = [("/test.py:1", 0.85)]
                mock_bm25.return_value = mock_engine

                # Mock database lookup
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_cursor.fetchone.return_value = (
                    "/test.py",
                    "test content",
                    1,
                    2,
                    "python",
                    "function_definition",
                    "test_func",
                    "TestClass",
                    "test_func",
                    None,
                    "str",
                    None,
                    None,
                    None,
                    1,
                    None,
                    None,
                    None,
                )
                mock_conn.execute.return_value = mock_cursor
                mock_sqlite3.connect.return_value = mock_conn

                service = SearchService()
                service.bm25_engine = mock_engine
                result = await service.search_by_bm25("test query", "python", 10)

                assert len(result) == 1
                mock_engine.search.assert_called_once()

    @patch("src.services.search_service.get_settings")
    async def test_search_by_bm25_no_engine(self, mock_get_settings, mock_config):
        """Test BM25 search when engine not available."""
        mock_get_settings.return_value = mock_config

        service = SearchService()
        service.bm25_engine = None
        result = await service.search_by_bm25("test query", "python", 10)

        assert result == []

    @patch("src.services.search_service.get_settings")
    @patch("src.services.search_service.SentenceTransformer")
    @patch("src.services.search_service.sqlite3")
    async def test_search_similar_embeddings_success(
        self, mock_sqlite3, mock_st, mock_get_settings, mock_config, sample_chunks
    ):
        """Test searching similar embeddings."""
        mock_get_settings.return_value = mock_config
        mock_model = Mock()
        mock_st.return_value = mock_model

        # Mock database to return embedding data
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (
                "/test/auth.py",
                "def authenticate_user():",
                1,
                2,
                "python",
                "function_definition",
                json.dumps([0.1, 0.2, 0.3]).encode(),
                "authenticate_user",
                None,
                "authenticate_user",
                None,
                "bool",
                None,
                None,
                None,
                1,
                None,
                None,
                None,
            )
        ]
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        service = SearchService()
        service.model = mock_model
        query_embedding = np.array([0.1, 0.2, 0.3])
        result = await service._search_similar_embeddings(query_embedding, "python", 0.7, 10)

        assert len(result) == 1
        assert result[0].file_path == "/test/auth.py"

    @patch("src.services.search_service.get_settings")
    async def test_search_by_keywords_basic(self, mock_get_settings, mock_config):
        """Test basic keyword search."""
        mock_get_settings.return_value = mock_config

        with patch("src.services.search_service.sqlite3") as mock_sqlite3:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_cursor.fetchall.return_value = [
                (
                    "/test/file.py",
                    "test content",
                    1,
                    2,
                    "python",
                    "function",
                    None,
                    "test_func",
                    None,
                    "test_func",
                    None,
                    None,
                    None,
                    None,
                    None,
                    1,
                    None,
                    None,
                    None,
                )
            ]
            mock_conn.execute.return_value = mock_cursor
            mock_sqlite3.connect.return_value = mock_conn

            service = SearchService()
            result = await service.search_by_keywords("test", "python")

            assert len(result) == 1
            assert result[0].file_path == "/test/file.py"
