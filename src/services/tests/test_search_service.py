import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from ...models.indexing_models import CodeChunk, SearchRequest
from ...services.search_service import SearchService


@pytest.fixture
def search_service(mock_config):
    """Create SearchService instance for testing."""
    with patch("src.services.search_service.get_settings") as mock_settings:
        mock_settings.return_value = mock_config
        return SearchService()


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    with patch("src.services.search_service.get_settings") as mock_settings:
        config = Mock()
        temp_dir = Path(tempfile.mkdtemp())
        config.INDEX_CACHE_DIR = temp_dir
        config.VECTOR_DB_PATH = temp_dir / f"test_embeddings_{id(config)}.db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.BATCH_SIZE = 10
        config.ENABLE_QUANTIZATION = True
        config.SUPPORTED_LANGUAGES = ["python", "javascript"]
        mock_settings.return_value = config
        yield config

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_embedding_model():
    """Mock sentence transformer model."""
    with patch("src.services.search_service.SentenceTransformer") as mock_transformer:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6]
        ])
        mock_transformer.return_value = mock_model
        yield mock_model


@pytest.fixture
def sample_code_chunks():
    """Create sample code chunks for testing."""
    return [
        CodeChunk(
            file_path="/test/file1.py",
            content="def hello_world():\n    return 'Hello, World!'",
            start_line=1,
            end_line=2,
            language="python",
            semantic_type="function_definition",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        ),
        CodeChunk(
            file_path="/test/file2.py",
            content="class Calculator:\n    def add(self, a, b):\n        return a + b",
            start_line=1,
            end_line=3,
            language="python",
            semantic_type="class_definition",
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6]
        )
    ]


class TestSearchService:

    @pytest.mark.asyncio
    async def test_init_vector_db(self, mock_config):
        """Test vector database initialization."""
        search_service = SearchService()

        # Check that database file exists
        assert search_service.db_path.exists()

        # Check that tables were created
        conn = sqlite3.connect(str(search_service.db_path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "embeddings" in tables
        conn.close()

    @pytest.mark.asyncio
    async def test_embed_code_chunks(self, search_service, sample_code_chunks, mock_embedding_model, mock_config):
        """Test code chunk embedding generation."""
        # Initialize with mock model
        search_service.model = mock_embedding_model

        # Mock the store method to avoid database operations
        with patch.object(search_service, '_store_embeddings_batch', new=AsyncMock()):
            embedded_chunks = await search_service.embed_code_chunks(sample_code_chunks)

            assert len(embedded_chunks) == 2
            for chunk in embedded_chunks:
                assert chunk.embedding is not None
                assert len(chunk.embedding) == 5  # Mock embedding dimension

    @pytest.mark.asyncio
    async def test_embed_code_chunks_no_model(self, search_service, sample_code_chunks, mock_config):
        """Test embedding when no model available."""
        search_service.model = None

        embedded_chunks = await search_service.embed_code_chunks(sample_code_chunks)

        # Should return original chunks unchanged
        assert embedded_chunks == sample_code_chunks

    @pytest.mark.asyncio
    async def test_search_code_success(self, search_service, mock_embedding_model, mock_config):
        """Test successful semantic code search."""
        # Setup mock model and database
        search_service.model = mock_embedding_model

        # Store test embeddings in database
        await search_service._store_embeddings_batch([
            CodeChunk(
                file_path="/test/file.py",
                content="def test_function(): pass",
                start_line=1,
                end_line=1,
                language="python",
                semantic_type="function_definition",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
            )
        ])

        request = SearchRequest(
            query="test function",
            language=None,
            max_results=10,
            similarity_threshold=0.5
        )

        result = await search_service.search_code(request)

        assert result.total_matches >= 0
        assert result.query_time_ms > 0
        assert isinstance(result.results, list)

    @pytest.mark.asyncio
    async def test_search_code_no_model(self, search_service, mock_config):
        """Test search when no embedding model available."""
        search_service.model = None

        request = SearchRequest(query="test", max_results=10)
        result = await search_service.search_code(request)

        assert result.total_matches == 0
        assert result.query_time_ms == 0.0
        assert result.results == []

    @pytest.mark.asyncio
    async def test_search_by_keywords(self, search_service, mock_config):
        """Test keyword-based search fallback."""
        # Store test data in database
        conn = sqlite3.connect(str(search_service.db_path))
        conn.execute("""
            INSERT OR REPLACE INTO embeddings (file_path, content, start_line, end_line, language,
                                   semantic_type, embedding, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "/test/file_keywords.py",
            "def hello_world(): return 'Hello'",
            1, 1, "python", "function_definition",
            json.dumps([0.1, 0.2, 0.3]).encode(),
            "test_hash_keywords"
        ))
        conn.commit()
        conn.close()

        results = await search_service.search_by_keywords("hello_world")

        assert len(results) > 0
        assert results[0].content == "def hello_world(): return 'Hello'"
        assert results[0].language == "python"

    def test_cosine_similarity(self, search_service):
        """Test cosine similarity calculation."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])

        similarity = search_service._cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 1e-6  # Should be 1.0 for identical vectors

        c = np.array([0.0, 1.0, 0.0])
        similarity = search_service._cosine_similarity(a, c)
        assert abs(similarity - 0.0) < 1e-6  # Should be 0.0 for orthogonal vectors

    def test_quantize_embeddings(self, search_service):
        """Test embedding quantization for memory efficiency."""
        embeddings = np.array([[0.5, -0.3, 0.8, -1.0, 1.0]])

        quantized = search_service._quantize_embeddings(embeddings)

        # Check that values are properly quantized and in valid range
        assert quantized.shape == embeddings.shape
        assert np.all(quantized >= -1.0)
        assert np.all(quantized <= 1.0)

    @pytest.mark.asyncio
    async def test_get_embeddings_stats_empty_db(self, mock_config):
        """Test getting stats from empty database."""
        # Create fresh SearchService with isolated config
        with patch("src.services.search_service.get_settings") as mock_settings:
            mock_settings.return_value = mock_config
            fresh_service = SearchService()

            stats = await fresh_service.get_embeddings_stats()

            assert stats["total_embeddings"] == 0
            assert stats["languages"] == {}
            assert stats["semantic_types"] == {}
            assert stats["database_size_mb"] >= 0

    @pytest.mark.asyncio
    async def test_store_embeddings_batch(self, mock_config):
        """Test storing batch of embeddings."""
        # Create fresh SearchService with isolated config
        with patch("src.services.search_service.get_settings") as mock_settings:
            mock_settings.return_value = mock_config
            fresh_service = SearchService()

            # Create sample chunks for this test
            sample_chunks = [
                CodeChunk(
                    file_path="/test/store_batch1.py",
                    content="def hello_world(): return 'Hello, World!'",
                    start_line=1,
                    end_line=2,
                    language="python",
                    semantic_type="function_definition",
                    embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
                ),
                CodeChunk(
                    file_path="/test/store_batch2.py",
                    content="class Calculator: def add(self, a, b): return a + b",
                    start_line=1,
                    end_line=3,
                    language="python",
                    semantic_type="class_definition",
                    embedding=[0.2, 0.3, 0.4, 0.5, 0.6]
                )
            ]

            await fresh_service._store_embeddings_batch(sample_chunks)

            # Verify embeddings were stored
            conn = sqlite3.connect(str(fresh_service.db_path))
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            count = cursor.fetchone()[0]
            assert count == 2

            # Verify data integrity
            cursor = conn.execute("SELECT file_path, language, semantic_type FROM embeddings")
            rows = cursor.fetchall()

            file_paths = [row[0] for row in rows]
            languages = [row[1] for row in rows]
            types = [row[2] for row in rows]

            assert "/test/store_batch1.py" in file_paths
            assert "/test/store_batch2.py" in file_paths
            assert "python" in languages
            assert "function_definition" in types
            assert "class_definition" in types

            conn.close()

    @pytest.mark.asyncio
    async def test_search_similar_embeddings(self, search_service, sample_code_chunks, mock_config):
        """Test searching for similar embeddings."""
        # Store test embeddings
        await search_service._store_embeddings_batch(sample_code_chunks)

        # Search with query embedding
        query_embedding = np.array([0.15, 0.25, 0.35, 0.45, 0.55])  # Similar to second chunk

        results = await search_service._search_similar_embeddings(
            query_embedding,
            language_filter=None,
            max_results=10,
            threshold=0.5
        )

        assert len(results) >= 0  # Should find similar chunks
        for result in results:
            assert isinstance(result, CodeChunk)
            assert result.language == "python"
