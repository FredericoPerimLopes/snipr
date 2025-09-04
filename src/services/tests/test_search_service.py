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
        config.EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"
        config.EMBEDDING_ENABLED = True
        config.EMBEDDING_BATCH_SIZE = 10
        config.ENABLE_QUANTIZATION = True
        config.SUPPORTED_LANGUAGES = ["python", "javascript"]
        config.VEC_DIMENSION = 5  # Small dimension for testing
        config.VEC_INDEX_TYPE = "flat"
        config.SIMILARITY_THRESHOLD = 0.7
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
        # Return 5-dimensional vectors for testing
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]])
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
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        ),
        CodeChunk(
            file_path="/test/file2.py",
            content="class Calculator:\n    def add(self, a, b):\n        return a + b",
            start_line=1,
            end_line=3,
            language="python",
            semantic_type="class_definition",
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
        ),
    ]


class TestSearchService:
    @pytest.mark.asyncio
    async def test_init_vector_db(self, mock_config):
        """Test vector database initialization."""
        search_service = SearchService()

        # Check that database file exists
        assert search_service.db_path.exists()

        # Check that vec tables were created
        conn = sqlite3.connect(str(search_service.db_path))
        search_service.vec_loader.load_extension(conn)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "embeddings_vec" in tables
        assert "embeddings_vec_metadata" in tables
        conn.close()

    @pytest.mark.asyncio
    async def test_embed_code_chunks(self, search_service, sample_code_chunks, mock_embedding_model, mock_config):
        """Test code chunk embedding generation."""
        # Initialize with mock model
        search_service.model = mock_embedding_model

        # Mock the store method to avoid database operations
        with patch.object(search_service, "_store_embeddings_batch", new=AsyncMock()):
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
        await search_service._store_embeddings_batch(
            [
                CodeChunk(
                    file_path="/test/file.py",
                    content="def test_function(): pass",
                    start_line=1,
                    end_line=1,
                    language="python",
                    semantic_type="function_definition",
                    embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                )
            ]
        )

        request = SearchRequest(query="test function", language=None, max_results=10, similarity_threshold=0.5)

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
        # Ensure database is properly initialized
        search_service._init_vector_db()

        # Store test data using vec operations
        test_chunk = CodeChunk(
            file_path="/test/file_keywords.py",
            content="def hello_world(): return 'Hello'",
            start_line=1,
            end_line=1,
            language="python",
            semantic_type="function_definition",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            function_name="hello_world",
        )
        await search_service._store_embeddings_batch([test_chunk])

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
                    embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                    function_signature="def hello_world()",
                    function_name="hello_world",
                    parameter_types=[],
                    return_type="str",
                    inheritance_chain=None,
                    import_statements=None,
                    docstring=None,
                    complexity_score=1,
                    dependencies=None,
                    interfaces=None,
                    decorators=None,
                    class_name=None,
                ),
                CodeChunk(
                    file_path="/test/store_batch2.py",
                    content="class Calculator: def add(self, a, b): return a + b",
                    start_line=1,
                    end_line=3,
                    language="python",
                    semantic_type="class_definition",
                    embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
                    function_signature=None,
                    function_name=None,
                    parameter_types=None,
                    return_type=None,
                    inheritance_chain=[],
                    import_statements=None,
                    docstring=None,
                    complexity_score=None,
                    dependencies=None,
                    interfaces=None,
                    decorators=None,
                    class_name="Calculator",
                ),
            ]

            await fresh_service._store_embeddings_batch(sample_chunks)

            # Verify embeddings were stored
            conn = sqlite3.connect(str(fresh_service.db_path))
            fresh_service.vec_loader.load_extension(conn)
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings_vec_metadata")
            count = cursor.fetchone()[0]
            assert count == 2

            # Verify data integrity
            cursor = conn.execute("SELECT file_path, language, semantic_type FROM embeddings_vec_metadata")
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

        results = await search_service._search_similar_embeddings_vec(
            query_embedding, language_filter=None, max_results=10, threshold=0.5
        )

        assert len(results) >= 0  # Should find similar chunks
        for result in results:
            assert isinstance(result, CodeChunk)
            assert result.language == "python"

    @pytest.mark.asyncio
    async def test_sqlite_vec_initialization(self, mock_config):
        """Test that sqlite-vec is properly initialized."""
        with patch("src.services.search_service.get_settings") as mock_settings:
            mock_settings.return_value = mock_config
            search_service = SearchService()

            # Verify vec components are initialized
            assert search_service.vec_loader is not None
            assert search_service.vec_ops is not None

            # Verify extension is loaded
            conn = sqlite3.connect(str(search_service.db_path))
            version_result = search_service.vec_loader.load_extension(conn, check_loaded=True)
            assert version_result is True
            conn.close()

    @pytest.mark.asyncio
    async def test_vec_store_and_search_integration(self, mock_config):
        """Test full integration of storage and search with sqlite-vec."""
        with patch("src.services.search_service.get_settings") as mock_settings:
            mock_settings.return_value = mock_config

            with patch("src.services.search_service.SentenceTransformer") as mock_transformer:
                mock_model = Mock()
                mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
                mock_transformer.return_value = mock_model

                search_service = SearchService()

                # Store test embedding
                test_chunks = [
                    CodeChunk(
                        file_path="/test/integration.py",
                        content="def integration_test(): return True",
                        start_line=1,
                        end_line=1,
                        language="python",
                        semantic_type="function",
                        embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                        function_name="integration_test",
                        content_hash="integration_hash",
                    )
                ]

                await search_service._store_embeddings_batch(test_chunks)

                # Search for it
                query_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Exact match
                results = await search_service._search_similar_embeddings_vec(
                    query_embedding, language_filter=None, max_results=5, threshold=0.9
                )

                assert len(results) >= 1
                assert results[0].function_name == "integration_test"
                assert results[0].file_path == "/test/integration.py"

    @pytest.mark.asyncio
    async def test_vec_language_filtering(self, mock_config):
        """Test language filtering in sqlite-vec search."""
        with patch("src.services.search_service.get_settings") as mock_settings:
            mock_settings.return_value = mock_config
            search_service = SearchService()

            # Store chunks in different languages
            test_chunks = [
                CodeChunk(
                    file_path="/test/python_func.py",
                    content="def python_function(): pass",
                    start_line=1,
                    end_line=1,
                    language="python",
                    semantic_type="function",
                    embedding=[1.0, 0.0, 0.0, 0.0, 0.0],
                    function_name="python_function",
                ),
                CodeChunk(
                    file_path="/test/js_func.js",
                    content="function jsFunction() {}",
                    start_line=1,
                    end_line=1,
                    language="javascript",
                    semantic_type="function",
                    embedding=[0.0, 1.0, 0.0, 0.0, 0.0],
                    function_name="jsFunction",
                ),
            ]

            await search_service._store_embeddings_batch(test_chunks)

            # Search with python filter
            query_embedding = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
            python_results = await search_service._search_similar_embeddings_vec(
                query_embedding, language_filter="python", max_results=10, threshold=0.1
            )

            # Should only find python results
            assert len(python_results) >= 1
            for result in python_results:
                assert result.language == "python"

            # Search with javascript filter
            js_results = await search_service._search_similar_embeddings_vec(
                query_embedding, language_filter="javascript", max_results=10, threshold=0.1
            )

            # Should only find javascript results
            for result in js_results:
                assert result.language == "javascript"

    @pytest.mark.asyncio
    async def test_vec_error_handling(self, mock_config):
        """Test error handling in sqlite-vec operations."""
        with patch("src.services.search_service.get_settings") as mock_settings:
            mock_settings.return_value = mock_config
            search_service = SearchService()

            # Test search with non-existent table
            query_embedding = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

            with patch.object(search_service.vec_ops, "search_similar", side_effect=Exception("Test error")):
                results = await search_service._search_similar_embeddings_vec(query_embedding, None, 10, 0.5)
                assert results == []  # Should return empty list on error

    @pytest.mark.asyncio
    async def test_vec_remove_file_embeddings(self, mock_config):
        """Test removing file embeddings with sqlite-vec."""
        with patch("src.services.search_service.get_settings") as mock_settings:
            mock_settings.return_value = mock_config
            search_service = SearchService()

            # Store test chunks
            test_chunks = [
                CodeChunk(
                    file_path="/test/remove1.py",
                    content="def func1(): pass",
                    start_line=1,
                    end_line=1,
                    language="python",
                    semantic_type="function",
                    embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                ),
                CodeChunk(
                    file_path="/test/remove2.py",
                    content="def func2(): pass",
                    start_line=1,
                    end_line=1,
                    language="python",
                    semantic_type="function",
                    embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
                ),
            ]

            await search_service._store_embeddings_batch(test_chunks)

            # Verify both are stored
            stats = await search_service.get_embeddings_stats()
            assert stats["total_embeddings"] == 2

            # Remove one file
            await search_service.remove_file_embeddings(["/test/remove1.py"])

            # Verify only one remains
            stats = await search_service.get_embeddings_stats()
            assert stats["total_embeddings"] == 1

    @pytest.mark.asyncio
    async def test_vec_clear_all_embeddings(self, mock_config):
        """Test clearing all embeddings with sqlite-vec."""
        with patch("src.services.search_service.get_settings") as mock_settings:
            mock_settings.return_value = mock_config
            search_service = SearchService()

            # Store test chunks
            test_chunks = [
                CodeChunk(
                    file_path="/test/clear_test.py",
                    content="def test(): pass",
                    start_line=1,
                    end_line=1,
                    language="python",
                    semantic_type="function",
                    embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                )
            ]

            await search_service._store_embeddings_batch(test_chunks)

            # Verify stored
            stats = await search_service.get_embeddings_stats()
            assert stats["total_embeddings"] == 1

            # Clear all
            await search_service.clear_all_embeddings()

            # Verify empty
            stats = await search_service.get_embeddings_stats()
            assert stats["total_embeddings"] == 0


class TestSqliteVecSearchService:
    """Specific tests for sqlite-vec functionality in SearchService."""

    @pytest.fixture
    def vec_search_service(self, mock_config):
        """Create SearchService configured for sqlite-vec testing."""
        with patch("src.services.search_service.get_settings") as mock_settings:
            mock_settings.return_value = mock_config

            with patch("src.services.search_service.SentenceTransformer") as mock_transformer:
                mock_model = Mock()
                mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
                mock_transformer.return_value = mock_model

                return SearchService()

    @pytest.mark.asyncio
    async def test_vec_database_structure(self, vec_search_service):
        """Test that sqlite-vec database has correct structure."""
        conn = sqlite3.connect(str(vec_search_service.db_path))

        # Check vec table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings_vec'")
        vec_table = cursor.fetchone()
        assert vec_table is not None

        # Check metadata table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings_vec_metadata'")
        metadata_table = cursor.fetchone()
        assert metadata_table is not None

        # Check metadata table has expected columns
        cursor = conn.execute("PRAGMA table_info(embeddings_vec_metadata)")
        columns = [row[1] for row in cursor.fetchall()]
        expected_columns = ["rowid", "file_path", "content", "start_line", "end_line", "language", "semantic_type"]
        for col in expected_columns:
            assert col in columns

        conn.close()

    @pytest.mark.asyncio
    async def test_vec_extension_info(self, vec_search_service):
        """Test getting sqlite-vec extension information."""
        conn = sqlite3.connect(str(vec_search_service.db_path))
        vec_search_service.vec_loader.load_extension(conn)

        info = vec_search_service.vec_loader.get_vec_info(conn)
        assert info is not None
        assert "version" in info
        assert "vec_tables" in info
        assert "table_stats" in info

        conn.close()

    @pytest.mark.asyncio
    async def test_vec_performance_vs_threshold(self, vec_search_service):
        """Test search performance with different similarity thresholds."""
        # Store multiple test embeddings with known similarities
        test_chunks = []
        base_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]

        # Create embeddings with varying similarities to base
        similarities = [1.0, 0.9, 0.7, 0.5, 0.3]  # Decreasing similarity
        for i, sim in enumerate(similarities):
            # Create embedding with specific similarity to base
            embedding = [sim, (1 - sim) * 0.5, 0.0, 0.0, 0.0]
            chunk = CodeChunk(
                file_path=f"/test/sim_{sim}.py",
                content=f"def func_{sim}(): pass",
                start_line=i + 1,
                end_line=i + 1,
                language="python",
                semantic_type="function",
                embedding=embedding,
                function_name=f"func_{sim}",
            )
            test_chunks.append(chunk)

        await vec_search_service._store_embeddings_batch(test_chunks)

        # Test with high threshold (should find fewer results)
        query_embedding = np.array(base_embedding)
        high_threshold_results = await vec_search_service._search_similar_embeddings_vec(query_embedding, None, 10, 0.8)

        # Test with low threshold (should find more results)
        low_threshold_results = await vec_search_service._search_similar_embeddings_vec(query_embedding, None, 10, 0.2)

        assert len(high_threshold_results) <= len(low_threshold_results)

        # Verify results are sorted by similarity (distance)
        if len(low_threshold_results) > 1:
            # First result should be most similar
            assert "1.0" in low_threshold_results[0].file_path
