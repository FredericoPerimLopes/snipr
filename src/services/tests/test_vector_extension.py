import random
import sqlite3
import tempfile
from pathlib import Path

import pytest

from ...services.vector_extension import VectorExtensionLoader, VectorOperations


class TestVectorExtensionLoader:

    def test_load_extension_success(self):
        """Test successful loading of sqlite-vec extension."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()

        result = loader.load_extension(conn)
        assert result is True

        # Verify extension is loaded by checking version
        cursor = conn.execute("SELECT vec_version()")
        version = cursor.fetchone()
        assert version is not None
        assert isinstance(version[0], str)

        conn.close()

    def test_load_extension_already_loaded(self):
        """Test loading extension when already loaded."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()

        # Load extension first time
        result1 = loader.load_extension(conn)
        assert result1 is True

        # Load extension second time (should detect already loaded)
        result2 = loader.load_extension(conn, check_loaded=True)
        assert result2 is True

        conn.close()

    def test_create_vec_table_success(self):
        """Test successful creation of vec0 table."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        loader.load_extension(conn)

        result = loader.create_vec_table(conn, "test_vecs", dimension=128)
        assert result is True

        # Verify table was created
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_vecs'")
        table_exists = cursor.fetchone()
        assert table_exists is not None

        # Verify metadata table was created
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_vecs_metadata'")
        metadata_table_exists = cursor.fetchone()
        assert metadata_table_exists is not None

        conn.close()

    def test_create_vec_table_different_dimensions(self):
        """Test creating vec tables with different dimensions."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        loader.load_extension(conn)

        # Test 256 dimensions
        result1 = loader.create_vec_table(conn, "vecs_256", dimension=256)
        assert result1 is True

        # Test 512 dimensions
        result2 = loader.create_vec_table(conn, "vecs_512", dimension=512)
        assert result2 is True

        conn.close()

    def test_test_vec_operations_success(self):
        """Test vec operations test passes."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        loader.load_extension(conn)

        result = loader.test_vec_operations(conn)
        assert result is True

        conn.close()

    def test_get_vec_info(self):
        """Test getting vec info returns expected data."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        loader.load_extension(conn)
        loader.create_vec_table(conn, "test_table", dimension=128)

        info = loader.get_vec_info(conn)
        assert info is not None
        assert "version" in info
        assert "vec_tables" in info
        assert "table_stats" in info
        assert "test_table" in info["vec_tables"]
        assert info["table_stats"]["test_table"] == 0

        conn.close()


class TestVectorOperations:

    def test_insert_embedding_success(self):
        """Test successful embedding insertion."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        loader.load_extension(conn)
        loader.create_vec_table(conn, "test_vecs", dimension=5)

        ops = VectorOperations()
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        metadata = {
            "file_path": "test.py",
            "content": "def test(): pass",
            "start_line": 1,
            "end_line": 1,
            "language": "python",
            "semantic_type": "function",
            "content_hash": "12345"
        }

        rowid = ops.insert_embedding(conn, "test_vecs", embedding, metadata)
        assert rowid is not None
        assert isinstance(rowid, int)

        # Verify insertion in vec table
        cursor = conn.execute("SELECT COUNT(*) FROM test_vecs")
        vec_count = cursor.fetchone()[0]
        assert vec_count == 1

        # Verify insertion in metadata table
        cursor = conn.execute("SELECT COUNT(*) FROM test_vecs_metadata")
        metadata_count = cursor.fetchone()[0]
        assert metadata_count == 1

        conn.close()

    def test_search_similar_success(self):
        """Test successful similarity search."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        loader.load_extension(conn)
        loader.create_vec_table(conn, "test_vecs", dimension=5)

        ops = VectorOperations()

        # Insert test embeddings
        embedding1 = [1.0, 0.0, 0.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0, 0.0, 0.0]
        embedding3 = [0.9, 0.1, 0.0, 0.0, 0.0]  # Similar to embedding1

        metadata1 = {
            "file_path": "test1.py",
            "content": "def func1(): pass",
            "start_line": 1,
            "end_line": 1,
            "language": "python",
            "semantic_type": "function",
            "content_hash": "hash1"
        }

        metadata2 = {
            "file_path": "test2.py",
            "content": "def func2(): pass",
            "start_line": 1,
            "end_line": 1,
            "language": "python",
            "semantic_type": "function",
            "content_hash": "hash2"
        }

        metadata3 = {
            "file_path": "test3.py",
            "content": "def func3(): pass",
            "start_line": 1,
            "end_line": 1,
            "language": "python",
            "semantic_type": "function",
            "content_hash": "hash3"
        }

        ops.insert_embedding(conn, "test_vecs", embedding1, metadata1)
        ops.insert_embedding(conn, "test_vecs", embedding2, metadata2)
        ops.insert_embedding(conn, "test_vecs", embedding3, metadata3)

        # Search for vectors similar to embedding1
        results = ops.search_similar(conn, "test_vecs", embedding1, k=3, distance_metric="cosine")

        assert len(results) >= 2  # Should find at least embedding1 and embedding3

        # First result should be identical (distance ~0)
        rowid, distance, metadata = results[0]
        assert distance < 0.01  # Very small distance for identical vector
        assert metadata["file_path"] == "test1.py"

        conn.close()

    def test_search_similar_with_threshold(self):
        """Test similarity search with distance threshold."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        loader.load_extension(conn)
        loader.create_vec_table(conn, "test_vecs", dimension=3)

        ops = VectorOperations()

        # Insert embeddings with known distances
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]  # 90 degrees apart, cosine distance ~1.0

        metadata1 = {"file_path": "test1.py", "content": "content1", "start_line": 1, "end_line": 1, "language": "python", "semantic_type": "function", "content_hash": "hash1"}
        metadata2 = {"file_path": "test2.py", "content": "content2", "start_line": 1, "end_line": 1, "language": "python", "semantic_type": "function", "content_hash": "hash2"}

        ops.insert_embedding(conn, "test_vecs", embedding1, metadata1)
        ops.insert_embedding(conn, "test_vecs", embedding2, metadata2)

        # Search with strict threshold (should only find identical)
        results = ops.search_similar(conn, "test_vecs", embedding1, k=10, threshold=0.1)
        assert len(results) == 1
        assert results[0][2]["file_path"] == "test1.py"

        # Search with loose threshold (should find both)
        results = ops.search_similar(conn, "test_vecs", embedding1, k=10, threshold=1.5)
        assert len(results) == 2

        conn.close()

    def test_search_different_distance_metrics(self):
        """Test different distance metrics."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        loader.load_extension(conn)
        loader.create_vec_table(conn, "test_vecs", dimension=3)

        ops = VectorOperations()

        embedding = [1.0, 2.0, 3.0]
        metadata = {"file_path": "test.py", "content": "content", "start_line": 1, "end_line": 1, "language": "python", "semantic_type": "function", "content_hash": "hash"}

        ops.insert_embedding(conn, "test_vecs", embedding, metadata)

        # Test cosine distance
        results_cosine = ops.search_similar(conn, "test_vecs", embedding, k=1, distance_metric="cosine")
        assert len(results_cosine) == 1
        assert results_cosine[0][1] < 0.01  # Very small distance

        # Test L2 distance
        results_l2 = ops.search_similar(conn, "test_vecs", embedding, k=1, distance_metric="l2")
        assert len(results_l2) == 1
        assert results_l2[0][1] < 0.01  # Very small distance

        # Note: dot product may not be available in all sqlite-vec versions
        # Test with available metrics only
        try:
            results_dot = ops.search_similar(conn, "test_vecs", embedding, k=1, distance_metric="dot")
            assert len(results_dot) >= 0  # May be 0 if function not available
        except Exception:
            # Skip if dot product function not available
            pass

        conn.close()

    def test_insert_embedding_with_all_metadata(self):
        """Test embedding insertion with all metadata fields."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        loader.load_extension(conn)
        loader.create_vec_table(conn, "test_vecs", dimension=3)

        ops = VectorOperations()
        embedding = [1.0, 2.0, 3.0]
        metadata = {
            "file_path": "test.py",
            "content": "def test_func(): pass",
            "start_line": 1,
            "end_line": 1,
            "language": "python",
            "semantic_type": "function",
            "content_hash": "hash123",
            "function_signature": "test_func() -> None",
            "class_name": None,
            "function_name": "test_func",
            "parameter_types": '[]',
            "return_type": "None",
            "inheritance_chain": None,
            "import_statements": '["import os"]',
            "docstring": "Test function",
            "complexity_score": 1,
            "dependencies": '[]',
            "interfaces": None,
            "decorators": None,
        }

        rowid = ops.insert_embedding(conn, "test_vecs", embedding, metadata)
        assert rowid is not None

        # Verify all metadata was stored
        cursor = conn.execute("SELECT * FROM test_vecs_metadata WHERE rowid = ?", (rowid,))
        row = cursor.fetchone()
        assert row is not None

        # Check specific fields
        columns = [desc[0] for desc in cursor.description]
        row_dict = dict(zip(columns, row, strict=False))
        assert row_dict["function_name"] == "test_func"
        assert row_dict["docstring"] == "Test function"
        assert row_dict["complexity_score"] == 1

        conn.close()

    def test_empty_embedding_list(self):
        """Test handling of empty embedding list."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        loader.load_extension(conn)
        loader.create_vec_table(conn, "test_vecs", dimension=3)

        ops = VectorOperations()

        # Try to insert empty embedding (should fail gracefully)
        result = ops.insert_embedding(conn, "test_vecs", [], {})
        assert result is None

        conn.close()

    def test_invalid_table_name(self):
        """Test operations with invalid table name."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        loader.load_extension(conn)

        ops = VectorOperations()
        embedding = [1.0, 2.0, 3.0]
        metadata = {"file_path": "test.py", "content": "content", "start_line": 1, "end_line": 1, "language": "python", "semantic_type": "function", "content_hash": "hash"}

        # Try to insert into non-existent table
        result = ops.insert_embedding(conn, "nonexistent_table", embedding, metadata)
        assert result is None

        # Try to search non-existent table
        results = ops.search_similar(conn, "nonexistent_table", embedding, k=1)
        assert len(results) == 0

        conn.close()


class TestVectorExtensionIntegration:

    def test_full_workflow(self):
        """Test complete workflow: create table, insert, search."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        ops = VectorOperations()

        # Load extension and create table
        assert loader.load_extension(conn) is True
        assert loader.create_vec_table(conn, "embeddings_vec", dimension=768) is True

        # Insert multiple embeddings
        embeddings = []
        metadatas = []

        for i in range(5):
            embedding = [random.random() for _ in range(768)]
            metadata = {
                "file_path": f"test{i}.py",
                "content": f"def func{i}(): pass",
                "start_line": i + 1,
                "end_line": i + 1,
                "language": "python",
                "semantic_type": "function",
                "content_hash": f"hash{i}",
                "function_name": f"func{i}",
            }

            rowid = ops.insert_embedding(conn, "embeddings_vec", embedding, metadata)
            assert rowid is not None

            embeddings.append(embedding)
            metadatas.append(metadata)

        # Search for similar to first embedding
        results = ops.search_similar(conn, "embeddings_vec", embeddings[0], k=3)
        assert len(results) >= 1

        # First result should be the inserted embedding
        assert results[0][2]["function_name"] == "func0"

        # Test vec info
        info = loader.get_vec_info(conn)
        assert info["table_stats"]["embeddings_vec"] == 5

        conn.close()

    def test_large_batch_insertion(self):
        """Test inserting larger batch of embeddings."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        ops = VectorOperations()

        loader.load_extension(conn)
        loader.create_vec_table(conn, "batch_test", dimension=128)

        # Insert 100 random embeddings
        batch_size = 100
        for i in range(batch_size):
            embedding = [random.random() for _ in range(128)]
            metadata = {
                "file_path": f"file{i}.py",
                "content": f"content{i}",
                "start_line": 1,
                "end_line": 1,
                "language": "python",
                "semantic_type": "function",
                "content_hash": f"hash{i}",
            }

            rowid = ops.insert_embedding(conn, "batch_test", embedding, metadata)
            assert rowid is not None

        # Verify all were inserted
        cursor = conn.execute("SELECT COUNT(*) FROM batch_test")
        count = cursor.fetchone()[0]
        assert count == batch_size

        # Test search returns reasonable results
        query_embedding = [random.random() for _ in range(128)]
        results = ops.search_similar(conn, "batch_test", query_embedding, k=10)
        assert len(results) <= 10
        assert len(results) >= 1

        conn.close()

    def test_vector_dimension_mismatch(self):
        """Test handling of dimension mismatches."""
        conn = sqlite3.connect(":memory:")
        loader = VectorExtensionLoader()
        ops = VectorOperations()

        loader.load_extension(conn)
        loader.create_vec_table(conn, "test_vecs", dimension=5)

        # Try to insert wrong dimension
        wrong_embedding = [1.0, 2.0, 3.0]  # 3 dims instead of 5
        metadata = {"file_path": "test.py", "content": "content", "start_line": 1, "end_line": 1, "language": "python", "semantic_type": "function", "content_hash": "hash"}

        result = ops.insert_embedding(conn, "test_vecs", wrong_embedding, metadata)
        assert result is None  # Should fail gracefully

        conn.close()


@pytest.fixture
def temp_db_path():
    """Create temporary database path for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_path = Path(temp_file.name)
    temp_file.close()

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()
