"""Tests for enhanced data models with scalability features."""

from datetime import datetime
from unittest import TestCase

from src.models.indexing_models import (
    ChunkType,
    CodeChunk,
    FileUpdateRecord,
    IndexUpdateResult,
)


class TestChunkType(TestCase):
    """Test ChunkType enum."""

    def test_chunk_type_values(self):
        """Test that all chunk types have correct string values."""
        assert ChunkType.FILE.value == "file"
        assert ChunkType.CLASS.value == "class"
        assert ChunkType.FUNCTION.value == "function"
        assert ChunkType.METHOD.value == "method"
        assert ChunkType.IMPORT_BLOCK.value == "import_block"
        assert ChunkType.VARIABLE.value == "variable"
        assert ChunkType.CODE_BLOCK.value == "code_block"

    def test_chunk_type_from_string(self):
        """Test creating ChunkType from string values."""
        assert ChunkType("function") == ChunkType.FUNCTION
        assert ChunkType("class") == ChunkType.CLASS


class TestEnhancedCodeChunk(TestCase):
    """Test enhanced CodeChunk model with new scalability fields."""

    def test_basic_code_chunk_creation(self):
        """Test creating basic code chunk with required fields."""
        chunk = CodeChunk(
            file_path="/test/file.py",
            content="def test(): pass",
            start_line=1,
            end_line=1,
            language="python",
            semantic_type="function_definition",
        )
        
        assert chunk.file_path == "/test/file.py"
        assert chunk.content == "def test(): pass"
        assert chunk.language == "python"
        assert chunk.semantic_type == "function_definition"

    def test_enhanced_fields_default_none(self):
        """Test that new enhanced fields default to None."""
        chunk = CodeChunk(
            file_path="/test/file.py",
            content="def test(): pass",
            start_line=1,
            end_line=1,
            language="python",
            semantic_type="function_definition",
        )
        
        assert chunk.chunk_type is None
        assert chunk.parent_chunk_id is None
        assert chunk.dependencies is None
        assert chunk.dependents is None
        assert chunk.complexity_score is None
        assert chunk.last_modified is None
        assert chunk.file_hash is None

    def test_enhanced_fields_with_values(self):
        """Test setting enhanced fields with values."""
        now = datetime.now()
        chunk = CodeChunk(
            file_path="/test/file.py",
            content="def test(): pass",
            start_line=1,
            end_line=1,
            language="python",
            semantic_type="function_definition",
            chunk_type=ChunkType.FUNCTION,
            parent_chunk_id="class_123",
            dependencies=["module_a", "module_b"],
            dependents=["test_file.py"],
            complexity_score=5.2,
            last_modified=now,
            file_hash="abc123",
        )
        
        assert chunk.chunk_type == ChunkType.FUNCTION
        assert chunk.parent_chunk_id == "class_123"
        assert chunk.dependencies == ["module_a", "module_b"]
        assert chunk.dependents == ["test_file.py"]
        assert chunk.complexity_score == 5.2
        assert chunk.last_modified == now
        assert chunk.file_hash == "abc123"

    def test_backward_compatibility(self):
        """Test that existing metadata fields still work."""
        chunk = CodeChunk(
            file_path="/test/file.py",
            content="class Test: pass",
            start_line=1,
            end_line=1,
            language="python",
            semantic_type="class_definition",
            function_signature=None,
            class_name="Test",
            function_name=None,
            parameter_types=None,
            return_type=None,
            inheritance_chain=["BaseClass"],
            import_statements=["import os"],
            docstring="Test class",
            interfaces=None,
            decorators=["@dataclass"],
        )
        
        assert chunk.class_name == "Test"
        assert chunk.inheritance_chain == ["BaseClass"]
        assert chunk.import_statements == ["import os"]
        assert chunk.docstring == "Test class"
        assert chunk.decorators == ["@dataclass"]


class TestFileUpdateRecord(TestCase):
    """Test FileUpdateRecord model."""

    def test_file_update_record_creation(self):
        """Test creating FileUpdateRecord."""
        now = datetime.now()
        record = FileUpdateRecord(
            file_path="/test/file.py",
            content_hash="abc123",
            last_indexed=now,
            chunk_ids=["chunk_1", "chunk_2"],
            dependencies=["module_a", "module_b"],
        )
        
        assert record.file_path == "/test/file.py"
        assert record.content_hash == "abc123"
        assert record.last_indexed == now
        assert record.chunk_ids == ["chunk_1", "chunk_2"]
        assert record.dependencies == ["module_a", "module_b"]

    def test_file_update_record_serialization(self):
        """Test JSON serialization of FileUpdateRecord."""
        now = datetime.now()
        record = FileUpdateRecord(
            file_path="/test/file.py",
            content_hash="abc123",
            last_indexed=now,
            chunk_ids=["chunk_1"],
            dependencies=["module_a"],
        )
        
        # Test that it can be serialized to dict
        data = record.model_dump()
        assert isinstance(data, dict)
        assert data["file_path"] == "/test/file.py"
        assert data["content_hash"] == "abc123"
        
        # Test that it can be reconstructed
        rebuilt = FileUpdateRecord.model_validate(data)
        assert rebuilt.file_path == record.file_path
        assert rebuilt.content_hash == record.content_hash


class TestIndexUpdateResult(TestCase):
    """Test IndexUpdateResult model."""

    def test_index_update_result_creation(self):
        """Test creating IndexUpdateResult."""
        result = IndexUpdateResult(
            updated_chunks=["chunk_1", "chunk_2"],
            deleted_chunks=["chunk_3"],
            affected_files=["/test/file1.py", "/test/file2.py"],
            processing_time=150.5,
        )
        
        assert result.updated_chunks == ["chunk_1", "chunk_2"]
        assert result.deleted_chunks == ["chunk_3"]
        assert result.affected_files == ["/test/file1.py", "/test/file2.py"]
        assert result.processing_time == 150.5

    def test_empty_update_result(self):
        """Test IndexUpdateResult with no changes."""
        result = IndexUpdateResult(
            updated_chunks=[],
            deleted_chunks=[],
            affected_files=[],
            processing_time=0.0,
        )
        
        assert len(result.updated_chunks) == 0
        assert len(result.deleted_chunks) == 0
        assert len(result.affected_files) == 0
        assert result.processing_time == 0.0

    def test_update_result_serialization(self):
        """Test JSON serialization of IndexUpdateResult."""
        result = IndexUpdateResult(
            updated_chunks=["chunk_1"],
            deleted_chunks=["chunk_2"],
            affected_files=["/test/file.py"],
            processing_time=100.0,
        )
        
        # Test serialization
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["updated_chunks"] == ["chunk_1"]
        assert data["processing_time"] == 100.0
        
        # Test reconstruction
        rebuilt = IndexUpdateResult.model_validate(data)
        assert rebuilt.updated_chunks == result.updated_chunks
        assert rebuilt.processing_time == result.processing_time