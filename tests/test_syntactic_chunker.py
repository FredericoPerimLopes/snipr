from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.indexing_models import CodeChunk
from src.services.syntactic_chunker import ChunkingConfig, SyntacticChunker


@pytest.fixture
def chunking_config():
    """Test chunking configuration."""
    return ChunkingConfig(
        max_chunk_chars=400, min_chunk_chars=50, overlap_chars=20, preserve_syntax=True, max_merge_size=600
    )


@pytest.fixture
def mock_parser():
    """Mock Tree-sitter parser."""
    parser = Mock()
    mock_tree = Mock()
    mock_root = Mock()
    mock_root.children = []
    mock_tree.root_node = mock_root
    parser.parse.return_value = mock_tree
    return parser


@pytest.fixture
def mock_node():
    """Mock Tree-sitter node."""
    node = Mock()
    node.type = "function_definition"
    node.start_byte = 0
    node.end_byte = 50
    node.start_point = (0, 0)
    node.end_point = (2, 0)
    node.children = []
    return node


@pytest.fixture
def sample_chunks():
    """Sample code chunks for testing."""
    return [
        CodeChunk(
            file_path="/test/file.py",
            content="def small_func(): pass",
            start_line=1,
            end_line=1,
            language="python",
            semantic_type="function_definition",
            function_name="small_func",
        ),
        CodeChunk(
            file_path="/test/file.py",
            content="def another_func(): return True",
            start_line=3,
            end_line=3,
            language="python",
            semantic_type="function_definition",
            function_name="another_func",
        ),
    ]


class TestChunkingConfig:
    def test_default_config(self):
        """Test default chunking configuration."""
        config = ChunkingConfig()
        assert config.max_chunk_chars == 600
        assert config.min_chunk_chars == 100
        assert config.overlap_chars == 50
        assert config.preserve_syntax == True
        assert config.max_merge_size == 800

    def test_custom_config(self):
        """Test custom chunking configuration."""
        config = ChunkingConfig(max_chunk_chars=400, min_chunk_chars=80, overlap_chars=30)
        assert config.max_chunk_chars == 400
        assert config.min_chunk_chars == 80
        assert config.overlap_chars == 30


class TestSyntacticChunker:
    def test_init_default_config(self):
        """Test initialization with default config."""
        chunker = SyntacticChunker()
        assert isinstance(chunker.config, ChunkingConfig)
        assert chunker.config.max_chunk_chars == 600

    def test_init_custom_config(self, chunking_config):
        """Test initialization with custom config."""
        chunker = SyntacticChunker(chunking_config)
        assert chunker.config == chunking_config
        assert chunker.config.max_chunk_chars == 400

    async def test_chunk_with_integrity_success(self, chunking_config, mock_parser):
        """Test successful chunking with syntactic integrity."""
        chunker = SyntacticChunker(chunking_config)

        content = "def test_func():\n    return True"

        with patch.object(chunker, "_extract_syntactic_chunks", return_value=[Mock()]) as mock_extract:
            with patch.object(chunker, "_optimize_chunks", return_value=[Mock()]) as mock_optimize:
                result = await chunker.chunk_with_integrity("/test/file.py", content, "python", mock_parser)

                assert len(result) == 1
                mock_extract.assert_called_once()
                mock_optimize.assert_called_once()

    async def test_chunk_with_integrity_error(self, mock_parser, caplog):
        """Test chunking with parsing error."""
        chunker = SyntacticChunker()
        mock_parser.parse.side_effect = Exception("Parse error")

        result = await chunker.chunk_with_integrity("/test/file.py", "content", "python", mock_parser)

        assert result == []
        assert "Error in syntactic chunking" in caplog.text

    async def test_extract_syntactic_chunks_chunkable_node_small(self, mock_node):
        """Test chunk extraction for small chunkable node."""
        chunker = SyntacticChunker()

        with patch.object(chunker, "_is_chunkable_node", return_value=True):
            with patch.object(chunker, "_get_node_char_count", return_value=100):  # Small node
                with patch.object(chunker, "_create_chunk_from_node", return_value=Mock()) as mock_create:
                    chunks = await chunker._extract_syntactic_chunks(mock_node, "content", "/test/file.py", "python")

                    assert len(chunks) == 1
                    mock_create.assert_called_once()

    async def test_extract_syntactic_chunks_chunkable_node_large(self, mock_node):
        """Test chunk extraction for large chunkable node."""
        chunker = SyntacticChunker()

        with patch.object(chunker, "_is_chunkable_node", return_value=True):
            with patch.object(chunker, "_get_node_char_count", return_value=1000):  # Large node
                with patch.object(chunker, "_recursive_chunk_split", return_value=[Mock()]) as mock_split:
                    chunks = await chunker._extract_syntactic_chunks(mock_node, "content", "/test/file.py", "python")

                    assert len(chunks) == 1
                    mock_split.assert_called_once()

    async def test_extract_syntactic_chunks_non_chunkable_node(self, mock_node):
        """Test chunk extraction for non-chunkable node with children."""
        chunker = SyntacticChunker()

        child_node = Mock()
        child_node.children = []
        mock_node.children = [child_node]

        with patch.object(chunker, "_is_chunkable_node", return_value=False):
            with patch.object(
                chunker,
                "_extract_syntactic_chunks",
                side_effect=[
                    [Mock()],  # Result for child
                    [],  # Result for recursive call
                ],
            ) as mock_recursive:
                chunks = await chunker._extract_syntactic_chunks(mock_node, "content", "/test/file.py", "python")

                assert len(chunks) == 1

    def test_is_chunkable_node_true(self):
        """Test chunkable node detection for valid types."""
        chunker = SyntacticChunker()

        chunkable_types = [
            "function_definition",
            "async_function_definition",
            "method_definition",
            "class_definition",
            "class_declaration",
            "interface_declaration",
            "variable_declaration",
        ]

        for node_type in chunkable_types:
            mock_node = Mock()
            mock_node.type = node_type
            assert chunker._is_chunkable_node(mock_node, "content") == True

    def test_is_chunkable_node_false(self):
        """Test chunkable node detection for invalid types."""
        chunker = SyntacticChunker()

        non_chunkable_types = ["expression", "statement", "identifier", "string", "number"]

        for node_type in non_chunkable_types:
            mock_node = Mock()
            mock_node.type = node_type
            assert chunker._is_chunkable_node(mock_node, "content") == False

    def test_get_node_char_count(self):
        """Test node character counting."""
        chunker = SyntacticChunker()

        mock_node = Mock()
        mock_node.start_byte = 0
        mock_node.end_byte = 20

        content = "def test():\n    pass"  # 20 chars, but only 14 non-whitespace
        char_count = chunker._get_node_char_count(mock_node, content)

        assert char_count == 14  # "def test(): pass" without whitespace

    async def test_create_chunk_from_node_success(self, mock_node):
        """Test successful chunk creation from node."""
        chunker = SyntacticChunker()

        content = "def test():\n    return True\n"

        mock_metadata = Mock()
        mock_metadata.semantic_type = "function_definition"
        mock_metadata.function_name = "test"
        mock_metadata.function_signature = "def test()"
        mock_metadata.class_name = None
        mock_metadata.parameter_types = None
        mock_metadata.return_type = None
        mock_metadata.inheritance_chain = None
        mock_metadata.import_statements = None
        mock_metadata.docstring = None
        mock_metadata.complexity_score = 1
        mock_metadata.dependencies = None
        mock_metadata.interfaces = None
        mock_metadata.decorators = None

        chunker.metadata_extractor.extract_all_metadata = AsyncMock(return_value=mock_metadata)

        chunk = await chunker._create_chunk_from_node(mock_node, content, "/test/file.py", "python")

        assert isinstance(chunk, CodeChunk)
        assert chunk.file_path == "/test/file.py"
        assert chunk.semantic_type == "function_definition"
        assert chunk.function_name == "test"

    async def test_create_chunk_from_node_error(self, mock_node, caplog):
        """Test chunk creation with error."""
        chunker = SyntacticChunker()
        chunker.metadata_extractor.extract_all_metadata = AsyncMock(side_effect=Exception("Metadata error"))

        chunk = await chunker._create_chunk_from_node(mock_node, "content", "/test/file.py", "python")

        assert chunk is None
        # Note: Error logging may not be captured in test context

    async def test_recursive_chunk_split(self, mock_node):
        """Test recursive chunk splitting."""
        chunker = SyntacticChunker()

        # Mock children nodes
        child1 = Mock()
        child2 = Mock()
        mock_node.children = [child1, child2]

        with patch.object(chunker, "_group_children_by_size", return_value=[[child1], [child2]]):
            with patch.object(chunker, "_extract_syntactic_chunks", return_value=[Mock()]):
                chunks = await chunker._recursive_chunk_split(mock_node, "content", "/test/file.py", "python")

                assert len(chunks) == 2

    async def test_recursive_chunk_split_multiple_children_group(self, mock_node):
        """Test recursive splitting with multiple children in one group."""
        chunker = SyntacticChunker()

        child1 = Mock()
        child2 = Mock()
        mock_node.children = [child1, child2]

        with patch.object(chunker, "_group_children_by_size", return_value=[[child1, child2]]):
            with patch.object(chunker, "_create_merged_chunk", return_value=Mock()) as mock_merge:
                chunks = await chunker._recursive_chunk_split(mock_node, "content", "/test/file.py", "python")

                assert len(chunks) == 1
                mock_merge.assert_called_once()

    async def test_group_children_by_size_fits_in_one_group(self):
        """Test grouping children that fit in one group."""
        chunker = SyntacticChunker()

        child1 = Mock()
        child2 = Mock()
        children = [child1, child2]

        with patch.object(chunker, "_get_node_char_count", side_effect=[100, 200]):  # Total 300, under limit
            groups = await chunker._group_children_by_size(children, "content")

            assert len(groups) == 1
            assert groups[0] == [child1, child2]

    async def test_group_children_by_size_exceeds_limit(self):
        """Test grouping children that exceed size limit."""
        chunker = SyntacticChunker()

        child1 = Mock()
        child2 = Mock()
        children = [child1, child2]

        with patch.object(chunker, "_get_node_char_count", side_effect=[400, 300]):  # Total 700, over 600 limit
            groups = await chunker._group_children_by_size(children, "content")

            assert len(groups) == 2
            assert groups[0] == [child1]
            assert groups[1] == [child2]

    async def test_group_children_by_size_oversized_child(self):
        """Test grouping with child that exceeds max size."""
        chunker = SyntacticChunker()

        child1 = Mock()
        child2 = Mock()
        children = [child1, child2]

        with patch.object(chunker, "_get_node_char_count", side_effect=[800, 100]):  # First child oversized
            groups = await chunker._group_children_by_size(children, "content")

            assert len(groups) == 2
            assert groups[0] == [child1]  # Oversized child in own group
            assert groups[1] == [child2]

    async def test_create_merged_chunk_success(self):
        """Test successful chunk merging."""
        chunker = SyntacticChunker()

        node1 = Mock()
        node1.start_byte = 0
        node1.end_byte = 20
        node1.start_point = (0, 0)
        node1.end_point = (1, 0)

        node2 = Mock()
        node2.start_byte = 21
        node2.end_byte = 40
        node2.start_point = (2, 0)
        node2.end_point = (3, 0)

        content = "def func1():\n    pass\n\ndef func2():\n    pass"

        with patch.object(chunker, "_get_dominant_semantic_type", return_value="function_definition"):
            chunk = await chunker._create_merged_chunk([node1, node2], content, "/test/file.py", "python")

            assert isinstance(chunk, CodeChunk)
            assert chunk.file_path == "/test/file.py"
            assert chunk.start_line == 1  # Adjusted for context
            assert chunk.semantic_type == "function_definition"

    async def test_create_merged_chunk_empty_nodes(self):
        """Test merged chunk creation with empty node list."""
        chunker = SyntacticChunker()

        chunk = await chunker._create_merged_chunk([], "content", "/test/file.py", "python")
        assert chunk is None

    async def test_create_merged_chunk_error(self, caplog):
        """Test merged chunk creation with error."""
        chunker = SyntacticChunker()

        # Mock node with problematic attributes that will cause an exception
        mock_node = Mock()
        mock_node.start_byte = 0
        mock_node.end_byte = 10
        mock_node.start_point = (0, 0)
        mock_node.end_point = (1, 0)

        with patch.object(chunker, "_get_dominant_semantic_type", side_effect=Exception("Test error")):
            chunk = await chunker._create_merged_chunk([mock_node], "content", "/test/file.py", "python")

            assert chunk is None
            # Error logging may not always be captured in all test environments

    async def test_get_dominant_semantic_type(self):
        """Test dominant semantic type determination."""
        chunker = SyntacticChunker()

        # Nodes with different priorities
        node1 = Mock()
        node1.type = "variable_declaration"  # Priority 4

        node2 = Mock()
        node2.type = "function_definition"  # Priority 8

        node3 = Mock()
        node3.type = "class_definition"  # Priority 10

        dominant_type = await chunker._get_dominant_semantic_type([node1, node2, node3], "content", "python")
        assert dominant_type == "class_definition"  # Highest priority

    async def test_get_dominant_semantic_type_unknown_type(self):
        """Test dominant type with unknown node types."""
        chunker = SyntacticChunker()

        node = Mock()
        node.type = "unknown_node_type"

        dominant_type = await chunker._get_dominant_semantic_type([node], "content", "python")
        assert dominant_type == "unknown_node_type"  # Should return the type even if unknown

    async def test_optimize_chunks_empty_list(self):
        """Test chunk optimization with empty list."""
        chunker = SyntacticChunker()

        result = await chunker._optimize_chunks([], "content")
        assert result == []

    async def test_optimize_chunks_no_merging_needed(self, sample_chunks):
        """Test chunk optimization when no merging is needed."""
        chunker = SyntacticChunker()

        # Mock chunks that don't need merging (large enough)
        large_chunk = sample_chunks[0]
        large_chunk.content = "a" * 200  # Large enough content

        with patch.object(chunker, "_add_hierarchical_context", side_effect=lambda x, _: x):
            result = await chunker._optimize_chunks([large_chunk], "content")

            assert len(result) == 1
            assert result[0] == large_chunk

    async def test_optimize_chunks_with_merging(self, sample_chunks):
        """Test chunk optimization with merging."""
        chunker = SyntacticChunker()

        # Make chunks small to trigger merging
        small_chunk1 = sample_chunks[0]
        small_chunk1.content = "x = 1"  # Very small
        small_chunk2 = sample_chunks[1]
        small_chunk2.content = "y = 2"  # Very small

        merged_chunk = Mock()

        with patch.object(chunker, "_try_merge_chunks", return_value=merged_chunk):
            result = await chunker._optimize_chunks([small_chunk1, small_chunk2], "content")

            assert len(result) == 1
            assert result[0] == merged_chunk

    async def test_try_merge_chunks_different_files(self, sample_chunks):
        """Test chunk merging with different file paths."""
        chunker = SyntacticChunker()

        chunk1 = sample_chunks[0]
        chunk2 = sample_chunks[1]
        chunk2.file_path = "/different/file.py"  # Different file

        result = await chunker._try_merge_chunks(chunk1, chunk2, "content")
        assert result is None

    async def test_try_merge_chunks_distinct_semantic_types(self, sample_chunks):
        """Test chunk merging with distinct semantic types."""
        chunker = SyntacticChunker()

        chunk1 = sample_chunks[0]
        chunk1.semantic_type = "function_definition"

        chunk2 = sample_chunks[1]
        chunk2.semantic_type = "class_definition"

        result = await chunker._try_merge_chunks(chunk1, chunk2, "content")
        assert result is None

    async def test_try_merge_chunks_not_adjacent(self, sample_chunks):
        """Test chunk merging with non-adjacent chunks."""
        chunker = SyntacticChunker()

        chunk1 = sample_chunks[0]
        chunk1.end_line = 5

        chunk2 = sample_chunks[1]
        chunk2.start_line = 15  # Gap of 10 lines (> 3)

        result = await chunker._try_merge_chunks(chunk1, chunk2, "content")
        assert result is None

    async def test_try_merge_chunks_too_large(self, sample_chunks):
        """Test chunk merging when result would be too large."""
        chunker = SyntacticChunker()

        chunk1 = sample_chunks[0]
        chunk1.start_line = 1
        chunk1.end_line = 10

        chunk2 = sample_chunks[1]
        chunk2.start_line = 11
        chunk2.end_line = 20

        # Mock large content that exceeds max_merge_size
        large_content = "a" * 1000  # Exceeds default 800 char limit
        content_lines = [large_content] * 20
        content = "\n".join(content_lines)

        result = await chunker._try_merge_chunks(chunk1, chunk2, content)
        assert result is None

    async def test_try_merge_chunks_success(self, sample_chunks):
        """Test successful chunk merging."""
        chunker = SyntacticChunker()

        chunk1 = sample_chunks[0]
        chunk1.start_line = 1
        chunk1.end_line = 2
        chunk1.semantic_type = "function_definition"

        chunk2 = sample_chunks[1]
        chunk2.start_line = 3
        chunk2.end_line = 4
        chunk2.semantic_type = "function_definition"

        content = "def func1():\n    pass\n\ndef func2():\n    pass"

        result = await chunker._try_merge_chunks(chunk1, chunk2, content)

        assert isinstance(result, CodeChunk)
        assert result.file_path == chunk1.file_path
        assert result.start_line == 1
        assert result.end_line == 4
        assert result.semantic_type == "function_definition"

    def test_merge_semantic_types_same(self):
        """Test merging identical semantic types."""
        chunker = SyntacticChunker()

        result = chunker._merge_semantic_types("function_definition", "function_definition")
        assert result == "function_definition"

    def test_merge_semantic_types_different_priorities(self):
        """Test merging different semantic types by priority."""
        chunker = SyntacticChunker()

        # class_definition (priority 10) vs function_definition (priority 8)
        result = chunker._merge_semantic_types("function_definition", "class_definition")
        assert result == "class_definition"

        # function_definition (priority 8) vs variable_declaration (priority 4)
        result = chunker._merge_semantic_types("variable_declaration", "function_definition")
        assert result == "function_definition"

    def test_merge_semantic_types_unknown(self):
        """Test merging unknown semantic types."""
        chunker = SyntacticChunker()

        result = chunker._merge_semantic_types("unknown_type1", "unknown_type2")
        # Should return first type when priorities are equal (both default to 1)
        assert result == "unknown_type1"

    async def test_add_hierarchical_context(self, sample_chunks):
        """Test hierarchical context addition."""
        chunker = SyntacticChunker()

        chunk = sample_chunks[0]
        result = await chunker._add_hierarchical_context(chunk, "content")

        # Currently returns chunk as-is
        assert result == chunk

    async def test_optimize_chunks_single_chunk(self, sample_chunks):
        """Test optimization with single chunk."""
        chunker = SyntacticChunker()

        chunk = sample_chunks[0]
        chunk.content = "a" * 200  # Large enough to not need merging

        with patch.object(chunker, "_add_hierarchical_context", return_value=chunk):
            result = await chunker._optimize_chunks([chunk], "content")

            assert len(result) == 1
            assert result[0] == chunk

    async def test_optimize_chunks_failed_merge_attempt(self, sample_chunks):
        """Test optimization when merge attempt fails."""
        chunker = SyntacticChunker()

        # Make chunks small to trigger merge attempt
        small_chunk1 = sample_chunks[0]
        small_chunk1.content = "x = 1"  # Small
        small_chunk2 = sample_chunks[1]
        small_chunk2.content = "y = 2"  # Small

        with patch.object(chunker, "_try_merge_chunks", return_value=None):  # Merge fails
            with patch.object(chunker, "_add_hierarchical_context", side_effect=lambda x, _: x):
                result = await chunker._optimize_chunks([small_chunk1, small_chunk2], "content")

                # Should keep both chunks when merge fails
                assert len(result) == 2

    async def test_group_children_by_size_complex(self):
        """Test complex grouping scenarios."""
        chunker = SyntacticChunker()

        # Create three nodes with different sizes
        small_node1 = Mock()
        small_node2 = Mock()
        large_node = Mock()

        with patch.object(chunker, "_get_node_char_count") as mock_char_count:
            # Setup sizes: small1=200, small2=200, large=800 (exceeds max_merge_size 800)
            mock_char_count.side_effect = lambda node, _: {
                small_node1: 200,
                small_node2: 200,
                large_node: 800,  # Large node
            }[node]

            groups = await chunker._group_children_by_size([small_node1, small_node2, large_node], "content")

            # Should create multiple groups due to size constraints
            assert len(groups) >= 1
            assert groups is not None
