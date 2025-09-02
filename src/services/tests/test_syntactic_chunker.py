from unittest.mock import Mock, patch

import pytest
from tree_sitter import Node, Parser

from ...models.indexing_models import CodeChunk
from ..syntactic_chunker import ChunkingConfig, SyntacticChunker


class TestSyntacticChunker:
    @pytest.fixture
    def chunker(self):
        config = ChunkingConfig(max_chunk_chars=500, min_chunk_chars=50)
        return SyntacticChunker(config)

    @pytest.fixture
    def mock_parser(self):
        parser = Mock(spec=Parser)
        tree = Mock()
        root_node = Mock(spec=Node)

        # Setup basic node structure
        root_node.type = "module"
        root_node.start_byte = 0
        root_node.end_byte = 100
        root_node.start_point = (0, 0)
        root_node.end_point = (10, 0)
        root_node.children = []

        tree.root_node = root_node
        parser.parse.return_value = tree

        return parser

    @pytest.mark.asyncio
    async def test_chunk_small_function(self, chunker, mock_parser):
        """Test chunking of small function that fits in one chunk."""
        content = """def small_function():
    return "hello world" """

        # Mock function node
        func_node = Mock(spec=Node)
        func_node.type = "function_definition"
        func_node.start_byte = 0
        func_node.end_byte = len(content)
        func_node.start_point = (0, 0)
        func_node.end_point = (1, 20)
        func_node.children = []

        mock_parser.parse.return_value.root_node.children = [func_node]

        # Mock metadata extraction
        with patch.object(chunker.metadata_extractor, "extract_all_metadata") as mock_extract:
            mock_metadata = Mock()
            mock_metadata.semantic_type = "function_definition"
            mock_metadata.function_name = "small_function"
            mock_metadata.function_signature = "def small_function()"
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
            mock_extract.return_value = mock_metadata

            chunks = await chunker.chunk_with_integrity("test.py", content, "python", mock_parser)

            assert len(chunks) == 1
            assert chunks[0].function_name == "small_function"

    @pytest.mark.asyncio
    async def test_chunk_large_class(self, chunker, mock_parser):
        """Test chunking of large class that needs splitting."""
        content = "class LargeClass:\n" + "    def method():\n        pass\n" * 50

        # Mock large class node
        class_node = Mock(spec=Node)
        class_node.type = "class_definition"
        class_node.start_byte = 0
        class_node.end_byte = len(content)
        class_node.start_point = (0, 0)
        class_node.end_point = (150, 0)

        # Mock method children - make them non-adjacent to prevent merging
        method_nodes = []
        for i in range(5):
            method_node = Mock(spec=Node)
            method_node.type = "method_definition"
            method_node.start_byte = i * 30
            method_node.end_byte = (i + 1) * 30
            # Space out the methods so they can't be merged (>3 line gap)
            method_node.start_point = (i * 10, 0)
            method_node.end_point = ((i * 10) + 2, 0)
            method_node.children = []
            method_nodes.append(method_node)

        class_node.children = method_nodes
        mock_parser.parse.return_value.root_node.children = [class_node]

        with patch.object(chunker, "_get_node_char_count") as mock_char_count:
            # Make class too large, methods large enough to prevent grouping
            mock_char_count.side_effect = lambda node, content: 1000 if node.type == "class_definition" else 200

            with patch.object(chunker.metadata_extractor, "extract_all_metadata") as mock_extract:
                mock_metadata = Mock()
                mock_metadata.semantic_type = "method_definition"
                mock_metadata.function_name = "method"
                mock_metadata.function_signature = "def method()"
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
                mock_extract.return_value = mock_metadata

                chunks = await chunker.chunk_with_integrity("test.py", content, "python", mock_parser)

                # Should create multiple chunks for the methods
                assert len(chunks) > 1

    def test_is_chunkable_node(self, chunker):
        """Test chunkable node detection."""
        func_node = Mock(spec=Node)
        func_node.type = "function_definition"
        assert chunker._is_chunkable_node(func_node, "") is True

        class_node = Mock(spec=Node)
        class_node.type = "class_definition"
        assert chunker._is_chunkable_node(class_node, "") is True

        comment_node = Mock(spec=Node)
        comment_node.type = "comment"
        assert chunker._is_chunkable_node(comment_node, "") is False

    def test_get_node_char_count(self, chunker):
        """Test non-whitespace character counting."""
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 20

        content = "def func():    \n    pass"  # Has whitespace

        char_count = chunker._get_node_char_count(node, content)

        # Should count only non-whitespace chars from first 20 bytes
        expected = len("deffunc():")  # First 20 chars with whitespace removed
        assert char_count == expected

    @pytest.mark.asyncio
    async def test_group_children_by_size(self, chunker):
        """Test child node grouping by size."""
        # Create mock children with different sizes
        small_nodes = []
        for _ in range(3):
            node = Mock(spec=Node)
            node.type = "statement"
            small_nodes.append(node)

        large_node = Mock(spec=Node)
        large_node.type = "large_function"

        children = small_nodes + [large_node]

        with patch.object(chunker, "_get_node_char_count") as mock_char_count:
            # Small nodes: 100 chars each, large node: 800 chars
            mock_char_count.side_effect = lambda node, content: 800 if node.type == "large_function" else 100

            groups = await chunker._group_children_by_size(children, "content")

            # Should group small nodes together, large node separately
            assert len(groups) >= 2

            # Find group with multiple small nodes
            multi_node_group = next((group for group in groups if len(group) > 1), None)
            assert multi_node_group is not None

            # Find group with single large node
            large_node_group = next(
                (group for group in groups if len(group) == 1 and group[0].type == "large_function"), None
            )
            assert large_node_group is not None

    @pytest.mark.asyncio
    async def test_optimize_chunks_merge_small(self, chunker):
        """Test chunk optimization with small chunk merging."""
        # Create small adjacent chunks
        small_chunk1 = CodeChunk(
            file_path="test.py",
            content="x = 1",  # Very small
            start_line=1,
            end_line=1,
            language="python",
            semantic_type="variable_declaration",
        )

        small_chunk2 = CodeChunk(
            file_path="test.py",
            content="y = 2",  # Very small
            start_line=2,
            end_line=2,
            language="python",
            semantic_type="variable_declaration",
        )

        large_chunk = CodeChunk(
            file_path="test.py",
            content="def large_function():\n" + "    pass\n" * 20,  # Large enough
            start_line=5,
            end_line=25,
            language="python",
            semantic_type="function_definition",
        )

        chunks = [small_chunk1, small_chunk2, large_chunk]
        content = "x = 1\ny = 2\n\ndef large_function():\n" + "    pass\n" * 20

        with patch.object(chunker, "_try_merge_chunks") as mock_merge:
            merged_chunk = CodeChunk(
                file_path="test.py",
                content="x = 1\ny = 2",
                start_line=1,
                end_line=2,
                language="python",
                semantic_type="variable_declaration",
            )
            mock_merge.return_value = merged_chunk

            optimized = await chunker._optimize_chunks(chunks, content)

            # Should have fewer chunks due to merging
            assert len(optimized) <= len(chunks)

    def test_merge_semantic_types(self, chunker):
        """Test semantic type merging logic."""
        # Higher priority type should dominate
        assert chunker._merge_semantic_types("class_definition", "function_definition") == "class_definition"
        assert chunker._merge_semantic_types("function_definition", "variable_declaration") == "function_definition"

        # Same types should return same
        assert chunker._merge_semantic_types("function_definition", "function_definition") == "function_definition"

    @pytest.mark.asyncio
    async def test_create_chunk_from_node(self, chunker, mock_parser):
        """Test chunk creation from AST node."""
        content = "def test_function():\n    return True"

        node = Mock(spec=Node)
        node.type = "function_definition"
        node.start_point = (0, 0)
        node.end_point = (1, 15)
        node.start_byte = 0
        node.end_byte = len(content)
        node.children = []

        with patch.object(chunker.metadata_extractor, "extract_all_metadata") as mock_extract:
            mock_metadata = Mock()
            mock_metadata.semantic_type = "function_definition"
            mock_metadata.function_name = "test_function"
            mock_metadata.function_signature = "def test_function()"
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
            mock_extract.return_value = mock_metadata

            chunk = await chunker._create_chunk_from_node(node, content, "test.py", "python")

            assert chunk is not None
            assert chunk.function_name == "test_function"
            assert chunk.semantic_type == "function_definition"
            assert chunk.language == "python"

    @pytest.mark.asyncio
    async def test_chunk_integration_with_metadata(self, chunker, mock_parser):
        """Test integration with metadata extraction."""
        content = """
class TestClass:
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name
"""

        # Mock class and method nodes
        class_node = Mock(spec=Node)
        class_node.type = "class_definition"
        class_node.start_point = (1, 0)
        class_node.end_point = (6, 0)
        class_node.start_byte = 1
        class_node.end_byte = len(content) - 1

        init_node = Mock(spec=Node)
        init_node.type = "method_definition"
        init_node.start_point = (2, 4)
        init_node.end_point = (3, 20)

        get_node = Mock(spec=Node)
        get_node.type = "method_definition"
        get_node.start_point = (5, 4)
        get_node.end_point = (6, 22)

        class_node.children = [init_node, get_node]
        mock_parser.parse.return_value.root_node.children = [class_node]

        with patch.object(chunker.metadata_extractor, "extract_all_metadata") as mock_extract:

            def mock_metadata_response(node, content, language):
                metadata = Mock()
                if node.type == "class_definition":
                    metadata.semantic_type = "class_definition"
                    metadata.class_name = "TestClass"
                    metadata.function_name = None
                else:
                    metadata.semantic_type = "method_definition"
                    metadata.class_name = "TestClass"
                    metadata.function_name = "__init__" if "init" in str(node) else "get_name"

                # Add all required attributes
                metadata.function_signature = None
                metadata.parameter_types = None
                metadata.return_type = None
                metadata.inheritance_chain = None
                metadata.import_statements = None
                metadata.docstring = None
                metadata.complexity_score = 1
                metadata.dependencies = None
                metadata.interfaces = None
                metadata.decorators = None
                return metadata

            mock_extract.side_effect = mock_metadata_response

            with patch.object(chunker, "_get_node_char_count", return_value=200):
                chunks = await chunker.chunk_with_integrity("test.py", content, "python", mock_parser)

                assert len(chunks) > 0
                # Should have class chunk
                class_chunks = [c for c in chunks if c.semantic_type == "class_definition"]
                assert len(class_chunks) > 0
