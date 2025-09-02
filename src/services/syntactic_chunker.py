import logging
import re
from dataclasses import dataclass

from tree_sitter import Node, Parser

from ..models.indexing_models import CodeChunk
from .metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    max_chunk_chars: int = 600
    min_chunk_chars: int = 100
    overlap_chars: int = 50
    preserve_syntax: bool = True
    max_merge_size: int = 800


class SyntacticChunker:
    """cAST-inspired chunking that preserves syntactic integrity."""

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.metadata_extractor = MetadataExtractor()

    async def chunk_with_integrity(
        self, file_path: str, content: str, language: str, parser: Parser
    ) -> list[CodeChunk]:
        """Main chunking method that preserves syntactic boundaries."""
        try:
            tree = parser.parse(content.encode())
            root_node = tree.root_node

            # Extract chunks with syntactic integrity
            raw_chunks = await self._extract_syntactic_chunks(root_node, content, file_path, language)

            # Apply merging and optimization
            optimized_chunks = await self._optimize_chunks(raw_chunks, content)

            logger.debug(f"Syntactic chunking: {len(raw_chunks)} -> {len(optimized_chunks)} chunks")
            return optimized_chunks

        except Exception as e:
            logger.error(f"Error in syntactic chunking: {e}")
            return []

    async def _extract_syntactic_chunks(
        self, node: Node, content: str, file_path: str, language: str
    ) -> list[CodeChunk]:
        """Extract chunks while preserving syntactic boundaries."""
        chunks = []

        # Check if current node is suitable for chunking
        if await self._is_chunkable_node(node, content):
            node_size = self._get_node_char_count(node, content)

            if node_size <= self.config.max_chunk_chars:
                # Node fits in single chunk
                chunk = await self._create_chunk_from_node(node, content, file_path, language)
                if chunk:
                    chunks.append(chunk)
            else:
                # Node too large, split recursively
                child_chunks = await self._recursive_chunk_split(node, content, file_path, language)
                chunks.extend(child_chunks)
        else:
            # Process children recursively
            for child in node.children:
                child_chunks = await self._extract_syntactic_chunks(child, content, file_path, language)
                chunks.extend(child_chunks)

        return chunks

    async def _is_chunkable_node(self, node: Node, content: str) -> bool:
        """Determine if a node represents a meaningful chunk boundary."""
        # Nodes that represent complete semantic units
        chunkable_types = {
            "function_definition",
            "async_function_definition",
            "method_definition",
            "class_definition",
            "class_declaration",
            "interface_declaration",
            "constructor_definition",
            "struct_declaration",
            "enum_declaration",
            "type_alias_declaration",
            "variable_declaration",
            "const_declaration",
        }

        return node.type in chunkable_types

    def _get_node_char_count(self, node: Node, content: str) -> int:
        """Count non-whitespace characters in node."""
        node_text = content[node.start_byte : node.end_byte]
        return len(re.sub(r"\s", "", node_text))

    async def _create_chunk_from_node(
        self, node: Node, content: str, file_path: str, language: str
    ) -> CodeChunk | None:
        """Create a CodeChunk from a Tree-sitter node with metadata."""
        try:
            lines = content.split("\n")
            start_line = node.start_point[0]
            end_line = node.end_point[0]

            # Add minimal context while preserving boundaries
            context_start = max(0, start_line - 1)
            context_end = min(len(lines), end_line + 2)

            chunk_content = "\n".join(lines[context_start:context_end])

            # Extract metadata
            metadata = await self.metadata_extractor.extract_all_metadata(node, content, language)

            return CodeChunk(
                file_path=file_path,
                content=chunk_content,
                start_line=context_start + 1,
                end_line=context_end,
                language=language,
                semantic_type=metadata.semantic_type,
                function_signature=metadata.function_signature,
                class_name=metadata.class_name,
                function_name=metadata.function_name,
                parameter_types=metadata.parameter_types,
                return_type=metadata.return_type,
                inheritance_chain=metadata.inheritance_chain,
                import_statements=metadata.import_statements,
                docstring=metadata.docstring,
                complexity_score=metadata.complexity_score,
                dependencies=metadata.dependencies,
                interfaces=metadata.interfaces,
                decorators=metadata.decorators,
            )

        except Exception as e:
            logger.debug(f"Error creating chunk from node: {e}")
            return None

    async def _recursive_chunk_split(self, node: Node, content: str, file_path: str, language: str) -> list[CodeChunk]:
        """Recursively split large nodes while preserving syntax."""
        chunks = []

        # Try to split at natural boundaries (direct children)
        child_groups = await self._group_children_by_size(node.children, content)

        for child_group in child_groups:
            if len(child_group) == 1:
                # Single child - recurse
                child_chunks = await self._extract_syntactic_chunks(child_group[0], content, file_path, language)
                chunks.extend(child_chunks)
            else:
                # Multiple children that fit together
                merged_chunk = await self._create_merged_chunk(child_group, content, file_path, language)
                if merged_chunk:
                    chunks.append(merged_chunk)

        return chunks

    async def _group_children_by_size(self, children: list[Node], content: str) -> list[list[Node]]:
        """Group child nodes to fit within chunk size limits."""
        groups = []
        current_group = []
        current_size = 0

        for child in children:
            child_size = self._get_node_char_count(child, content)

            if child_size > self.config.max_chunk_chars:
                # Child too large - needs its own processing
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_size = 0
                groups.append([child])
            elif current_size + child_size <= self.config.max_chunk_chars:
                # Child fits in current group
                current_group.append(child)
                current_size += child_size
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [child]
                current_size = child_size

        if current_group:
            groups.append(current_group)

        return groups

    async def _create_merged_chunk(
        self, nodes: list[Node], content: str, file_path: str, language: str
    ) -> CodeChunk | None:
        """Create a chunk from multiple adjacent nodes."""
        if not nodes:
            return None

        try:
            # Get span of all nodes
            _ = min(node.start_byte for node in nodes)
            _ = max(node.end_byte for node in nodes)

            lines = content.split("\n")
            start_line = min(node.start_point[0] for node in nodes)
            end_line = max(node.end_point[0] for node in nodes)

            # Add minimal context
            context_start = max(0, start_line - 1)
            context_end = min(len(lines), end_line + 2)

            chunk_content = "\n".join(lines[context_start:context_end])

            # Determine dominant semantic type
            semantic_type = await self._get_dominant_semantic_type(nodes, content, language)

            return CodeChunk(
                file_path=file_path,
                content=chunk_content,
                start_line=context_start + 1,
                end_line=context_end,
                language=language,
                semantic_type=semantic_type,
            )

        except Exception as e:
            logger.debug(f"Error creating merged chunk: {e}")
            return None

    async def _get_dominant_semantic_type(self, nodes: list[Node], content: str, language: str) -> str:
        """Determine the dominant semantic type for a group of nodes."""
        # Priority order for semantic types
        priority_map = {
            "class_definition": 10,
            "class_declaration": 10,
            "function_definition": 8,
            "async_function_definition": 8,
            "method_definition": 7,
            "constructor_definition": 7,
            "interface_declaration": 6,
            "struct_declaration": 6,
            "variable_declaration": 4,
            "const_declaration": 4,
            "import_statement": 2,
            "import_declaration": 2,
        }

        highest_priority = 0
        dominant_type = "code_block"

        for node in nodes:
            priority = priority_map.get(node.type, 1)
            if priority > highest_priority:
                highest_priority = priority
                dominant_type = node.type

        return dominant_type

    async def _optimize_chunks(self, chunks: list[CodeChunk], content: str) -> list[CodeChunk]:
        """Optimize chunks by merging small adjacent chunks and adding context."""
        if not chunks:
            return chunks

        optimized = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]

            # Check if current chunk is too small and can be merged
            if len(current_chunk.content) < self.config.min_chunk_chars and i + 1 < len(chunks):
                # Try to merge with next chunk
                next_chunk = chunks[i + 1]
                merged = await self._try_merge_chunks(current_chunk, next_chunk, content)

                if merged:
                    optimized.append(merged)
                    i += 2  # Skip next chunk since it was merged
                    continue

            # Add context to chunk if beneficial
            contextualized = await self._add_hierarchical_context(current_chunk, content)
            optimized.append(contextualized)
            i += 1

        return optimized

    async def _try_merge_chunks(self, chunk1: CodeChunk, chunk2: CodeChunk, content: str) -> CodeChunk | None:
        """Try to merge two adjacent chunks if beneficial."""
        # Only merge chunks from same file
        if chunk1.file_path != chunk2.file_path:
            return None

        # Don't merge distinct semantic constructs (function with class, etc.)
        distinct_types = {
            "function_definition", "class_definition", "interface_declaration", 
            "method_definition", "constructor_definition"
        }
        if (chunk1.semantic_type in distinct_types and 
            chunk2.semantic_type in distinct_types and 
            chunk1.semantic_type != chunk2.semantic_type):
            return None

        # Check if chunks are adjacent or nearly adjacent
        if abs(chunk1.end_line - chunk2.start_line) > 3:
            return None

        # Check merged size
        lines = content.split("\n")
        merged_start = min(chunk1.start_line, chunk2.start_line) - 1
        merged_end = max(chunk1.end_line, chunk2.end_line)
        merged_content = "\n".join(lines[merged_start:merged_end])

        if len(merged_content) > self.config.max_merge_size:
            return None

        # Create merged chunk
        return CodeChunk(
            file_path=chunk1.file_path,
            content=merged_content,
            start_line=merged_start + 1,
            end_line=merged_end,
            language=chunk1.language,
            semantic_type=self._merge_semantic_types(chunk1.semantic_type, chunk2.semantic_type),
            # Merge metadata intelligently
            function_name=chunk1.function_name or chunk2.function_name,
            class_name=chunk1.class_name or chunk2.class_name,
            function_signature=chunk1.function_signature or chunk2.function_signature,
            docstring=chunk1.docstring or chunk2.docstring,
        )

    def _merge_semantic_types(self, type1: str, type2: str) -> str:
        """Merge two semantic types intelligently."""
        # Priority for merged types
        if type1 == type2:
            return type1

        priority = {"class_definition": 10, "function_definition": 8, "method_definition": 7, "variable_declaration": 4}

        priority1 = priority.get(type1, 1)
        priority2 = priority.get(type2, 1)

        return type1 if priority1 >= priority2 else type2

    async def _add_hierarchical_context(self, chunk: CodeChunk, content: str) -> CodeChunk:
        """Add hierarchical context information to chunks."""
        # For now, return chunk as-is
        # This could be enhanced to add parent/sibling/child context
        return chunk
