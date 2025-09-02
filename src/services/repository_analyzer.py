import json
import logging
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings
from ..models.indexing_models import CodeChunk

logger = logging.getLogger(__name__)


@dataclass
class RepoNode:
    id: str
    type: str  # "file", "class", "function", "module"
    name: str
    file_path: str
    metadata: dict[str, any]


@dataclass
class RepoEdge:
    source: str
    target: str
    relationship: str  # "imports", "calls", "inherits", "contains"
    weight: float = 1.0


class ModuleGraph:
    """Represents module and dependency relationships."""

    def __init__(self):
        self.nodes: dict[str, RepoNode] = {}
        self.edges: dict[str, list[RepoEdge]] = defaultdict(list)
        self.reverse_edges: dict[str, list[RepoEdge]] = defaultdict(list)

    def add_node(self, node: RepoNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

    def add_edge(self, edge: RepoEdge) -> None:
        """Add an edge to the graph."""
        self.edges[edge.source].append(edge)
        self.reverse_edges[edge.target].append(edge)

    def get_neighbors(self, node_id: str, relationship: str = None) -> list[RepoNode]:
        """Get neighboring nodes with optional relationship filter."""
        neighbors = []

        for edge in self.edges.get(node_id, []):
            if relationship is None or edge.relationship == relationship:
                if edge.target in self.nodes:
                    neighbors.append(self.nodes[edge.target])

        return neighbors

    def find_shortest_path(self, source: str, target: str) -> list[str]:
        """Find shortest path between two nodes."""
        if source not in self.nodes or target not in self.nodes:
            return []

        if source == target:
            return [source]

        # BFS for shortest path
        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            for edge in self.edges.get(current, []):
                if edge.target == target:
                    return path + [target]

                if edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, path + [edge.target]))

        return []

    def get_neighborhood(self, entity: str, radius: int) -> set[str]:
        """Get all entities within relationship radius."""
        if entity not in self.nodes:
            return set()

        neighborhood = {entity}
        current_level = {entity}

        for _ in range(radius):
            next_level = set()

            for node_id in current_level:
                # Add outgoing neighbors
                for edge in self.edges.get(node_id, []):
                    if edge.target not in neighborhood:
                        next_level.add(edge.target)

                # Add incoming neighbors
                for edge in self.reverse_edges.get(node_id, []):
                    if edge.source not in neighborhood:
                        next_level.add(edge.source)

            neighborhood.update(next_level)
            current_level = next_level

            if not current_level:
                break

        return neighborhood


class RepositoryAnalyzer:
    """Analyzes repository structure and cross-file relationships."""

    def __init__(self):
        self.config = get_settings()
        self.db_path = self.config.VECTOR_DB_PATH

    async def build_module_graph(self, codebase_path: str) -> ModuleGraph:
        """Build complete module dependency graph."""
        logger.info("Building repository module graph...")

        graph = ModuleGraph()

        try:
            # Get all chunks from database
            chunks = await self._get_all_chunks()
        except Exception as e:
            logger.error(f"Error getting chunks for module graph: {e}")
            return graph

        # Build nodes for files, classes, and functions
        file_nodes = {}

        for chunk in chunks:
            # Create file node if not exists
            file_id = f"file:{chunk.file_path}"
            if file_id not in file_nodes:
                file_node = RepoNode(
                    id=file_id,
                    type="file",
                    name=Path(chunk.file_path).name,
                    file_path=chunk.file_path,
                    metadata={"language": chunk.language},
                )
                graph.add_node(file_node)
                file_nodes[file_id] = file_node

            # Create function node
            if chunk.function_name:
                func_id = f"function:{chunk.file_path}:{chunk.function_name}"
                func_node = RepoNode(
                    id=func_id,
                    type="function",
                    name=chunk.function_name,
                    file_path=chunk.file_path,
                    metadata={
                        "signature": chunk.function_signature,
                        "return_type": chunk.return_type,
                        "complexity": chunk.complexity_score,
                    },
                )
                graph.add_node(func_node)

                # Add containment edge
                graph.add_edge(RepoEdge(file_id, func_id, "contains"))

            # Create class node
            if chunk.class_name:
                class_id = f"class:{chunk.file_path}:{chunk.class_name}"
                class_node = RepoNode(
                    id=class_id,
                    type="class",
                    name=chunk.class_name,
                    file_path=chunk.file_path,
                    metadata={"inheritance": chunk.inheritance_chain, "interfaces": chunk.interfaces},
                )
                graph.add_node(class_node)

                # Add containment edge
                graph.add_edge(RepoEdge(file_id, class_id, "contains"))

        # Add import relationships
        await self._add_import_relationships(graph, chunks)

        # Add inheritance relationships
        await self._add_inheritance_relationships(graph, chunks)

        total_edges = sum(len(edges) for edges in graph.edges.values())
        logger.info(f"Built module graph with {len(graph.nodes)} nodes and {total_edges} edges")
        return graph

    async def _get_all_chunks(self) -> list[CodeChunk]:
        """Get all chunks from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            cursor = conn.execute("""
                SELECT file_path, content, start_line, end_line, language, semantic_type,
                       function_signature, class_name, function_name, parameter_types, return_type,
                       inheritance_chain, import_statements, docstring, complexity_score,
                       dependencies, interfaces, decorators
                FROM embeddings
            """)

            chunks = []
            for row in cursor.fetchall():
                # Deserialize metadata
                parameter_types = json.loads(row[9]) if row[9] else None
                inheritance_chain = json.loads(row[11]) if row[11] else None
                import_statements = json.loads(row[12]) if row[12] else None
                dependencies = json.loads(row[15]) if row[15] else None
                interfaces = json.loads(row[16]) if row[16] else None
                decorators = json.loads(row[17]) if row[17] else None

                chunk = CodeChunk(
                    file_path=row[0],
                    content=row[1],
                    start_line=row[2],
                    end_line=row[3],
                    language=row[4],
                    semantic_type=row[5],
                    function_signature=row[6],
                    class_name=row[7],
                    function_name=row[8],
                    parameter_types=parameter_types,
                    return_type=row[10],
                    inheritance_chain=inheritance_chain,
                    import_statements=import_statements,
                    docstring=row[13],
                    complexity_score=row[14],
                    dependencies=dependencies,
                    interfaces=interfaces,
                    decorators=decorators,
                )
                chunks.append(chunk)

            conn.close()
            return chunks

        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            return []

    async def _add_import_relationships(self, graph: ModuleGraph, chunks: list[CodeChunk]) -> None:
        """Add import dependency relationships to graph."""
        for chunk in chunks:
            if not chunk.import_statements:
                continue

            source_file = f"file:{chunk.file_path}"

            for import_stmt in chunk.import_statements:
                # Parse import to find target module
                target_module = self._parse_import_target(import_stmt, chunk.language)

                if target_module:
                    # Try to find target file in graph
                    target_file = self._find_file_by_module(graph, target_module)

                    if target_file:
                        graph.add_edge(RepoEdge(source_file, target_file, "imports"))

    async def _add_inheritance_relationships(self, graph: ModuleGraph, chunks: list[CodeChunk]) -> None:
        """Add class inheritance relationships to graph."""
        for chunk in chunks:
            if not chunk.class_name or not chunk.inheritance_chain:
                continue

            source_class = f"class:{chunk.file_path}:{chunk.class_name}"

            for base_class in chunk.inheritance_chain:
                # Find base class in graph
                target_class = self._find_class_by_name(graph, base_class)

                if target_class:
                    graph.add_edge(RepoEdge(source_class, target_class, "inherits"))

    def _parse_import_target(self, import_stmt: str, language: str) -> str | None:
        """Parse import statement to extract target module."""
        try:
            if language == "python":
                if import_stmt.startswith("import "):
                    return import_stmt.replace("import ", "").split(" as ")[0].split(".")[0]
                elif import_stmt.startswith("from "):
                    return import_stmt.split("from ")[1].split(" import")[0].split(".")[0]
            elif language in ["javascript", "typescript"]:
                if "from " in import_stmt:
                    module = import_stmt.split("from ")[1].strip().strip("'\"")
                    return module.split("/")[-1].replace(".js", "").replace(".ts", "")

            return None
        except Exception:
            return None

    def _find_file_by_module(self, graph: ModuleGraph, module_name: str) -> str | None:
        """Find file node by module name."""
        for node_id, node in graph.nodes.items():
            if node.type == "file" and (module_name in node.name or node.name.startswith(module_name)):
                return node_id
        return None

    def _find_class_by_name(self, graph: ModuleGraph, class_name: str) -> str | None:
        """Find class node by name."""
        for node_id, node in graph.nodes.items():
            if node.type == "class" and node.name == class_name:
                return node_id
        return None


class CrossFileAnalyzer:
    """Analyzes cross-file relationships and dependencies."""

    def __init__(self, repository_analyzer: RepositoryAnalyzer):
        self.repo_analyzer = repository_analyzer
        self.config = get_settings()

    async def find_related_code(self, chunk: CodeChunk, graph: ModuleGraph) -> list[CodeChunk]:
        """Find related code across files based on relationships."""
        related_chunks = []

        # Find related through imports
        if chunk.import_statements:
            import_related = await self._find_import_related_code(chunk, graph)
            related_chunks.extend(import_related)

        # Find related through class inheritance
        if chunk.class_name and chunk.inheritance_chain:
            inheritance_related = await self._find_inheritance_related_code(chunk, graph)
            related_chunks.extend(inheritance_related)

        # Find related through function calls
        call_related = await self._find_call_related_code(chunk, graph)
        related_chunks.extend(call_related)

        # Remove duplicates and original chunk
        unique_related = []
        seen_ids = {f"{chunk.file_path}:{chunk.start_line}"}

        for related_chunk in related_chunks:
            chunk_id = f"{related_chunk.file_path}:{related_chunk.start_line}"
            if chunk_id not in seen_ids:
                unique_related.append(related_chunk)
                seen_ids.add(chunk_id)

        return unique_related[:10]  # Limit to top 10 related chunks

    async def _find_import_related_code(self, chunk: CodeChunk, graph: ModuleGraph) -> list[CodeChunk]:
        """Find code related through import dependencies."""
        related = []

        if not chunk.import_statements:
            return related

        # Get chunks from imported modules
        for import_stmt in chunk.import_statements:
            target_module = self.repo_analyzer._parse_import_target(import_stmt, chunk.language)

            if target_module:
                target_file = self.repo_analyzer._find_file_by_module(graph, target_module)

                if target_file:
                    # Get chunks from target file
                    file_path = graph.nodes[target_file].file_path
                    file_chunks = await self._get_chunks_from_file(file_path)
                    related.extend(file_chunks[:3])  # Limit per file

        return related

    async def _find_inheritance_related_code(self, chunk: CodeChunk, graph: ModuleGraph) -> list[CodeChunk]:
        """Find code related through class inheritance."""
        related = []

        if not chunk.class_name or not chunk.inheritance_chain:
            return related

        for base_class in chunk.inheritance_chain:
            base_class_id = self.repo_analyzer._find_class_by_name(graph, base_class)

            if base_class_id:
                base_node = graph.nodes[base_class_id]
                base_chunks = await self._get_chunks_from_file(base_node.file_path)

                # Filter to class-related chunks
                class_chunks = [c for c in base_chunks if c.class_name == base_class]
                related.extend(class_chunks[:2])

        return related

    async def _find_call_related_code(self, chunk: CodeChunk, graph: ModuleGraph) -> list[CodeChunk]:
        """Find code related through function calls."""
        related = []

        # Extract function calls from chunk content
        function_calls = self._extract_function_calls(chunk.content, chunk.language)

        for called_function in function_calls:
            # Find function definitions with matching names
            matching_chunks = await self._find_function_by_name(called_function)
            related.extend(matching_chunks[:1])  # One per function

        return related

    def _extract_function_calls(self, content: str, language: str) -> list[str]:
        """Extract function call names from code content."""
        import re

        calls = []

        if language == "python":
            # Match function calls: word followed by (
            pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
            matches = re.findall(pattern, content)
            calls.extend(matches)

        elif language in ["javascript", "typescript"]:
            # Match JS function calls
            pattern = r"\b([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\("
            matches = re.findall(pattern, content)
            calls.extend(matches)

        # Filter out common keywords
        filtered_calls = []
        keywords = {"if", "while", "for", "try", "catch", "with", "super", "print", "len", "str", "int", "float"}

        for call in calls:
            if call not in keywords and len(call) > 2:
                filtered_calls.append(call)

        return list(set(filtered_calls))  # Remove duplicates

    async def _get_chunks_from_file(self, file_path: str) -> list[CodeChunk]:
        """Get all chunks from a specific file."""
        try:
            conn = sqlite3.connect(str(self.config.VECTOR_DB_PATH))

            cursor = conn.execute(
                """
                SELECT file_path, content, start_line, end_line, language, semantic_type,
                       function_signature, class_name, function_name, parameter_types, return_type,
                       inheritance_chain, import_statements, docstring, complexity_score,
                       dependencies, interfaces, decorators
                FROM embeddings
                WHERE file_path = ?
            """,
                (file_path,),
            )

            chunks = []
            for row in cursor.fetchall():
                # Deserialize metadata
                parameter_types = json.loads(row[9]) if row[9] else None
                inheritance_chain = json.loads(row[11]) if row[11] else None
                import_statements = json.loads(row[12]) if row[12] else None
                dependencies = json.loads(row[15]) if row[15] else None
                interfaces = json.loads(row[16]) if row[16] else None
                decorators = json.loads(row[17]) if row[17] else None

                chunk = CodeChunk(
                    file_path=row[0],
                    content=row[1],
                    start_line=row[2],
                    end_line=row[3],
                    language=row[4],
                    semantic_type=row[5],
                    function_signature=row[6],
                    class_name=row[7],
                    function_name=row[8],
                    parameter_types=parameter_types,
                    return_type=row[10],
                    inheritance_chain=inheritance_chain,
                    import_statements=import_statements,
                    docstring=row[13],
                    complexity_score=row[14],
                    dependencies=dependencies,
                    interfaces=interfaces,
                    decorators=decorators,
                )
                chunks.append(chunk)

            conn.close()
            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks from file {file_path}: {e}")
            return []

    async def _find_function_by_name(self, function_name: str) -> list[CodeChunk]:
        """Find function definitions by name."""
        try:
            conn = sqlite3.connect(str(self.config.VECTOR_DB_PATH))

            cursor = conn.execute(
                """
                SELECT file_path, content, start_line, end_line, language, semantic_type,
                       function_signature, class_name, function_name, parameter_types, return_type,
                       inheritance_chain, import_statements, docstring, complexity_score,
                       dependencies, interfaces, decorators
                FROM embeddings
                WHERE function_name = ? AND semantic_type LIKE '%function%'
                LIMIT 5
            """,
                (function_name,),
            )

            chunks = []
            for row in cursor.fetchall():
                # Deserialize metadata
                parameter_types = json.loads(row[9]) if row[9] else None
                inheritance_chain = json.loads(row[11]) if row[11] else None
                import_statements = json.loads(row[12]) if row[12] else None
                dependencies = json.loads(row[15]) if row[15] else None
                interfaces = json.loads(row[16]) if row[16] else None
                decorators = json.loads(row[17]) if row[17] else None

                chunk = CodeChunk(
                    file_path=row[0],
                    content=row[1],
                    start_line=row[2],
                    end_line=row[3],
                    language=row[4],
                    semantic_type=row[5],
                    function_signature=row[6],
                    class_name=row[7],
                    function_name=row[8],
                    parameter_types=parameter_types,
                    return_type=row[10],
                    inheritance_chain=inheritance_chain,
                    import_statements=import_statements,
                    docstring=row[13],
                    complexity_score=row[14],
                    dependencies=dependencies,
                    interfaces=interfaces,
                    decorators=decorators,
                )
                chunks.append(chunk)

            conn.close()
            return chunks

        except Exception as e:
            logger.error(f"Error finding function {function_name}: {e}")
            return []

    async def get_context_chunks(self, chunk: CodeChunk, context_radius: int = 2) -> list[CodeChunk]:
        """Get surrounding context chunks from same and related files."""
        context_chunks = []

        # Get chunks from same file (file-level context)
        file_chunks = await self._get_chunks_from_file(chunk.file_path)

        # Find current chunk position
        current_index = -1
        for i, file_chunk in enumerate(file_chunks):
            if file_chunk.start_line == chunk.start_line and file_chunk.end_line == chunk.end_line:
                current_index = i
                break

        # Add surrounding chunks from same file
        if current_index >= 0:
            start_idx = max(0, current_index - context_radius)
            end_idx = min(len(file_chunks), current_index + context_radius + 1)

            for i in range(start_idx, end_idx):
                if i != current_index:  # Exclude the original chunk
                    context_chunks.append(file_chunks[i])

        # Add related chunks from other files
        graph = await self.repo_analyzer.build_module_graph(str(Path(chunk.file_path).parent))
        related_chunks = await self.find_related_code(chunk, graph)
        context_chunks.extend(related_chunks[:5])  # Limit cross-file context

        return context_chunks


class ContextualReranker:
    """Reranks search results based on repository context."""

    def __init__(self, repository_analyzer: RepositoryAnalyzer):
        self.repo_analyzer = repository_analyzer

    async def rerank_with_repository_context(
        self, results: list[CodeChunk], query: str, current_file: str = None
    ) -> list[CodeChunk]:
        """Rerank results based on repository context and relationships."""
        if not results:
            return results

        # Build repository graph
        graph = await self.repo_analyzer.build_module_graph(str(Path(results[0].file_path).parent))

        # Calculate context scores for each result
        scored_results = []

        for chunk in results:
            context_score = await self._calculate_context_score(chunk, query, graph, current_file)
            scored_results.append((context_score, chunk))

        # Sort by context score and return
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_results]

    async def _calculate_context_score(
        self, chunk: CodeChunk, query: str, graph: ModuleGraph, current_file: str = None
    ) -> float:
        """Calculate context relevance score for a chunk."""
        score = 0.0

        # File proximity scoring
        if current_file:
            file_distance = self._calculate_file_distance(chunk.file_path, current_file, graph)
            if file_distance >= 0:
                score += 0.3 / (file_distance + 1)  # Closer files get higher score

        # Import relationship scoring
        if chunk.import_statements:
            import_score = self._calculate_import_relevance(chunk.import_statements, query)
            score += import_score * 0.2

        # Function call relationship scoring
        call_score = self._calculate_call_relevance(chunk.content, query)
        score += call_score * 0.2

        # Module clustering similarity
        cluster_score = await self._calculate_cluster_similarity(chunk, graph)
        score += cluster_score * 0.3

        return min(1.0, score)

    def _calculate_file_distance(self, file1: str, file2: str, graph: ModuleGraph) -> int:
        """Calculate distance between files in dependency graph."""
        file1_id = f"file:{file1}"
        file2_id = f"file:{file2}"

        if file1_id not in graph.nodes or file2_id not in graph.nodes:
            return -1

        path = graph.find_shortest_path(file1_id, file2_id)
        return len(path) - 1 if path else -1

    def _calculate_import_relevance(self, import_statements: list[str], query: str) -> float:
        """Calculate relevance based on import patterns."""
        query_words = set(query.lower().split())
        relevance = 0.0

        for import_stmt in import_statements:
            import_words = set(import_stmt.lower().split())
            overlap = query_words.intersection(import_words)
            if overlap:
                relevance += len(overlap) / len(query_words)

        return min(1.0, relevance)

    def _calculate_call_relevance(self, content: str, query: str) -> float:
        """Calculate relevance based on function calls in content."""
        query_words = set(query.lower().split())
        content_lower = content.lower()
        
        # Count matches using substring matching
        matches = 0
        for query_word in query_words:
            if query_word in content_lower:
                matches += 1
        
        return matches / len(query_words) if query_words else 0.0

    async def _calculate_cluster_similarity(self, chunk: CodeChunk, graph: ModuleGraph) -> float:
        """Calculate similarity based on module clustering."""
        # Simple heuristic: files in same directory are more similar
        file_path = Path(chunk.file_path)
        parent_dir = file_path.parent.name

        # Files in common directories get similarity boost
        common_dirs = {"auth", "database", "api", "models", "services", "utils", "components"}

        if parent_dir.lower() in common_dirs:
            return 0.5

        return 0.0


class RepoKnowledgeGraph:
    """Repository-wide knowledge graph for code understanding."""

    def __init__(self):
        self.nodes: dict[str, RepoNode] = {}
        self.edges: dict[str, RepoEdge] = {}
        self.entity_index: dict[str, set[str]] = defaultdict(set)

    async def build_from_chunks(self, chunks: list[CodeChunk]) -> None:
        """Build knowledge graph from code chunks."""
        logger.info("Building repository knowledge graph...")

        # Add entity nodes
        for chunk in chunks:
            # Add file entity
            file_id = f"file:{chunk.file_path}"
            if file_id not in self.nodes:
                self.nodes[file_id] = RepoNode(
                    id=file_id,
                    type="file",
                    name=Path(chunk.file_path).name,
                    file_path=chunk.file_path,
                    metadata={"language": chunk.language},
                )
                self.entity_index["file"].add(file_id)

            # Add function entity
            if chunk.function_name:
                func_id = f"function:{chunk.function_name}"
                self.nodes[func_id] = RepoNode(
                    id=func_id,
                    type="function",
                    name=chunk.function_name,
                    file_path=chunk.file_path,
                    metadata={"signature": chunk.function_signature, "complexity": chunk.complexity_score},
                )
                self.entity_index["function"].add(func_id)

                # Add containment relationship
                edge_id = f"{file_id}-contains-{func_id}"
                self.edges[edge_id] = RepoEdge(file_id, func_id, "contains")

            # Add class entity
            if chunk.class_name:
                class_id = f"class:{chunk.class_name}"
                self.nodes[class_id] = RepoNode(
                    id=class_id,
                    type="class",
                    name=chunk.class_name,
                    file_path=chunk.file_path,
                    metadata={"inheritance": chunk.inheritance_chain},
                )
                self.entity_index["class"].add(class_id)

                # Add containment relationship
                edge_id = f"{file_id}-contains-{class_id}"
                self.edges[edge_id] = RepoEdge(file_id, class_id, "contains")

        logger.info(f"Knowledge graph built with {len(self.nodes)} entities and {len(self.edges)} relationships")

    def query_entities(self, entity_type: str, name_pattern: str = None) -> list[RepoNode]:
        """Query entities by type and optional name pattern."""
        matching_entities = []

        for entity_id in self.entity_index.get(entity_type, set()):
            entity = self.nodes[entity_id]

            if name_pattern is None or name_pattern.lower() in entity.name.lower():
                matching_entities.append(entity)

        return matching_entities

    def get_relationships(self, entity_id: str, relationship_type: str = None) -> list[RepoEdge]:
        """Get relationships for an entity."""
        relationships = []

        for edge in self.edges.values():
            if edge.source == entity_id or edge.target == entity_id:
                if relationship_type is None or edge.relationship == relationship_type:
                    relationships.append(edge)

        return relationships
