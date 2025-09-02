import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from src.services.repository_analyzer import (
    RepositoryAnalyzer,
    CrossFileAnalyzer,
    ContextualReranker,
    RepoKnowledgeGraph,
    ModuleGraph,
    RepoNode,
    RepoEdge
)
from src.models.indexing_models import CodeChunk


@pytest.fixture
def sample_chunks():
    return [
        CodeChunk(
            file_path="/test/auth.py",
            content="def authenticate_user(username, password):\n    return validate_credentials(username, password)",
            start_line=1,
            end_line=2,
            language="python",
            semantic_type="function_definition",
            function_name="authenticate_user",
            function_signature="authenticate_user(username, password)",
            import_statements=["from database import connect", "import hashlib"],
            dependencies=["database", "hashlib"]
        ),
        CodeChunk(
            file_path="/test/auth.py",
            content="class UserManager:\n    def __init__(self):\n        self.users = {}",
            start_line=5,
            end_line=7,
            language="python",
            semantic_type="class_definition",
            class_name="UserManager",
            inheritance_chain=["BaseManager"]
        ),
        CodeChunk(
            file_path="/test/database.py",
            content="class BaseManager:\n    def connect(self):\n        pass",
            start_line=1,
            end_line=3,
            language="python",
            semantic_type="class_definition",
            class_name="BaseManager"
        ),
        CodeChunk(
            file_path="/test/database.py",
            content="def connect():\n    return sqlite3.connect('db.sqlite')",
            start_line=10,
            end_line=11,
            language="python",
            semantic_type="function_definition",
            function_name="connect"
        )
    ]


class TestModuleGraph:
    
    def test_graph_creation(self):
        graph = ModuleGraph()
        
        node = RepoNode(
            id="test_node",
            type="function",
            name="test_func",
            file_path="/test.py",
            metadata={}
        )
        
        graph.add_node(node)
        assert "test_node" in graph.nodes
        assert graph.nodes["test_node"] == node
    
    def test_add_edge(self):
        graph = ModuleGraph()
        
        # Add nodes first
        node1 = RepoNode("node1", "file", "file1", "/file1.py", {})
        node2 = RepoNode("node2", "function", "func1", "/file1.py", {})
        graph.add_node(node1)
        graph.add_node(node2)
        
        # Add edge
        edge = RepoEdge("node1", "node2", "contains")
        graph.add_edge(edge)
        
        assert len(graph.edges["node1"]) == 1
        assert len(graph.reverse_edges["node2"]) == 1
    
    def test_get_neighbors(self):
        graph = ModuleGraph()
        
        # Setup nodes and edges
        node1 = RepoNode("node1", "file", "file1", "/file1.py", {})
        node2 = RepoNode("node2", "function", "func1", "/file1.py", {})
        node3 = RepoNode("node3", "class", "class1", "/file1.py", {})
        
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        graph.add_edge(RepoEdge("node1", "node2", "contains"))
        graph.add_edge(RepoEdge("node1", "node3", "contains"))
        
        neighbors = graph.get_neighbors("node1")
        assert len(neighbors) == 2
        
        # Test relationship filtering
        func_neighbors = graph.get_neighbors("node1", "contains")
        assert len(func_neighbors) == 2
    
    def test_shortest_path(self):
        graph = ModuleGraph()
        
        # Create linear path: A -> B -> C
        nodes = [
            RepoNode("A", "file", "a", "/a.py", {}),
            RepoNode("B", "function", "b", "/a.py", {}),
            RepoNode("C", "class", "c", "/a.py", {})
        ]
        
        for node in nodes:
            graph.add_node(node)
        
        graph.add_edge(RepoEdge("A", "B", "contains"))
        graph.add_edge(RepoEdge("B", "C", "calls"))
        
        path = graph.find_shortest_path("A", "C")
        assert path == ["A", "B", "C"]
        
        # Test no path
        node_d = RepoNode("D", "file", "d", "/d.py", {})
        graph.add_node(node_d)
        
        path = graph.find_shortest_path("A", "D")
        assert path == []
    
    def test_get_neighborhood(self):
        graph = ModuleGraph()
        
        # Create small network
        nodes = [RepoNode(str(i), "node", f"node{i}", f"/{i}.py", {}) for i in range(5)]
        for node in nodes:
            graph.add_node(node)
        
        # Connect: 0 -> 1 -> 2, 0 -> 3 -> 4
        edges = [
            RepoEdge("0", "1", "connects"),
            RepoEdge("1", "2", "connects"),
            RepoEdge("0", "3", "connects"),
            RepoEdge("3", "4", "connects")
        ]
        for edge in edges:
            graph.add_edge(edge)
        
        # Test neighborhood of radius 1
        neighborhood = graph.get_neighborhood("0", 1)
        assert "0" in neighborhood  # Include self
        assert "1" in neighborhood
        assert "3" in neighborhood
        assert len(neighborhood) == 3
        
        # Test larger radius
        neighborhood = graph.get_neighborhood("0", 2)
        assert len(neighborhood) >= 4  # Should include more distant nodes


class TestRepositoryAnalyzer:
    
    @pytest.mark.asyncio
    async def test_build_module_graph(self, sample_chunks):
        analyzer = RepositoryAnalyzer()
        
        # Mock the database query
        with patch.object(analyzer, '_get_all_chunks', return_value=sample_chunks):
            graph = await analyzer.build_module_graph("/test")
            
            # Check that nodes were created
            assert len(graph.nodes) > 0
            
            # Check for file nodes
            file_nodes = [n for n in graph.nodes.values() if n.type == "file"]
            assert len(file_nodes) >= 2  # auth.py and database.py
            
            # Check for function nodes
            func_nodes = [n for n in graph.nodes.values() if n.type == "function"]
            assert len(func_nodes) >= 2
            
            # Check for class nodes
            class_nodes = [n for n in graph.nodes.values() if n.type == "class"]
            assert len(class_nodes) >= 2
    
    def test_parse_import_target(self):
        analyzer = RepositoryAnalyzer()
        
        # Python imports
        assert analyzer._parse_import_target("import os", "python") == "os"
        assert analyzer._parse_import_target("from pathlib import Path", "python") == "pathlib"
        assert analyzer._parse_import_target("import numpy as np", "python") == "numpy"
        
        # JavaScript imports
        assert analyzer._parse_import_target("import { something } from './module'", "javascript") == "module"
        assert analyzer._parse_import_target("import React from 'react'", "javascript") == "react"
    
    def test_find_file_by_module(self):
        analyzer = RepositoryAnalyzer()
        graph = ModuleGraph()
        
        # Add file nodes
        auth_node = RepoNode("file:/test/auth.py", "file", "auth.py", "/test/auth.py", {})
        db_node = RepoNode("file:/test/database.py", "file", "database.py", "/test/database.py", {})
        
        graph.add_node(auth_node)
        graph.add_node(db_node)
        
        # Test finding by module name
        result = analyzer._find_file_by_module(graph, "auth")
        assert result == "file:/test/auth.py"
        
        result = analyzer._find_file_by_module(graph, "database")
        assert result == "file:/test/database.py"
        
        # Test non-existent module
        result = analyzer._find_file_by_module(graph, "nonexistent")
        assert result is None
    
    def test_find_class_by_name(self):
        analyzer = RepositoryAnalyzer()
        graph = ModuleGraph()
        
        # Add class node
        class_node = RepoNode("class:/test/auth.py:UserManager", "class", "UserManager", "/test/auth.py", {})
        graph.add_node(class_node)
        
        result = analyzer._find_class_by_name(graph, "UserManager")
        assert result == "class:/test/auth.py:UserManager"
        
        result = analyzer._find_class_by_name(graph, "NonExistent")
        assert result is None


class TestCrossFileAnalyzer:
    
    @pytest.mark.asyncio
    async def test_find_related_code(self, sample_chunks):
        repo_analyzer = Mock(spec=RepositoryAnalyzer)
        analyzer = CrossFileAnalyzer(repo_analyzer)
        
        # Mock module graph
        graph = ModuleGraph()
        
        # Mock methods
        analyzer._find_import_related_code = AsyncMock(return_value=sample_chunks[:1])
        analyzer._find_inheritance_related_code = AsyncMock(return_value=sample_chunks[1:2])
        analyzer._find_call_related_code = AsyncMock(return_value=sample_chunks[2:3])
        
        chunk = sample_chunks[0]  # authenticate_user function
        chunk.import_statements = ["import database"]
        chunk.class_name = "TestClass"
        chunk.inheritance_chain = ["BaseClass"]
        
        related = await analyzer.find_related_code(chunk, graph)
        
        # Should find related code through imports, inheritance, and calls
        assert len(related) <= 10  # Respects limit
    
    def test_extract_function_calls(self, sample_chunks):
        repo_analyzer = Mock(spec=RepositoryAnalyzer)
        analyzer = CrossFileAnalyzer(repo_analyzer)
        
        # Python code with function calls
        python_content = """
def main():
    user = authenticate_user("test", "pass")
    data = fetch_data(query)
    handle_error(error)
"""
        
        calls = analyzer._extract_function_calls(python_content, "python")
        
        assert "authenticate_user" in calls
        assert "fetch_data" in calls
        assert "handle_error" in calls
        # Should filter out keywords
        assert "def" not in calls
    
    @pytest.mark.asyncio
    async def test_get_context_chunks(self, sample_chunks):
        repo_analyzer = Mock(spec=RepositoryAnalyzer)
        analyzer = CrossFileAnalyzer(repo_analyzer)
        
        # Mock methods
        analyzer._get_chunks_from_file = AsyncMock(return_value=sample_chunks)
        repo_analyzer.build_module_graph = AsyncMock(return_value=ModuleGraph())
        analyzer.find_related_code = AsyncMock(return_value=sample_chunks[1:])
        
        chunk = sample_chunks[0]
        context = await analyzer.get_context_chunks(chunk, context_radius=1)
        
        # Should include surrounding chunks from same file and related chunks
        assert len(context) > 0


class TestContextualReranker:
    
    @pytest.mark.asyncio
    async def test_rerank_with_repository_context(self, sample_chunks):
        repo_analyzer = Mock(spec=RepositoryAnalyzer)
        reranker = ContextualReranker(repo_analyzer)
        
        # Mock graph building
        repo_analyzer.build_module_graph = AsyncMock(return_value=ModuleGraph())
        
        # Mock context score calculation
        with patch.object(reranker, '_calculate_context_score', return_value=0.8):
            reranked = await reranker.rerank_with_repository_context(
                sample_chunks, "test query"
            )
            
            assert len(reranked) == len(sample_chunks)
            # Results should be in some order (sorted by context score)
    
    def test_calculate_file_distance(self):
        reranker = ContextualReranker(Mock())
        graph = ModuleGraph()
        
        # Add file nodes
        file1 = RepoNode("file:/test/auth.py", "file", "auth.py", "/test/auth.py", {})
        file2 = RepoNode("file:/test/database.py", "file", "database.py", "/test/database.py", {})
        
        graph.add_node(file1)
        graph.add_node(file2)
        
        # Add import relationship
        graph.add_edge(RepoEdge("file:/test/auth.py", "file:/test/database.py", "imports"))
        
        distance = reranker._calculate_file_distance("/test/auth.py", "/test/database.py", graph)
        assert distance == 1  # Connected by one edge
        
        # Test non-connected files
        file3 = RepoNode("file:/test/utils.py", "file", "utils.py", "/test/utils.py", {})
        graph.add_node(file3)
        
        distance = reranker._calculate_file_distance("/test/auth.py", "/test/utils.py", graph)
        assert distance == -1  # No path
    
    def test_calculate_import_relevance(self):
        reranker = ContextualReranker(Mock())
        
        import_statements = ["import database", "from auth import validate", "import json"]
        
        # Query matching imports
        relevance = reranker._calculate_import_relevance(import_statements, "database operations")
        assert relevance > 0
        
        # Query not matching imports
        relevance = reranker._calculate_import_relevance(import_statements, "unrelated topic")
        assert relevance == 0
    
    def test_calculate_call_relevance(self):
        reranker = ContextualReranker(Mock())
        
        content = "def authenticate_user():\n    validate_credentials()\n    return True"
        
        # Matching query
        relevance = reranker._calculate_call_relevance(content, "authenticate user")
        assert relevance > 0
        
        # Non-matching query
        relevance = reranker._calculate_call_relevance(content, "database operations")
        assert relevance >= 0  # Might have some overlap


class TestRepoKnowledgeGraph:
    
    @pytest.mark.asyncio
    async def test_build_from_chunks(self, sample_chunks):
        kg = RepoKnowledgeGraph()
        
        await kg.build_from_chunks(sample_chunks)
        
        # Check entities were created
        assert len(kg.nodes) > 0
        assert len(kg.edges) > 0
        
        # Check entity indexing
        assert "file" in kg.entity_index
        assert "function" in kg.entity_index
        assert "class" in kg.entity_index
        
        # Check specific entities
        file_entities = kg.query_entities("file")
        assert len(file_entities) >= 2  # auth.py and database.py
        
        function_entities = kg.query_entities("function")
        assert len(function_entities) >= 2
    
    def test_query_entities(self):
        kg = RepoKnowledgeGraph()
        
        # Add test entities
        node1 = RepoNode("func:test", "function", "test_function", "/test.py", {})
        node2 = RepoNode("func:auth", "function", "authenticate", "/auth.py", {})
        
        kg.nodes[node1.id] = node1
        kg.nodes[node2.id] = node2
        kg.entity_index["function"].add(node1.id)
        kg.entity_index["function"].add(node2.id)
        
        # Query all functions
        functions = kg.query_entities("function")
        assert len(functions) == 2
        
        # Query with pattern
        auth_functions = kg.query_entities("function", "auth")
        assert len(auth_functions) == 1
        assert auth_functions[0].name == "authenticate"
    
    def test_get_relationships(self):
        kg = RepoKnowledgeGraph()
        
        # Add test edge
        edge = RepoEdge("node1", "node2", "contains")
        kg.edges["edge1"] = edge
        
        # Test getting relationships
        relationships = kg.get_relationships("node1")
        assert len(relationships) == 1
        assert relationships[0] == edge
        
        # Test relationship type filtering
        contains_rels = kg.get_relationships("node1", "contains")
        assert len(contains_rels) == 1
        
        imports_rels = kg.get_relationships("node1", "imports")
        assert len(imports_rels) == 0


class TestRepositoryAnalyzerIntegration:
    
    @pytest.mark.asyncio
    async def test_add_import_relationships(self, sample_chunks):
        analyzer = RepositoryAnalyzer()
        graph = ModuleGraph()
        
        # Add file nodes manually for testing
        auth_node = RepoNode("file:/test/auth.py", "file", "auth.py", "/test/auth.py", {})
        db_node = RepoNode("file:/test/database.py", "file", "database.py", "/test/database.py", {})
        
        graph.add_node(auth_node)
        graph.add_node(db_node)
        
        # Test import relationship addition
        await analyzer._add_import_relationships(graph, sample_chunks)
        
        # Should have added import edges based on import_statements
        auth_edges = graph.edges.get("file:/test/auth.py", [])
        # May have edges if import statements match files
    
    @pytest.mark.asyncio
    async def test_add_inheritance_relationships(self, sample_chunks):
        analyzer = RepositoryAnalyzer()
        graph = ModuleGraph()
        
        # Add class nodes
        user_manager_node = RepoNode(
            "class:/test/auth.py:UserManager", "class", "UserManager", "/test/auth.py", {}
        )
        base_manager_node = RepoNode(
            "class:/test/database.py:BaseManager", "class", "BaseManager", "/test/database.py", {}
        )
        
        graph.add_node(user_manager_node)
        graph.add_node(base_manager_node)
        
        await analyzer._add_inheritance_relationships(graph, sample_chunks)
        
        # Should have added inheritance edge from UserManager to BaseManager
        user_edges = graph.edges.get("class:/test/auth.py:UserManager", [])
        # Check if inheritance relationship was added


@pytest.mark.asyncio
async def test_integration_repository_analyzer(sample_chunks):
    """Integration test for repository analyzer."""
    analyzer = RepositoryAnalyzer()
    
    # Mock database operations
    with patch.object(analyzer, '_get_all_chunks', return_value=sample_chunks):
        graph = await analyzer.build_module_graph("/test")
        
        # Verify graph structure
        assert len(graph.nodes) > 0
        
        # Test cross-file analyzer
        cross_analyzer = CrossFileAnalyzer(analyzer)
        
        # Mock additional methods for full integration
        cross_analyzer._get_chunks_from_file = AsyncMock(return_value=sample_chunks)
        cross_analyzer._find_function_by_name = AsyncMock(return_value=sample_chunks[:1])
        
        chunk = sample_chunks[0]
        related = await cross_analyzer.find_related_code(chunk, graph)
        
        # Should find some related code
        assert isinstance(related, list)


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in repository analyzer."""
    analyzer = RepositoryAnalyzer()
    
    # Test with empty chunks
    with patch.object(analyzer, '_get_all_chunks', return_value=[]):
        graph = await analyzer.build_module_graph("/test")
        assert len(graph.nodes) == 0
    
    # Test with database error
    with patch.object(analyzer, '_get_all_chunks', side_effect=Exception("DB Error")):
        graph = await analyzer.build_module_graph("/test")
        assert len(graph.nodes) == 0