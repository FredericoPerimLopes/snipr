import pytest
from unittest.mock import Mock, patch
from tree_sitter import Node

from ..metadata_extractor import MetadataExtractor, ExtractedMetadata


class TestMetadataExtractor:
    @pytest.fixture
    def extractor(self):
        return MetadataExtractor()

    @pytest.fixture
    def mock_python_node(self):
        """Mock Tree-sitter node for Python function."""
        node = Mock(spec=Node)
        node.type = "function_definition"
        node.start_byte = 0
        node.end_byte = 50
        node.start_point = (0, 0)
        node.end_point = (5, 0)
        
        # Mock children for function components
        identifier_child = Mock(spec=Node)
        identifier_child.type = "identifier"
        identifier_child.start_byte = 4
        identifier_child.end_byte = 12
        
        parameters_child = Mock(spec=Node)
        parameters_child.type = "parameters"
        parameters_child.start_byte = 12
        parameters_child.end_byte = 20
        parameters_child.children = []
        
        node.children = [identifier_child, parameters_child]
        return node

    @pytest.mark.asyncio
    async def test_extract_python_metadata(self, extractor, mock_python_node):
        """Test Python metadata extraction."""
        content = "def test_func(param1: str) -> bool:\n    pass"
        
        metadata = await extractor.extract_all_metadata(mock_python_node, content, "python")
        
        assert metadata.semantic_type == "function_definition"
        assert metadata.function_name == "test_func"

    @pytest.mark.asyncio
    async def test_extract_function_signature(self, extractor):
        """Test function signature extraction."""
        # Create a more realistic mock
        with patch.object(extractor, '_get_node_text') as mock_get_text:
            mock_get_text.side_effect = lambda node, child_type, content: {
                "identifier": "calculate_sum",
                "parameters": "(a: int, b: int)"
            }.get(child_type)
            
            node = Mock(spec=Node)
            node.type = "function_definition"
            node.children = [Mock(), Mock()]  # Mock children
            
            content = "def calculate_sum(a: int, b: int) -> int:\n    return a + b"
            
            signature = extractor._extract_python_function_signature(node, content)
            assert "def calculate_sum" in signature
            assert "(a: int, b: int)" in signature

    def test_extract_import_statements_python(self, extractor):
        """Test Python import statement extraction."""
        content = """
import os
import sys
from typing import List, Dict
from pathlib import Path
import numpy as np

def main():
    pass
"""
        
        imports = extractor._extract_import_statements(content, "python")
        
        assert imports is not None
        assert len(imports) == 4
        assert "import os" in imports
        assert "from typing import List, Dict" in imports

    def test_extract_import_statements_javascript(self, extractor):
        """Test JavaScript import statement extraction."""
        content = """
import React from 'react';
import { useState, useEffect } from 'react';
const fs = require('fs');

function App() {
    return null;
}
"""
        
        imports = extractor._extract_import_statements(content, "javascript")
        
        assert imports is not None
        assert len(imports) == 3
        assert any("React" in imp for imp in imports)
        assert any("useState" in imp for imp in imports)

    def test_calculate_complexity(self, extractor):
        """Test cyclomatic complexity calculation."""
        # Mock node with decision points
        node = Mock(spec=Node)
        node.type = "function_definition"
        
        # Create mock children with decision points
        if_child = Mock(spec=Node)
        if_child.type = "if_statement"
        if_child.children = []
        
        while_child = Mock(spec=Node)
        while_child.type = "while_statement"
        while_child.children = []
        
        node.children = [if_child, while_child]
        
        complexity = extractor._calculate_complexity(node)
        
        # Base complexity (1) + if (1) + while (1) = 3
        assert complexity == 3

    @pytest.mark.asyncio
    async def test_extract_file_level_metadata(self, extractor):
        """Test file-level metadata extraction."""
        content = """
import os
import sys
from typing import List

class TestClass:
    def method1(self):
        pass
        
    def method2(self):
        pass

def function1():
    pass

def function2():
    pass
"""
        
        with patch.object(extractor, 'parsers') as mock_parsers:
            mock_parser = Mock()
            mock_tree = Mock()
            mock_root = Mock()
            
            # Mock the parsing structure
            mock_parser.parse.return_value = mock_tree
            mock_tree.root_node = mock_root
            
            # Mock nodes for counting
            class_node = Mock()
            class_node.type = "class_definition"
            class_node.children = []
            
            func_nodes = [Mock(), Mock()]
            for node in func_nodes:
                node.type = "function_definition"
                node.children = []
            
            mock_root.children = [class_node] + func_nodes
            
            def count_constructs(node):
                # Simulate the counting logic
                pass
            
            mock_parsers.__getitem__.return_value = mock_parser
            
            metadata = await extractor.extract_file_level_metadata("test.py", content)
            
            assert metadata["language"] == "python"
            assert "imports" in metadata
            assert metadata["total_lines"] > 0

    def test_detect_language_from_path(self, extractor):
        """Test language detection from file paths."""
        assert extractor._detect_language_from_path("test.py") == "python"
        assert extractor._detect_language_from_path("app.js") == "javascript"
        assert extractor._detect_language_from_path("component.tsx") == "typescript"
        assert extractor._detect_language_from_path("main.go") == "go"
        assert extractor._detect_language_from_path("lib.rs") == "rust"
        assert extractor._detect_language_from_path("App.java") == "java"

    def test_get_node_text(self, extractor):
        """Test node text extraction."""
        content = "def test_function():\n    pass"
        
        node = Mock(spec=Node)
        child = Mock(spec=Node)
        child.type = "identifier"
        child.start_byte = 4
        child.end_byte = 17
        
        node.children = [child]
        
        text = extractor._get_node_text(node, "identifier", content)
        assert text == "test_function"

    def test_extract_dependencies_python(self, extractor):
        """Test dependency extraction from Python imports."""
        content = """
import os
import numpy as np
from sklearn.metrics import accuracy_score
from .local_module import helper
"""
        
        dependencies = extractor._extract_dependencies(content, "python")
        
        assert dependencies is not None
        assert "os" in dependencies
        assert "numpy" in dependencies
        assert "sklearn" in dependencies

    def test_extract_dependencies_javascript(self, extractor):
        """Test dependency extraction from JavaScript imports."""
        content = """
import React from 'react';
import { axios } from 'axios';
const fs = require('fs');
const path = require('path');
"""
        
        dependencies = extractor._extract_dependencies(content, "javascript")
        
        assert dependencies is not None
        assert "react" in dependencies
        assert "axios" in dependencies
        assert "fs" in dependencies
        assert "path" in dependencies