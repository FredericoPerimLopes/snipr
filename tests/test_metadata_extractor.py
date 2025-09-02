from unittest.mock import Mock, patch

import pytest

from src.services.metadata_extractor import ExtractedMetadata, MetadataExtractor


@pytest.fixture
def mock_node():
    """Mock Tree-sitter node for testing."""
    node = Mock()
    node.type = "function_definition"
    node.start_byte = 0
    node.end_byte = 50
    node.children = []
    return node


@pytest.fixture
def python_function_content():
    """Sample Python function content."""
    return '''def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user with credentials."""
    return validate_credentials(username, password)'''


@pytest.fixture
def python_class_content():
    """Sample Python class content."""
    return '''class UserManager(BaseManager):
    """Manages user operations."""
    
    def __init__(self):
        super().__init__()'''


class TestMetadataExtractor:
    def test_init_basic(self):
        """Test basic initialization."""
        with patch.object(MetadataExtractor, "_init_parsers"):
            extractor = MetadataExtractor()
            assert isinstance(extractor.parsers, dict)
            assert isinstance(extractor.languages, dict)

    @patch("src.services.metadata_extractor.tspython")
    def test_init_parsers_success(self, mock_tspython):
        """Test successful parser initialization."""
        mock_language_func = Mock()
        mock_tspython.language = mock_language_func

        with (
            patch("src.services.metadata_extractor.Language") as mock_lang_class,
            patch("src.services.metadata_extractor.Parser") as mock_parser_class,
        ):
            mock_lang = Mock()
            mock_lang_class.return_value = mock_lang
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser

            extractor = MetadataExtractor()

            assert "python" in extractor.languages
            assert "python" in extractor.parsers

    @patch("src.services.metadata_extractor.tspython", None)
    def test_init_parsers_missing_module(self, caplog):
        """Test parser initialization with missing language module."""
        extractor = MetadataExtractor()

        # Should skip unavailable languages
        assert "python" not in extractor.parsers

    async def test_extract_all_metadata_python(self, mock_node, python_function_content):
        """Test comprehensive metadata extraction for Python."""
        extractor = MetadataExtractor()

        with patch.object(extractor, "_extract_python_metadata") as mock_extract:
            mock_extract.return_value = None

            metadata = await extractor.extract_all_metadata(mock_node, python_function_content, "python")

            assert isinstance(metadata, ExtractedMetadata)
            assert metadata.semantic_type == "function_definition"
            mock_extract.assert_called_once()

    async def test_extract_all_metadata_javascript(self, mock_node):
        """Test metadata extraction for JavaScript."""
        extractor = MetadataExtractor()

        with patch.object(extractor, "_extract_js_ts_metadata") as mock_extract:
            mock_extract.return_value = None

            metadata = await extractor.extract_all_metadata(mock_node, "function test() {}", "javascript")

            assert metadata.semantic_type == "function_definition"
            mock_extract.assert_called_once()

    async def test_extract_all_metadata_typescript(self, mock_node):
        """Test metadata extraction for TypeScript."""
        extractor = MetadataExtractor()

        with patch.object(extractor, "_extract_js_ts_metadata") as mock_extract:
            mock_extract.return_value = None

            metadata = await extractor.extract_all_metadata(mock_node, "function test(): void {}", "typescript")

            assert metadata.semantic_type == "function_definition"
            mock_extract.assert_called_once()

    async def test_extract_all_metadata_java(self, mock_node):
        """Test metadata extraction for Java."""
        extractor = MetadataExtractor()

        with patch.object(extractor, "_extract_java_metadata") as mock_extract:
            mock_extract.return_value = None

            metadata = await extractor.extract_all_metadata(mock_node, "public void test() {}", "java")

            assert metadata.semantic_type == "function_definition"
            mock_extract.assert_called_once()

    async def test_extract_python_metadata_function(self, python_function_content):
        """Test Python function metadata extraction."""
        extractor = MetadataExtractor()

        # Mock node for function
        mock_node = Mock()
        mock_node.type = "function_definition"
        mock_node.children = [
            Mock(type="identifier", start_byte=4, end_byte=20),  # function name
            Mock(type="parameters", start_byte=21, end_byte=51),  # parameters
        ]

        metadata = ExtractedMetadata(semantic_type="function_definition")

        with patch.object(
            extractor, "_get_node_text", side_effect=["authenticate_user", "(username: str, password: str)"]
        ):
            with patch.object(
                extractor,
                "_extract_python_function_signature",
                return_value="def authenticate_user(username: str, password: str)",
            ):
                with patch.object(extractor, "_extract_python_parameter_types", return_value=["str", "str"]):
                    with patch.object(extractor, "_extract_python_return_type", return_value="bool"):
                        with patch.object(extractor, "_extract_python_decorators", return_value=None):
                            with patch.object(extractor, "_extract_python_docstring", return_value="Authenticate user"):
                                with patch.object(extractor, "_calculate_complexity", return_value=2):
                                    with patch.object(
                                        extractor, "_extract_import_statements", return_value=["import os"]
                                    ):
                                        await extractor._extract_python_metadata(
                                            mock_node, python_function_content, metadata
                                        )

                                        assert metadata.function_name == "authenticate_user"
                                        assert (
                                            metadata.function_signature
                                            == "def authenticate_user(username: str, password: str)"
                                        )
                                        assert metadata.parameter_types == ["str", "str"]
                                        assert metadata.return_type == "bool"
                                        assert metadata.docstring == "Authenticate user"
                                        assert metadata.complexity_score == 2

    async def test_extract_python_metadata_class(self, python_class_content):
        """Test Python class metadata extraction."""
        extractor = MetadataExtractor()

        mock_node = Mock()
        mock_node.type = "class_definition"
        mock_node.children = [
            Mock(type="identifier", start_byte=6, end_byte=17),  # class name
            Mock(type="argument_list", start_byte=18, end_byte=30),  # inheritance
        ]

        metadata = ExtractedMetadata(semantic_type="class_definition")

        with patch.object(extractor, "_get_node_text", return_value="UserManager"):
            with patch.object(extractor, "_extract_python_inheritance", return_value=["BaseManager"]):
                with patch.object(extractor, "_extract_python_docstring", return_value="Manages users"):
                    with patch.object(extractor, "_extract_import_statements", return_value=None):
                        await extractor._extract_python_metadata(mock_node, python_class_content, metadata)

                        assert metadata.class_name == "UserManager"
                        assert metadata.inheritance_chain == ["BaseManager"]
                        assert metadata.docstring == "Manages users"

    def test_get_node_text_found(self):
        """Test getting node text when child exists."""
        extractor = MetadataExtractor()

        child_node = Mock()
        child_node.type = "identifier"
        child_node.start_byte = 4
        child_node.end_byte = 12

        parent_node = Mock()
        parent_node.children = [child_node]

        content = "def test_func():"
        result = extractor._get_node_text(parent_node, "identifier", content)

        assert result == "test_func"

    def test_get_node_text_not_found(self):
        """Test getting node text when child doesn't exist."""
        extractor = MetadataExtractor()

        parent_node = Mock()
        parent_node.children = []

        result = extractor._get_node_text(parent_node, "identifier", "def test():")
        assert result is None

    def test_extract_python_function_signature_basic(self):
        """Test Python function signature extraction."""
        extractor = MetadataExtractor()

        mock_node = Mock()
        mock_node.children = [
            Mock(type="identifier", start_byte=4, end_byte=8),  # "test"
            Mock(type="parameters", start_byte=8, end_byte=10),  # "()"
        ]

        with patch.object(extractor, "_get_node_text", side_effect=["test", "()"]):
            signature = extractor._extract_python_function_signature(mock_node, "def test():")
            assert signature == "def test()"

    def test_extract_python_function_signature_with_return_type(self):
        """Test Python function signature with return type."""
        extractor = MetadataExtractor()

        mock_node = Mock()
        type_child = Mock()
        type_child.type = "type"
        type_child.start_byte = 15
        type_child.end_byte = 19
        mock_node.children = [Mock(type="identifier"), Mock(type="parameters"), type_child]

        content = "def test() -> bool:"

        with patch.object(extractor, "_get_node_text", side_effect=["test", "()"]):
            signature = extractor._extract_python_function_signature(mock_node, content)
            assert signature == "def test() -> bool"

    def test_extract_python_function_signature_error(self):
        """Test function signature extraction with error."""
        extractor = MetadataExtractor()

        mock_node = Mock()
        mock_node.children = []

        with patch.object(extractor, "_get_node_text", side_effect=Exception("Error")):
            signature = extractor._extract_python_function_signature(mock_node, "invalid")
            assert signature is None

    def test_extract_python_parameter_types(self):
        """Test Python parameter type extraction."""
        extractor = MetadataExtractor()

        # Mock parameters node with typed parameters
        param_node = Mock()
        param_node.type = "parameters"
        type_param = Mock()
        type_param.type = "typed_parameter"
        type_annotation = Mock()
        type_annotation.type = "type"
        type_annotation.start_byte = 20
        type_annotation.end_byte = 23
        type_param.children = [type_annotation]
        param_node.children = [type_param]

        mock_node = Mock()
        mock_node.children = [param_node]

        content = "def test(name: str):"
        param_types = extractor._extract_python_parameter_types(mock_node, content)

        assert param_types == ["str"]

    def test_extract_python_parameter_types_no_types(self):
        """Test parameter type extraction with no type annotations."""
        extractor = MetadataExtractor()

        mock_node = Mock()
        mock_node.children = []

        result = extractor._extract_python_parameter_types(mock_node, "def test(name):")
        assert result is None

    def test_extract_python_return_type(self):
        """Test Python return type extraction."""
        extractor = MetadataExtractor()

        type_child = Mock()
        type_child.type = "type"
        type_child.start_byte = 15
        type_child.end_byte = 18

        mock_node = Mock()
        mock_node.children = [type_child]

        content = "def test() -> int:"
        return_type = extractor._extract_python_return_type(mock_node, content)

        assert return_type == "int"

    def test_extract_python_decorators(self):
        """Test Python decorator extraction."""
        extractor = MetadataExtractor()

        decorator_child = Mock()
        decorator_child.type = "decorator"
        decorator_child.start_byte = 0
        decorator_child.end_byte = 9

        mock_node = Mock()
        mock_node.children = [decorator_child]

        content = "@property\ndef test():"
        decorators = extractor._extract_python_decorators(mock_node, content)

        assert decorators == ["@property"]

    def test_extract_python_docstring(self):
        """Test Python docstring extraction."""
        extractor = MetadataExtractor()

        # Mock block with expression statement containing string
        string_node = Mock()
        string_node.type = "string"
        string_node.start_byte = 20
        string_node.end_byte = 45

        expr_stmt = Mock()
        expr_stmt.type = "expression_statement"
        expr_stmt.children = [string_node]

        block_node = Mock()
        block_node.type = "block"
        block_node.children = [expr_stmt]

        mock_node = Mock()
        mock_node.children = [block_node]

        content = 'def test():\n    """Test docstring"""\n    pass'
        docstring = extractor._extract_python_docstring(mock_node, content)

        assert docstring == "Test docstring"

    def test_extract_python_inheritance(self):
        """Test Python inheritance extraction."""
        extractor = MetadataExtractor()

        # Mock argument list with identifier
        identifier = Mock()
        identifier.type = "identifier"
        identifier.start_byte = 15
        identifier.end_byte = 26

        arg_list = Mock()
        arg_list.type = "argument_list"
        arg_list.children = [identifier]

        mock_node = Mock()
        mock_node.children = [arg_list]

        content = "class Test(BaseClass):"
        inheritance = extractor._extract_python_inheritance(mock_node, content)

        assert inheritance == ["BaseClass"]

    def test_extract_js_function_signature(self):
        """Test JavaScript function signature extraction."""
        extractor = MetadataExtractor()

        mock_node = Mock()
        mock_node.start_byte = 0
        mock_node.end_byte = 25

        content = "function test(name) { return name; }"
        signature = extractor._extract_js_function_signature(mock_node, content)

        assert signature == "function test(name)"

    def test_extract_js_parameter_types_typescript(self):
        """Test TypeScript parameter type extraction."""
        extractor = MetadataExtractor()

        # Mock formal parameters with type annotation
        type_annotation = Mock()
        type_annotation.type = "type_annotation"
        type_annotation.start_byte = 15
        type_annotation.end_byte = 22

        required_param = Mock()
        required_param.type = "required_parameter"
        required_param.children = [type_annotation]

        formal_params = Mock()
        formal_params.type = "formal_parameters"
        formal_params.children = [required_param]

        mock_node = Mock()
        mock_node.children = [formal_params]

        content = "function test(name: string) {}"
        param_types = extractor._extract_js_parameter_types(mock_node, content)

        # For TypeScript, should extract types; for JavaScript, should return None
        # Mock as non-TypeScript for this test
        with patch("str", return_value="javascript"):
            param_types = extractor._extract_js_parameter_types(mock_node, content)
            assert param_types is None

    def test_extract_js_return_type(self):
        """Test JavaScript/TypeScript return type extraction."""
        extractor = MetadataExtractor()

        type_annotation = Mock()
        type_annotation.type = "type_annotation"
        type_annotation.start_byte = 20
        type_annotation.end_byte = 26

        mock_node = Mock()
        mock_node.children = [type_annotation]

        content = "function test(): string {}"
        return_type = extractor._extract_js_return_type(mock_node, content)

        assert return_type == "string"

    def test_extract_js_inheritance(self):
        """Test JavaScript class inheritance extraction."""
        extractor = MetadataExtractor()

        identifier = Mock()
        identifier.type = "identifier"
        identifier.start_byte = 20
        identifier.end_byte = 29

        class_heritage = Mock()
        class_heritage.type = "class_heritage"
        class_heritage.children = [identifier]

        mock_node = Mock()
        mock_node.children = [class_heritage]

        content = "class Test extends BaseClass {}"
        inheritance = extractor._extract_js_inheritance(mock_node, content)

        assert inheritance == ["BaseClass"]

    def test_extract_java_method_signature(self):
        """Test Java method signature extraction."""
        extractor = MetadataExtractor()

        modifiers = Mock()
        modifiers.type = "modifiers"
        modifiers.start_byte = 0
        modifiers.end_byte = 6

        return_type = Mock()
        return_type.type = "type_identifier"
        return_type.start_byte = 7
        return_type.end_byte = 11

        identifier = Mock()
        identifier.type = "identifier"
        identifier.start_byte = 12
        identifier.end_byte = 16

        params = Mock()
        params.type = "formal_parameters"
        params.start_byte = 16
        params.end_byte = 18

        mock_node = Mock()
        mock_node.children = [modifiers, return_type, identifier, params]

        content = "public void test() {}"

        with patch.object(extractor, "_get_node_text", side_effect=["test", "()"]):
            signature = extractor._extract_java_method_signature(mock_node, content)
            assert signature == "public void test ()"

    def test_extract_java_parameter_types(self):
        """Test Java parameter type extraction."""
        extractor = MetadataExtractor()

        type_id = Mock()
        type_id.type = "type_identifier"
        type_id.start_byte = 15
        type_id.end_byte = 21

        formal_param = Mock()
        formal_param.type = "formal_parameter"
        formal_param.children = [type_id]

        formal_params = Mock()
        formal_params.type = "formal_parameters"
        formal_params.children = [formal_param]

        mock_node = Mock()
        mock_node.children = [formal_params]

        content = "public void test(String name) {}"
        param_types = extractor._extract_java_parameter_types(mock_node, content)

        assert param_types == ["String"]

    def test_extract_java_return_type(self):
        """Test Java return type extraction."""
        extractor = MetadataExtractor()

        type_id = Mock()
        type_id.type = "type_identifier"
        type_id.start_byte = 7
        type_id.end_byte = 13

        mock_node = Mock()
        mock_node.children = [type_id]

        content = "public String test() {}"
        return_type = extractor._extract_java_return_type(mock_node, content)

        assert return_type == "String"

    def test_extract_java_inheritance(self):
        """Test Java inheritance extraction."""
        extractor = MetadataExtractor()

        type_id = Mock()
        type_id.type = "type_identifier"
        type_id.start_byte = 20
        type_id.end_byte = 29

        superclass = Mock()
        superclass.type = "superclass"
        superclass.children = [type_id]

        mock_node = Mock()
        mock_node.children = [superclass]

        content = "class Test extends BaseClass {}"
        inheritance = extractor._extract_java_inheritance(mock_node, content)

        assert inheritance == ["BaseClass"]

    def test_extract_java_interfaces(self):
        """Test Java interface extraction."""
        extractor = MetadataExtractor()

        type_id = Mock()
        type_id.type = "type_identifier"
        type_id.start_byte = 25
        type_id.end_byte = 34

        super_interfaces = Mock()
        super_interfaces.type = "super_interfaces"
        super_interfaces.children = [type_id]

        mock_node = Mock()
        mock_node.children = [super_interfaces]

        content = "class Test implements Runnable {}"
        interfaces = extractor._extract_java_interfaces(mock_node, content)

        assert interfaces == ["Runnable"]

    def test_extract_go_function_signature(self):
        """Test Go function signature extraction."""
        extractor = MetadataExtractor()

        mock_node = Mock()
        mock_node.start_byte = 0
        mock_node.end_byte = 25

        content = "func test(name string) { return name }"
        signature = extractor._extract_go_function_signature(mock_node, content)

        assert signature == "func test(name string)"

    def test_extract_rust_function_signature(self):
        """Test Rust function signature extraction."""
        extractor = MetadataExtractor()

        mock_node = Mock()
        mock_node.start_byte = 0
        mock_node.end_byte = 25

        content = "fn test(name: &str) -> bool { true }"
        signature = extractor._extract_rust_function_signature(mock_node, content)

        assert signature == "fn test(name: &str) -> bool"

    def test_extract_import_statements_python(self):
        """Test Python import extraction."""
        extractor = MetadataExtractor()

        content = """import os
import sys
from pathlib import Path
from typing import Optional
"""
        imports = extractor._extract_import_statements(content, "python")

        assert "import os" in imports
        assert "import sys" in imports
        assert "from pathlib import Path" in imports
        assert "from typing import Optional" in imports

    def test_extract_import_statements_javascript(self):
        """Test JavaScript import extraction."""
        extractor = MetadataExtractor()

        content = """import React from 'react';
import { useState } from 'react';
const fs = require('fs');
"""
        imports = extractor._extract_import_statements(content, "javascript")

        assert "import React from 'react';" in imports
        assert "import { useState } from 'react';" in imports
        assert "const fs = require('fs');" in imports

    def test_extract_import_statements_java(self):
        """Test Java import extraction."""
        extractor = MetadataExtractor()

        content = """import java.util.List;
import java.io.IOException;
public class Test {}
"""
        imports = extractor._extract_import_statements(content, "java")

        assert "import java.util.List;" in imports
        assert "import java.io.IOException;" in imports

    def test_extract_import_statements_go(self):
        """Test Go import extraction."""
        extractor = MetadataExtractor()

        content = """import "fmt"
import "os"
import (
    "encoding/json"
)
"""
        imports = extractor._extract_import_statements(content, "go")

        assert 'import "fmt"' in imports
        assert 'import "os"' in imports
        assert '"encoding/json"' in imports

    def test_extract_import_statements_rust(self):
        """Test Rust import extraction."""
        extractor = MetadataExtractor()

        content = """use std::collections::HashMap;
use serde::{Serialize, Deserialize};
"""
        imports = extractor._extract_import_statements(content, "rust")

        assert "use std::collections::HashMap;" in imports
        assert "use serde::{Serialize, Deserialize};" in imports

    def test_extract_import_statements_no_imports(self):
        """Test import extraction with no imports."""
        extractor = MetadataExtractor()

        content = "def test(): pass"
        imports = extractor._extract_import_statements(content, "python")

        assert imports is None

    def test_calculate_complexity_simple(self):
        """Test complexity calculation for simple function."""
        extractor = MetadataExtractor()

        mock_node = Mock()
        mock_node.children = []

        complexity = extractor._calculate_complexity(mock_node)
        assert complexity == 1  # Base complexity

    def test_calculate_complexity_with_decisions(self):
        """Test complexity calculation with decision points."""
        extractor = MetadataExtractor()

        # Mock if statement child
        if_child = Mock()
        if_child.type = "if_statement"
        if_child.children = []

        # Mock while statement child
        while_child = Mock()
        while_child.type = "while_statement"
        while_child.children = []

        mock_node = Mock()
        mock_node.children = [if_child, while_child]

        complexity = extractor._calculate_complexity(mock_node)
        assert complexity == 3  # Base (1) + if (1) + while (1)

    def test_calculate_complexity_error(self):
        """Test complexity calculation with error."""
        extractor = MetadataExtractor()

        mock_node = Mock()
        mock_node.children = Mock(side_effect=Exception("Error"))

        complexity = extractor._calculate_complexity(mock_node)
        assert complexity is None

    async def test_extract_file_level_metadata_success(self):
        """Test file-level metadata extraction."""
        extractor = MetadataExtractor()
        extractor.parsers = {"python": Mock()}

        mock_parser = extractor.parsers["python"]
        mock_tree = Mock()
        mock_root = Mock()
        mock_root.children = []
        mock_tree.root_node = mock_root
        mock_parser.parse.return_value = mock_tree

        content = """import os
def test_func():
    pass

class TestClass:
    pass
"""

        with patch.object(extractor, "_detect_language_from_path", return_value="python"):
            with patch.object(extractor, "_extract_import_statements", return_value=["import os"]):
                with patch.object(extractor, "_extract_dependencies", return_value=["os"]):
                    metadata = await extractor.extract_file_level_metadata("/test/file.py", content)

                    assert metadata["file_path"] == "/test/file.py"
                    assert metadata["language"] == "python"
                    assert metadata["imports"] == ["import os"]
                    assert metadata["dependencies"] == ["os"]
                    assert metadata["total_lines"] == 6

    async def test_extract_file_level_metadata_unsupported_language(self):
        """Test file-level metadata for unsupported language."""
        extractor = MetadataExtractor()
        extractor.parsers = {}  # No parsers

        with patch.object(extractor, "_detect_language_from_path", return_value="unknown"):
            metadata = await extractor.extract_file_level_metadata("/test/file.xyz", "content")
            assert metadata == {}

    async def test_extract_file_level_metadata_error(self, caplog):
        """Test file-level metadata extraction with error."""
        extractor = MetadataExtractor()
        extractor.parsers = {"python": Mock()}

        mock_parser = extractor.parsers["python"]
        mock_parser.parse.side_effect = Exception("Parse error")

        with patch.object(extractor, "_detect_language_from_path", return_value="python"):
            metadata = await extractor.extract_file_level_metadata("/test/file.py", "content")

            assert metadata == {}
            assert "Error extracting file metadata" in caplog.text

    def test_extract_dependencies_python(self):
        """Test Python dependency extraction."""
        extractor = MetadataExtractor()

        content = """import os
import sys
from pathlib import Path
from mypackage.submodule import function
"""

        with patch.object(
            extractor,
            "_extract_import_statements",
            return_value=[
                "import os",
                "import sys",
                "from pathlib import Path",
                "from mypackage.submodule import function",
            ],
        ):
            deps = extractor._extract_dependencies(content, "python")

            assert "os" in deps
            assert "sys" in deps
            assert "pathlib" in deps
            assert "mypackage" in deps

    def test_extract_dependencies_javascript(self):
        """Test JavaScript dependency extraction."""
        extractor = MetadataExtractor()

        content = """import React from 'react';
const fs = require('fs');
"""

        with patch.object(
            extractor,
            "_extract_import_statements",
            return_value=["import React from 'react';", "const fs = require('fs');"],
        ):
            deps = extractor._extract_dependencies(content, "javascript")

            assert "react" in deps
            assert "fs" in deps

    def test_extract_dependencies_no_imports(self):
        """Test dependency extraction with no imports."""
        extractor = MetadataExtractor()

        with patch.object(extractor, "_extract_import_statements", return_value=None):
            deps = extractor._extract_dependencies("def test(): pass", "python")
            assert deps is None

    def test_extract_dependencies_error(self):
        """Test dependency extraction with error."""
        extractor = MetadataExtractor()

        with patch.object(extractor, "_extract_import_statements", side_effect=Exception("Error")):
            deps = extractor._extract_dependencies("content", "python")
            assert deps is None

    def test_detect_language_from_path(self):
        """Test language detection from file path."""
        extractor = MetadataExtractor()

        assert extractor._detect_language_from_path("/test/file.py") == "python"
        assert extractor._detect_language_from_path("/test/file.js") == "javascript"
        assert extractor._detect_language_from_path("/test/file.jsx") == "javascript"
        assert extractor._detect_language_from_path("/test/file.ts") == "typescript"
        assert extractor._detect_language_from_path("/test/file.tsx") == "typescript"
        assert extractor._detect_language_from_path("/test/file.go") == "go"
        assert extractor._detect_language_from_path("/test/file.rs") == "rust"
        assert extractor._detect_language_from_path("/test/file.java") == "java"
        assert extractor._detect_language_from_path("/test/file.txt") == "text"
