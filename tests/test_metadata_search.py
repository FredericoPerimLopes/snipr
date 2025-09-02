from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.models.indexing_models import CodeChunk
from src.services.metadata_search import MetadataSearchEngine


@pytest.fixture
def mock_config():
    """Mock configuration for tests."""
    config = Mock()
    config.VECTOR_DB_PATH = Path("/tmp/test.db")
    return config


@pytest.fixture
def sample_db_rows():
    """Sample database rows for testing."""
    return [
        # Function row
        (
            "/test/auth.py",
            "def authenticate_user(username, password):",
            1,
            2,
            "python",
            "function_definition",
            "def authenticate_user(username, password)",
            None,
            "authenticate_user",
            '["str", "str"]',
            "bool",
            None,
            '["import hashlib"]',
            "Authenticate user",
            2,
            '["hashlib"]',
            None,
            None,
        ),
        # Class row
        (
            "/test/user.py",
            "class UserManager:",
            5,
            10,
            "python",
            "class_definition",
            None,
            "UserManager",
            None,
            None,
            None,
            '["BaseManager"]',
            '["import os"]',
            "Manages users",
            None,
            '["os"]',
            None,
            None,
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Sample code chunks for testing."""
    return [
        CodeChunk(
            file_path="/test/auth.py",
            content="def authenticate_user(username, password):",
            start_line=1,
            end_line=2,
            language="python",
            semantic_type="function_definition",
            function_name="authenticate_user",
            function_signature="def authenticate_user(username, password)",
            parameter_types=["str", "str"],
            return_type="bool",
            docstring="Authenticate user",
            complexity_score=2,
        ),
        CodeChunk(
            file_path="/test/user.py",
            content="class UserManager:",
            start_line=5,
            end_line=10,
            language="python",
            semantic_type="class_definition",
            class_name="UserManager",
            inheritance_chain=["BaseManager"],
            docstring="Manages users",
        ),
    ]


class TestMetadataSearchEngine:
    @patch("src.services.metadata_search.get_settings")
    def test_init(self, mock_get_settings, mock_config):
        """Test metadata search engine initialization."""
        mock_get_settings.return_value = mock_config

        engine = MetadataSearchEngine()
        assert engine.config == mock_config
        assert engine.db_path == mock_config.VECTOR_DB_PATH

    @patch("src.services.metadata_search.get_settings")
    async def test_search_function_query(self, mock_get_settings, mock_config, sample_chunks):
        """Test search with function-type query."""
        mock_get_settings.return_value = mock_config
        engine = MetadataSearchEngine()

        with patch.object(
            engine, "_parse_metadata_query", return_value={"type": "function", "function_name": "authenticate"}
        ):
            with patch.object(engine, "search_functions", return_value=sample_chunks[:1]):
                results = await engine.search("function authenticate")

                assert len(results) == 1
                assert results[0].function_name == "authenticate_user"

    @patch("src.services.metadata_search.get_settings")
    async def test_search_class_query(self, mock_get_settings, mock_config, sample_chunks):
        """Test search with class-type query."""
        mock_get_settings.return_value = mock_config
        engine = MetadataSearchEngine()

        with patch.object(engine, "_parse_metadata_query", return_value={"type": "class", "class_name": "UserManager"}):
            with patch.object(engine, "search_classes", return_value=sample_chunks[1:]):
                results = await engine.search("class UserManager")

                assert len(results) == 1
                assert results[0].class_name == "UserManager"

    @patch("src.services.metadata_search.get_settings")
    async def test_search_import_query(self, mock_get_settings, mock_config, sample_chunks):
        """Test search with import-type query."""
        mock_get_settings.return_value = mock_config
        engine = MetadataSearchEngine()

        with patch.object(engine, "_parse_metadata_query", return_value={"type": "import", "import_module": "os"}):
            with patch.object(engine, "search_by_imports", return_value=sample_chunks):
                results = await engine.search("import os")

                assert len(results) == 2

    @patch("src.services.metadata_search.get_settings")
    async def test_search_general_query(self, mock_get_settings, mock_config, sample_chunks):
        """Test search with general query."""
        mock_get_settings.return_value = mock_config
        engine = MetadataSearchEngine()

        with patch.object(engine, "_parse_metadata_query", return_value={"type": "general"}):
            with patch.object(engine, "search_general_metadata", return_value=sample_chunks):
                results = await engine.search("user management")

                assert len(results) == 2

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_functions_success(self, mock_sqlite3, mock_get_settings, mock_config, sample_db_rows):
        """Test successful function search."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [sample_db_rows[0]]  # Function row only
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        results = await engine.search_functions(function_name="authenticate")

        assert len(results) == 1
        assert results[0].function_name == "authenticate_user"
        mock_sqlite3.connect.assert_called_with(str(mock_config.VECTOR_DB_PATH))

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_functions_with_filters(self, mock_sqlite3, mock_get_settings, mock_config, sample_db_rows):
        """Test function search with multiple filters."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [sample_db_rows[0]]
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        results = await engine.search_functions(
            function_name="authenticate", return_type="bool", parameter_types=["str"], language="python"
        )

        assert len(results) == 1
        # Verify query was called with all filters
        call_args = mock_conn.execute.call_args[0]
        assert "function_name LIKE ?" in call_args[0]
        assert "return_type LIKE ?" in call_args[0]
        assert "parameter_types LIKE ?" in call_args[0]
        assert "language = ?" in call_args[0]

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_functions_error(self, mock_sqlite3, mock_get_settings, mock_config, caplog):
        """Test function search with database error."""
        mock_get_settings.return_value = mock_config
        mock_sqlite3.connect.side_effect = Exception("DB error")

        engine = MetadataSearchEngine()
        results = await engine.search_functions(function_name="test")

        assert results == []
        assert "Error in function search" in caplog.text

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_classes_success(self, mock_sqlite3, mock_get_settings, mock_config, sample_db_rows):
        """Test successful class search."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [sample_db_rows[1]]  # Class row only
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        results = await engine.search_classes(class_name="UserManager")

        assert len(results) == 1
        assert results[0].class_name == "UserManager"

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_classes_with_inheritance(self, mock_sqlite3, mock_get_settings, mock_config, sample_db_rows):
        """Test class search with inheritance filter."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [sample_db_rows[1]]
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        results = await engine.search_classes(
            class_name="UserManager", inherits_from="BaseManager", implements=["Interface1"], language="python"
        )

        assert len(results) == 1
        # Verify inheritance and interface filters in query
        call_args = mock_conn.execute.call_args[0]
        assert "inheritance_chain LIKE ?" in call_args[0]
        assert "interfaces LIKE ?" in call_args[0]

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_by_imports_success(self, mock_sqlite3, mock_get_settings, mock_config, sample_db_rows):
        """Test successful import-based search."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = sample_db_rows
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        results = await engine.search_by_imports(import_module="os", uses_function="path")

        assert len(results) == 2
        # Verify import and function usage filters
        call_args = mock_conn.execute.call_args[0]
        assert "import_statements LIKE ?" in call_args[0] or "dependencies LIKE ?" in call_args[0]
        assert "content LIKE ?" in call_args[0]

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_general_metadata_success(self, mock_sqlite3, mock_get_settings, mock_config, sample_db_rows):
        """Test successful general metadata search."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = sample_db_rows
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        results = await engine.search_general_metadata("user", language="python")

        assert len(results) == 2
        # Verify general search across multiple fields
        call_args = mock_conn.execute.call_args[0]
        assert "function_name LIKE ?" in call_args[0]
        assert "class_name LIKE ?" in call_args[0]
        assert "language = ?" in call_args[0]

    def test_parse_metadata_query_function_with_return(self):
        """Test parsing function query with return type."""
        engine = MetadataSearchEngine()

        query = "function that returns bool"
        parsed = engine._parse_metadata_query(query)

        assert parsed["type"] == "function"
        assert parsed["return_type"] == "bool"

    def test_parse_metadata_query_function_by_name(self):
        """Test parsing function query by name."""
        engine = MetadataSearchEngine()

        query = "function authenticate"
        parsed = engine._parse_metadata_query(query)

        assert parsed["type"] == "function"
        assert parsed["function_name"] == "authenticate"

    def test_parse_metadata_query_class_with_inheritance(self):
        """Test parsing class query with inheritance."""
        engine = MetadataSearchEngine()

        query = "class that inherits from BaseManager"
        parsed = engine._parse_metadata_query(query)

        assert parsed["type"] == "class"
        assert parsed["inherits_from"] == "BaseManager"

    def test_parse_metadata_query_class_by_name(self):
        """Test parsing class query by name."""
        engine = MetadataSearchEngine()

        query = "class UserManager"
        parsed = engine._parse_metadata_query(query)

        assert parsed["type"] == "class"
        assert parsed["class_name"] == "UserManager"

    def test_parse_metadata_query_import(self):
        """Test parsing import query."""
        engine = MetadataSearchEngine()

        query = "import pandas"
        parsed = engine._parse_metadata_query(query)

        assert parsed["type"] == "import"
        assert parsed["import_module"] == "pandas"

    def test_parse_metadata_query_general(self):
        """Test parsing general query."""
        engine = MetadataSearchEngine()

        query = "user management logic"
        parsed = engine._parse_metadata_query(query)

        assert parsed["type"] == "general"

    def test_row_to_chunk_success(self, sample_db_rows):
        """Test successful row to chunk conversion."""
        engine = MetadataSearchEngine()

        chunk = engine._row_to_chunk(sample_db_rows[0])

        assert isinstance(chunk, CodeChunk)
        assert chunk.file_path == "/test/auth.py"
        assert chunk.function_name == "authenticate_user"
        assert chunk.parameter_types == ["str", "str"]
        assert chunk.return_type == "bool"
        assert chunk.complexity_score == 2

    def test_row_to_chunk_invalid_json(self, caplog):
        """Test row to chunk conversion with invalid JSON."""
        engine = MetadataSearchEngine()

        # Row with invalid JSON in parameter_types field
        invalid_row = (
            "/test/file.py",
            "content",
            1,
            2,
            "python",
            "function_definition",
            "signature",
            None,
            "test_func",
            "invalid_json",
            "void",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        chunk = engine._row_to_chunk(invalid_row)
        assert chunk is None
        assert "Error converting row to chunk" in caplog.text

    def test_tokenize_signature_basic(self):
        """Test signature tokenization."""
        engine = MetadataSearchEngine()

        signature = "def authenticate_user(username: str, password: str) -> bool"
        tokens = engine._tokenize_signature(signature)

        assert "authenticate_user" in tokens
        assert "username" in tokens
        assert "password" in tokens
        assert "str" in tokens
        assert "bool" in tokens
        # Should filter out common keywords
        assert "def" not in tokens

    def test_tokenize_signature_with_stop_words(self):
        """Test signature tokenization filters stop words."""
        engine = MetadataSearchEngine()

        signature = "public static void main(String[] args)"
        tokens = engine._tokenize_signature(signature)

        assert "main" in tokens
        assert "String" in tokens
        assert "args" in tokens
        # Should filter out stop words
        assert "public" not in tokens
        assert "static" not in tokens
        assert "void" not in tokens

    def test_tokenize_signature_empty(self):
        """Test signature tokenization with empty input."""
        engine = MetadataSearchEngine()

        tokens = engine._tokenize_signature("")
        assert tokens == []

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_by_complexity_range(self, mock_sqlite3, mock_get_settings, mock_config, sample_db_rows):
        """Test search by complexity range."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [sample_db_rows[0]]
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        results = await engine.search_by_complexity(min_complexity=1, max_complexity=5)

        assert len(results) == 1
        # Verify complexity filters in query
        call_args = mock_conn.execute.call_args[0]
        assert "complexity_score >= ?" in call_args[0]
        assert "complexity_score <= ?" in call_args[0]

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_by_complexity_with_language(
        self, mock_sqlite3, mock_get_settings, mock_config, sample_db_rows
    ):
        """Test complexity search with language filter."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [sample_db_rows[0]]
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        results = await engine.search_by_complexity(min_complexity=1, max_complexity=10, language="python")

        assert len(results) == 1
        call_args = mock_conn.execute.call_args[0]
        assert "language = ?" in call_args[0]

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_by_complexity_error(self, mock_sqlite3, mock_get_settings, mock_config, caplog):
        """Test complexity search with database error."""
        mock_get_settings.return_value = mock_config
        mock_sqlite3.connect.side_effect = Exception("DB error")

        engine = MetadataSearchEngine()
        results = await engine.search_by_complexity(min_complexity=1)

        assert results == []
        assert "Error in complexity search" in caplog.text

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_similar_functions_success(self, mock_sqlite3, mock_get_settings, mock_config, sample_db_rows):
        """Test successful similar function search."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [sample_db_rows[0]]
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()

        with patch.object(engine, "_tokenize_signature", return_value=["authenticate", "user", "password"]):
            results = await engine.search_similar_functions("def authenticate_user(password)")

            assert len(results) == 1
            assert results[0].function_name == "authenticate_user"

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_similar_functions_error(self, mock_sqlite3, mock_get_settings, mock_config, caplog):
        """Test similar function search with error."""
        mock_get_settings.return_value = mock_config
        mock_sqlite3.connect.side_effect = Exception("DB error")

        engine = MetadataSearchEngine()
        results = await engine.search_similar_functions("def test()")

        assert results == []
        assert "Error in similar function search" in caplog.text

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_get_inheritance_tree_success(self, mock_sqlite3, mock_get_settings, mock_config):
        """Test successful inheritance tree building."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ("UserManager", '["BaseManager"]', '["Serializable"]')
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()

        # Mock recursive call to avoid infinite recursion
        with patch.object(
            engine,
            "get_inheritance_tree",
            side_effect=[
                # First call (actual)
                {"class_name": "UserManager", "parents": ["BaseManager"], "interfaces": ["Serializable"]},
                # Recursive call for BaseManager
                {"class_name": "BaseManager", "parents": [], "interfaces": []},
            ],
        ) as mock_recursive:
            # Only call the original method for the first call
            mock_recursive.side_effect[0] = await engine.get_inheritance_tree.__wrapped__(engine, "UserManager")

            tree = await engine.get_inheritance_tree("UserManager")

            assert tree["class_name"] == "UserManager"
            assert tree["parents"] == ["BaseManager"]
            assert tree["interfaces"] == ["Serializable"]

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_get_inheritance_tree_not_found(self, mock_sqlite3, mock_get_settings, mock_config):
        """Test inheritance tree when class not found."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        tree = await engine.get_inheritance_tree("NonExistentClass")

        assert tree["class_name"] == "NonExistentClass"
        assert tree["parents"] == []
        assert tree["interfaces"] == []

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_get_inheritance_tree_error(self, mock_sqlite3, mock_get_settings, mock_config, caplog):
        """Test inheritance tree with database error."""
        mock_get_settings.return_value = mock_config
        mock_sqlite3.connect.side_effect = Exception("DB error")

        engine = MetadataSearchEngine()
        tree = await engine.get_inheritance_tree("TestClass")

        assert tree["class_name"] == "TestClass"
        assert tree["parents"] == []
        assert tree["interfaces"] == []
        assert "Error building inheritance tree" in caplog.text

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_get_dependency_graph_success(self, mock_sqlite3, mock_get_settings, mock_config):
        """Test successful dependency graph building."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ('["import os", "import sys"]', '["os", "sys"]', "test_func", "TestClass"),
            ('["import json"]', '["json"]', "parse_data", None),
        ]
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        graph = await engine.get_dependency_graph("/test/file.py")

        assert graph["file_path"] == "/test/file.py"
        assert "os" in graph["imports"]
        assert "sys" in graph["imports"]
        assert "json" in graph["imports"]
        assert "os" in graph["dependencies"]
        assert "test_func" in graph["functions"]
        assert "TestClass" in graph["classes"]

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_get_dependency_graph_error(self, mock_sqlite3, mock_get_settings, mock_config, caplog):
        """Test dependency graph with database error."""
        mock_get_settings.return_value = mock_config
        mock_sqlite3.connect.side_effect = Exception("DB error")

        engine = MetadataSearchEngine()
        graph = await engine.get_dependency_graph("/test/file.py")

        assert graph["file_path"] == "/test/file.py"
        assert graph["imports"] == []
        assert graph["dependencies"] == []
        assert graph["functions"] == []
        assert graph["classes"] == []
        assert "Error building dependency graph" in caplog.text

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_functions_no_results(self, mock_sqlite3, mock_get_settings, mock_config):
        """Test function search with no results."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        results = await engine.search_functions(function_name="nonexistent")

        assert results == []

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_classes_no_results(self, mock_sqlite3, mock_get_settings, mock_config):
        """Test class search with no results."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        results = await engine.search_classes(class_name="NonExistent")

        assert results == []

    @patch("src.services.metadata_search.get_settings")
    async def test_search_without_filters(self, mock_get_settings, mock_config):
        """Test search methods without any filters."""
        mock_get_settings.return_value = mock_config
        engine = MetadataSearchEngine()

        with patch("src.services.metadata_search.sqlite3") as mock_sqlite3:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_cursor.fetchall.return_value = []
            mock_conn.execute.return_value = mock_cursor
            mock_sqlite3.connect.return_value = mock_conn

            # Test functions without filters
            results = await engine.search_functions()
            assert results == []

            # Test classes without filters
            results = await engine.search_classes()
            assert results == []

            # Test imports without filters
            results = await engine.search_by_imports()
            assert results == []

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_by_imports_no_filters(self, mock_sqlite3, mock_get_settings, mock_config):
        """Test import search without specific filters."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        results = await engine.search_by_imports()

        # Should still execute query (with WHERE 1=1)
        assert results == []
        call_args = mock_conn.execute.call_args[0]
        assert "1=1" in call_args[0]

    @patch("src.services.metadata_search.get_settings")
    @patch("src.services.metadata_search.sqlite3")
    async def test_search_general_metadata_with_ranking(
        self, mock_sqlite3, mock_get_settings, mock_config, sample_db_rows
    ):
        """Test general metadata search uses proper ranking."""
        mock_get_settings.return_value = mock_config

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = sample_db_rows
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        engine = MetadataSearchEngine()
        results = await engine.search_general_metadata("user")

        assert len(results) == 2
        # Verify ranking in ORDER BY clause
        call_args = mock_conn.execute.call_args[0]
        assert "ORDER BY" in call_args[0]
        assert "CASE" in call_args[0]
        assert "function_name LIKE ?" in call_args[0]

    def test_row_to_chunk_with_null_values(self):
        """Test row to chunk conversion with NULL database values."""
        engine = MetadataSearchEngine()

        # Row with many NULL values
        row = (
            "/test/simple.py",
            "x = 1",
            1,
            1,
            "python",
            "variable_declaration",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        chunk = engine._row_to_chunk(row)

        assert isinstance(chunk, CodeChunk)
        assert chunk.file_path == "/test/simple.py"
        assert chunk.content == "x = 1"
        assert chunk.function_name is None
        assert chunk.class_name is None
        assert chunk.parameter_types is None

    @patch("src.services.metadata_search.get_settings")
    async def test_search_edge_cases(self, mock_get_settings, mock_config):
        """Test search with edge case inputs."""
        mock_get_settings.return_value = mock_config
        engine = MetadataSearchEngine()

        with patch("src.services.metadata_search.sqlite3") as mock_sqlite3:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_cursor.fetchall.return_value = []
            mock_conn.execute.return_value = mock_cursor
            mock_sqlite3.connect.return_value = mock_conn

            # Empty query
            results = await engine.search("")
            assert results == []

            # Very long query
            long_query = "a" * 1000
            results = await engine.search(long_query)
            assert results == []

            # Special characters in query
            results = await engine.search("function!@#$%^&*()")
            assert results == []
