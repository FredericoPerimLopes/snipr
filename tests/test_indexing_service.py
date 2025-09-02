import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

from src.models.indexing_models import CodeChunk, IndexingRequest, IndexingStatus
from src.services.indexing_service import IndexingService


@pytest.fixture
def mock_config():
    """Mock configuration for tests."""
    config = Mock()
    config.MAX_FILE_SIZE_MB = 10
    config.INDEX_CACHE_DIR = Path("/tmp/test_cache")
    config.DEFAULT_EXCLUDE_PATTERNS = ["*.pyc", "*.git*", "node_modules/*"]
    return config


@pytest.fixture
def sample_chunks():
    """Sample code chunks for testing."""
    return [
        CodeChunk(
            file_path="/test/auth.py",
            content="def authenticate_user(username, password):\n    return validate_credentials(username, password)",
            start_line=1,
            end_line=2,
            language="python",
            semantic_type="function_definition",
            function_name="authenticate_user",
        ),
        CodeChunk(
            file_path="/test/utils.py",
            content="class Logger:\n    def log(self, message):\n        print(message)",
            start_line=5,
            end_line=7,
            language="python",
            semantic_type="class_definition",
            class_name="Logger",
        ),
    ]


@pytest.fixture
def mock_metadata():
    """Mock index metadata."""
    return {
        "codebase_path": "/test/codebase",
        "total_chunks": 10,
        "last_indexed": "2023-01-01T12:00:00",
        "file_hashes": {"/test/file1.py": "hash1", "/test/file2.py": "hash2"},
    }


class TestIndexingService:
    @patch("src.services.indexing_service.get_settings")
    def test_init_basic(self, mock_get_settings, mock_config):
        """Test basic initialization."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            assert service.config == mock_config
            assert isinstance(service.parsers, dict)
            assert isinstance(service.languages, dict)
            assert isinstance(service.code_splitters, dict)

    @patch("src.services.indexing_service.get_settings")
    @patch("src.services.indexing_service.tspython")
    def test_init_parsers_success(self, mock_tspython, mock_get_settings, mock_config):
        """Test successful parser initialization."""
        mock_get_settings.return_value = mock_config

        # Mock tree-sitter language module
        mock_language_func = Mock()
        mock_tspython.language = mock_language_func

        with (
            patch("src.services.indexing_service.Language") as mock_lang_class,
            patch("src.services.indexing_service.Parser") as mock_parser_class,
        ):
            mock_lang = Mock()
            mock_lang_class.return_value = mock_lang
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser

            service = IndexingService()

            # Should initialize python parser
            assert "python" in service.languages
            assert "python" in service.parsers

    @patch("src.services.indexing_service.get_settings")
    @patch("src.services.indexing_service.CodeSplitter")
    def test_init_code_splitters_success(self, mock_code_splitter, mock_get_settings, mock_config):
        """Test successful code splitter initialization."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"):
            service = IndexingService()
            service.parsers = {"python": Mock()}  # Mock parser exists

            mock_splitter = Mock()
            mock_code_splitter.return_value = mock_splitter

            service._init_code_splitters()

            assert "python" in service.code_splitters
            mock_code_splitter.assert_called_with(
                language="python", chunk_lines=15, chunk_lines_overlap=3, max_chars=600
            )

    @patch("src.services.indexing_service.get_settings")
    def test_init_code_splitters_unavailable(self, mock_get_settings, mock_config, caplog):
        """Test code splitter initialization when CodeSplitter is None."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch("src.services.indexing_service.CodeSplitter", None):
            service = IndexingService()

            assert len(service.code_splitters) == 0
            assert "LlamaIndex CodeSplitter not available" in caplog.text

    @patch("src.services.indexing_service.get_settings")
    @patch("src.services.indexing_service.validate_codebase_path")
    async def test_index_codebase_no_changes(self, mock_validate, mock_get_settings, mock_config):
        """Test indexing when no files have changed."""
        mock_get_settings.return_value = mock_config
        mock_validate.return_value = Path("/test/codebase")

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            # Mock no changed files
            service.get_changed_files = AsyncMock(return_value=([], [], []))
            service.get_indexing_status = AsyncMock(
                return_value=IndexingStatus(is_indexed=True, total_files=5, total_chunks=50, index_size_mb=1.0)
            )

            request = IndexingRequest(codebase_path="/test/codebase")
            result = await service.index_codebase(request)

            assert result.indexed_files == 0
            assert result.total_chunks == 50
            assert result.status == "up_to_date"

    @patch("src.services.indexing_service.get_settings")
    @patch("src.services.indexing_service.validate_codebase_path")
    async def test_index_codebase_with_changes(self, mock_validate, mock_get_settings, mock_config, sample_chunks):
        """Test indexing with file changes."""
        mock_get_settings.return_value = mock_config
        mock_validate.return_value = Path("/test/codebase")

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            # Mock changed files
            service.get_changed_files = AsyncMock(
                return_value=([Path("/test/modified.py")], [Path("/test/new.py")], [Path("/test/deleted.py")])
            )
            service._parse_file = AsyncMock(return_value=sample_chunks)
            service._update_index_metadata = AsyncMock()

            # Mock SearchService
            mock_search_service = Mock()
            mock_search_service.remove_file_embeddings = AsyncMock()
            mock_search_service.embed_code_chunks = AsyncMock(return_value=sample_chunks)

            with patch("src.services.indexing_service.SearchService", return_value=mock_search_service):
                request = IndexingRequest(codebase_path="/test/codebase")
                result = await service.index_codebase(request)

                assert result.indexed_files == 2  # modified + new
                assert result.total_chunks == 4  # 2 files * 2 chunks each
                assert result.status == "success"

    def test_detect_language(self):
        """Test language detection from file extensions."""
        with patch("src.services.indexing_service.get_settings"):
            with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
                service = IndexingService()

                assert service._detect_language(Path("test.py")) == "python"
                assert service._detect_language(Path("test.js")) == "javascript"
                assert service._detect_language(Path("test.tsx")) == "typescript"
                assert service._detect_language(Path("test.go")) == "go"
                assert service._detect_language(Path("test.rs")) == "rust"
                assert service._detect_language(Path("test.java")) == "java"
                assert service._detect_language(Path("test.txt")) == "text"

    @patch("src.services.indexing_service.get_settings")
    async def test_parse_file_success(self, mock_get_settings, mock_config, sample_chunks):
        """Test successful file parsing."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()
            service.parsers = {"python": Mock()}
            service.syntactic_chunker.chunk_with_integrity = AsyncMock(return_value=sample_chunks)

            file_path = Path("/test/auth.py")
            with patch.object(file_path, "read_text", return_value="def test(): pass"):
                chunks = await service._parse_file(file_path)

                assert len(chunks) == 2
                assert all(isinstance(chunk, CodeChunk) for chunk in chunks)

    @patch("src.services.indexing_service.get_settings")
    async def test_parse_file_unsupported_language(self, mock_get_settings, mock_config):
        """Test parsing file with unsupported language."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()
            service.parsers = {}  # No parsers available

            file_path = Path("/test/unknown.xyz")
            chunks = await service._parse_file(file_path)

            assert chunks == []

    @patch("src.services.indexing_service.get_settings")
    async def test_parse_file_read_error(self, mock_get_settings, mock_config, caplog):
        """Test parsing file with read error."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()
            service.parsers = {"python": Mock()}

            file_path = Path("/test/auth.py")
            with patch.object(file_path, "read_text", side_effect=Exception("Read failed")):
                chunks = await service._parse_file(file_path)

                assert chunks == []
                assert "Error parsing" in caplog.text

    @patch("src.services.indexing_service.get_settings")
    async def test_basic_chunking(self, mock_get_settings, mock_config):
        """Test basic line-based chunking."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            content = "\n".join([f"line {i}" for i in range(20)])  # 20 lines
            file_path = Path("/test/file.txt")

            chunks = await service._basic_chunking(file_path, content, "text")

            assert len(chunks) > 0
            assert all(chunk.semantic_type == "code_block" for chunk in chunks)
            assert all(chunk.language == "text" for chunk in chunks)

    @patch("src.services.indexing_service.get_settings")
    async def test_basic_chunking_empty_content(self, mock_get_settings, mock_config):
        """Test basic chunking with empty content."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            chunks = await service._basic_chunking(Path("/test/empty.txt"), "", "text")
            assert chunks == []

    def test_should_exclude_path(self):
        """Test path exclusion logic."""
        with patch("src.services.indexing_service.get_settings"):
            with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
                service = IndexingService()

                exclude_patterns = ["*.pyc", "*/node_modules/*", "*.git*"]

                assert service._should_exclude_path(Path("/test/file.pyc"), exclude_patterns) == True
                assert service._should_exclude_path(Path("/test/node_modules/lib.js"), exclude_patterns) == True
                assert service._should_exclude_path(Path("/test/.git/config"), exclude_patterns) == True
                assert service._should_exclude_path(Path("/test/main.py"), exclude_patterns) == False

    @patch("src.services.indexing_service.get_settings")
    async def test_discover_source_files(self, mock_get_settings, mock_config):
        """Test source file discovery."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            # Mock file system structure
            with patch("os.walk") as mock_walk:
                mock_walk.return_value = [
                    ("/test", ["subdir"], ["main.py", "test.js", "config.json"]),
                    ("/test/subdir", [], ["utils.py", "large.py"]),
                ]

                # Mock file stats
                def mock_stat(size):
                    stat_result = Mock()
                    stat_result.st_size = size
                    return stat_result

                with patch("pathlib.Path.stat") as mock_path_stat:
                    mock_path_stat.return_value = mock_stat(1000)  # Small file

                    files = await service._discover_source_files(Path("/test"), ["python", "javascript"], [])

                    # Should find .py and .js files
                    py_files = [f for f in files if f.suffix == ".py"]
                    js_files = [f for f in files if f.suffix == ".js"]
                    assert len(py_files) >= 1
                    assert len(js_files) >= 1

    @patch("src.services.indexing_service.get_settings")
    async def test_discover_source_files_size_limit(self, mock_get_settings, mock_config):
        """Test file discovery respects size limits."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            with patch("os.walk") as mock_walk:
                mock_walk.return_value = [("/test", [], ["small.py", "large.py"])]

                def mock_stat_by_name(self):
                    if "large" in str(self):
                        result = Mock()
                        result.st_size = 20 * 1024 * 1024  # 20MB (over 10MB limit)
                        return result
                    else:
                        result = Mock()
                        result.st_size = 1000  # Small file
                        return result

                with patch("pathlib.Path.stat", mock_stat_by_name):
                    files = await service._discover_source_files(Path("/test"), None, [])

                    # Should only include small file
                    file_names = [f.name for f in files]
                    assert "small.py" in file_names
                    assert "large.py" not in file_names

    @patch("src.services.indexing_service.get_settings")
    async def test_get_semantic_type(self, mock_get_settings, mock_config):
        """Test semantic type detection."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            # Mock parser and tree
            mock_parser = Mock()
            mock_tree = Mock()
            mock_root = Mock()
            mock_root.type = "function_definition"
            mock_root.children = []
            mock_tree.root_node = mock_root
            mock_parser.parse.return_value = mock_tree

            semantic_type = await service._get_semantic_type("def test(): pass", mock_parser, "python")
            assert semantic_type == "function_definition"

    @patch("src.services.indexing_service.get_settings")
    async def test_get_semantic_type_error(self, mock_get_settings, mock_config, caplog):
        """Test semantic type detection with parsing error."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            mock_parser = Mock()
            mock_parser.parse.side_effect = Exception("Parse error")

            semantic_type = await service._get_semantic_type("invalid code", mock_parser, "python")
            assert semantic_type == "code_block"
            assert "Error determining semantic type" in caplog.text

    @patch("src.services.indexing_service.get_settings")
    async def test_extract_chunks_from_node(self, mock_get_settings, mock_config):
        """Test chunk extraction from AST node."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()
            service.metadata_extractor.extract_all_metadata = AsyncMock(
                return_value=Mock(
                    semantic_type="function_definition",
                    function_signature="def test()",
                    class_name=None,
                    function_name="test",
                    parameter_types=None,
                    return_type=None,
                    inheritance_chain=None,
                    import_statements=None,
                    docstring=None,
                    complexity_score=None,
                    dependencies=None,
                    interfaces=None,
                    decorators=None,
                )
            )

            # Mock node
            mock_node = Mock()
            mock_node.type = "function_definition"
            mock_node.start_point = (0, 0)
            mock_node.end_point = (2, 0)
            mock_node.children = []

            content = "def test():\n    pass\n"
            chunks = []

            await service._extract_chunks_from_node(mock_node, content, "/test/file.py", "python", chunks)

            assert len(chunks) == 1
            assert chunks[0].semantic_type == "function_definition"
            assert chunks[0].function_name == "test"

    @patch("src.services.indexing_service.get_settings")
    async def test_update_index_metadata(self, mock_get_settings, mock_config, mock_metadata):
        """Test index metadata update."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            metadata_path = mock_config.INDEX_CACHE_DIR / "index_metadata.json"

            # Mock existing metadata file
            with patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.read_text", return_value="new content"):
                        updated_files = [Path("/test/new_file.py")]
                        deleted_files = [Path("/test/file2.py")]

                        await service._update_index_metadata(Path("/test/codebase"), updated_files, deleted_files)

    @patch("src.services.indexing_service.get_settings")
    async def test_get_indexing_status_exists(self, mock_get_settings, mock_config, mock_metadata):
        """Test getting indexing status when metadata exists."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            metadata_path = mock_config.INDEX_CACHE_DIR / "index_metadata.json"

            with patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.iterdir", return_value=[]):
                        with patch("src.services.indexing_service.validate_codebase_path"):
                            status = await service.get_indexing_status("/test/codebase")

                            assert status.is_indexed == True
                            assert status.total_files == 2
                            assert status.total_chunks == 10

    @patch("src.services.indexing_service.get_settings")
    async def test_get_indexing_status_not_exists(self, mock_get_settings, mock_config):
        """Test getting indexing status when metadata doesn't exist."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            with patch("pathlib.Path.exists", return_value=False):
                with patch("src.services.indexing_service.validate_codebase_path"):
                    status = await service.get_indexing_status("/test/codebase")

                    assert status.is_indexed == False
                    assert status.total_files == 0
                    assert status.total_chunks == 0

    @patch("src.services.indexing_service.get_settings")
    async def test_needs_reindexing_no_metadata(self, mock_get_settings, mock_config):
        """Test reindexing check when no metadata exists."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            with patch("pathlib.Path.exists", return_value=False):
                needs_reindex = await service.needs_reindexing("/test/codebase")
                assert needs_reindex == True

    @patch("src.services.indexing_service.get_settings")
    async def test_needs_reindexing_file_changed(self, mock_get_settings, mock_config, mock_metadata):
        """Test reindexing check when file content changed."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            with patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.read_text", return_value="changed content"):
                        needs_reindex = await service.needs_reindexing("/test/codebase")
                        assert needs_reindex == True

    @patch("src.services.indexing_service.get_settings")
    async def test_needs_reindexing_file_deleted(self, mock_get_settings, mock_config, mock_metadata):
        """Test reindexing check when tracked file is deleted."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            with patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))):
                with patch("pathlib.Path.exists", side_effect=lambda: False):  # File doesn't exist
                    needs_reindex = await service.needs_reindexing("/test/codebase")
                    assert needs_reindex == True

    @patch("src.services.indexing_service.get_settings")
    async def test_get_changed_files_no_metadata(self, mock_get_settings, mock_config):
        """Test change detection when no metadata exists."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()
            service._discover_source_files = AsyncMock(return_value=[Path("/test/file.py")])

            with patch("pathlib.Path.exists", return_value=False):
                modified, new, deleted = await service.get_changed_files("/test/codebase")

                assert modified == []
                assert len(new) == 1
                assert deleted == []

    @patch("src.services.indexing_service.get_settings")
    async def test_get_changed_files_with_changes(self, mock_get_settings, mock_config, mock_metadata):
        """Test change detection with actual changes."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            # Mock current files (file1 modified, file3 new, file2 deleted)
            current_files = [Path("/test/file1.py"), Path("/test/file3.py")]
            service._discover_source_files = AsyncMock(return_value=current_files)

            with patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))):
                with patch("pathlib.Path.exists", return_value=True):
                    # Mock file1 as changed (different hash)
                    with patch("pathlib.Path.read_text", return_value="modified content"):
                        modified, new, deleted = await service.get_changed_files("/test/codebase")

                        assert len(modified) == 1  # file1.py modified
                        assert len(new) == 1  # file3.py new
                        assert len(deleted) == 1  # file2.py deleted

    @patch("src.services.indexing_service.get_settings")
    async def test_get_changed_files_error(self, mock_get_settings, mock_config, caplog):
        """Test change detection with error."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()
            service._discover_source_files = AsyncMock(return_value=[Path("/test/file.py")])

            with patch("builtins.open", side_effect=Exception("JSON error")):
                modified, new, deleted = await service.get_changed_files("/test/codebase")

                # Should treat all as new files on error
                assert modified == []
                assert len(new) == 1
                assert deleted == []
                assert "Error detecting changed files" in caplog.text

    @patch("src.services.indexing_service.get_settings")
    async def test_store_index_metadata(self, mock_get_settings, mock_config, sample_chunks):
        """Test storing index metadata."""
        mock_get_settings.return_value = mock_config

        with patch.object(IndexingService, "_init_parsers"), patch.object(IndexingService, "_init_code_splitters"):
            service = IndexingService()

            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.read_text", return_value="file content"):
                    with patch("builtins.open", mock_open()) as mock_file:
                        await service._store_index_metadata(Path("/test/codebase"), sample_chunks)

                        # Should write metadata JSON
                        mock_file.assert_called_once()
                        write_calls = mock_file().write.call_args_list
                        written_content = "".join(call[0][0] for call in write_calls)

                        # Should contain expected metadata structure
                        assert "codebase_path" in written_content
                        assert "total_chunks" in written_content
                        assert "file_hashes" in written_content
