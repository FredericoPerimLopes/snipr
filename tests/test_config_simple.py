from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import get_settings, validate_codebase_path


class TestConfigBasic:
    def test_get_settings_basic(self):
        """Test basic settings retrieval."""
        with patch("src.config.Path.exists", return_value=False):
            settings = get_settings()
            assert hasattr(settings, "EMBEDDING_ENABLED")
            assert hasattr(settings, "EMBEDDING_MODEL")

    def test_validate_codebase_path_valid(self):
        """Test validation of valid codebase path."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                result = validate_codebase_path("/valid/path")
                assert isinstance(result, Path)

    def test_validate_codebase_path_not_exists(self):
        """Test validation when path doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ValueError):
                validate_codebase_path("/nonexistent/path")

    def test_validate_codebase_path_not_directory(self):
        """Test validation when path is not a directory."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=False):
                with pytest.raises(ValueError):
                    validate_codebase_path("/path/to/file.txt")

    @pytest.mark.parametrize("path", ["   "])
    def test_validate_codebase_path_invalid(self, path):
        """Test validation of invalid codebase paths."""
        with pytest.raises(ValueError):
            validate_codebase_path(path)

    def test_validate_codebase_path_none(self):
        """Test validation with None path."""
        with pytest.raises(TypeError):
            validate_codebase_path(None)

    def test_validate_codebase_path_empty(self):
        """Test validation with empty string."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ValueError):
                validate_codebase_path("empty_path")

    def test_get_settings_cache_dir_creation(self):
        """Test settings cache directory creation."""
        with patch("src.config.Path.mkdir") as mock_mkdir:
            with patch("src.config.Path.exists", return_value=False):
                settings = get_settings()
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                assert hasattr(settings, "INDEX_CACHE_DIR")
