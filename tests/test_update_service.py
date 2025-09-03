"""Integration tests for IncrementalUpdateService."""

import tempfile
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from unittest import TestCase

from src.models.indexing_models import FileUpdateRecord
from src.services.update_service import IncrementalUpdateService


class TestIncrementalUpdateService(TestCase):
    """Test incremental update service functionality."""

    def setUp(self):
        """Set up test environment with temporary git repository."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.update_service = IncrementalUpdateService()
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=self.temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.temp_dir, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.temp_dir, check=True)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_detect_new_files_in_git_repo(self):
        """Test detecting new files in git repository."""
        # Create a new Python file
        test_file = self.temp_dir / "test.py"
        test_file.write_text("def hello(): pass")
        
        # Run change detection
        import asyncio
        modified, new, deleted = asyncio.run(
            self.update_service.detect_changes(str(self.temp_dir))
        )
        
        assert len(modified) == 0
        assert len(new) == 1
        assert str(test_file) in new
        assert len(deleted) == 0

    def test_detect_modified_files_in_git_repo(self):
        """Test detecting modified files in git repository."""
        # Create and commit a file
        test_file = self.temp_dir / "test.py"
        test_file.write_text("def hello(): pass")
        subprocess.run(["git", "add", "test.py"], cwd=self.temp_dir, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.temp_dir, check=True)
        
        # Modify the file
        test_file.write_text("def hello(): return 'world'")
        
        # Run change detection
        import asyncio
        modified, new, deleted = asyncio.run(
            self.update_service.detect_changes(str(self.temp_dir))
        )
        
        assert len(modified) == 1
        assert str(test_file) in modified
        assert len(new) == 0
        assert len(deleted) == 0

    def test_detect_deleted_files_in_git_repo(self):
        """Test detecting deleted files in git repository."""
        # Create and commit a file
        test_file = self.temp_dir / "test.py"
        test_file.write_text("def hello(): pass")
        subprocess.run(["git", "add", "test.py"], cwd=self.temp_dir, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.temp_dir, check=True)
        
        # Delete the file
        test_file.unlink()
        
        # Run change detection
        import asyncio
        modified, new, deleted = asyncio.run(
            self.update_service.detect_changes(str(self.temp_dir))
        )
        
        assert len(modified) == 0
        assert len(new) == 0
        assert len(deleted) == 1
        assert str(test_file) in deleted

    def test_python_import_extraction(self):
        """Test extracting Python import dependencies."""
        content = '''
import os
import sys
from pathlib import Path
from typing import Dict, List
from .local_module import function
from ..parent_module import Class
        '''.strip()
        
        imports = self.update_service._extract_python_imports(content)
        expected = ["os", "sys", "pathlib", "typing"]
        
        for expected_import in expected:
            assert expected_import in imports

    def test_javascript_import_extraction(self):
        """Test extracting JavaScript import dependencies."""
        content = '''
import React from 'react';
import { useState } from 'react';
import axios from 'axios';
const fs = require('fs');
const path = require('path');
        '''.strip()
        
        imports = self.update_service._extract_js_imports(content)
        expected = ["react", "axios", "fs", "path"]
        
        for expected_import in expected:
            assert expected_import in imports

    def test_dependency_graph_building(self):
        """Test building dependency graph from files."""
        # Create test files with dependencies
        file1 = self.temp_dir / "module1.py"
        file1.write_text("import os\nfrom pathlib import Path")
        
        file2 = self.temp_dir / "module2.py"  
        file2.write_text("import module1\nfrom module1 import function")
        
        # Build dependency graph
        import asyncio
        asyncio.run(self.update_service.build_dependency_graph(str(self.temp_dir)))
        
        # Check that dependencies were built
        assert isinstance(self.update_service.dependency_graph, dict)

    def test_file_hash_calculation(self):
        """Test file content hash calculation."""
        test_file = self.temp_dir / "test.py"
        content = "def hello(): pass"
        test_file.write_text(content)
        
        hash1 = self.update_service.calculate_file_hash(str(test_file))
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex length
        
        # Same content should produce same hash
        hash2 = self.update_service.calculate_file_hash(str(test_file))
        assert hash1 == hash2
        
        # Different content should produce different hash
        test_file.write_text("def hello(): return 'world'")
        hash3 = self.update_service.calculate_file_hash(str(test_file))
        assert hash1 != hash3

    def test_should_update_index_with_changes(self):
        """Test checking if index needs updating when files changed."""
        # Create a new file
        test_file = self.temp_dir / "test.py"
        test_file.write_text("def hello(): pass")
        
        # Should need update for new file
        import asyncio
        should_update = asyncio.run(
            self.update_service.should_update_index(str(self.temp_dir))
        )
        assert should_update is True

    def test_should_update_index_no_changes(self):
        """Test checking if index needs updating with no changes."""
        # Empty directory with git repo - no source files
        import asyncio
        should_update = asyncio.run(
            self.update_service.should_update_index(str(self.temp_dir))
        )
        # Should be False since no source files to index
        assert should_update is False

    def test_file_update_record_management(self):
        """Test updating and retrieving file records."""
        file_path = "/test/file.py"
        content_hash = "abc123"
        chunk_ids = ["chunk_1", "chunk_2"]
        dependencies = ["module_a"]
        
        # Update file record
        import asyncio
        asyncio.run(self.update_service.update_file_record(
            file_path, content_hash, chunk_ids, dependencies
        ))
        
        # Check record was stored
        assert file_path in self.update_service.file_records
        record = self.update_service.file_records[file_path]
        assert record.content_hash == content_hash
        assert record.chunk_ids == chunk_ids
        assert record.dependencies == dependencies

    def test_non_git_fallback_detection(self):
        """Test fallback to filesystem detection for non-git directories."""
        # Create regular directory (not git)
        non_git_dir = self.temp_dir / "non_git"
        non_git_dir.mkdir()
        
        test_file = non_git_dir / "test.py"
        test_file.write_text("def hello(): pass")
        
        # Should still detect changes using filesystem approach
        import asyncio
        modified, new, deleted = asyncio.run(
            self.update_service.detect_changes(str(non_git_dir))
        )
        
        # All files should be considered new in non-git directory with no previous index
        assert len(new) >= 1  # At least our test file
        assert str(test_file) in new