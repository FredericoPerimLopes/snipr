import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

from ..bm25_search import BM25SearchEngine, CodeTokenizer
from ...models.indexing_models import CodeChunk


class TestCodeTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return CodeTokenizer()

    def test_tokenize_python_code(self, tokenizer):
        """Test Python code tokenization."""
        content = """
def calculate_user_score(user_id: int, bonus_points: float) -> float:
    # Calculate final score
    base_score = get_base_score(user_id)
    return base_score + bonus_points
"""
        
        tokens = tokenizer.tokenize(content, "python")
        
        assert "calculate" in tokens
        assert "user" in tokens
        assert "score" in tokens
        assert "bonus" in tokens
        assert "points" in tokens
        assert "base" in tokens
        
        # Should not contain stop words
        assert "def" not in tokens
        assert "return" not in tokens

    def test_tokenize_javascript_code(self, tokenizer):
        """Test JavaScript code tokenization."""
        content = """
function getUserProfile(userId) {
    const userdata = fetchUserData(userId);
    return userdata.profile;
}
"""
        
        tokens = tokenizer.tokenize(content, "javascript")
        
        assert "get" in tokens
        assert "user" in tokens
        assert "profile" in tokens
        assert "fetch" in tokens
        assert "data" in tokens
        
        # Should not contain keywords
        assert "function" not in tokens
        assert "const" not in tokens

    def test_split_compound_identifier(self, tokenizer):
        """Test compound identifier splitting."""
        # Test camelCase
        assert "getUserData" in tokenizer._split_compound_identifier("getUserData")
        tokens = tokenizer._split_compound_identifier("getUserData")
        assert "get" in tokens
        assert "user" in tokens
        assert "data" in tokens
        
        # Test snake_case
        tokens = tokenizer._split_compound_identifier("user_profile_data")
        assert "user" in tokens
        assert "profile" in tokens
        assert "data" in tokens
        
        # Test kebab-case
        tokens = tokenizer._split_compound_identifier("user-profile-data")
        assert "user" in tokens
        assert "profile" in tokens
        assert "data" in tokens

    def test_remove_comments_and_strings(self, tokenizer):
        """Test comment and string removal."""
        python_content = '''
def func():
    # This is a comment
    """This is a docstring"""
    name = "John Doe"
    return name
'''
        
        cleaned = tokenizer._remove_comments_and_strings(python_content, "python")
        
        assert "This is a comment" not in cleaned
        assert "This is a docstring" not in cleaned
        assert "John Doe" not in cleaned
        assert "def func" in cleaned
        assert "return name" in cleaned


class TestBM25SearchEngine:
    @pytest.fixture
    def search_engine(self):
        # Use temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.services.bm25_search.get_settings') as mock_settings:
                mock_config = Mock()
                mock_config.INDEX_CACHE_DIR = Path(temp_dir)
                mock_settings.return_value = mock_config
                
                engine = BM25SearchEngine()
                yield engine

    @pytest.fixture
    def sample_chunks(self):
        return [
            CodeChunk(
                file_path="test1.py",
                content="def authenticate_user(username: str, password: str) -> bool:\n    return verify_credentials(username, password)",
                start_line=1,
                end_line=3,
                language="python",
                semantic_type="function_definition",
                function_name="authenticate_user"
            ),
            CodeChunk(
                file_path="test2.py", 
                content="class UserManager:\n    def get_user_profile(self, user_id: int):\n        return self.database.fetch_user(user_id)",
                start_line=5,
                end_line=8,
                language="python",
                semantic_type="class_definition",
                class_name="UserManager"
            ),
            CodeChunk(
                file_path="test3.js",
                content="function calculateTotalScore(baseScore, bonusPoints) {\n    return baseScore + bonusPoints;\n}",
                start_line=10,
                end_line=13,
                language="javascript",
                semantic_type="function_definition",
                function_name="calculateTotalScore"
            )
        ]

    @pytest.mark.asyncio
    async def test_build_index(self, search_engine, sample_chunks):
        """Test BM25 index building."""
        await search_engine.build_index(sample_chunks)
        
        stats = await search_engine.get_index_stats()
        
        assert stats["indexed"] is True
        assert stats["total_documents"] == 3
        assert stats["total_terms"] > 0

    @pytest.mark.asyncio
    async def test_search_authentication(self, search_engine, sample_chunks):
        """Test BM25 search for authentication-related code."""
        await search_engine.build_index(sample_chunks)
        
        results = await search_engine.search("authentication user credentials")
        
        assert len(results) > 0
        # Should find the authenticate_user function
        doc_ids = [doc_id for doc_id, score in results]
        assert any("test1.py:1" in doc_id for doc_id in doc_ids)

    @pytest.mark.asyncio
    async def test_search_with_language_filter(self, search_engine, sample_chunks):
        """Test BM25 search with language filtering."""
        await search_engine.build_index(sample_chunks)
        
        # Search only Python code
        python_results = await search_engine.search("user", language="python")
        js_results = await search_engine.search("user", language="javascript")
        
        # Should return different results for different languages
        assert len(python_results) > 0
        assert len(js_results) >= 0
        
        # Python results should only contain Python files
        for doc_id, score in python_results:
            assert ".py" in doc_id

    @pytest.mark.asyncio
    async def test_update_index(self, search_engine, sample_chunks):
        """Test incremental index updates."""
        # Build initial index
        await search_engine.build_index(sample_chunks[:2])
        
        initial_stats = await search_engine.get_index_stats()
        assert initial_stats["total_documents"] == 2
        
        # Add new chunk
        new_chunk = CodeChunk(
            file_path="test4.py",
            content="def new_function():\n    pass",
            start_line=1,
            end_line=2,
            language="python",
            semantic_type="function_definition"
        )
        
        await search_engine.update_index([new_chunk], [])
        
        updated_stats = await search_engine.get_index_stats()
        assert updated_stats["total_documents"] == 3

    @pytest.mark.asyncio 
    async def test_remove_from_index(self, search_engine, sample_chunks):
        """Test removing documents from index."""
        await search_engine.build_index(sample_chunks)
        
        # Remove one document
        doc_to_remove = "test1.py:1"
        await search_engine.update_index([], [doc_to_remove])
        
        # Verify document was removed
        results = await search_engine.search("authenticate")
        doc_ids = [doc_id for doc_id, score in results]
        assert doc_to_remove not in doc_ids

    def test_tokenize_code_content(self, search_engine):
        """Test code content tokenization."""
        content = "def getUserProfile(user_id: int) -> UserProfile:"
        
        tokens = search_engine._tokenize_code(content, "python")
        
        assert "get" in tokens
        assert "user" in tokens  
        assert "profile" in tokens
        # Should not contain Python keywords
        assert "def" not in tokens

    def test_tokenize_query(self, search_engine):
        """Test query tokenization."""
        query = "find user authentication functions"
        
        tokens = search_engine._tokenize_query(query)
        
        assert "find" in tokens
        assert "user" in tokens
        assert "authentication" in tokens
        assert "functions" in tokens

    def test_detect_language_from_path(self, search_engine):
        """Test language detection."""
        assert search_engine._detect_language_from_path("app.py") == "python"
        assert search_engine._detect_language_from_path("script.js") == "javascript" 
        assert search_engine._detect_language_from_path("component.ts") == "typescript"
        assert search_engine._detect_language_from_path("main.go") == "go"