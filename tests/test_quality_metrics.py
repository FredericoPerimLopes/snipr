import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from src.services.quality_metrics import (
    SearchQualityMetrics,
    CodeRetrievalMetrics,
    PerformanceBenchmark,
    SearchQualityDashboard,
    BenchmarkQuery,
    SearchQualityResult
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
            parameter_types=["str", "str"],
            return_type="bool"
        ),
        CodeChunk(
            file_path="/test/auth.py",
            content="class UserManager:\n    def __init__(self):\n        self.users = {}",
            start_line=5,
            end_line=7,
            language="python",
            semantic_type="class_definition",
            class_name="UserManager"
        ),
        CodeChunk(
            file_path="/test/database.py",
            content="async def fetch_data(query):\n    return await db.execute(query)",
            start_line=10,
            end_line=11,
            language="python",
            semantic_type="async_function_definition",
            function_name="fetch_data"
        )
    ]


@pytest.fixture
def benchmark_query():
    return BenchmarkQuery(
        query="user authentication logic",
        expected_functions=["authenticate_user", "verify_credentials", "login"],
        expected_classes=["UserManager", "AuthService"],
        expected_files=["auth.py", "authentication.py"],
        language="python",
        query_type="function"
    )


class TestSearchQualityMetrics:
    
    def test_load_benchmark_queries(self):
        metrics = SearchQualityMetrics()
        queries = metrics.benchmark_queries
        
        assert len(queries) >= 5
        assert all(isinstance(q, BenchmarkQuery) for q in queries)
        assert any(q.query == "user authentication logic" for q in queries)
    
    @pytest.mark.asyncio
    async def test_evaluate_query_results(self, sample_chunks, benchmark_query):
        metrics = SearchQualityMetrics()
        
        result = await metrics._evaluate_query_results(
            benchmark_query, sample_chunks, query_time=50.0
        )
        
        assert isinstance(result, SearchQualityResult)
        assert result.query_time_ms == 50.0
        assert 0 <= result.precision_at_5 <= 1
        assert 0 <= result.recall_at_5 <= 1
        assert 0 <= result.ndcg_at_5 <= 1
    
    def test_precision_at_k(self, sample_chunks):
        metrics = SearchQualityMetrics()
        
        relevant_items = {"function:authenticate_user", "class:UserManager"}
        retrieved_items = {"function:authenticate_user", "function:fetch_data"}
        
        precision = metrics._precision_at_k(retrieved_items, relevant_items, 5, sample_chunks)
        
        # Should find matches in the sample chunks
        assert precision > 0
    
    def test_recall_at_k(self, sample_chunks):
        metrics = SearchQualityMetrics()
        
        relevant_items = {"function:authenticate_user", "class:UserManager"}
        retrieved_items = {"function:authenticate_user"}
        
        recall = metrics._recall_at_k(retrieved_items, relevant_items, 5, sample_chunks)
        
        # Should be 0.5 since we found 1 out of 2 relevant items
        assert 0 < recall <= 1
    
    def test_ndcg_calculation(self):
        metrics = SearchQualityMetrics()
        
        relevance_scores = [1.0, 0.5, 0.0, 1.0, 0.5]
        ndcg = metrics._ndcg_at_k(relevance_scores, 5)
        
        assert 0 <= ndcg <= 1
    
    def test_relevance_score_calculation(self):
        metrics = SearchQualityMetrics()
        
        relevant_set = {"function:authenticate_user", "class:UserManager"}
        
        # Exact match
        assert metrics._calculate_relevance_score("function:authenticate_user", relevant_set) == 1.0
        
        # Partial match
        assert metrics._calculate_relevance_score("function:authenticateuser", relevant_set) == 0.5
        
        # No match
        assert metrics._calculate_relevance_score("function:unrelated", relevant_set) == 0.0
    
    @pytest.mark.asyncio
    async def test_evaluate_search_method(self, sample_chunks):
        metrics = SearchQualityMetrics()
        
        # Mock search function
        async def mock_search(query, language):
            return sample_chunks
        
        # Test with single query
        test_queries = [BenchmarkQuery(
            query="test query",
            expected_functions=["authenticate_user"],
            expected_classes=["UserManager"],
            expected_files=["auth.py"],
            language="python",
            query_type="function"
        )]
        
        result = await metrics.evaluate_search_method(
            mock_search, "test_method", test_queries
        )
        
        assert result["method"] == "test_method"
        assert result["total_queries"] == 1
        assert result["successful_queries"] == 1
        assert "aggregate_metrics" in result
        assert "individual_results" in result


class TestCodeRetrievalMetrics:
    
    def test_semantic_relevance_score(self, sample_chunks):
        chunk = sample_chunks[0]  # authenticate_user function
        
        # High relevance query
        score = CodeRetrievalMetrics.semantic_relevance_score(
            "user authentication function", chunk
        )
        assert score > 0.5
        
        # Low relevance query
        score = CodeRetrievalMetrics.semantic_relevance_score(
            "database operations", chunk
        )
        assert score < 0.5
    
    def test_syntactic_completeness_score(self, sample_chunks):
        # Function chunk
        func_chunk = sample_chunks[0]
        score = CodeRetrievalMetrics.syntactic_completeness_score(func_chunk)
        assert score >= 0.5
        
        # Class chunk
        class_chunk = sample_chunks[1]
        score = CodeRetrievalMetrics.syntactic_completeness_score(class_chunk)
        assert score >= 0.5
    
    def test_cross_file_context_score(self, sample_chunks):
        chunk = sample_chunks[0]
        chunk.import_statements = ["import database", "from auth import validate"]
        chunk.dependencies = ["database", "auth"]
        
        score = CodeRetrievalMetrics.cross_file_context_score(chunk, "database operations")
        assert score > 0


class TestPerformanceBenchmark:
    
    @pytest.mark.asyncio
    async def test_benchmark_search_methods(self, sample_chunks):
        benchmark = PerformanceBenchmark()
        
        # Mock search services
        async def fast_search(query):
            await asyncio.sleep(0.01)  # 10ms
            return sample_chunks[:2]
        
        async def slow_search(query):
            await asyncio.sleep(0.05)  # 50ms
            return sample_chunks
        
        search_services = {
            "fast_method": fast_search,
            "slow_method": slow_search
        }
        
        results = await benchmark.benchmark_search_methods(
            search_services, ["test query"]
        )
        
        assert "fast_method" in results
        assert "slow_method" in results
        assert results["fast_method"]["avg_response_time_ms"] < results["slow_method"]["avg_response_time_ms"]
    
    def test_generate_test_chunks(self):
        benchmark = PerformanceBenchmark()
        
        chunks = benchmark._generate_test_chunks(5)
        
        assert len(chunks) == 5
        assert all(isinstance(chunk, CodeChunk) for chunk in chunks)
        assert all(chunk.language == "python" for chunk in chunks)
    
    def test_percentile_calculation(self):
        benchmark = PerformanceBenchmark()
        
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        p50 = benchmark._percentile(values, 50)
        p95 = benchmark._percentile(values, 95)
        
        assert p50 == 5
        assert p95 == 9


class TestSearchQualityDashboard:
    
    @pytest.mark.asyncio
    async def test_generate_quality_report(self, sample_chunks):
        dashboard = SearchQualityDashboard()
        
        # Mock search service
        mock_service = Mock()
        mock_service.search = AsyncMock(return_value=sample_chunks)
        
        search_services = {"test_method": mock_service}
        
        with patch('src.services.quality_metrics.SearchQualityMetrics.evaluate_search_method') as mock_eval:
            mock_eval.return_value = {
                "method": "test_method",
                "total_queries": 5,
                "successful_queries": 5,
                "aggregate_metrics": {"avg_precision_at_5": 0.8}
            }
            
            report = await dashboard.generate_quality_report(search_services)
            
            assert "timestamp" in report
            assert "search_methods" in report
            assert "performance_benchmarks" in report
            assert "test_method" in report["search_methods"]
    
    @pytest.mark.asyncio
    async def test_compare_search_methods(self, sample_chunks):
        dashboard = SearchQualityDashboard()
        
        # Mock baseline and enhanced services
        baseline_service = Mock()
        enhanced_service = Mock()
        
        with patch('src.services.quality_metrics.SearchQualityMetrics.evaluate_search_method') as mock_eval:
            mock_eval.side_effect = [
                {
                    "method": "baseline",
                    "aggregate_metrics": {"avg_precision_at_5": 0.6, "avg_recall_at_5": 0.5}
                },
                {
                    "method": "enhanced", 
                    "aggregate_metrics": {"avg_precision_at_5": 0.8, "avg_recall_at_5": 0.7}
                }
            ]
            
            comparison = await dashboard.compare_search_methods(
                baseline_service, enhanced_service, ["test query"]
            )
            
            assert "baseline_results" in comparison
            assert "enhanced_results" in comparison
            assert "improvements" in comparison
            
            # Check improvement calculation
            improvements = comparison["improvements"]
            if "avg_precision_at_5" in improvements:
                assert improvements["avg_precision_at_5"]["improvement_percent"] > 0


@pytest.mark.asyncio
async def test_integration_quality_metrics(sample_chunks):
    """Integration test for quality metrics with mock search service."""
    metrics = SearchQualityMetrics()
    
    # Mock search function that returns our sample chunks
    async def mock_search_function(query, language):
        return sample_chunks
    
    # Test evaluation
    result = await metrics.evaluate_search_method(
        mock_search_function, "mock_search"
    )
    
    assert result["method"] == "mock_search"
    assert result["total_queries"] > 0
    assert result["successful_queries"] >= 0
    assert "aggregate_metrics" in result
    assert "individual_results" in result


@pytest.mark.asyncio
async def test_quality_metrics_with_empty_results():
    """Test quality metrics with empty search results."""
    metrics = SearchQualityMetrics()
    
    async def empty_search_function(query, language):
        return []
    
    result = await metrics.evaluate_search_method(
        empty_search_function, "empty_search"
    )
    
    assert result["method"] == "empty_search"
    assert result["successful_queries"] >= 0
    # Should handle empty results gracefully


@pytest.mark.asyncio 
async def test_quality_metrics_error_handling():
    """Test quality metrics error handling."""
    metrics = SearchQualityMetrics()
    
    async def failing_search_function(query, language):
        raise Exception("Search failed")
    
    result = await metrics.evaluate_search_method(
        failing_search_function, "failing_search"
    )
    
    assert result["method"] == "failing_search"
    # Should handle search errors gracefully
    assert any("error" in r for r in result["individual_results"])