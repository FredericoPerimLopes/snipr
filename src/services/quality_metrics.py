import math
import logging
import time
import asyncio
from typing import Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from ..models.indexing_models import CodeChunk, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)


@dataclass
class SearchQualityResult:
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    mean_average_precision: float
    ndcg_at_5: float
    ndcg_at_10: float
    query_time_ms: float


@dataclass
class BenchmarkQuery:
    query: str
    expected_functions: list[str]
    expected_classes: list[str]
    expected_files: list[str]
    language: str
    query_type: str  # "function", "class", "general", "cross_file"


class SearchQualityMetrics:
    """Comprehensive search quality measurement and benchmarking system."""
    
    def __init__(self):
        self.benchmark_queries = self._load_benchmark_queries()

    def _load_benchmark_queries(self) -> list[BenchmarkQuery]:
        """Load predefined benchmark queries for evaluation."""
        return [
            BenchmarkQuery(
                query="user authentication logic",
                expected_functions=["authenticate_user", "verify_credentials", "login"],
                expected_classes=["UserManager", "AuthService", "LoginHandler"],
                expected_files=["auth.py", "authentication.py", "login.py"],
                language="python",
                query_type="function"
            ),
            BenchmarkQuery(
                query="async database operations",
                expected_functions=["fetch_data", "save_data", "query_db"],
                expected_classes=["DatabaseManager", "AsyncRepository"],
                expected_files=["database.py", "db.py", "repository.py"],
                language="python",
                query_type="function"
            ),
            BenchmarkQuery(
                query="React component state management",
                expected_functions=["useState", "useEffect", "setState"],
                expected_classes=["Component", "StateManager"],
                expected_files=["component.jsx", "state.js", "hooks.js"],
                language="javascript",
                query_type="class"
            ),
            BenchmarkQuery(
                query="error handling patterns",
                expected_functions=["handle_error", "catch_exception", "log_error"],
                expected_classes=["ErrorHandler", "Exception", "CustomError"],
                expected_files=["errors.py", "exceptions.py", "error_handler.py"],
                language="python",
                query_type="general"
            ),
            BenchmarkQuery(
                query="configuration management",
                expected_functions=["load_config", "get_setting", "parse_config"],
                expected_classes=["Config", "Settings", "ConfigManager"],
                expected_files=["config.py", "settings.py", "configuration.py"],
                language="python",
                query_type="general"
            )
        ]

    async def evaluate_search_method(
        self, 
        search_function,
        method_name: str,
        queries: list[BenchmarkQuery] = None
    ) -> dict[str, Any]:
        """Evaluate a search method against benchmark queries."""
        if queries is None:
            queries = self.benchmark_queries
            
        results = []
        total_time = 0
        
        for query in queries:
            start_time = time.time()
            
            try:
                # Execute search
                search_results = await search_function(query.query, query.language)
                
                query_time = (time.time() - start_time) * 1000
                total_time += query_time
                
                # Evaluate results
                quality_result = await self._evaluate_query_results(
                    query, search_results, query_time
                )
                
                results.append({
                    "query": query.query,
                    "language": query.language,
                    "query_type": query.query_type,
                    "metrics": asdict(quality_result)
                })
                
            except Exception as e:
                logger.error(f"Error evaluating query '{query.query}': {e}")
                results.append({
                    "query": query.query,
                    "error": str(e),
                    "metrics": None
                })
        
        # Calculate aggregate metrics
        valid_results = [r for r in results if r.get("metrics")]
        
        if not valid_results:
            return {
                "method": method_name,
                "total_queries": len(queries),
                "successful_queries": 0,
                "aggregate_metrics": None,
                "individual_results": results
            }
        
        aggregate_metrics = self._calculate_aggregate_metrics(valid_results)
        
        return {
            "method": method_name,
            "total_queries": len(queries),
            "successful_queries": len(valid_results),
            "average_query_time_ms": total_time / len(queries),
            "aggregate_metrics": aggregate_metrics,
            "individual_results": results
        }

    async def _evaluate_query_results(
        self, 
        query: BenchmarkQuery, 
        results: list[CodeChunk], 
        query_time: float
    ) -> SearchQualityResult:
        """Evaluate search results against expected outcomes."""
        # Create relevance sets
        relevant_items = set()
        
        # Add expected functions
        for func in query.expected_functions:
            relevant_items.add(f"function:{func}")
            
        # Add expected classes  
        for cls in query.expected_classes:
            relevant_items.add(f"class:{cls}")
            
        # Add expected files
        for file in query.expected_files:
            relevant_items.add(f"file:{file}")
        
        # Extract retrieved items
        retrieved_items = set()
        relevance_scores = []
        
        for i, chunk in enumerate(results):
            # Check function relevance
            if chunk.function_name:
                retrieved_items.add(f"function:{chunk.function_name}")
                relevance_scores.append(self._calculate_relevance_score(
                    f"function:{chunk.function_name}", relevant_items
                ))
            
            # Check class relevance
            if chunk.class_name:
                retrieved_items.add(f"class:{chunk.class_name}")
                relevance_scores.append(self._calculate_relevance_score(
                    f"class:{chunk.class_name}", relevant_items
                ))
            
            # Check file relevance
            file_name = Path(chunk.file_path).name
            retrieved_items.add(f"file:{file_name}")
            relevance_scores.append(self._calculate_relevance_score(
                f"file:{file_name}", relevant_items
            ))
        
        # Calculate metrics
        precision_at_5 = self._precision_at_k(retrieved_items, relevant_items, 5, results)
        precision_at_10 = self._precision_at_k(retrieved_items, relevant_items, 10, results)
        recall_at_5 = self._recall_at_k(retrieved_items, relevant_items, 5, results)
        recall_at_10 = self._recall_at_k(retrieved_items, relevant_items, 10, results)
        
        map_score = self._mean_average_precision([query], [results], [relevant_items])
        ndcg_5 = self._ndcg_at_k(relevance_scores[:5], 5)
        ndcg_10 = self._ndcg_at_k(relevance_scores[:10], 10)
        
        return SearchQualityResult(
            precision_at_5=precision_at_5,
            precision_at_10=precision_at_10,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            mean_average_precision=map_score,
            ndcg_at_5=ndcg_5,
            ndcg_at_10=ndcg_10,
            query_time_ms=query_time
        )

    def _precision_at_k(self, retrieved: set[str], relevant: set[str], k: int, results: list[CodeChunk]) -> float:
        """Calculate Precision@K."""
        if not results or k == 0:
            return 0.0
            
        # Get top-k retrieved items
        top_k_items = set()
        for i, chunk in enumerate(results[:k]):
            if chunk.function_name:
                top_k_items.add(f"function:{chunk.function_name}")
            if chunk.class_name:
                top_k_items.add(f"class:{chunk.class_name}")
            file_name = Path(chunk.file_path).name
            top_k_items.add(f"file:{file_name}")
        
        relevant_retrieved = top_k_items.intersection(relevant)
        return len(relevant_retrieved) / min(k, len(results)) if results else 0.0

    def _recall_at_k(self, retrieved: set[str], relevant: set[str], k: int, results: list[CodeChunk]) -> float:
        """Calculate Recall@K."""
        if not relevant:
            return 0.0
            
        # Get top-k retrieved items
        top_k_items = set()
        for i, chunk in enumerate(results[:k]):
            if chunk.function_name:
                top_k_items.add(f"function:{chunk.function_name}")
            if chunk.class_name:
                top_k_items.add(f"class:{chunk.class_name}")
            file_name = Path(chunk.file_path).name
            top_k_items.add(f"file:{file_name}")
        
        relevant_retrieved = top_k_items.intersection(relevant)
        return len(relevant_retrieved) / len(relevant)

    def _mean_average_precision(
        self, 
        queries: list[BenchmarkQuery], 
        results_list: list[list[CodeChunk]], 
        relevant_sets: list[set[str]]
    ) -> float:
        """Calculate Mean Average Precision across queries."""
        if not queries:
            return 0.0
            
        average_precisions = []
        
        for results, relevant in zip(results_list, relevant_sets):
            if not relevant:
                continue
                
            precision_sum = 0.0
            relevant_count = 0
            
            for i, chunk in enumerate(results):
                # Check if current item is relevant
                chunk_items = set()
                if chunk.function_name:
                    chunk_items.add(f"function:{chunk.function_name}")
                if chunk.class_name:
                    chunk_items.add(f"class:{chunk.class_name}")
                file_name = Path(chunk.file_path).name
                chunk_items.add(f"file:{file_name}")
                
                if chunk_items.intersection(relevant):
                    relevant_count += 1
                    precision_at_i = relevant_count / (i + 1)
                    precision_sum += precision_at_i
            
            if relevant_count > 0:
                average_precision = precision_sum / len(relevant)
                average_precisions.append(average_precision)
        
        return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0

    def _ndcg_at_k(self, relevance_scores: list[float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""
        if not relevance_scores or k == 0:
            return 0.0
            
        # Calculate DCG@K
        dcg = 0.0
        for i, score in enumerate(relevance_scores[:k]):
            if i == 0:
                dcg += score
            else:
                dcg += score / math.log2(i + 1)
        
        # Calculate IDCG@K (ideal DCG)
        ideal_scores = sorted(relevance_scores[:k], reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            if i == 0:
                idcg += score
            else:
                idcg += score / math.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_relevance_score(self, item: str, relevant_set: set[str]) -> float:
        """Calculate relevance score for an item."""
        if item in relevant_set:
            return 1.0
            
        # Partial matching for similar terms
        item_clean = item.lower().replace("_", "").replace("-", "")
        for relevant_item in relevant_set:
            relevant_clean = relevant_item.lower().replace("_", "").replace("-", "")
            if item_clean in relevant_clean or relevant_clean in item_clean:
                return 0.5
                
        return 0.0

    def _calculate_aggregate_metrics(self, results: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate aggregate metrics across all queries."""
        metrics = {}
        metric_names = [
            "precision_at_5", "precision_at_10", "recall_at_5", "recall_at_10",
            "mean_average_precision", "ndcg_at_5", "ndcg_at_10", "query_time_ms"
        ]
        
        for metric_name in metric_names:
            values = [r["metrics"][metric_name] for r in results if r["metrics"]]
            metrics[f"avg_{metric_name}"] = sum(values) / len(values) if values else 0.0
            metrics[f"min_{metric_name}"] = min(values) if values else 0.0
            metrics[f"max_{metric_name}"] = max(values) if values else 0.0
        
        return metrics


class CodeRetrievalMetrics:
    """Code-specific quality metrics for retrieval systems."""

    @staticmethod
    def semantic_relevance_score(query: str, chunk: CodeChunk) -> float:
        """Measure semantic alignment between query and code chunk."""
        query_lower = query.lower()
        content_lower = chunk.content.lower()
        
        # Basic keyword overlap scoring
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        overlap = query_words.intersection(content_words)
        if not query_words:
            return 0.0
            
        keyword_score = len(overlap) / len(query_words)
        
        # Boost score for metadata matches
        metadata_boost = 0.0
        if chunk.function_name and any(word in chunk.function_name.lower() for word in query_words):
            metadata_boost += 0.3
        if chunk.class_name and any(word in chunk.class_name.lower() for word in query_words):
            metadata_boost += 0.3
        if chunk.docstring and any(word in chunk.docstring.lower() for word in query_words):
            metadata_boost += 0.2
            
        return min(1.0, keyword_score + metadata_boost)

    @staticmethod
    def syntactic_completeness_score(chunk: CodeChunk) -> float:
        """Measure if chunk contains complete syntactic units."""
        content = chunk.content.strip()
        
        # Check for complete function definitions
        if chunk.semantic_type in ["function_definition", "async_function_definition"]:
            # Should have function signature and body
            has_signature = "def " in content or "function " in content or "fn " in content
            has_body = "{" in content or ":" in content
            return 1.0 if has_signature and has_body else 0.5
            
        # Check for complete class definitions
        elif chunk.semantic_type in ["class_definition", "class_declaration"]:
            has_class_keyword = "class " in content or "struct " in content
            has_body = "{" in content or ":" in content
            return 1.0 if has_class_keyword and has_body else 0.5
            
        # For other types, check for balanced braces/brackets
        else:
            open_chars = content.count("{") + content.count("(") + content.count("[")
            close_chars = content.count("}") + content.count(")") + content.count("]")
            
            if open_chars == 0 and close_chars == 0:
                return 1.0  # No braces needed
            elif open_chars == close_chars:
                return 1.0  # Balanced
            else:
                return 0.7  # Potentially incomplete

    @staticmethod
    def cross_file_context_score(chunk: CodeChunk, query: str) -> float:
        """Measure repository-level context relevance."""
        score = 0.0
        
        # Import relevance
        if chunk.import_statements:
            query_words = set(query.lower().split())
            for import_stmt in chunk.import_statements:
                import_words = set(import_stmt.lower().split())
                if query_words.intersection(import_words):
                    score += 0.2
        
        # Dependency relevance
        if chunk.dependencies:
            query_words = set(query.lower().split())
            for dep in chunk.dependencies:
                if any(word in dep.lower() for word in query_words):
                    score += 0.15
        
        # File path relevance
        file_name = Path(chunk.file_path).stem.lower()
        query_words = set(query.lower().split())
        if any(word in file_name for word in query_words):
            score += 0.3
            
        return min(1.0, score)


class PerformanceBenchmark:
    """Performance benchmarking for search operations."""
    
    def __init__(self):
        self.results_history = []

    async def benchmark_search_methods(
        self, 
        search_services: Dict[str, Any],
        test_queries: list[str] = None
    ) -> dict[str, Any]:
        """Compare performance across different search methods."""
        if test_queries is None:
            test_queries = [
                "user authentication",
                "database operations", 
                "error handling",
                "async functions",
                "class inheritance"
            ]
        
        benchmark_results = {}
        
        for method_name, search_service in search_services.items():
            logger.info(f"Benchmarking {method_name}...")
            
            method_results = {
                "total_queries": len(test_queries),
                "response_times": [],
                "success_rate": 0.0,
                "avg_results_per_query": 0.0
            }
            
            successful_queries = 0
            total_results = 0
            
            for query in test_queries:
                start_time = time.time()
                
                try:
                    if hasattr(search_service, 'search'):
                        request = SearchRequest(query=query, max_results=10)
                        response = await search_service.search(request)
                        results = response.results if hasattr(response, 'results') else response
                    else:
                        results = await search_service(query)
                    
                    query_time = (time.time() - start_time) * 1000
                    method_results["response_times"].append(query_time)
                    
                    if results:
                        successful_queries += 1
                        total_results += len(results)
                        
                except Exception as e:
                    logger.warning(f"Query '{query}' failed for {method_name}: {e}")
                    method_results["response_times"].append(0.0)
            
            # Calculate aggregate stats
            response_times = [t for t in method_results["response_times"] if t > 0]
            method_results.update({
                "success_rate": successful_queries / len(test_queries),
                "avg_results_per_query": total_results / successful_queries if successful_queries > 0 else 0,
                "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
                "p95_response_time_ms": self._percentile(response_times, 95) if response_times else 0,
                "p99_response_time_ms": self._percentile(response_times, 99) if response_times else 0
            })
            
            benchmark_results[method_name] = method_results
        
        return benchmark_results

    async def benchmark_scalability(
        self, 
        search_service, 
        chunk_counts: list[int] = None
    ) -> dict[str, Any]:
        """Test search performance with different index sizes."""
        if chunk_counts is None:
            chunk_counts = [100, 1000, 5000, 10000]
            
        scalability_results = {}
        
        for count in chunk_counts:
            logger.info(f"Testing scalability with {count} chunks...")
            
            # Generate test chunks
            test_chunks = self._generate_test_chunks(count)
            
            # Measure indexing time
            index_start = time.time()
            await search_service.embed_code_chunks(test_chunks)
            index_time = (time.time() - index_start) * 1000
            
            # Measure search time
            test_queries = ["test function", "example class", "sample code"]
            search_times = []
            
            for query in test_queries:
                search_start = time.time()
                try:
                    request = SearchRequest(query=query, max_results=10)
                    await search_service.search_code(request)
                    search_time = (time.time() - search_start) * 1000
                    search_times.append(search_time)
                except Exception as e:
                    logger.warning(f"Search failed for {count} chunks: {e}")
            
            scalability_results[count] = {
                "indexing_time_ms": index_time,
                "avg_search_time_ms": sum(search_times) / len(search_times) if search_times else 0,
                "chunks_per_second": count / (index_time / 1000) if index_time > 0 else 0
            }
        
        return scalability_results

    def _generate_test_chunks(self, count: int) -> list[CodeChunk]:
        """Generate test chunks for scalability testing."""
        chunks = []
        
        for i in range(count):
            chunk = CodeChunk(
                file_path=f"test_{i // 100}.py",
                content=f"def test_function_{i}():\n    return {i}",
                start_line=i * 3,
                end_line=i * 3 + 2,
                language="python",
                semantic_type="function_definition",
                function_name=f"test_function_{i}"
            )
            chunks.append(chunk)
            
        return chunks

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
            
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]


class SearchQualityDashboard:
    """Dashboard for monitoring search quality metrics."""
    
    def __init__(self):
        self.metrics_history = []

    async def generate_quality_report(
        self, 
        search_services: Dict[str, Any],
        output_path: str = None
    ) -> dict[str, Any]:
        """Generate comprehensive quality report."""
        quality_metrics = SearchQualityMetrics()
        performance_benchmark = PerformanceBenchmark()
        code_metrics = CodeRetrievalMetrics()
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "search_methods": {},
            "performance_benchmarks": {},
            "code_quality_analysis": {}
        }
        
        # Evaluate each search method
        for method_name, service in search_services.items():
            logger.info(f"Evaluating {method_name} search quality...")
            
            evaluation = await quality_metrics.evaluate_search_method(
                service, method_name
            )
            report["search_methods"][method_name] = evaluation
        
        # Performance benchmarking
        report["performance_benchmarks"] = await performance_benchmark.benchmark_search_methods(
            search_services
        )
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report

    async def compare_search_methods(
        self, 
        baseline_service,
        enhanced_service,
        test_queries: list[str] = None
    ) -> dict[str, Any]:
        """Compare baseline vs enhanced search performance."""
        if test_queries is None:
            test_queries = [
                "authentication functions",
                "database connection", 
                "error handling classes",
                "async operations"
            ]
        
        comparison = {
            "baseline_results": [],
            "enhanced_results": [], 
            "improvements": {}
        }
        
        quality_metrics = SearchQualityMetrics()
        
        # Test baseline
        baseline_eval = await quality_metrics.evaluate_search_method(
            baseline_service, "baseline"
        )
        comparison["baseline_results"] = baseline_eval
        
        # Test enhanced
        enhanced_eval = await quality_metrics.evaluate_search_method(
            enhanced_service, "enhanced"
        )
        comparison["enhanced_results"] = enhanced_eval
        
        # Calculate improvements
        if (baseline_eval.get("aggregate_metrics") and 
            enhanced_eval.get("aggregate_metrics")):
            
            baseline_metrics = baseline_eval["aggregate_metrics"]
            enhanced_metrics = enhanced_eval["aggregate_metrics"]
            
            for metric in baseline_metrics:
                if metric.startswith("avg_"):
                    baseline_val = baseline_metrics[metric]
                    enhanced_val = enhanced_metrics.get(metric, 0)
                    
                    if baseline_val > 0:
                        improvement = ((enhanced_val - baseline_val) / baseline_val) * 100
                        comparison["improvements"][metric] = {
                            "baseline": baseline_val,
                            "enhanced": enhanced_val,
                            "improvement_percent": improvement
                        }
        
        return comparison