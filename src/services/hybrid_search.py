import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass

from ..models.indexing_models import CodeChunk, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    bm25_weight: float = 0.4
    semantic_weight: float = 0.4
    metadata_weight: float = 0.2
    rrf_k_parameter: int = 60
    enable_query_expansion: bool = True


class HybridSearchService:
    def __init__(self, search_service, metadata_search_service=None):
        self.search_service = search_service
        self.metadata_search_service = metadata_search_service
        self.config = HybridSearchConfig()

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute hybrid search using RRF to combine multiple search methods."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Run searches in parallel
            search_tasks = [
                self._run_bm25_search(request),
                self._run_semantic_search(request),
            ]

            # Add metadata search if available and query suggests it
            if self.metadata_search_service and self._should_use_metadata_search(request.query):
                search_tasks.append(self._run_metadata_search(request))

            # Execute all searches in parallel
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Filter out exceptions and empty results
            valid_results = []
            for result in search_results:
                if isinstance(result, Exception):
                    logger.warning(f"Search method failed: {result}")
                elif result:
                    valid_results.append(result)

            if not valid_results:
                return SearchResponse(results=[], total_matches=0, query_time_ms=0.0)

            # Apply Reciprocal Rank Fusion
            combined_results = self._reciprocal_rank_fusion(valid_results)

            # Apply final filtering and ranking
            final_results = await self._post_process_results(combined_results, request)

            query_time = (asyncio.get_event_loop().time() - start_time) * 1000

            return SearchResponse(
                results=final_results[: request.max_results],
                total_matches=len(combined_results),
                query_time_ms=round(query_time, 2),
            )

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return SearchResponse(results=[], total_matches=0, query_time_ms=0.0)

    def _reciprocal_rank_fusion(self, results_list: list[list[CodeChunk]], k: int = None) -> list[CodeChunk]:
        """Combine multiple ranked result lists using Reciprocal Rank Fusion."""
        if k is None:
            k = self.config.rrf_k_parameter

        score_dict = {}

        for results in results_list:
            for rank, chunk in enumerate(results):
                chunk_id = f"{chunk.file_path}:{chunk.start_line}"
                if chunk_id not in score_dict:
                    score_dict[chunk_id] = {"chunk": chunk, "score": 0.0}
                score_dict[chunk_id]["score"] += 1.0 / (k + rank + 1)

        # Sort by combined score
        sorted_results = sorted(score_dict.values(), key=lambda x: x["score"], reverse=True)
        return [item["chunk"] for item in sorted_results]

    async def _run_bm25_search(self, request: SearchRequest) -> list[CodeChunk]:
        """Run BM25 lexical search."""
        try:
            results = await self.search_service.search_by_bm25(request.query, request.language, max_results=50)
            logger.debug(f"BM25 search returned {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    async def _run_semantic_search(self, request: SearchRequest) -> list[CodeChunk]:
        """Run semantic embedding search."""
        try:
            response = await self.search_service.search_code(request)
            logger.debug(f"Semantic search returned {len(response.results)} results")
            return response.results
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    async def _run_metadata_search(self, request: SearchRequest) -> list[CodeChunk]:
        """Run metadata-based search."""
        if not self.metadata_search_service:
            return []

        try:
            results = await self.metadata_search_service.search(request.query, request.language)
            logger.debug(f"Metadata search returned {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"Metadata search failed: {e}")
            return []

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query for intelligent routing."""
        query_lower = query.lower()

        # Code-specific patterns - check more specific patterns first
        if any(pattern in query_lower for pattern in ["return type", "parameter type", "function type"]):
            return "type_search"
        elif any(pattern in query_lower for pattern in ["import", "require", "include"]):
            return "import_search"
        elif any(pattern in query_lower for pattern in ["class", "interface", "struct"]):
            return "class_search"
        elif any(pattern in query_lower for pattern in ["function", "method", "def ", "async def"]):
            return "function_search"
        elif any(pattern in query_lower for pattern in ["type", "parameter"]):
            return "type_search"
        else:
            return "general_search"

    def _should_use_metadata_search(self, query: str) -> bool:
        """Determine if metadata search should be included."""
        query_lower = query.lower()

        # Use metadata search for specific query types
        metadata_indicators = [
            "function",
            "method",
            "class",
            "interface",
            "type",
            "return",
            "parameter",
            "inherit",
            "extend",
            "implement",
            "decorator",
            "complexity",
            "signature",
        ]

        return any(indicator in query_lower for indicator in metadata_indicators)

    async def _post_process_results(self, results: list[CodeChunk], request: SearchRequest) -> list[CodeChunk]:
        """Post-process results with additional ranking and filtering."""
        # Apply diversity filtering to avoid too many results from same file
        diverse_results = self._apply_diversity_filter(results)

        # Apply contextual scoring boost
        contextual_results = await self._apply_contextual_scoring(diverse_results, request.query)

        return contextual_results

    def _apply_diversity_filter(self, results: list[CodeChunk], max_per_file: int = 3) -> list[CodeChunk]:
        """Ensure diversity by limiting results per file."""
        file_counts = defaultdict(int)
        diverse_results = []

        for chunk in results:
            if file_counts[chunk.file_path] < max_per_file:
                diverse_results.append(chunk)
                file_counts[chunk.file_path] += 1

        return diverse_results

    async def _apply_contextual_scoring(self, results: list[CodeChunk], query: str) -> list[CodeChunk]:
        """Apply contextual scoring based on relationships and relevance."""
        # For now, return as-is - this could be enhanced with:
        # - File proximity scoring based on import relationships
        # - Function call relationship scoring
        # - Module clustering similarity
        return results

    async def expand_query(self, query: str, context_file: str = None) -> list[str]:
        """Expand query with related terms based on context."""
        if not self.config.enable_query_expansion:
            return [query]

        expanded_queries = [query]

        # Basic query expansion based on common programming concepts
        query_lower = query.lower()

        expansion_map = {
            "auth": ["authenticate", "login", "credential", "token", "session"],
            "user": ["account", "profile", "member", "person"],
            "data": ["information", "record", "entity", "model"],
            "save": ["store", "persist", "write", "update"],
            "get": ["fetch", "retrieve", "load", "read"],
            "delete": ["remove", "destroy", "drop"],
            "error": ["exception", "failure", "issue", "problem"],
            "config": ["configuration", "setting", "option", "parameter"],
        }

        for base_term, expansions in expansion_map.items():
            if base_term in query_lower:
                for expansion in expansions:
                    if expansion not in query_lower:
                        expanded_queries.append(f"{query} {expansion}")

        return expanded_queries[:5]  # Limit to 5 expanded queries
