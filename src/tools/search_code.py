import json
import logging

from ..models.indexing_models import SearchRequest
from ..services.search_service import SearchService

logger = logging.getLogger(__name__)


# Global service instance
search_service = SearchService()


async def search_code(
    query: str, language: str | None = None, max_results: int = 10, similarity_threshold: float = 0.7
) -> str:
    """Search for semantically similar code chunks.

    Args:
        query: Natural language or code query to search for
        language: Filter results by programming language (optional)
        max_results: Maximum number of results to return (1-100)
        similarity_threshold: Minimum similarity score (0.0-1.0)

    Returns:
        JSON string with search results and metadata
    """
    try:
        # Validate and create search request
        request = SearchRequest(
            query=query,
            language=language,
            max_results=max(1, min(100, max_results)),  # Clamp to valid range
            similarity_threshold=max(0.0, min(1.0, similarity_threshold)),  # Clamp to valid range
        )

        # Perform semantic search
        logger.info(f"Searching for: '{query}' (language: {language or 'any'})")
        search_result = await search_service.search_code(request)

        # Prepare response with additional metadata
        response = {
            "status": "success",
            "search_result": search_result.model_dump(),
            "query": query,
            "filters": {
                "language": language,
                "max_results": request.max_results,
                "similarity_threshold": request.similarity_threshold,
            },
            "message": f"Found {search_result.total_matches} results in {search_result.query_time_ms:.1f}ms",
        }

        logger.info(f"Search completed: {search_result.total_matches} results in {search_result.query_time_ms:.1f}ms")

        return json.dumps(response)

    except Exception as e:
        logger.error(f"Error during code search: {e}")
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to search code: {e!s}",
                "error_type": type(e).__name__,
                "query": query,
            }
        )


async def search_by_type(semantic_type: str, language: str | None = None, max_results: int = 20) -> str:
    """Search for code chunks by semantic type (function, class, etc.).

    Args:
        semantic_type: Type of code construct (function_definition, class_definition, etc.)
        language: Filter results by programming language (optional)
        max_results: Maximum number of results to return

    Returns:
        JSON string with filtered results by semantic type
    """
    try:
        # Use keyword search for semantic type filtering
        # This is a simpler approach than embedding-based search for type filtering
        results = await search_service.search_by_keywords(query=semantic_type, language_filter=language)

        # Filter by exact semantic type match
        filtered_results = [result for result in results if result.semantic_type == semantic_type][:max_results]

        response = {
            "status": "success",
            "results": [chunk.model_dump() for chunk in filtered_results],
            "total_matches": len(filtered_results),
            "semantic_type": semantic_type,
            "language_filter": language,
            "message": f"Found {len(filtered_results)} {semantic_type} examples",
        }

        logger.info(f"Type search completed: {len(filtered_results)} {semantic_type} results")

        return json.dumps(response)

    except Exception as e:
        logger.error(f"Error searching by type: {e}")
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to search by type: {e!s}",
                "error_type": type(e).__name__,
                "semantic_type": semantic_type,
            }
        )


async def search_in_file(file_path: str, query: str, max_results: int = 5) -> str:
    """Search for code patterns within a specific file.

    Args:
        file_path: Absolute path to the file to search in
        query: Search query for semantic matching
        max_results: Maximum number of results to return

    Returns:
        JSON string with file-specific search results
    """
    try:
        # Perform general search
        search_request = SearchRequest(
            query=query,
            max_results=max_results * 3,  # Get more results to filter
        )

        search_result = await search_service.search_code(search_request)

        # Filter results to specific file
        file_results = [chunk for chunk in search_result.results if chunk.file_path == file_path][:max_results]

        response = {
            "status": "success",
            "results": [chunk.model_dump() for chunk in file_results],
            "total_matches": len(file_results),
            "file_path": file_path,
            "query": query,
            "message": f"Found {len(file_results)} matches in {file_path}",
        }

        logger.info(f"File search completed: {len(file_results)} results in {file_path}")

        return json.dumps(response)

    except Exception as e:
        logger.error(f"Error searching in file: {e}")
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to search in file: {e!s}",
                "error_type": type(e).__name__,
                "file_path": file_path,
            }
        )


async def get_search_stats() -> str:
    """Get statistics about the indexed codebase and search capabilities.

    Returns:
        JSON string with comprehensive indexing and search statistics
    """
    try:
        embeddings_stats = await search_service.get_embeddings_stats()

        response = {
            "status": "success",
            "stats": embeddings_stats,
            "capabilities": {
                "semantic_search": search_service.model is not None,
                "supported_languages": list(search_service.config.SUPPORTED_LANGUAGES),
                "quantization_enabled": search_service.config.ENABLE_QUANTIZATION,
                "max_file_size_mb": search_service.config.MAX_FILE_SIZE_MB,
            },
            "message": "Successfully retrieved search statistics",
        }

        return json.dumps(response)

    except Exception as e:
        logger.error(f"Error getting search stats: {e}")
        return json.dumps(
            {"status": "error", "message": f"Failed to get search stats: {e!s}", "error_type": type(e).__name__}
        )
