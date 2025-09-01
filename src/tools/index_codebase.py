import json
import logging

from ..models.indexing_models import IndexingRequest
from ..services.indexing_service import IndexingService
from ..services.search_service import SearchService

logger = logging.getLogger(__name__)


# Global service instances
indexing_service = IndexingService()
search_service = SearchService()


async def index_codebase(
    codebase_path: str,
    languages: str | None = None,
    exclude_patterns: str | None = None
) -> str:
    """Index a codebase for semantic search.

    Args:
        codebase_path: Absolute path to the codebase root directory
        languages: Comma-separated list of languages to index (optional)
        exclude_patterns: Comma-separated list of glob patterns to exclude (optional)

    Returns:
        JSON string with indexing results and statistics
    """
    try:
        # Parse optional parameters
        languages_list = None
        if languages:
            languages_list = [lang.strip() for lang in languages.split(",")]

        exclude_list = None
        if exclude_patterns:
            exclude_list = [pattern.strip() for pattern in exclude_patterns.split(",")]

        # Create indexing request
        request = IndexingRequest(
            codebase_path=codebase_path,
            languages=languages_list,
            exclude_patterns=exclude_list
        )

        # Check if reindexing is needed
        needs_reindex = await indexing_service.needs_reindexing(codebase_path)

        if not needs_reindex:
            # Return current status if no reindexing needed
            status = await indexing_service.get_indexing_status(codebase_path)
            return json.dumps({
                "status": "already_indexed",
                "message": "Codebase is already up to date",
                "indexing_status": status.model_dump()
            })

        # Perform indexing
        logger.info(f"Starting indexing for {codebase_path}")
        indexing_result = await indexing_service.index_codebase(request)

        # Generate embeddings for indexed chunks
        # Get chunks from indexing service (we need to modify the service to return chunks)
        # For now, let's get them from database after indexing

        # Get final status
        final_status = await indexing_service.get_indexing_status(codebase_path)
        embeddings_stats = await search_service.get_embeddings_stats()

        response = {
            "status": "success",
            "indexing_result": indexing_result.model_dump(),
            "indexing_status": final_status.model_dump(),
            "embeddings_stats": embeddings_stats,
            "message": (
                f"Successfully indexed {indexing_result.indexed_files} files "
                f"with {indexing_result.total_chunks} code chunks"
            )
        }

        logger.info(
            f"Indexing completed: {indexing_result.total_chunks} chunks "
            f"in {indexing_result.processing_time_ms:.1f}ms"
        )

        return json.dumps(response)

    except Exception as e:
        logger.error(f"Error indexing codebase: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to index codebase: {e!s}",
            "error_type": type(e).__name__
        })


async def get_indexing_status(codebase_path: str) -> str:
    """Get current indexing status for a codebase.

    Args:
        codebase_path: Absolute path to the codebase root directory

    Returns:
        JSON string with current indexing status and statistics
    """
    try:
        status = await indexing_service.get_indexing_status(codebase_path)
        embeddings_stats = await search_service.get_embeddings_stats()

        response = {
            "status": "success",
            "indexing_status": status.model_dump(),
            "embeddings_stats": embeddings_stats,
            "message": "Successfully retrieved indexing status"
        }

        return json.dumps(response)

    except Exception as e:
        logger.error(f"Error getting indexing status: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to get indexing status: {e!s}",
            "error_type": type(e).__name__
        })


async def clear_index() -> str:
    """Clear all indexing data and start fresh.

    Returns:
        JSON string with clearing results
    """
    try:
        import shutil

        config = indexing_service.config

        # Remove index cache directory
        if config.INDEX_CACHE_DIR.exists():
            shutil.rmtree(config.INDEX_CACHE_DIR)

        # Recreate empty cache directory
        config.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Reinitialize vector database
        search_service._init_vector_db()

        response = {
            "status": "success",
            "message": "Successfully cleared all indexing data"
        }

        logger.info("Index cache cleared successfully")
        return json.dumps(response)

    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to clear index: {e!s}",
            "error_type": type(e).__name__
        })
