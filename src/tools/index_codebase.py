import asyncio
import json
import logging
from datetime import datetime

from ..models.indexing_models import IndexingRequest, IndexingTask, TaskStatus
from ..services.indexing_service import IndexingService
from ..services.logging_service import create_indexing_logger
from ..services.search_service import SearchService
from ..services.task_registry import task_registry

logger = logging.getLogger(__name__)


# Global service instances
indexing_service = IndexingService()
search_service = SearchService()


async def index_codebase(codebase_path: str, languages: str | None = None, exclude_patterns: str | None = None) -> str:
    """Index a codebase for semantic search asynchronously.

    Args:
        codebase_path: Absolute path to the codebase root directory
        languages: Comma-separated list of languages to index (optional)
        exclude_patterns: Comma-separated list of glob patterns to exclude (optional)

    Returns:
        JSON string with task ID for tracking progress
    """
    try:
        # Parse optional parameters
        languages_list = None
        if languages:
            languages_list = [lang.strip() for lang in languages.split(",")]

        exclude_list = None
        if exclude_patterns:
            exclude_list = [pattern.strip() for pattern in exclude_patterns.split(",")]

        # Check if there's already an active task for this codebase
        existing_task = task_registry.get_active_task_for_codebase(codebase_path)
        if existing_task:
            return json.dumps(
                {
                    "status": "already_running",
                    "task_id": existing_task.task_id,
                    "message": f"Indexing already in progress for {codebase_path}",
                    "progress": existing_task.progress.model_dump(mode="json"),
                }
            )

        # Check if reindexing is needed
        needs_reindex = await indexing_service.needs_reindexing(codebase_path)

        if not needs_reindex:
            # Return current status if no reindexing needed
            status = await indexing_service.get_indexing_status(codebase_path)
            return json.dumps(
                {
                    "status": "already_indexed",
                    "message": "Codebase is already up to date",
                    "indexing_status": status.model_dump(mode="json"),
                }
            )

        # Create indexing task
        task = IndexingTask(codebase_path=codebase_path)
        task_id = task_registry.register_task(task)

        # Create indexing request
        request = IndexingRequest(codebase_path=codebase_path, languages=languages_list, exclude_patterns=exclude_list)

        # Start background indexing
        task_registry.start_background_task(task_id, _run_indexing_task(task_id, request))

        logger.info(f"Started background indexing task {task_id} for {codebase_path}")

        return json.dumps(
            {
                "status": "started",
                "task_id": task_id,
                "message": f"Indexing started for {codebase_path}. Use get_indexing_status to check progress.",
                "codebase_path": codebase_path,
            }
        )

    except Exception as e:
        logger.error(f"Error starting indexing task: {e}")
        return json.dumps(
            {"status": "error", "message": f"Failed to start indexing: {e!s}", "error_type": type(e).__name__}
        )


async def _run_indexing_task(task_id: str, request: IndexingRequest) -> None:
    """Run the actual indexing task in the background."""
    task = task_registry.get_task(task_id)
    if not task:
        logger.error(f"Task {task_id} not found in registry")
        return

    # Initialize indexing logger for this background task
    indexing_logger = create_indexing_logger(f"bg_{task_id}")

    try:
        # Mark task as running
        task.status = TaskStatus.RUNNING
        task_registry.update_task(task)

        indexing_logger.log_info(f"Background indexing task {task_id} started", codebase_path=request.codebase_path)
        logger.info(f"Starting background indexing for task {task_id}")

        # Perform indexing with progress tracking
        indexing_result = await indexing_service.index_codebase_with_progress(request, task)

        # Mark task as completed
        task.status = TaskStatus.COMPLETED
        task.result = indexing_result
        task.completed_at = datetime.now()
        task_registry.update_task(task)

        indexing_logger.log_info(
            f"Background indexing task {task_id} completed successfully",
            indexed_files=indexing_result.indexed_files,
            total_chunks=indexing_result.total_chunks,
            processing_time_ms=indexing_result.processing_time_ms,
        )
        logger.info(f"Background indexing task {task_id} completed successfully")

    except asyncio.CancelledError:
        # Handle task cancellation
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        task_registry.update_task(task)
        indexing_logger.log_warning(f"Indexing task {task_id} was cancelled")
        logger.info(f"Indexing task {task_id} was cancelled")

    except Exception as e:
        # Handle task failure
        task.status = TaskStatus.FAILED
        task.error_message = str(e)
        task.completed_at = datetime.now()
        task_registry.update_task(task)
        indexing_logger.log_error(f"Background indexing task {task_id} failed", error=e)
        logger.error(f"Background indexing task {task_id} failed: {e}")

    finally:
        indexing_logger.close()


async def get_indexing_status(codebase_path: str) -> str:
    """Get current indexing status for a codebase, including active background tasks.

    Args:
        codebase_path: Absolute path to the codebase root directory

    Returns:
        JSON string with current indexing status, statistics, and task progress
    """
    try:
        # Check for active task
        active_task = task_registry.get_active_task_for_codebase(codebase_path)

        # Get base indexing status
        base_status = await indexing_service.get_indexing_status(codebase_path)
        embeddings_stats = await search_service.get_embeddings_stats()

        # Create response with base status (don't modify the base_status object)
        response = {
            "status": "success",
            "indexing_status": base_status.model_dump(mode="json"),
            "embeddings_stats": embeddings_stats,
            "message": "Successfully retrieved indexing status",
        }

        # Add task-specific information
        if active_task:
            response["active_task"] = {
                "task_id": active_task.task_id,
                "status": active_task.status.value,
                "progress": active_task.progress.model_dump(mode="json"),
                "created_at": active_task.created_at.isoformat(),
            }

            if active_task.status == TaskStatus.RUNNING:
                response["message"] = f"Indexing in progress: {active_task.progress.progress_percentage:.1f}% complete"
            elif active_task.status == TaskStatus.COMPLETED:
                response["message"] = "Indexing completed successfully"
            elif active_task.status == TaskStatus.FAILED:
                response["message"] = f"Indexing failed: {active_task.error_message}"

        return json.dumps(response)

    except Exception as e:
        logger.error(f"Error getting indexing status: {e}")
        return json.dumps(
            {"status": "error", "message": f"Failed to get indexing status: {e!s}", "error_type": type(e).__name__}
        )


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

        response = {"status": "success", "message": "Successfully cleared all indexing data"}

        logger.info("Index cache cleared successfully")
        return json.dumps(response)

    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        return json.dumps(
            {"status": "error", "message": f"Failed to clear index: {e!s}", "error_type": type(e).__name__}
        )


async def cancel_indexing_task(task_id: str) -> str:
    """Cancel an active indexing task.

    Args:
        task_id: ID of the task to cancel

    Returns:
        JSON string with cancellation result
    """
    try:
        success = task_registry.cancel_task(task_id)

        if success:
            response = {
                "status": "success",
                "message": f"Indexing task {task_id} has been cancelled",
                "task_id": task_id,
            }
        else:
            response = {
                "status": "not_found",
                "message": f"Task {task_id} not found or already completed",
                "task_id": task_id,
            }

        return json.dumps(response)

    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        return json.dumps(
            {"status": "error", "message": f"Failed to cancel task: {e!s}", "error_type": type(e).__name__}
        )
