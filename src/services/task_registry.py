"""Global task registry for tracking async indexing operations."""

import asyncio
import logging

from ..models.indexing_models import IndexingTask, TaskStatus

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Registry for managing async indexing tasks."""

    def __init__(self):
        self._tasks: dict[str, IndexingTask] = {}
        self._background_tasks: dict[str, asyncio.Task] = {}

    def register_task(self, task: IndexingTask) -> str:
        """Register a new indexing task."""
        self._tasks[task.task_id] = task
        logger.info(f"Registered indexing task {task.task_id} for {task.codebase_path}")
        return task.task_id

    def get_task(self, task_id: str) -> IndexingTask | None:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def update_task(self, task: IndexingTask) -> None:
        """Update an existing task."""
        self._tasks[task.task_id] = task

    def get_active_task_for_codebase(self, codebase_path: str) -> IndexingTask | None:
        """Get active task for a specific codebase."""
        for task in self._tasks.values():
            if task.codebase_path == codebase_path and task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                return task
        return None

    def start_background_task(self, task_id: str, coroutine) -> None:
        """Start a background asyncio task."""
        background_task = asyncio.create_task(coroutine)
        self._background_tasks[task_id] = background_task

        # Add done callback to clean up completed tasks
        background_task.add_done_callback(lambda t: self._background_tasks.pop(task_id, None))

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        task = self.get_task(task_id)
        if not task:
            return False

        # Cancel background asyncio task if running
        if task_id in self._background_tasks:
            self._background_tasks[task_id].cancel()

        # Update task status
        task.status = TaskStatus.CANCELLED
        self.update_task(task)

        logger.info(f"Cancelled indexing task {task_id}")
        return True

    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> None:
        """Clean up old completed tasks."""
        from datetime import datetime, timedelta

        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        completed_tasks = [
            task_id
            for task_id, task in self._tasks.items()
            if (
                task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                and task.completed_at
                and task.completed_at < cutoff_time
            )
        ]

        for task_id in completed_tasks:
            del self._tasks[task_id]

        if completed_tasks:
            logger.info(f"Cleaned up {len(completed_tasks)} old completed tasks")


# Global task registry instance
task_registry = TaskRegistry()
