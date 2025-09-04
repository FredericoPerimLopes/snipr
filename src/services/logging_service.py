"""
Enhanced logging service for indexing operations.
Provides detailed file-based logging with configurable levels and disable option.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import get_settings


class IndexingLogger:
    """Dedicated logger for indexing operations with file output and metrics tracking."""

    def __init__(self, session_id: str | None = None):
        self.config = get_settings()
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enabled = self.config.ENABLE_INDEXING_LOGS
        self.logger = None
        self.file_handler = None
        self.start_time = time.time()
        self.metrics = {
            "files_discovered": 0,
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "database_operations": 0,
            "total_processing_time": 0.0,
            "per_file_times": [],
        }

        if self.enabled:
            self._setup_file_logger()

    def _setup_file_logger(self) -> None:
        """Setup file-based logger for indexing operations."""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Create log file with timestamp
        log_file = log_dir / f"indexing_{self.session_id}.log"

        # Setup logger
        self.logger = logging.getLogger(f"indexing_session_{self.session_id}")
        self.logger.setLevel(getattr(logging, self.config.INDEXING_LOG_LEVEL))

        # Clear any existing handlers
        self.logger.handlers.clear()

        # File handler with detailed formatting
        self.file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

        # Log session start
        self.log_info("=== INDEXING SESSION STARTED ===")
        self.log_info(f"Session ID: {self.session_id}")
        self.log_info(f"Log file: {log_file}")
        self.log_info(f"Log level: {self.config.INDEXING_LOG_LEVEL}")

    def log_info(self, message: str, **kwargs) -> None:
        """Log info level message with optional metrics."""
        if not self.enabled or not self.logger:
            return

        if kwargs:
            message = f"{message} | {self._format_kwargs(kwargs)}"
        self.logger.info(message)

    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug level message with optional metrics."""
        if not self.enabled or not self.logger:
            return

        if kwargs:
            message = f"{message} | {self._format_kwargs(kwargs)}"
        self.logger.debug(message)

    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning level message with optional metrics."""
        if not self.enabled or not self.logger:
            return

        if kwargs:
            message = f"{message} | {self._format_kwargs(kwargs)}"
        self.logger.warning(message)

    def log_error(self, message: str, error: Exception | None = None, **kwargs) -> None:
        """Log error level message with optional exception and metrics."""
        if not self.enabled or not self.logger:
            return

        if error:
            message = f"{message} | Error: {error}"
        if kwargs:
            message = f"{message} | {self._format_kwargs(kwargs)}"
        self.logger.error(message)

    def log_phase_start(self, phase: str, **kwargs) -> None:
        """Log the start of a major indexing phase."""
        self.log_info(f"PHASE START: {phase}", **kwargs)

    def log_phase_end(self, phase: str, duration_ms: float, **kwargs) -> None:
        """Log the end of a major indexing phase with duration."""
        self.log_info(f"PHASE END: {phase} | Duration: {duration_ms:.1f}ms", **kwargs)

    def log_file_processed(
        self, file_path: str, chunks_count: int, processing_time_ms: float, language: str, **kwargs
    ) -> None:
        """Log successful file processing with metrics."""
        self.metrics["files_processed"] += 1
        self.metrics["chunks_created"] += chunks_count
        self.metrics["per_file_times"].append(processing_time_ms)

        self.log_debug(
            f"FILE PROCESSED: {file_path}",
            chunks=chunks_count,
            time_ms=f"{processing_time_ms:.1f}",
            language=language,
            **kwargs,
        )

    def log_file_failed(self, file_path: str, error: Exception, **kwargs) -> None:
        """Log failed file processing."""
        self.metrics["files_failed"] += 1
        self.log_error(f"FILE FAILED: {file_path}", error=error, **kwargs)

    def log_database_operation(self, operation: str, duration_ms: float, records_affected: int = 0, **kwargs) -> None:
        """Log database operations with timing."""
        self.metrics["database_operations"] += 1
        self.log_debug(
            f"DB OPERATION: {operation}", duration_ms=f"{duration_ms:.1f}", records=records_affected, **kwargs
        )

    def log_embedding_batch(self, batch_size: int, duration_ms: float, **kwargs) -> None:
        """Log embedding generation batch completion."""
        self.metrics["embeddings_generated"] += batch_size
        self.log_info(
            f"EMBEDDINGS GENERATED: {batch_size} embeddings",
            duration_ms=f"{duration_ms:.1f}",
            total_embeddings=self.metrics["embeddings_generated"],
            **kwargs,
        )

    def log_session_summary(self, **kwargs) -> None:
        """Log comprehensive session summary with all metrics."""
        if not self.enabled or not self.logger:
            return

        total_time = (time.time() - self.start_time) * 1000
        self.metrics["total_processing_time"] = total_time

        # Calculate averages
        avg_file_time = (
            sum(self.metrics["per_file_times"]) / len(self.metrics["per_file_times"])
            if self.metrics["per_file_times"]
            else 0
        )

        summary_lines = [
            "=== INDEXING SESSION SUMMARY ===",
            f"Session ID: {self.session_id}",
            f"Total Duration: {total_time:.1f}ms ({total_time / 1000:.1f}s)",
            f"Files Discovered: {self.metrics['files_discovered']}",
            f"Files Processed: {self.metrics['files_processed']}",
            f"Files Failed: {self.metrics['files_failed']}",
            f"Chunks Created: {self.metrics['chunks_created']}",
            f"Embeddings Generated: {self.metrics['embeddings_generated']}",
            f"Database Operations: {self.metrics['database_operations']}",
            f"Average File Processing Time: {avg_file_time:.1f}ms",
            f"Success Rate: {(self.metrics['files_processed'] / max(1, self.metrics['files_discovered']) * 100):.1f}%",
        ]

        if kwargs:
            summary_lines.append(f"Additional Metrics: {self._format_kwargs(kwargs)}")

        for line in summary_lines:
            self.log_info(line)

    def update_metric(self, key: str, value: Any) -> None:
        """Update a specific metric value."""
        self.metrics[key] = value

    def increment_metric(self, key: str, increment: int = 1) -> None:
        """Increment a metric counter."""
        self.metrics[key] = self.metrics.get(key, 0) + increment

    def _format_kwargs(self, kwargs: dict[str, Any]) -> str:
        """Format kwargs as key=value pairs."""
        return " | ".join(f"{k}={v}" for k, v in kwargs.items())

    def close(self) -> None:
        """Close the logger and file handlers."""
        if self.enabled and self.file_handler:
            self.log_info("=== INDEXING SESSION ENDED ===")
            self.file_handler.close()
            if self.logger:
                self.logger.removeHandler(self.file_handler)


def create_indexing_logger(session_id: str | None = None) -> IndexingLogger:
    """Factory function to create an indexing logger."""
    return IndexingLogger(session_id)
