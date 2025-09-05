#!/usr/bin/env python3
"""
Code Indexing MCP Server

A Model Context Protocol server that provides semantic code indexing and search
capabilities for AI coding assistants like Claude Code.
"""

import json
import logging
import sys

try:
    from fastmcp import FastMCP
except ImportError:
    print("Error: fastmcp not found. Please install with: uv add fastmcp")
    sys.exit(1)

# Import tools
from .tools.index_codebase import cancel_indexing_task, clear_index, get_indexing_status, index_codebase
from .tools.search_code import (
    get_search_stats,
    search_by_type,
    search_code,
    search_in_file,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Initialize FastMCP server
app = FastMCP("Code Indexer")


# Register indexing tools
@app.tool()
async def index_codebase_tool(codebase_path: str, languages: str = None, exclude_patterns: str = None) -> str:
    """Index a codebase for semantic search."""
    return await index_codebase(codebase_path, languages, exclude_patterns)


@app.tool()
async def get_indexing_status_tool(codebase_path: str) -> str:
    """Get current indexing status for a codebase."""
    return await get_indexing_status(codebase_path)


@app.tool()
async def clear_index_tool() -> str:
    """Clear all indexing data and start fresh."""
    return await clear_index()


@app.tool()
async def cancel_indexing_task_tool(task_id: str) -> str:
    """Cancel an active indexing task."""
    return await cancel_indexing_task(task_id)


# Register search tools
@app.tool()
async def search_code_tool(
    query: str, language: str = None, max_results: int = 10, similarity_threshold: float = 0.7
) -> str:
    """Search for semantically similar code chunks."""
    return await search_code(query, language, max_results, similarity_threshold)


@app.tool()
async def search_by_type_tool(semantic_type: str, language: str = None, max_results: int = 20) -> str:
    """Search for code chunks by semantic type."""
    return await search_by_type(semantic_type, language, max_results)


@app.tool()
async def search_in_file_tool(file_path: str, query: str, max_results: int = 5) -> str:
    """Search for code patterns within a specific file."""
    return await search_in_file(file_path, query, max_results)


@app.tool()
async def get_search_stats_tool() -> str:
    """Get statistics about indexed codebase and search capabilities."""
    return await get_search_stats()


# Health check tool
@app.tool()
async def health_check() -> str:
    """Check server health and capabilities."""
    try:
        from .services.indexing_service import IndexingService
        from .services.search_service import SearchService

        indexing_service = IndexingService()
        search_service = SearchService()

        health_status = {
            "status": "healthy",
            "capabilities": {
                "tree_sitter_parsers": len(indexing_service.parsers),
                "supported_languages": len(indexing_service.config.SUPPORTED_LANGUAGES),
                "embedding_model_loaded": search_service.model is not None,
                "vector_db_available": search_service.db_path.exists(),
            },
            "server_info": {"name": "Code Indexer MCP Server", "version": "0.1.0", "tools_available": 7},
        }

        return json.dumps(health_status)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return json.dumps({"status": "unhealthy", "error": str(e), "error_type": type(e).__name__})


def main():
    """Run the MCP server."""
    logger.info("Starting Code Indexer MCP Server...")

    try:
        # Run the FastMCP server
        app.run()

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
