# SNIPR Code Indexer

A semantic code indexing service that augments AI coding assistants with intelligent, contextually-relevant codebase information through Model Context Protocol (MCP) integration.

## Features

- **Semantic Code Parsing**: Tree-sitter integration for accurate AST-based code analysis
- **Vector Embeddings**: Advanced semantic search using sentence transformers
- **MCP Integration**: Direct integration with Claude Code and other AI assistants
- **Multi-Language Support**: Python, JavaScript, TypeScript, Go, Rust, Java
- **Incremental Indexing**: Efficient updates based on file change detection
- **Performance Optimized**: Quantized embeddings, batch processing, <200ms search latency

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd snipr

# Install dependencies
uv sync

# Install development dependencies (optional)
uv sync --extra dev
```

### Usage

#### 1. Start the MCP Server

```bash
uv run python -m src.main
```

#### 2. Configure Claude Code

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "code-indexer": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.main"],
      "cwd": "/path/to/snipr",
      "env": {
        "INDEX_CACHE_DIR": ".index_cache",
        "ENABLE_QUANTIZATION": "true"
      }
    }
  }
}
```

#### 3. Index Your Codebase

Use the `index_codebase_tool` in Claude Code:

```python
# Index entire codebase
await index_codebase("/path/to/your/project")

# Index specific languages
await index_codebase("/path/to/your/project", languages="python,javascript")

# Exclude patterns
await index_codebase("/path/to/your/project", exclude_patterns="**/node_modules/**,**/.git/**")
```

#### 4. Search Your Code

```python
# Semantic search
await search_code("authentication logic", language="python")

# Search by code type
await search_by_type("function_definition", language="python")

# Search within specific file
await search_in_file("/path/to/file.py", "error handling")

# Get indexing statistics
await get_search_stats()
```

## Architecture

### Core Components

- **IndexingService**: Tree-sitter parsing and code chunk extraction
- **SearchService**: Vector embeddings and semantic search
- **MCP Tools**: FastMCP-based tool implementations
- **Configuration**: Environment-based settings management

### Data Models

- **CodeChunk**: Represents indexed code segments with metadata
- **SearchRequest/Response**: API contracts for search operations
- **IndexingRequest/Response**: API contracts for indexing operations

### Supported Languages

| Language   | Extension | Tree-sitter Support |
|------------|-----------|-------------------|
| Python     | `.py`     | ✅ Full support   |
| JavaScript | `.js`     | ✅ Full support   |
| TypeScript | `.ts`     | 🔄 Auto-detected  |
| Go         | `.go`     | 🔄 Auto-detected  |
| Rust       | `.rs`     | 🔄 Auto-detected  |
| Java       | `.java`   | 🔄 Auto-detected  |

## Configuration

### Environment Variables

```bash
# Cache directory for index storage
INDEX_CACHE_DIR=".index_cache"

# Enable/disable embedding generation
EMBEDDING_ENABLED="true"

# Enable quantized embeddings (8x memory reduction)
ENABLE_QUANTIZATION="true"

# Maximum file size for indexing (MB)
MAX_FILE_SIZE_MB="5"

# Embedding model for semantic search
EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE="32"
```

### Performance Tuning

- **Quantization**: Reduces memory usage by 8x with minimal accuracy loss
- **Batch Processing**: Configurable batch sizes for optimal throughput
- **File Filtering**: Automatic exclusion of binary files and large files
- **Incremental Updates**: Only re-index changed files

## Development

### Running Tests

```bash
# Run all tests
uv run python -m pytest src/ -v

# Run with coverage
uv run python -m pytest src/ -v --cov=src --cov-report=term-missing

# Run specific test file
uv run python -m pytest src/services/tests/test_indexing_service.py -v
```

### Code Quality

```bash
# Lint code
uv run ruff check src/ --fix

# Type checking
uv run mypy src/

# Format code
uv run black src/
```

### Project Structure

```
src/
├── models/              # Pydantic data models
│   └── indexing_models.py
├── services/            # Core business logic
│   ├── indexing_service.py
│   └── search_service.py
├── tools/               # MCP tool implementations
│   ├── index_codebase.py
│   └── search_code.py
├── config.py           # Configuration management
└── main.py            # FastMCP server entry point
```

## MCP Tools Reference

### `index_codebase_tool`

Index a codebase for semantic search.

**Parameters:**
- `codebase_path` (string): Absolute path to codebase root
- `languages` (string, optional): Comma-separated languages to index
- `exclude_patterns` (string, optional): Comma-separated glob patterns to exclude

### `search_code_tool`

Search for semantically similar code chunks.

**Parameters:**
- `query` (string): Natural language or code query
- `language` (string, optional): Filter by programming language
- `max_results` (number): Maximum results (1-100)
- `similarity_threshold` (number): Minimum similarity score (0.0-1.0)

### `search_by_type_tool`

Search for specific code constructs (functions, classes, etc.).

**Parameters:**
- `semantic_type` (string): Code construct type
- `language` (string, optional): Filter by programming language
- `max_results` (number): Maximum results

### `search_in_file_tool`

Search within a specific file.

**Parameters:**
- `file_path` (string): Absolute path to file
- `query` (string): Search query
- `max_results` (number): Maximum results

### `get_indexing_status_tool`

Get current indexing status for a codebase.

**Parameters:**
- `codebase_path` (string): Absolute path to codebase root

### `clear_index_tool`

Clear all indexing data and start fresh.

### `get_search_stats_tool`

Get comprehensive indexing and search statistics.

## Performance Benchmarks

- **Search Latency**: <200ms for typical queries
- **Memory Usage**: 8x reduction with quantization enabled
- **Index Size**: ~5MB per 10k lines of code
- **Supported Files**: Up to 5MB per file (configurable)

## Troubleshooting

### Tree-sitter Issues

If you encounter Tree-sitter import errors:

```bash
# Install specific language parsers
uv add tree-sitter-python tree-sitter-javascript
```

### Memory Issues

Enable quantization for large codebases:

```bash
export ENABLE_QUANTIZATION="true"
```

### Performance Issues

Reduce batch size for lower memory usage:

```bash
export BATCH_SIZE="16"
```

## Requirements

- Python 3.11+
- uv package manager
- Tree-sitter language parsers
- sentence-transformers (optional, for semantic search)

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `uv run python -m pytest src/ -v`
4. Submit a pull request

---

Built with ❤️ for AI-powered development workflows.