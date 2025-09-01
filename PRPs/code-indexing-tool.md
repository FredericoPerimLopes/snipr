name: "Code Indexing Tool for AI Coding Assistant Augmentation"
description: |

---

## Goal

**Feature Goal**: Build a semantic code indexing service that augments AI coding tools like Claude Code with intelligent, contextually-relevant codebase information

**Deliverable**: MCP server that provides real-time semantic code search, intelligent chunking, and contextual code retrieval for AI coding assistants

**Success Definition**: AI coding assistants can query for relevant code context, reducing hallucinations and improving code generation accuracy by 40%+ through semantic understanding

## User Persona

**Target User**: Software developers using AI coding assistants (Claude Code, Cursor, GitHub Copilot)

**Use Case**: Developer asks AI assistant to "implement authentication middleware" and the indexing tool automatically provides relevant existing auth patterns, security configurations, and integration examples

**User Journey**: 
1. Developer writes code request to AI assistant
2. AI assistant queries indexing tool for relevant context
3. Tool returns semantically similar code, patterns, and documentation
4. AI generates more accurate, context-aware code following existing patterns

**Pain Points Addressed**: 
- AI assistants lacking project-specific context
- Inconsistent code patterns across large codebases
- Manual context gathering slowing development
- AI hallucinations due to insufficient codebase understanding

## Why

- **Business Value**: 40%+ improvement in AI coding accuracy reduces debugging time and technical debt
- **Integration**: Seamlessly works with existing Claude Code MCP ecosystem
- **Problems Solved**: Context window limitations, pattern inconsistency, manual context curation for developers and teams

## What

Real-time semantic code indexing service that AI assistants query for contextually relevant code examples, patterns, and documentation.

### Success Criteria

- [ ] Indexes codebases with <200ms search latency for 100K+ LOC
- [ ] Provides semantic similarity search with 99%+ accuracy 
- [ ] Integrates with Claude Code via MCP protocol
- [ ] Supports incremental indexing for real-time code changes
- [ ] Handles 30+ programming languages via Tree-sitter

## All Needed Context

### Context Completeness Check

_"If someone knew nothing about this codebase, would they have everything needed to implement this successfully?"_

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- url: https://tree-sitter.github.io/tree-sitter/using-parsers
  why: Core parsing technology for semantic code understanding
  critical: Incremental parsing enables real-time updates, error-tolerant AST generation essential for robust indexing

- url: https://modelcontextprotocol.io/specification/2025-03-26
  why: MCP protocol specification for AI tool integration
  critical: JSON-RPC 2.0 communication standard, three core primitives (tools, resources, prompts)

- file: /home/flopes/snipr/PRPs/prp_base.md
  why: Implementation patterns for FastMCP, service structure, validation approach
  pattern: Async service methods with Pydantic models, MCP tool decorators
  gotcha: Must follow domain-driven directory structure, snake_case naming for files

- file: /home/flopes/snipr/PRPs/scripts/prp_runner.py
  why: Project's Python conventions and CLI tool patterns
  pattern: argparse usage, Path handling, subprocess execution
  gotcha: Uses 'uv run' for Python package management, absolute paths required

- docfile: /home/flopes/snipr/PRPs/ai_docs/cc_mcp.md
  why: Claude Code MCP integration patterns and authentication
  section: All sections - covers server setup, OAuth, resources, prompts

- url: https://docs.anthropic.com/en/docs/claude-code/mcp
  why: Claude Code MCP integration best practices
  critical: Security considerations, tool allowlists, scope management for project vs user configurations

- url: https://github.com/zilliztech/claude-context
  why: Reference implementation of semantic code search for Claude
  critical: Vector embedding strategies, 40% token reduction while maintaining retrieval quality

- url: https://github.com/modelcontextprotocol/servers/tree/main/src
  why: Official MCP server implementation patterns
  pattern: Server architecture, tool registration, resource handling
  gotcha: Error handling patterns, logging standards, configuration management
```

### Current Codebase tree

```bash
/home/flopes/snipr/
├── PRPs/
│   ├── README.md
│   ├── ai_docs/         # Claude Code documentation for context
│   ├── prp_*.md         # PRP templates and examples
│   └── scripts/
│       └── prp_runner.py # Python CLI with uv, argparse patterns
└── .claude/             # Claude Code configuration
```

### Desired Codebase tree with files to be added and responsibility of file

```bash
/home/flopes/snipr/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── indexing_models.py      # Pydantic models for indexing requests/responses
│   ├── services/
│   │   ├── __init__.py
│   │   ├── indexing_service.py     # Core indexing logic with Tree-sitter integration
│   │   ├── search_service.py       # Semantic search with vector embeddings
│   │   └── tests/
│   │       ├── test_indexing_service.py
│   │       └── test_search_service.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── index_codebase.py       # MCP tool for indexing operations
│   │   ├── search_code.py          # MCP tool for semantic code search
│   │   └── tests/
│   │       ├── test_index_codebase.py
│   │       └── test_search_code.py
│   ├── main.py                     # FastMCP server with tool registration
│   └── config.py                   # Configuration and environment management
├── pyproject.toml                  # uv project configuration with dependencies
└── .env.example                    # Environment variable template
```

### Known Gotchas of our codebase & Library Quirks

```python
# CRITICAL: This project uses 'uv' as package manager, not pip
# All commands must use: uv run python, uv run pytest, etc.

# CRITICAL: FastMCP requires async functions for all tool definitions
# Use @app.tool() decorator with async def functions

# CRITICAL: Tree-sitter requires language-specific grammars
# Must install tree-sitter-python, tree-sitter-javascript, etc.

# CRITICAL: Vector embeddings require significant memory management
# Use quantized representations for 8x memory reduction (250MB for 100M LOC)

# CRITICAL: MCP tools return JSON strings, not Python objects
# Always return json.dumps(response_dict) from MCP tool functions

# CRITICAL: File paths in this codebase must be absolute
# Use Path.resolve() and avoid relative path assumptions
```

## Implementation Blueprint

### Data models and structure

Create the core data models to ensure type safety and consistency.

```python
# src/models/indexing_models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class IndexingRequest(BaseModel):
    codebase_path: str = Field(..., description="Absolute path to codebase root")
    languages: Optional[List[str]] = Field(default=None, description="Languages to index (auto-detect if None)")
    exclude_patterns: Optional[List[str]] = Field(default=None, description="Glob patterns to exclude")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Semantic search query")
    language: Optional[str] = Field(default=None, description="Filter by programming language")
    max_results: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class CodeChunk(BaseModel):
    file_path: str
    content: str
    start_line: int
    end_line: int
    language: str
    semantic_type: str  # function, class, variable, etc.
    embedding: Optional[List[float]] = None

class SearchResponse(BaseModel):
    results: List[CodeChunk]
    total_matches: int
    query_time_ms: float
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE pyproject.toml
  - IMPLEMENT: uv project configuration with all required dependencies
  - DEPENDENCIES: tree-sitter, sentence-transformers, fastmcp, pydantic
  - NAMING: Standard Python project structure
  - PLACEMENT: Project root

Task 2: CREATE src/models/indexing_models.py
  - IMPLEMENT: IndexingRequest, SearchRequest, CodeChunk, SearchResponse Pydantic models
  - FOLLOW pattern: /home/flopes/snipr/PRPs/prp_base.md (field validation approach)
  - NAMING: CamelCase for classes, snake_case for fields
  - PLACEMENT: Domain-specific model file in src/models/

Task 3: CREATE src/config.py
  - IMPLEMENT: Configuration management with environment variables
  - FOLLOW pattern: /home/flopes/snipr/PRPs/scripts/prp_runner.py (Path handling, environment)
  - NAMING: CONFIG constants, get_settings() function
  - PLACEMENT: Project configuration in src/

Task 4: CREATE src/services/indexing_service.py
  - IMPLEMENT: IndexingService class with Tree-sitter integration
  - FOLLOW pattern: /home/flopes/snipr/PRPs/prp_base.md (async service structure)
  - NAMING: IndexingService class, async def parse_*, index_*, get_* methods
  - DEPENDENCIES: tree-sitter parsers, models from Task 2
  - CRITICAL: Incremental parsing for real-time updates, error-tolerant AST processing
  - PLACEMENT: Service layer in src/services/

Task 5: CREATE src/services/search_service.py
  - IMPLEMENT: SearchService class with vector embeddings and semantic search
  - FOLLOW pattern: /home/flopes/snipr/PRPs/prp_base.md (service structure, error handling)
  - NAMING: SearchService class, async def search_*, embed_*, similarity_* methods
  - DEPENDENCIES: sentence-transformers, vector database (SQLite-vec), models from Task 2
  - CRITICAL: Quantized vector search for memory efficiency, hybrid BM25+semantic search
  - PLACEMENT: Service layer in src/services/

Task 6: CREATE src/tools/index_codebase.py
  - IMPLEMENT: MCP tool for indexing codebase operations
  - FOLLOW pattern: /home/flopes/snipr/PRPs/ai_docs/cc_mcp.md (FastMCP tool structure)
  - NAMING: index_codebase.py file, index_codebase() tool function
  - DEPENDENCIES: IndexingService from Task 4
  - CRITICAL: Must return JSON string, async def with @app.tool() decorator
  - PLACEMENT: Tool layer in src/tools/

Task 7: CREATE src/tools/search_code.py
  - IMPLEMENT: MCP tool for semantic code search
  - FOLLOW pattern: /home/flopes/snipr/PRPs/ai_docs/cc_mcp.md (MCP tool patterns)
  - NAMING: search_code.py file, search_code() tool function  
  - DEPENDENCIES: SearchService from Task 5
  - CRITICAL: Return JSON with CodeChunk results, handle similarity thresholds
  - PLACEMENT: Tool layer in src/tools/

Task 8: CREATE src/main.py
  - IMPLEMENT: FastMCP server with tool registration
  - FOLLOW pattern: /home/flopes/snipr/PRPs/ai_docs/cc_mcp.md (server setup and tool registration)
  - INTEGRATE: Register index_codebase and search_code tools
  - NAMING: Standard FastMCP server patterns
  - CRITICAL: Proper tool allowlists, environment configuration
  - PLACEMENT: Server entry point in src/

Task 9: CREATE src/services/tests/test_indexing_service.py
  - IMPLEMENT: Unit tests for IndexingService (parsing, incremental updates, error handling)
  - FOLLOW pattern: /home/flopes/snipr/PRPs/prp_base.md (pytest fixture usage, async testing)
  - NAMING: test_{method}_{scenario} function naming
  - COVERAGE: All public methods with positive and negative test cases
  - CRITICAL: Test Tree-sitter parser integration, large file handling
  - PLACEMENT: Tests alongside the code they test

Task 10: CREATE src/services/tests/test_search_service.py
  - IMPLEMENT: Unit tests for SearchService (semantic search, embeddings, performance)
  - FOLLOW pattern: /home/flopes/snipr/PRPs/prp_base.md (assertion patterns, mocking)
  - COVERAGE: Vector embeddings, search accuracy, performance thresholds
  - CRITICAL: Test vector quantization, similarity thresholds, result ranking
  - PLACEMENT: Service tests in src/services/tests/

Task 11: CREATE src/tools/tests/test_index_codebase.py
  - IMPLEMENT: Unit tests for index_codebase MCP tool
  - FOLLOW pattern: /home/flopes/snipr/PRPs/prp_base.md (MCP tool testing approach)
  - MOCK: IndexingService dependencies
  - COVERAGE: Tool input validation, JSON response format, error handling
  - PLACEMENT: Tool tests in src/tools/tests/

Task 12: CREATE src/tools/tests/test_search_code.py
  - IMPLEMENT: Unit tests for search_code MCP tool
  - FOLLOW pattern: /home/flopes/snipr/PRPs/prp_base.md (testing approach)
  - MOCK: SearchService dependencies  
  - COVERAGE: Search parameters, result formatting, performance validation
  - PLACEMENT: Tool tests in src/tools/tests/
```

### Implementation Patterns & Key Details

```python
# Service method pattern for Tree-sitter integration
async def parse_file(self, file_path: str, language: str) -> List[CodeChunk]:
    # PATTERN: Input validation first (Path validation, language support check)
    validated_path = Path(file_path).resolve()
    
    # CRITICAL: Tree-sitter requires specific language grammar loading
    import tree_sitter_python as tspython
    parser = Language(tspython.language(), "python")
    
    # GOTCHA: Tree-sitter memory management - must clean up parser resources
    # PATTERN: Error handling for malformed files (continue parsing through errors)
    # CRITICAL: Incremental parsing for performance - only reparse changed sections
    
    return [CodeChunk(file_path=str(validated_path), content=chunk_content, ...)]

# Vector embedding pattern for semantic search  
async def embed_code(self, code: str, language: str) -> List[float]:
    # PATTERN: Model loading and caching (avoid reloading embeddings model)
    # CRITICAL: Quantized representations for 8x memory reduction
    # GOTCHA: Context window limits - chunk code intelligently using AST boundaries
    
    embedding = self.model.encode(code, normalize_embeddings=True)
    return embedding.tolist()  # Convert numpy to JSON-serializable

# MCP tool pattern
@app.tool()
async def search_code(query: str, language: str = None, max_results: int = 10) -> str:
    # PATTERN: Tool validation and service delegation (see FastMCP docs)
    request = SearchRequest(query=query, language=language, max_results=max_results)
    
    # CRITICAL: All MCP tools must return JSON strings, not Python objects
    response = await search_service.search(request)
    return json.dumps(response.model_dump())
```

### Integration Points

```yaml
DEPENDENCIES:
  - add to: pyproject.toml
  - pattern: "tree-sitter = '^0.21.0'"
  - pattern: "sentence-transformers = '^2.2.0'"
  - pattern: "fastmcp = '^1.0.0'"
  - pattern: "sqlite-vec = '^0.1.0'"

CONFIG:
  - add to: src/config.py  
  - pattern: "INDEX_CACHE_DIR = Path(os.getenv('INDEX_CACHE_DIR', '.index_cache'))"
  - pattern: "EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')"

MCP_REGISTRATION:
  - add to: src/main.py
  - pattern: "from src.tools.index_codebase import index_codebase"
  - pattern: "from src.tools.search_code import search_code"
  - pattern: "app.register_tool(index_codebase)"
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# Run after each file creation - fix before proceeding
uv run ruff check src/ --fix                    # Auto-format and fix linting issues
uv run mypy src/                                # Type checking for all source files
uv run ruff format src/                         # Ensure consistent formatting

# Expected: Zero errors. If errors exist, READ output and fix before proceeding.
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test each component as it's created
uv run pytest src/services/tests/test_indexing_service.py -v
uv run pytest src/services/tests/test_search_service.py -v
uv run pytest src/tools/tests/test_index_codebase.py -v
uv run pytest src/tools/tests/test_search_code.py -v

# Full test suite validation
uv run pytest src/ -v --cov=src --cov-report=term-missing

# Expected: All tests pass with >90% coverage. If failing, debug root cause and fix implementation.
```

### Level 3: Integration Testing (System Validation)

```bash
# MCP server startup validation
uv run python src/main.py &
sleep 3  # Allow startup time

# Test indexing tool functionality
echo '{"method": "tools/call", "params": {"name": "index_codebase", "arguments": {"codebase_path": "/home/flopes/snipr", "languages": ["python"]}}}' | uv run python src/main.py

# Test search tool functionality  
echo '{"method": "tools/call", "params": {"name": "search_code", "arguments": {"query": "async function definition", "language": "python", "max_results": 5}}}' | uv run python src/main.py

# Performance validation - index medium codebase
time uv run python -c "
import asyncio
from src.services.indexing_service import IndexingService
async def test():
    service = IndexingService()
    result = await service.index_codebase('/home/flopes/snipr')
    print(f'Indexed {len(result)} chunks')
asyncio.run(test())
"

# Expected: <200ms search latency, successful MCP tool responses, no connection errors
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Claude Code Integration Validation
# Test with real Claude Code session
claude mcp add code-indexer -- uv run python src/main.py
claude mcp list  # Verify server is registered

# Real-world usage test with Claude Code
echo "Can you search for async function examples in this codebase?" | claude

# Performance benchmarking
# Test with larger codebase (if available)
uv run python -c "
import time
from src.services.search_service import SearchService
service = SearchService()
start = time.time()
results = service.search('authentication middleware patterns')
duration = (time.time() - start) * 1000
print(f'Search completed in {duration:.1f}ms with {len(results)} results')
"

# Memory usage validation
# Monitor memory with large codebase indexing
ps aux | grep python  # Monitor memory usage during indexing

# Vector database validation
sqlite3 .index_cache/embeddings.db "SELECT COUNT(*) FROM embeddings;"

# Expected: <200ms search time, memory usage <500MB for 100K LOC, accurate semantic results
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] All tests pass: `uv run pytest src/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] No formatting issues: `uv run ruff format src/ --check`

### Feature Validation

- [ ] Codebase indexing completes successfully with Tree-sitter parsing
- [ ] Semantic search returns relevant results with <200ms latency
- [ ] MCP integration works with Claude Code: `claude mcp add code-indexer`
- [ ] Incremental indexing updates only changed files
- [ ] Vector embeddings provide semantic similarity matching
- [ ] Tool error cases handled gracefully with informative messages

### Code Quality Validation

- [ ] Follows project Python conventions with uv package management
- [ ] File placement matches desired codebase tree structure
- [ ] Uses async/await patterns consistently throughout
- [ ] Pydantic models provide proper input validation
- [ ] FastMCP tool patterns followed with JSON string returns
- [ ] Configuration management uses environment variables

### Documentation & Deployment

- [ ] Tool functions have clear docstrings and type hints
- [ ] Environment variables documented in .env.example
- [ ] MCP server registration follows Claude Code patterns
- [ ] Logging provides useful information without being verbose

---

## Anti-Patterns to Avoid

- ❌ Don't use sync functions in async MCP context
- ❌ Don't return Python objects from MCP tools - use JSON strings
- ❌ Don't ignore Tree-sitter parsing errors - handle gracefully
- ❌ Don't load embedding models repeatedly - cache properly
- ❌ Don't index without incremental update support
- ❌ Don't use relative paths - always resolve to absolute
- ❌ Don't skip vector quantization for large codebases
- ❌ Don't forget to clean up Tree-sitter parser resources