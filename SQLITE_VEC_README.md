# SQLite-vec Integration Guide for Snipr

This guide covers the integration of the sqlite-vec extension for optimized vector search in the Snipr codebase indexing tool.

## 🚀 Quick Start

### Run with SQLite-vec

```bash
# Run Snipr with sqlite-vec (enabled by default)
uv run python src/main.py
```

### Verify Installation

```bash
# Test sqlite-vec functionality
uv run python -c "
from src.services.vector_extension import VectorExtensionLoader
import sqlite3

conn = sqlite3.connect(':memory:')
loader = VectorExtensionLoader()
if loader.load_extension(conn):
    print('✓ sqlite-vec is working!')
else:
    print('✗ Failed to load sqlite-vec')
"
```

## 📦 Installation

The sqlite-vec extension is already added to the project dependencies:

```toml
# pyproject.toml
dependencies = [
    ...
    "sqlite-vec>=0.1.6",
]
```

To install/update:

```bash
uv pip install sqlite-vec
```

## 🔄 Migration Guide

### Migrating Existing Embeddings

If you have existing embeddings in the legacy format, use the migration script:

```bash
# Dry run to see what would be migrated
uv run python -m src.services.migration_to_vec --dry-run

# Perform actual migration
uv run python -m src.services.migration_to_vec

# Migrate with custom batch size
uv run python -m src.services.migration_to_vec --batch-size 500

# Verify migration
uv run python -m src.services.migration_to_vec --verify-only
```

### Migration Options

- `--dry-run`: Simulate migration without making changes
- `--batch-size N`: Process N embeddings per batch (default: 100)
- `--verify-only`: Check if migration was successful

## ⚙️ Configuration

### Environment Variables

```bash
# Vector index type (flat, ivf, hnsw)
export VEC_INDEX_TYPE=flat

# Other settings are in src/config.py
```

### Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `VEC_DIMENSION` | 768 | Embedding vector dimensions |
| `VEC_INDEX_TYPE` | flat | Index type (flat/ivf/hnsw) |
| `VECTOR_DB_PATH` | .index_cache/embeddings_vec.db | Vec database location |

## 🏗️ Architecture

### SQLite-vec Architecture

The implementation uses sqlite-vec for all vector operations:

```python
# SQLite-vec is used by default for all operations
results = vec_ops.search_similar(...)
```

### Database Structure

```sql
-- Vec0 virtual table for embeddings
CREATE VIRTUAL TABLE embeddings_vec USING vec0(
    embedding float[768]
);

-- Metadata table
CREATE TABLE embeddings_vec_metadata (
    rowid INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,
    content TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    language TEXT NOT NULL,
    semantic_type TEXT NOT NULL,
    -- ... other metadata fields
);
```

## 📈 Performance

### Expected Improvements

| Metric | Legacy System | SQLite-vec | Improvement |
|--------|--------------|------------|-------------|
| Search Speed | ~500ms | ~50ms | 10x faster |
| Memory Usage | 1GB/100k | 500MB/100k | 50% less |
| Storage Size | 2GB/100k | 1.2GB/100k | 40% less |

### Index Types

#### Flat Index (Default)
- Best for: < 100k vectors
- Accuracy: 100% (exact search)
- Speed: Linear O(n)

#### IVF Index
- Best for: 100k - 1M vectors
- Accuracy: 95-99% (approximate)
- Speed: ~O(√n)

To change index type:
```bash
export VEC_INDEX_TYPE=ivf
```

#### HNSW Index
- Best for: > 1M vectors
- Accuracy: 95-99% (approximate)
- Speed: O(log n)

```bash
export VEC_INDEX_TYPE=hnsw
```

## 🧪 Testing

### Run Tests

```bash
# Test vec extension functionality
uv run pytest src/services/tests/test_sqlite_vec.py -v

# Test search service
uv run pytest src/services/tests/test_search_service.py -v
```

### Manual Testing

```python
from src.services.vector_extension import VectorExtensionLoader, VectorOperations
import sqlite3
import random

# Create test database
conn = sqlite3.connect('test.db')
loader = VectorExtensionLoader()
loader.load_extension(conn)
loader.create_vec_table(conn, 'test_vecs', dimension=768)

# Insert test embedding
ops = VectorOperations()
embedding = [random.random() for _ in range(768)]
metadata = {
    'file_path': 'test.py',
    'content': 'def test(): pass',
    'start_line': 1,
    'end_line': 1,
    'language': 'python',
    'semantic_type': 'function',
    'content_hash': '12345'
}

rowid = ops.insert_embedding(conn, 'test_vecs', embedding, metadata)
print(f"Inserted embedding with rowid: {rowid}")

# Search for similar
results = ops.search_similar(conn, 'test_vecs', embedding, k=5)
print(f"Found {len(results)} similar embeddings")
```

## 🚀 Deployment

### Development Environment

```bash
# Clone and checkout the feature branch
git clone <repository>
cd snipr
git checkout feature/sqlite-vec-integration

# Install dependencies
uv pip install -r pyproject.toml

# Run the application (sqlite-vec enabled by default)
uv run python src/main.py
```

### Production Deployment

**Production Migration**

```bash
# 1. Backup existing database
cp .index_cache/embeddings.db .index_cache/embeddings_backup.db

# 2. Run migration
uv run python -m src.services.migration_to_vec

# 3. Verify migration
uv run python -m src.services.migration_to_vec --verify-only

# 4. Deploy application (sqlite-vec enabled by default)
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11

# Install sqlite-vec dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    build-essential

# Copy application
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install uv
RUN uv pip install -r pyproject.toml

# SQLite-vec enabled by default

CMD ["uv", "run", "python", "src/main.py"]
```

## 🔄 Rollback

If you need to rollback to the legacy system:

```bash
# Rollback requires code changes to disable sqlite-vec
# Legacy database remains intact for migration compatibility
```

## 📊 Monitoring

### Check Vec Status

```python
from src.services.vector_extension import VectorExtensionLoader
import sqlite3

conn = sqlite3.connect('.index_cache/embeddings_vec.db')
loader = VectorExtensionLoader()
loader.load_extension(conn)

# Get vec info
info = loader.get_vec_info(conn)
print(f"Version: {info['version']}")
print(f"Tables: {info['vec_tables']}")
print(f"Vector count: {info['table_stats']}")
```

### Performance Metrics

Monitor these metrics to track improvement:

1. **Query latency** - Should decrease by ~10x
2. **Memory usage** - Should decrease by ~50%
3. **Database size** - Should decrease by ~40%
4. **CPU usage** - Should decrease during searches

## 🐛 Troubleshooting

### Common Issues

#### Extension fails to load

```bash
# Check sqlite-vec installation
uv pip show sqlite-vec

# Reinstall if needed
uv pip install --force-reinstall sqlite-vec
```

#### Migration fails

```bash
# Check legacy database
sqlite3 .index_cache/embeddings.db "SELECT COUNT(*) FROM embeddings;"

# Try with smaller batch size
uv run python -m src.services.migration_to_vec --batch-size 10
```

#### Search returns no results

```bash
# Verify vec database has data
sqlite3 .index_cache/embeddings_vec.db "SELECT COUNT(*) FROM embeddings_vec;"

# SQLite-vec is enabled by default
# Check if working by running a test search
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed vec operations
from src.services.search_service import SearchService
service = SearchService()
```

## 📚 Additional Resources

- [sqlite-vec Documentation](https://github.com/asg017/sqlite-vec)
- [Performance Benchmarks](sqlite-vec-comparison.md)
- [Integration Plan](sqlite-vec-integration-plan.md)
- [Migration Script](src/services/migration_to_vec.py)
- [Vector Extension Module](src/services/vector_extension.py)

## 🤝 Contributing

To contribute to the sqlite-vec integration:

1. Test thoroughly with your use case
2. Report performance metrics
3. Submit bug reports with debug logs
4. Propose optimizations

## 📄 License

This integration maintains compatibility with the existing Snipr license.