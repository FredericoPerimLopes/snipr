import json
import logging
import sqlite3
import time
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.warning("sentence-transformers not available")
    SentenceTransformer = None

from ..config import get_settings
from ..models.indexing_models import CodeChunk, SearchRequest, SearchResponse
from .bm25_search import BM25SearchEngine
from .db_migration import DatabaseMigration
from .vector_extension import VectorExtensionLoader, VectorOperations

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self):
        self.config = get_settings()
        self.model: SentenceTransformer | None = None
        
        # Initialize db paths based on configuration
        if self.config.USE_SQLITE_VEC:
            self.db_path = self.config.VEC_DB_PATH
            self.vec_enabled = True
            self.vec_loader = VectorExtensionLoader()
            self.vec_ops = VectorOperations()
        else:
            self.db_path = self.config.VECTOR_DB_PATH
            self.vec_enabled = False
            self.vec_loader = None
            self.vec_ops = None
        
        self.bm25_engine = BM25SearchEngine()
        self._init_embedding_model()
        self._init_vector_db()

    def _init_embedding_model(self) -> None:
        """Initialize sentence transformer model for embeddings."""
        if not self.config.EMBEDDING_ENABLED:
            logger.info("Embedding generation disabled by configuration")
            return
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not available, semantic search disabled")
            return

        try:
            self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            logger.info(f"Initialized embedding model: {self.config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.model = None

    def _init_vector_db(self) -> None:
        """Initialize SQLite vector database for embeddings storage."""
        try:
            # Create cache directory if it doesn't exist
            self.config.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            if self.vec_enabled:
                # Initialize with sqlite-vec extension
                conn = sqlite3.connect(str(self.db_path))
                
                # Load the sqlite-vec extension
                if self.vec_loader.load_extension(conn):
                    logger.info("Successfully loaded sqlite-vec extension")
                    
                    # Create vec0 tables
                    if self.vec_loader.create_vec_table(
                        conn,
                        "embeddings_vec",
                        self.config.VEC_DIMENSION,
                        self.config.VEC_INDEX_TYPE
                    ):
                        logger.info("Successfully created vec0 tables")
                    
                    # Test vec operations
                    if self.vec_loader.test_vec_operations(conn):
                        logger.info("Vec operations test passed")
                    
                    # Get vec info
                    info = self.vec_loader.get_vec_info(conn)
                    if info:
                        logger.info(f"Vec info: {info}")
                else:
                    logger.warning("Failed to load sqlite-vec extension, falling back to legacy mode")
                    self.vec_enabled = False
                
                conn.close()
            
            # Also initialize legacy database for dual-system support
            if not self.vec_enabled or self.config.USE_SQLITE_VEC:
                # Run database migration first
                migration = DatabaseMigration()
                import asyncio
                
                # Properly handle async migration in sync context
                try:
                    # Try to get the running loop
                    loop = asyncio.get_running_loop()
                    # Schedule the migration as a task (fire and forget)
                    asyncio.create_task(migration.migrate_to_metadata_schema())
                except RuntimeError:
                    # No running loop, run synchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(migration.migrate_to_metadata_schema())

                # Initialize legacy SQLite database
                legacy_conn = sqlite3.connect(str(self.config.VECTOR_DB_PATH))

                # Create embeddings table with metadata fields
                legacy_conn.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT NOT NULL,
                        content TEXT NOT NULL,
                        start_line INTEGER NOT NULL,
                        end_line INTEGER NOT NULL,
                        language TEXT NOT NULL,
                        semantic_type TEXT NOT NULL,
                        embedding BLOB,
                        content_hash TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        function_signature TEXT,
                        class_name TEXT,
                        function_name TEXT,
                        parameter_types TEXT,
                        return_type TEXT,
                        inheritance_chain TEXT,
                        import_statements TEXT,
                        docstring TEXT,
                        complexity_score INTEGER,
                        dependencies TEXT,
                        interfaces TEXT,
                        decorators TEXT,
                        UNIQUE(file_path, start_line, end_line)
                    )
                """)

                # Create indexes for performance
                legacy_conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON embeddings(language)")
                legacy_conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_type ON embeddings(semantic_type)")
                legacy_conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON embeddings(file_path)")
                legacy_conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON embeddings(content_hash)")
                legacy_conn.execute("CREATE INDEX IF NOT EXISTS idx_function_name ON embeddings(function_name)")
                legacy_conn.execute("CREATE INDEX IF NOT EXISTS idx_class_name ON embeddings(class_name)")
                legacy_conn.execute("CREATE INDEX IF NOT EXISTS idx_complexity ON embeddings(complexity_score)")
                legacy_conn.execute("CREATE INDEX IF NOT EXISTS idx_return_type ON embeddings(return_type)")

                legacy_conn.commit()
                legacy_conn.close()

                logger.info(f"Initialized legacy database at {self.config.VECTOR_DB_PATH}")

        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")

    async def embed_code_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Generate embeddings for code chunks and store in database."""
        if self.model is None:
            logger.warning("No embedding model available")
            return chunks

        start_time = time.time()
        embedded_chunks = []

        try:
            # Prepare content for embedding
            texts = [f"{chunk.semantic_type}: {chunk.content}" for chunk in chunks]

            # Generate embeddings in batches
            batch_size = self.config.EMBEDDING_BATCH_SIZE
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_chunks = chunks[i : i + batch_size]

                # Generate embeddings
                embeddings = self.model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)

                # Quantize embeddings if enabled
                if self.config.ENABLE_QUANTIZATION:
                    embeddings = self._quantize_embeddings(embeddings)

                # Update chunks with embeddings
                for chunk, embedding in zip(batch_chunks, embeddings, strict=False):
                    chunk.embedding = embedding.tolist()
                    embedded_chunks.append(chunk)

                # Store in database
                await self._store_embeddings_batch(batch_chunks)

            # Build BM25 index for all chunks
            await self.bm25_engine.build_index(embedded_chunks)

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Generated embeddings for {len(chunks)} chunks in {processing_time:.1f}ms")

            return embedded_chunks

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return chunks

    def _quantize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Quantize embeddings for 8x memory reduction."""
        # Simple quantization to int8
        # Normalize to [-1, 1] range then scale to int8
        normalized = np.clip(embeddings, -1.0, 1.0)
        quantized = (normalized * 127).astype(np.int8)
        return quantized.astype(np.float32) / 127.0  # Convert back to float32

    async def _store_embeddings_batch(self, chunks: list[CodeChunk]) -> None:
        """Store batch of embeddings in vector database."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            for chunk in chunks:
                if chunk.embedding is None:
                    continue

                # Convert embedding to binary
                embedding_blob = json.dumps(chunk.embedding).encode()

                # Calculate content hash for deduplication
                content_hash = hash(chunk.content)

                # Serialize metadata fields
                parameter_types_json = json.dumps(chunk.parameter_types) if chunk.parameter_types else None
                inheritance_chain_json = json.dumps(chunk.inheritance_chain) if chunk.inheritance_chain else None
                import_statements_json = json.dumps(chunk.import_statements) if chunk.import_statements else None
                dependencies_json = json.dumps(chunk.dependencies) if chunk.dependencies else None
                interfaces_json = json.dumps(chunk.interfaces) if chunk.interfaces else None
                decorators_json = json.dumps(chunk.decorators) if chunk.decorators else None

                # Insert or replace embedding with metadata
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings
                    (file_path, content, start_line, end_line, language, semantic_type,
                     embedding, content_hash, function_signature, class_name, function_name,
                     parameter_types, return_type, inheritance_chain, import_statements,
                     docstring, complexity_score, dependencies, interfaces, decorators)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        chunk.file_path,
                        chunk.content,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.language,
                        chunk.semantic_type,
                        embedding_blob,
                        str(content_hash),
                        chunk.function_signature,
                        chunk.class_name,
                        chunk.function_name,
                        parameter_types_json,
                        chunk.return_type,
                        inheritance_chain_json,
                        import_statements_json,
                        chunk.docstring,
                        chunk.complexity_score,
                        dependencies_json,
                        interfaces_json,
                        decorators_json,
                    ),
                )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")

    async def search_code(self, request: SearchRequest) -> SearchResponse:
        """Perform semantic code search with optional language filtering."""
        start_time = time.time()

        if self.model is None:
            logger.warning("No embedding model available for semantic search")
            return SearchResponse(results=[], total_matches=0, query_time_ms=0.0)

        try:
            # Generate query embedding
            query_embedding = self.model.encode([request.query], normalize_embeddings=True)[0]

            # Quantize query embedding if enabled
            if self.config.ENABLE_QUANTIZATION:
                query_embedding = self._quantize_embeddings(query_embedding.reshape(1, -1))[0]

            # Search database for similar embeddings
            results = await self._search_similar_embeddings(
                query_embedding, request.language, request.max_results, request.similarity_threshold
            )

            query_time = (time.time() - start_time) * 1000

            return SearchResponse(results=results, total_matches=len(results), query_time_ms=round(query_time, 2))

        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return SearchResponse(results=[], total_matches=0, query_time_ms=0.0)

    async def _search_similar_embeddings(
        self, query_embedding: np.ndarray, language_filter: str | None, max_results: int, threshold: float
    ) -> list[CodeChunk]:
        """Search for similar embeddings in vector database."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            # Build query with optional language filter
            base_query = """
                SELECT file_path, content, start_line, end_line, language, semantic_type, embedding,
                       function_signature, class_name, function_name, parameter_types, return_type,
                       inheritance_chain, import_statements, docstring, complexity_score,
                       dependencies, interfaces, decorators
                FROM embeddings
            """

            params = []
            if language_filter:
                base_query += " WHERE language = ?"
                params.append(language_filter)

            cursor = conn.execute(base_query)
            rows = cursor.fetchall()

            # Calculate similarities
            similarities: list[tuple[float, CodeChunk]] = []

            for row in rows:
                try:
                    # Decode embedding
                    embedding_blob = row[6]
                    stored_embedding = np.array(json.loads(embedding_blob.decode()))

                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)

                    if similarity >= threshold:
                        # Deserialize metadata fields
                        parameter_types = json.loads(row[10]) if row[10] else None
                        inheritance_chain = json.loads(row[12]) if row[12] else None
                        import_statements = json.loads(row[13]) if row[13] else None
                        dependencies = json.loads(row[16]) if row[16] else None
                        interfaces = json.loads(row[17]) if row[17] else None
                        decorators = json.loads(row[18]) if row[18] else None

                        chunk = CodeChunk(
                            file_path=row[0],
                            content=row[1],
                            start_line=row[2],
                            end_line=row[3],
                            language=row[4],
                            semantic_type=row[5],
                            embedding=stored_embedding.tolist(),
                            function_signature=row[7],
                            class_name=row[8],
                            function_name=row[9],
                            parameter_types=parameter_types,
                            return_type=row[11],
                            inheritance_chain=inheritance_chain,
                            import_statements=import_statements,
                            docstring=row[14],
                            complexity_score=row[15],
                            dependencies=dependencies,
                            interfaces=interfaces,
                            decorators=decorators,
                        )
                        similarities.append((similarity, chunk))

                except Exception as e:
                    logger.debug(f"Error processing embedding row: {e}")
                    continue

            # Sort by similarity and return top results
            similarities.sort(reverse=True)
            results = [chunk for _, chunk in similarities[:max_results]]

            conn.close()
            return results

        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Ensure vectors are normalized
            a_norm = a / (np.linalg.norm(a) + 1e-8)
            b_norm = b / (np.linalg.norm(b) + 1e-8)

            # Calculate cosine similarity
            similarity = np.dot(a_norm, b_norm)
            return float(similarity)

        except Exception as e:
            logger.debug(f"Error calculating similarity: {e}")
            return 0.0

    async def search_by_bm25(
        self, query: str, language_filter: str | None = None, max_results: int = 50
    ) -> list[CodeChunk]:
        """BM25-based lexical search."""
        try:
            # Get BM25 results (document IDs with scores)
            bm25_results = await self.bm25_engine.search(query, language_filter, max_results)

            if not bm25_results:
                return []

            # Convert document IDs back to CodeChunks
            chunks = []
            conn = sqlite3.connect(str(self.db_path))

            for doc_id, _ in bm25_results:
                # Parse document ID to get file_path and start_line
                file_path, start_line_str = doc_id.rsplit(":", 1)
                start_line = int(start_line_str)

                # Query database for full chunk data
                cursor = conn.execute(
                    """
                    SELECT file_path, content, start_line, end_line, language, semantic_type,
                           function_signature, class_name, function_name, parameter_types, return_type,
                           inheritance_chain, import_statements, docstring, complexity_score,
                           dependencies, interfaces, decorators
                    FROM embeddings
                    WHERE file_path = ? AND start_line = ?
                    """,
                    (file_path, start_line),
                )

                row = cursor.fetchone()
                if row:
                    # Deserialize metadata fields
                    parameter_types = json.loads(row[9]) if row[9] else None
                    inheritance_chain = json.loads(row[11]) if row[11] else None
                    import_statements = json.loads(row[12]) if row[12] else None
                    dependencies = json.loads(row[15]) if row[15] else None
                    interfaces = json.loads(row[16]) if row[16] else None
                    decorators = json.loads(row[17]) if row[17] else None

                    chunk = CodeChunk(
                        file_path=row[0],
                        content=row[1],
                        start_line=row[2],
                        end_line=row[3],
                        language=row[4],
                        semantic_type=row[5],
                        function_signature=row[6],
                        class_name=row[7],
                        function_name=row[8],
                        parameter_types=parameter_types,
                        return_type=row[10],
                        inheritance_chain=inheritance_chain,
                        import_statements=import_statements,
                        docstring=row[13],
                        complexity_score=row[14],
                        dependencies=dependencies,
                        interfaces=interfaces,
                        decorators=decorators,
                    )
                    chunks.append(chunk)

            conn.close()
            return chunks

        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []

    async def search_by_keywords(self, query: str, language_filter: str | None = None) -> list[CodeChunk]:
        """Simple keyword-based search as fallback."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            # Build keyword search query
            base_query = """
                SELECT file_path, content, start_line, end_line, language, semantic_type,
                       function_signature, class_name, function_name, parameter_types, return_type,
                       inheritance_chain, import_statements, docstring, complexity_score,
                       dependencies, interfaces, decorators
                FROM embeddings
                WHERE content LIKE ?
            """

            params = [f"%{query}%"]
            if language_filter:
                base_query += " AND language = ?"
                params.append(language_filter)

            base_query += " LIMIT 50"

            cursor = conn.execute(base_query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                # Deserialize metadata
                parameter_types = json.loads(row[9]) if row[9] else None
                inheritance_chain = json.loads(row[11]) if row[11] else None
                import_statements = json.loads(row[12]) if row[12] else None
                dependencies = json.loads(row[15]) if row[15] else None
                interfaces = json.loads(row[16]) if row[16] else None
                decorators = json.loads(row[17]) if row[17] else None

                chunk = CodeChunk(
                    file_path=row[0],
                    content=row[1],
                    start_line=row[2],
                    end_line=row[3],
                    language=row[4],
                    semantic_type=row[5],
                    embedding=None,
                    function_signature=row[6],
                    class_name=row[7],
                    function_name=row[8],
                    parameter_types=parameter_types,
                    return_type=row[10],
                    inheritance_chain=inheritance_chain,
                    import_statements=import_statements,
                    docstring=row[13],
                    complexity_score=row[14],
                    dependencies=dependencies,
                    interfaces=interfaces,
                    decorators=decorators,
                )
                results.append(chunk)

            conn.close()
            return results

        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

    async def get_embeddings_stats(self) -> dict[str, Any]:
        """Get statistics about stored embeddings."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            # Get total count
            total_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

            # Get language distribution
            lang_stats = conn.execute("""
                SELECT language, COUNT(*) as count
                FROM embeddings
                GROUP BY language
                ORDER BY count DESC
            """).fetchall()

            # Get semantic type distribution
            type_stats = conn.execute("""
                SELECT semantic_type, COUNT(*) as count
                FROM embeddings
                GROUP BY semantic_type
                ORDER BY count DESC
            """).fetchall()

            conn.close()

            return {
                "total_embeddings": total_count,
                "languages": dict(lang_stats),
                "semantic_types": dict(type_stats),
                "database_size_mb": (
                    round(self.db_path.stat().st_size / (1024 * 1024), 2) if self.db_path.exists() else 0
                ),
            }

        except Exception as e:
            logger.error(f"Error getting embeddings stats: {e}")
            return {"total_embeddings": 0, "languages": {}, "semantic_types": {}, "database_size_mb": 0}

    async def remove_file_embeddings(self, file_paths: list[str]) -> None:
        """Remove embeddings for specific files from the database."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            # Delete embeddings for specified files
            placeholders = ",".join("?" * len(file_paths))
            conn.execute(f"DELETE FROM embeddings WHERE file_path IN ({placeholders})", file_paths)

            conn.commit()
            conn.close()

            logger.info(f"Removed embeddings for {len(file_paths)} files")

        except Exception as e:
            logger.error(f"Error removing file embeddings: {e}")

    async def clear_all_embeddings(self) -> None:
        """Clear all embeddings from the database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("DELETE FROM embeddings")
            conn.commit()
            conn.close()

            logger.info("Cleared all embeddings from database")

        except Exception as e:
            logger.error(f"Error clearing embeddings: {e}")
