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

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self):
        self.config = get_settings()
        self.model: SentenceTransformer | None = None
        self.db_path = self.config.VECTOR_DB_PATH
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

            # Initialize SQLite database with vector extension
            conn = sqlite3.connect(str(self.db_path))

            # Create embeddings table
            conn.execute("""
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
                    UNIQUE(file_path, start_line, end_line)
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON embeddings(language)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_type ON embeddings(semantic_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON embeddings(file_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON embeddings(content_hash)")

            conn.commit()
            conn.close()

            logger.info(f"Initialized vector database at {self.db_path}")

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

                # Insert or replace embedding
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings
                    (file_path, content, start_line, end_line, language,
                     semantic_type, embedding, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                SELECT file_path, content, start_line, end_line, language, semantic_type, embedding
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
                        chunk = CodeChunk(
                            file_path=row[0],
                            content=row[1],
                            start_line=row[2],
                            end_line=row[3],
                            language=row[4],
                            semantic_type=row[5],
                            embedding=stored_embedding.tolist(),
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

    async def search_by_keywords(self, query: str, language_filter: str | None = None) -> list[CodeChunk]:
        """Simple keyword-based search as fallback."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            # Build keyword search query
            base_query = """
                SELECT file_path, content, start_line, end_line, language, semantic_type
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
                chunk = CodeChunk(
                    file_path=row[0],
                    content=row[1],
                    start_line=row[2],
                    end_line=row[3],
                    language=row[4],
                    semantic_type=row[5],
                    embedding=None,
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
