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
from .vector_extension import VectorExtensionLoader, VectorOperations

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self):
        self.config = get_settings()
        self.model: SentenceTransformer | None = None

        # Initialize sqlite-vec database
        self.db_path = self.config.VECTOR_DB_PATH
        self.vec_loader = VectorExtensionLoader()
        self.vec_ops = VectorOperations()

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
            # Initialize with specified device (cpu or cuda)
            device = self.config.DEVICE
            self.model = SentenceTransformer(self.config.EMBEDDING_MODEL, device=device, trust_remote_code=True)
            logger.info(f"Initialized embedding model: {self.config.EMBEDDING_MODEL} on device: {device}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.model = None

    def _init_vector_db(self) -> None:
        """Initialize SQLite vector database for embeddings storage."""
        try:
            # Create cache directory if it doesn't exist
            self.config.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            # Initialize with sqlite-vec extension
            conn = sqlite3.connect(str(self.db_path))

            # Load the sqlite-vec extension
            if self.vec_loader.load_extension(conn):
                logger.info("Successfully loaded sqlite-vec extension")

                # Create vec0 tables
                if self.vec_loader.create_vec_table(
                    conn, "embeddings_vec", self.config.VEC_DIMENSION, self.config.VEC_INDEX_TYPE
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
                logger.error("Failed to load sqlite-vec extension")
                raise RuntimeError("SQLite-vec extension is required")

            conn.close()

            logger.info("Vector database initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")

    async def embed_code_chunks(self, chunks: list[CodeChunk], indexing_logger=None) -> list[CodeChunk]:
        """Generate embeddings for code chunks and store in database."""
        if self.model is None:
            logger.warning("No embedding model available")
            return chunks

        start_time = time.time()
        embedded_chunks = []

        try:
            logger.info(f"Starting embedding generation for {len(chunks)} chunks")
            if indexing_logger:
                indexing_logger.log_info(f"Starting embedding generation for {len(chunks)} chunks")

            # Prepare content for embedding
            texts = [f"{chunk.semantic_type}: {chunk.content}" for chunk in chunks]
            logger.info(f"Prepared {len(texts)} texts for embedding")
            if indexing_logger:
                indexing_logger.log_info(f"Prepared {len(texts)} texts for embedding")

            # Generate embeddings in batches
            batch_size = self.config.EMBEDDING_BATCH_SIZE
            total_batches = (len(texts) + batch_size - 1) // batch_size
            logger.info(f"Processing {total_batches} batches of size {batch_size}")
            if indexing_logger:
                indexing_logger.log_info(f"Processing {total_batches} batches of size {batch_size}")

            for i in range(0, len(texts), batch_size):
                batch_num = i // batch_size + 1
                batch_texts = texts[i : i + batch_size]
                batch_chunks = chunks[i : i + batch_size]

                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} chunks)")
                if indexing_logger:
                    indexing_logger.log_info(
                        f"Processing embedding batch {batch_num}/{total_batches}", batch_size=len(batch_texts)
                    )
                batch_start = time.time()

                # Generate embeddings
                embeddings = self.model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)

                encode_time = (time.time() - batch_start) * 1000
                logger.info(f"Batch {batch_num} encoding completed in {encode_time:.1f}ms")
                if indexing_logger:
                    indexing_logger.log_info(f"Batch {batch_num} encoding completed", encoding_time_ms=encode_time)

                # Quantize embeddings if enabled
                if self.config.ENABLE_QUANTIZATION:
                    quant_start = time.time()
                    embeddings = self._quantize_embeddings(embeddings)
                    quant_time = (time.time() - quant_start) * 1000
                    logger.info(f"Batch {batch_num} quantization completed in {quant_time:.1f}ms")
                    if indexing_logger:
                        indexing_logger.log_info(
                            f"Batch {batch_num} quantization completed", quantization_time_ms=quant_time
                        )

                # Update chunks with embeddings
                for chunk, embedding in zip(batch_chunks, embeddings, strict=False):
                    chunk.embedding = embedding.tolist()
                    embedded_chunks.append(chunk)

                # Store in database
                store_start = time.time()
                await self._store_embeddings_batch(batch_chunks)
                store_time = (time.time() - store_start) * 1000

                batch_total_time = (time.time() - batch_start) * 1000
                logger.info(
                    f"Batch {batch_num} storage completed in {store_time:.1f}ms (total: {batch_total_time:.1f}ms)"
                )
                if indexing_logger:
                    indexing_logger.log_info(
                        f"Batch {batch_num} storage completed",
                        storage_time_ms=store_time,
                        total_batch_time_ms=batch_total_time,
                    )

            # Build BM25 index for all chunks
            logger.info("Building BM25 index for all embedded chunks")
            if indexing_logger:
                indexing_logger.log_info("Building BM25 index for all embedded chunks")
            bm25_start = time.time()
            await self.bm25_engine.build_index(embedded_chunks)
            bm25_time = (time.time() - bm25_start) * 1000
            logger.info(f"BM25 index building completed in {bm25_time:.1f}ms")
            if indexing_logger:
                indexing_logger.log_info("BM25 index building completed", bm25_build_time_ms=bm25_time)

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Generated embeddings for {len(chunks)} chunks in {processing_time:.1f}ms")
            if indexing_logger:
                indexing_logger.log_info(
                    "Complete embedding generation finished",
                    total_chunks=len(embedded_chunks),
                    total_processing_time_ms=processing_time,
                )

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
        """Store batch of embeddings in vector database using sqlite-vec."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            self.vec_loader.load_extension(conn)

            for chunk in chunks:
                if chunk.embedding is None:
                    continue

                # Calculate content hash for deduplication
                content_hash = hash(chunk.content)

                # Serialize metadata fields
                parameter_types_json = json.dumps(chunk.parameter_types) if chunk.parameter_types else None
                inheritance_chain_json = json.dumps(chunk.inheritance_chain) if chunk.inheritance_chain else None
                import_statements_json = json.dumps(chunk.import_statements) if chunk.import_statements else None
                dependencies_json = json.dumps(chunk.dependencies) if chunk.dependencies else None
                interfaces_json = json.dumps(chunk.interfaces) if chunk.interfaces else None
                decorators_json = json.dumps(chunk.decorators) if chunk.decorators else None

                # Prepare metadata
                metadata = {
                    "file_path": chunk.file_path,
                    "content": chunk.content,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                    "semantic_type": chunk.semantic_type,
                    "content_hash": str(content_hash),
                    "function_signature": chunk.function_signature,
                    "class_name": chunk.class_name,
                    "function_name": chunk.function_name,
                    "parameter_types": parameter_types_json,
                    "return_type": chunk.return_type,
                    "inheritance_chain": inheritance_chain_json,
                    "import_statements": import_statements_json,
                    "docstring": chunk.docstring,
                    "complexity_score": chunk.complexity_score,
                    "dependencies": dependencies_json,
                    "interfaces": interfaces_json,
                    "decorators": decorators_json,
                }

                # Insert using sqlite-vec operations
                self.vec_ops.insert_embedding(conn, "embeddings_vec", chunk.embedding, metadata)

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

            # Search database for similar embeddings using sqlite-vec
            results = await self._search_similar_embeddings_vec(
                query_embedding, request.language, request.max_results, request.similarity_threshold
            )

            query_time = (time.time() - start_time) * 1000

            return SearchResponse(results=results, total_matches=len(results), query_time_ms=round(query_time, 2))

        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return SearchResponse(results=[], total_matches=0, query_time_ms=0.0)

    async def _search_similar_embeddings_vec(
        self, query_embedding: np.ndarray, language_filter: str | None, max_results: int, threshold: float
    ) -> list[CodeChunk]:
        """Search for similar embeddings using sqlite-vec."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            self.vec_loader.load_extension(conn)

            # Convert similarity threshold to distance threshold (cosine distance)
            # sqlite-vec returns distance where 0 = identical, higher = less similar
            distance_threshold = 1.0 - threshold

            # Use sqlite-vec search
            results = self.vec_ops.search_similar(
                conn,
                "embeddings_vec",
                query_embedding.tolist(),
                k=max_results,
                distance_metric="cosine",
                threshold=distance_threshold,
            )

            chunks = []
            for _rowid, distance, metadata in results:
                # Apply language filter if specified
                if language_filter and metadata.get("language") != language_filter:
                    continue

                # Convert distance back to similarity score (0 distance = 1.0 similarity)
                similarity = 1.0 - distance

                # Deserialize JSON metadata fields
                parameter_types = json.loads(metadata["parameter_types"]) if metadata["parameter_types"] else None
                inheritance_chain = json.loads(metadata["inheritance_chain"]) if metadata["inheritance_chain"] else None
                import_statements = json.loads(metadata["import_statements"]) if metadata["import_statements"] else None
                dependencies = json.loads(metadata["dependencies"]) if metadata["dependencies"] else None
                interfaces = json.loads(metadata["interfaces"]) if metadata["interfaces"] else None
                decorators = json.loads(metadata["decorators"]) if metadata["decorators"] else None

                chunk = CodeChunk(
                    file_path=metadata["file_path"],
                    content=metadata["content"],
                    start_line=metadata["start_line"],
                    end_line=metadata["end_line"],
                    language=metadata["language"],
                    semantic_type=metadata["semantic_type"],
                    embedding=None,  # Don't return embedding data for search results
                    similarity=round(similarity, 3),
                    function_signature=metadata["function_signature"],
                    class_name=metadata["class_name"],
                    function_name=metadata["function_name"],
                    parameter_types=parameter_types,
                    return_type=metadata["return_type"],
                    inheritance_chain=inheritance_chain,
                    import_statements=import_statements,
                    docstring=metadata["docstring"],
                    complexity_score=metadata["complexity_score"],
                    dependencies=dependencies,
                    interfaces=interfaces,
                    decorators=decorators,
                )
                chunks.append(chunk)

            conn.close()
            return chunks

        except Exception as e:
            logger.error(f"Error searching embeddings with sqlite-vec: {e}")
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
                    FROM embeddings_vec_metadata
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
                FROM embeddings_vec_metadata
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
            total_count = conn.execute("SELECT COUNT(*) FROM embeddings_vec_metadata").fetchone()[0]

            # Get language distribution
            lang_stats = conn.execute("""
                SELECT language, COUNT(*) as count
                FROM embeddings_vec_metadata
                GROUP BY language
                ORDER BY count DESC
            """).fetchall()

            # Get semantic type distribution
            type_stats = conn.execute("""
                SELECT semantic_type, COUNT(*) as count
                FROM embeddings_vec_metadata
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
            self.vec_loader.load_extension(conn)

            # Delete embeddings for specified files
            placeholders = ",".join("?" * len(file_paths))
            # Delete from both vec table and metadata table
            conn.execute(
                f"DELETE FROM embeddings_vec WHERE rowid IN "
                f"(SELECT rowid FROM embeddings_vec_metadata WHERE file_path IN ({placeholders}))",
                file_paths,
            )
            conn.execute(f"DELETE FROM embeddings_vec_metadata WHERE file_path IN ({placeholders})", file_paths)

            conn.commit()
            conn.close()

            logger.info(f"Removed embeddings for {len(file_paths)} files")

        except Exception as e:
            logger.error(f"Error removing file embeddings: {e}")

    async def clear_all_embeddings(self) -> None:
        """Clear all embeddings from the database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            self.vec_loader.load_extension(conn)
            # Clear both vec table and metadata table
            conn.execute("DELETE FROM embeddings_vec")
            conn.execute("DELETE FROM embeddings_vec_metadata")
            conn.commit()
            conn.close()

            logger.info("Cleared all embeddings from database")

        except Exception as e:
            logger.error(f"Error clearing embeddings: {e}")
