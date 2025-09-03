"""
SQLite-vec extension loader and utilities.
Handles loading and configuration of the sqlite-vec extension.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import sqlite_vec

logger = logging.getLogger(__name__)


class VectorExtensionLoader:
    """Manages sqlite-vec extension loading and configuration."""
    
    @staticmethod
    def load_extension(conn: sqlite3.Connection, check_loaded: bool = True) -> bool:
        """
        Load sqlite-vec extension into the SQLite connection.
        
        Args:
            conn: SQLite database connection
            check_loaded: Whether to check if extension is already loaded
            
        Returns:
            True if successfully loaded or already loaded, False otherwise
        """
        try:
            # Check if extension is already loaded
            if check_loaded:
                cursor = conn.execute("SELECT vec_version()")
                version = cursor.fetchone()
                if version:
                    logger.debug(f"sqlite-vec already loaded, version: {version[0]}")
                    return True
        except sqlite3.OperationalError:
            # Extension not loaded yet, continue with loading
            pass
        
        try:
            # Enable extension loading
            conn.enable_load_extension(True)
            
            # Load sqlite-vec extension using the Python package
            sqlite_vec.load(conn)
            
            # Verify extension loaded successfully
            cursor = conn.execute("SELECT vec_version()")
            version = cursor.fetchone()
            logger.info(f"Successfully loaded sqlite-vec extension, version: {version[0]}")
            
            # Disable extension loading for security
            conn.enable_load_extension(False)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load sqlite-vec extension: {e}")
            return False
    
    @staticmethod
    def create_vec_table(
        conn: sqlite3.Connection,
        table_name: str = "embeddings_vec",
        dimension: int = 768,
        index_type: str = "flat"
    ) -> bool:
        """
        Create a vec0 virtual table for storing embeddings.
        
        Args:
            conn: SQLite database connection
            table_name: Name of the vector table
            dimension: Dimension of vectors
            index_type: Type of index (flat, ivf, hnsw)
            
        Returns:
            True if table created successfully, False otherwise
        """
        try:
            # Drop existing table if it exists
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Create vec0 virtual table
            create_sql = f"""
                CREATE VIRTUAL TABLE {table_name} USING vec0(
                    embedding float[{dimension}]
                )
            """
            conn.execute(create_sql)
            
            # Create associated metadata table
            metadata_table = f"{table_name}_metadata"
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {metadata_table} (
                    rowid INTEGER PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    language TEXT NOT NULL,
                    semantic_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(file_path, start_line, end_line)
                )
            """)
            
            # Create indexes on metadata table
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{metadata_table}_file_path ON {metadata_table}(file_path)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{metadata_table}_language ON {metadata_table}(language)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{metadata_table}_semantic_type ON {metadata_table}(semantic_type)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{metadata_table}_function_name ON {metadata_table}(function_name)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{metadata_table}_class_name ON {metadata_table}(class_name)")
            
            conn.commit()
            logger.info(f"Successfully created vec0 table '{table_name}' with dimension {dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vec0 table: {e}")
            return False
    
    @staticmethod
    def test_vec_operations(conn: sqlite3.Connection) -> bool:
        """
        Test basic vec operations to ensure extension is working.
        
        Args:
            conn: SQLite database connection
            
        Returns:
            True if all tests pass, False otherwise
        """
        try:
            # Test 1: Create temporary test table
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS test_vec USING vec0(test_embedding float[3])")
            
            # Test 2: Insert test vectors (need to convert to bytes)
            import struct
            vec1 = struct.pack('3f', 1.0, 2.0, 3.0)
            vec2 = struct.pack('3f', 4.0, 5.0, 6.0)
            conn.execute("INSERT INTO test_vec(test_embedding) VALUES (?)", (vec1,))
            conn.execute("INSERT INTO test_vec(test_embedding) VALUES (?)", (vec2,))
            
            # Test 3: Query with distance function
            query_vec = struct.pack('3f', 1.0, 2.0, 3.0)
            cursor = conn.execute("""
                SELECT rowid, vec_distance_cosine(test_embedding, ?) as distance
                FROM test_vec
                ORDER BY distance
                LIMIT 1
            """, (query_vec,))
            
            result = cursor.fetchone()
            if result:
                logger.debug(f"Vec operations test passed: rowid={result[0]}, distance={result[1]}")
            
            # Cleanup test table
            conn.execute("DROP TABLE IF EXISTS test_vec")
            conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Vec operations test failed: {e}")
            return False
    
    @staticmethod
    def get_vec_info(conn: sqlite3.Connection) -> Optional[dict]:
        """
        Get information about sqlite-vec extension and tables.
        
        Args:
            conn: SQLite database connection
            
        Returns:
            Dictionary with vec info or None if not available
        """
        try:
            info = {}
            
            # Get version
            cursor = conn.execute("SELECT vec_version()")
            info["version"] = cursor.fetchone()[0]
            
            # Get vec tables
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND sql LIKE '%vec0%'
            """)
            info["vec_tables"] = [row[0] for row in cursor.fetchall()]
            
            # Get table stats for each vec table
            table_stats = {}
            for table in info["vec_tables"]:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                table_stats[table] = count
            info["table_stats"] = table_stats
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get vec info: {e}")
            return None


class VectorOperations:
    """Utility functions for vector operations using sqlite-vec."""
    
    @staticmethod
    def insert_embedding(
        conn: sqlite3.Connection,
        table_name: str,
        embedding: list[float],
        metadata: dict
    ) -> Optional[int]:
        """
        Insert an embedding with metadata.
        
        Args:
            conn: SQLite database connection
            table_name: Name of the vec table
            embedding: The embedding vector
            metadata: Associated metadata
            
        Returns:
            The rowid of the inserted record or None if failed
        """
        try:
            # Convert embedding to bytes format for sqlite-vec
            import struct
            embedding_bytes = struct.pack(f'{len(embedding)}f', *embedding)
            
            # Insert embedding into vec table
            cursor = conn.execute(
                f"INSERT INTO {table_name}(embedding) VALUES (?)",
                (embedding_bytes,)
            )
            rowid = cursor.lastrowid
            
            # Insert metadata
            metadata_table = f"{table_name}_metadata"
            metadata["rowid"] = rowid
            
            columns = list(metadata.keys())
            placeholders = ",".join(["?" for _ in columns])
            columns_str = ",".join(columns)
            
            conn.execute(
                f"INSERT INTO {metadata_table} ({columns_str}) VALUES ({placeholders})",
                list(metadata.values())
            )
            
            return rowid
            
        except Exception as e:
            logger.error(f"Failed to insert embedding: {e}")
            return None
    
    @staticmethod
    def search_similar(
        conn: sqlite3.Connection,
        table_name: str,
        query_embedding: list[float],
        k: int = 10,
        distance_metric: str = "cosine",
        threshold: Optional[float] = None
    ) -> list[tuple]:
        """
        Search for similar embeddings using vec distance functions.
        
        Args:
            conn: SQLite database connection
            table_name: Name of the vec table
            query_embedding: The query embedding vector
            k: Number of results to return
            distance_metric: Distance metric (cosine, l2, dot)
            threshold: Optional similarity threshold
            
        Returns:
            List of (rowid, distance, metadata) tuples
        """
        try:
            # Map distance metric to function name
            distance_functions = {
                "cosine": "vec_distance_cosine",
                "l2": "vec_distance_l2",
                "dot": "-vec_distance_dot"  # Negative for similarity
            }
            
            distance_func = distance_functions.get(distance_metric, "vec_distance_cosine")
            
            # Convert query embedding to bytes format for sqlite-vec
            import struct
            query_bytes = struct.pack(f'{len(query_embedding)}f', *query_embedding)
            
            # Build query
            metadata_table = f"{table_name}_metadata"
            
            query = f"""
                SELECT
                    v.rowid,
                    {distance_func}(v.embedding, ?) as distance,
                    m.*
                FROM {table_name} v
                JOIN {metadata_table} m ON v.rowid = m.rowid
            """
            
            if threshold is not None:
                if distance_metric == "cosine":
                    # For cosine, lower is better (0 = identical)
                    query += f" WHERE {distance_func}(v.embedding, ?) <= {threshold}"
                else:
                    query += f" WHERE {distance_func}(v.embedding, ?) <= {threshold}"
            
            query += f" ORDER BY distance LIMIT {k}"
            
            # Execute query
            if threshold is not None:
                cursor = conn.execute(query, (query_bytes, query_bytes))
            else:
                cursor = conn.execute(query, (query_bytes,))
            
            # Fetch results
            results = []
            for row in cursor.fetchall():
                rowid = row[0]
                distance = row[1]
                # Convert row to dict for metadata
                metadata = dict(zip([d[0] for d in cursor.description[2:]], row[2:]))
                results.append((rowid, distance, metadata))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {e}")
            return []