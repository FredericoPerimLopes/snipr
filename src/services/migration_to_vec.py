#!/usr/bin/env python
"""
Migration script to convert existing embeddings to sqlite-vec format.
This script migrates data from the legacy embeddings table to the new vec0 tables.
"""

import json
import logging
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple

from ..config import get_settings
from .vector_extension import VectorExtensionLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingsMigrator:
    """Handles migration of embeddings to sqlite-vec format."""
    
    def __init__(self, dry_run: bool = False):
        """
        Initialize the migrator.
        
        Args:
            dry_run: If True, only simulate migration without making changes
        """
        self.config = get_settings()
        self.dry_run = dry_run
        self.vec_loader = VectorExtensionLoader()
        
        # Database paths
        self.legacy_db_path = self.config.VECTOR_DB_PATH
        self.vec_db_path = self.config.VEC_DB_PATH
        
        if dry_run:
            logger.info("Running in DRY RUN mode - no changes will be made")
    
    def check_prerequisites(self) -> bool:
        """Check if migration can proceed."""
        # Check legacy database exists
        if not self.legacy_db_path.exists():
            logger.error(f"Legacy database not found at {self.legacy_db_path}")
            return False
        
        # Check if vec database already exists
        if self.vec_db_path.exists() and not self.dry_run:
            logger.warning(f"Vec database already exists at {self.vec_db_path}")
            response = input("Overwrite existing vec database? (y/n): ")
            if response.lower() != 'y':
                logger.info("Migration cancelled by user")
                return False
        
        # Test sqlite-vec extension
        test_conn = sqlite3.connect(":memory:")
        if not self.vec_loader.load_extension(test_conn):
            logger.error("Failed to load sqlite-vec extension")
            test_conn.close()
            return False
        test_conn.close()
        
        logger.info("Prerequisites check passed")
        return True
    
    def count_embeddings(self) -> Tuple[int, int]:
        """Count embeddings in legacy database."""
        conn = sqlite3.connect(str(self.legacy_db_path))
        
        # Total embeddings
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        total = cursor.fetchone()[0]
        
        # Embeddings with actual vectors
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings WHERE embedding IS NOT NULL")
        with_vectors = cursor.fetchone()[0]
        
        conn.close()
        return total, with_vectors
    
    def create_vec_database(self) -> bool:
        """Create the new vec database with tables."""
        if self.dry_run:
            logger.info("[DRY RUN] Would create vec database at %s", self.vec_db_path)
            return True
        
        try:
            # Create database directory
            self.vec_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect and load extension
            conn = sqlite3.connect(str(self.vec_db_path))
            if not self.vec_loader.load_extension(conn):
                logger.error("Failed to load sqlite-vec extension")
                return False
            
            # Create vec table
            success = self.vec_loader.create_vec_table(
                conn,
                "embeddings_vec",
                dimension=self.config.VEC_DIMENSION,
                index_type=self.config.VEC_INDEX_TYPE
            )
            
            conn.close()
            
            if success:
                logger.info("Created vec database successfully")
            return success
            
        except Exception as e:
            logger.error(f"Failed to create vec database: {e}")
            return False
    
    def migrate_batch(self, conn_legacy: sqlite3.Connection, 
                     conn_vec: sqlite3.Connection,
                     offset: int, batch_size: int) -> int:
        """
        Migrate a batch of embeddings.
        
        Returns:
            Number of successfully migrated embeddings
        """
        query = """
            SELECT id, file_path, content, start_line, end_line, language,
                   semantic_type, embedding, content_hash, function_signature,
                   class_name, function_name, parameter_types, return_type,
                   inheritance_chain, import_statements, docstring,
                   complexity_score, dependencies, interfaces, decorators
            FROM embeddings
            WHERE embedding IS NOT NULL
            ORDER BY id
            LIMIT ? OFFSET ?
        """
        
        cursor = conn_legacy.execute(query, (batch_size, offset))
        rows = cursor.fetchall()
        
        migrated = 0
        for row in rows:
            try:
                # Extract embedding
                embedding_blob = row[7]
                if not embedding_blob:
                    continue
                
                # Decode from JSON
                embedding_list = json.loads(embedding_blob.decode())
                
                # Convert to bytes for sqlite-vec
                embedding_bytes = struct.pack(f'{len(embedding_list)}f', *embedding_list)
                
                # Insert into vec table
                cursor_vec = conn_vec.execute(
                    "INSERT INTO embeddings_vec(embedding) VALUES (?)",
                    (embedding_bytes,)
                )
                rowid = cursor_vec.lastrowid
                
                # Insert metadata
                metadata_values = (
                    rowid,  # rowid to link with vec table
                    row[1],  # file_path
                    row[2],  # content
                    row[3],  # start_line
                    row[4],  # end_line
                    row[5],  # language
                    row[6],  # semantic_type
                    row[8],  # content_hash
                    row[9],  # function_signature
                    row[10], # class_name
                    row[11], # function_name
                    row[12], # parameter_types
                    row[13], # return_type
                    row[14], # inheritance_chain
                    row[15], # import_statements
                    row[16], # docstring
                    row[17], # complexity_score
                    row[18], # dependencies
                    row[19], # interfaces
                    row[20], # decorators
                )
                
                conn_vec.execute("""
                    INSERT INTO embeddings_vec_metadata
                    (rowid, file_path, content, start_line, end_line, language,
                     semantic_type, content_hash, function_signature, class_name,
                     function_name, parameter_types, return_type, inheritance_chain,
                     import_statements, docstring, complexity_score, dependencies,
                     interfaces, decorators)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, metadata_values)
                
                migrated += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate embedding ID {row[0]}: {e}")
                continue
        
        if not self.dry_run:
            conn_vec.commit()
        
        return migrated
    
    def migrate(self, batch_size: int = 100) -> bool:
        """
        Perform the migration.
        
        Args:
            batch_size: Number of embeddings to migrate per batch
            
        Returns:
            True if migration succeeded, False otherwise
        """
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Count embeddings
        total, with_vectors = self.count_embeddings()
        logger.info(f"Found {total} total embeddings, {with_vectors} with vectors")
        
        if with_vectors == 0:
            logger.info("No embeddings to migrate")
            return True
        
        # Create vec database
        if not self.create_vec_database():
            return False
        
        # Open connections
        conn_legacy = sqlite3.connect(str(self.legacy_db_path))
        
        if self.dry_run:
            conn_vec = None
            logger.info("[DRY RUN] Would migrate %d embeddings", with_vectors)
        else:
            conn_vec = sqlite3.connect(str(self.vec_db_path))
            self.vec_loader.load_extension(conn_vec)
        
        # Migrate in batches
        total_migrated = 0
        offset = 0
        
        while offset < with_vectors:
            if self.dry_run:
                # Simulate batch migration
                batch_count = min(batch_size, with_vectors - offset)
                logger.info("[DRY RUN] Would migrate batch %d-%d", 
                           offset + 1, offset + batch_count)
                total_migrated += batch_count
            else:
                batch_migrated = self.migrate_batch(
                    conn_legacy, conn_vec, offset, batch_size
                )
                total_migrated += batch_migrated
                logger.info(f"Migrated batch {offset//batch_size + 1}: "
                           f"{batch_migrated} embeddings "
                           f"(total: {total_migrated}/{with_vectors})")
            
            offset += batch_size
        
        # Close connections
        conn_legacy.close()
        if conn_vec:
            conn_vec.close()
        
        # Report results
        if self.dry_run:
            logger.info(f"[DRY RUN] Would have migrated {total_migrated} embeddings")
        else:
            logger.info(f"Successfully migrated {total_migrated}/{with_vectors} embeddings")
            
            if total_migrated < with_vectors:
                logger.warning(f"Failed to migrate {with_vectors - total_migrated} embeddings")
                return False
        
        return True
    
    def verify_migration(self) -> bool:
        """Verify the migration was successful."""
        if self.dry_run:
            logger.info("[DRY RUN] Skipping verification")
            return True
        
        if not self.vec_db_path.exists():
            logger.error("Vec database does not exist")
            return False
        
        # Count embeddings in both databases
        legacy_total, legacy_with_vectors = self.count_embeddings()
        
        # Count in vec database
        conn_vec = sqlite3.connect(str(self.vec_db_path))
        self.vec_loader.load_extension(conn_vec)
        
        cursor = conn_vec.execute("SELECT COUNT(*) FROM embeddings_vec")
        vec_count = cursor.fetchone()[0]
        
        cursor = conn_vec.execute("SELECT COUNT(*) FROM embeddings_vec_metadata")
        metadata_count = cursor.fetchone()[0]
        
        conn_vec.close()
        
        logger.info(f"Legacy database: {legacy_with_vectors} embeddings with vectors")
        logger.info(f"Vec database: {vec_count} vectors, {metadata_count} metadata records")
        
        if vec_count != metadata_count:
            logger.error("Vector count doesn't match metadata count")
            return False
        
        if vec_count < legacy_with_vectors:
            logger.warning(f"Vec database has fewer embeddings ({vec_count} vs {legacy_with_vectors})")
            return False
        
        logger.info("Migration verification passed")
        return True


def main():
    """Main entry point for migration script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate existing embeddings to sqlite-vec format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without making changes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of embeddings to migrate per batch (default: 100)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing migration"
    )
    
    args = parser.parse_args()
    
    # Create migrator
    migrator = EmbeddingsMigrator(dry_run=args.dry_run)
    
    if args.verify_only:
        # Just verify
        success = migrator.verify_migration()
        sys.exit(0 if success else 1)
    
    # Perform migration
    success = migrator.migrate(batch_size=args.batch_size)
    
    if success and not args.dry_run:
        # Verify migration
        migrator.verify_migration()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()