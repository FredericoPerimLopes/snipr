import logging
import sqlite3

from ..config import get_settings

logger = logging.getLogger(__name__)


class DatabaseMigration:
    def __init__(self):
        self.config = get_settings()
        self.db_path = self.config.VECTOR_DB_PATH

    async def migrate_to_metadata_schema(self) -> bool:
        """Migrate existing database to include metadata fields."""
        if not self.db_path.exists():
            logger.info("No existing database found - new schema will be created")
            return True

        try:
            conn = sqlite3.connect(str(self.db_path))

            # Check if metadata columns already exist
            cursor = conn.execute("PRAGMA table_info(embeddings)")
            columns = [row[1] for row in cursor.fetchall()]

            metadata_columns = [
                "function_signature",
                "class_name",
                "function_name",
                "parameter_types",
                "return_type",
                "inheritance_chain",
                "import_statements",
                "docstring",
                "complexity_score",
                "dependencies",
                "interfaces",
                "decorators",
            ]

            # Add missing columns
            for column in metadata_columns:
                if column not in columns:
                    if column == "complexity_score":
                        conn.execute(f"ALTER TABLE embeddings ADD COLUMN {column} INTEGER")
                    else:
                        conn.execute(f"ALTER TABLE embeddings ADD COLUMN {column} TEXT")
                    logger.info(f"Added column: {column}")

            # Create metadata indexes if they don't exist
            conn.execute("CREATE INDEX IF NOT EXISTS idx_function_name ON embeddings(function_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_class_name ON embeddings(class_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_complexity ON embeddings(complexity_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_return_type ON embeddings(return_type)")

            conn.commit()
            conn.close()

            logger.info("Database migration completed successfully")
            return True

        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            return False
