import json
import logging
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from ..config import get_settings
from ..models.indexing_models import CodeChunk

logger = logging.getLogger(__name__)


class BM25SearchEngine:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b  # Length normalization parameter
        self.config = get_settings()
        self.db_path = self.config.INDEX_CACHE_DIR / "bm25_index.db"
        self._init_database()

    def _init_database(self) -> None:
        """Initialize BM25 index database."""
        self.config.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bm25_index (
                    term TEXT PRIMARY KEY,
                    document_frequency INTEGER,
                    posting_list TEXT  -- JSON list of [doc_id, term_frequency]
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_stats (
                    document_id TEXT PRIMARY KEY,
                    document_length INTEGER,
                    file_path TEXT,
                    language TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_bm25_term ON bm25_index(term)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_stats ON document_stats(document_id)")

    async def build_index(self, chunks: list[CodeChunk]) -> None:
        """Build BM25 inverted index from code chunks."""
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        # Clear existing index
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM bm25_index")
            conn.execute("DELETE FROM document_stats")
            conn.execute("DELETE FROM index_metadata")

        # Build inverted index
        term_doc_frequencies = defaultdict(dict)  # term -> {doc_id: frequency}
        document_lengths = {}
        total_documents = len(chunks)

        for chunk in chunks:
            doc_id = f"{chunk.file_path}:{chunk.start_line}"

            # Tokenize and count terms
            terms = self._tokenize_code(chunk.content, chunk.language)
            term_counts = Counter(terms)
            document_lengths[doc_id] = len(terms)

            # Update term frequencies
            for term, freq in term_counts.items():
                term_doc_frequencies[term][doc_id] = freq

        # Calculate average document length
        avg_doc_length = sum(document_lengths.values()) / len(document_lengths) if document_lengths else 0

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            # Store document stats
            for doc_id, length in document_lengths.items():
                file_path = doc_id.split(":")[0]
                language = self._detect_language_from_path(file_path)
                conn.execute(
                    "INSERT INTO document_stats "
                    "(document_id, document_length, file_path, language) "
                    "VALUES (?, ?, ?, ?)",
                    (doc_id, length, file_path, language),
                )

            # Store inverted index
            for term, doc_frequencies in term_doc_frequencies.items():
                document_frequency = len(doc_frequencies)
                posting_list = json.dumps([[doc_id, freq] for doc_id, freq in doc_frequencies.items()])

                conn.execute(
                    "INSERT INTO bm25_index (term, document_frequency, posting_list) VALUES (?, ?, ?)",
                    (term, document_frequency, posting_list),
                )

            # Store metadata
            conn.execute(
                "INSERT INTO index_metadata (key, value) VALUES (?, ?)", ("total_documents", str(total_documents))
            )
            conn.execute(
                "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
                ("average_document_length", str(avg_doc_length)),
            )

        logger.info(f"BM25 index built with {len(term_doc_frequencies)} unique terms")

    async def search(self, query: str, language: str = None, max_results: int = 50) -> list[tuple[str, float]]:
        """Search using BM25 algorithm."""
        query_terms = self._tokenize_query(query, language)
        if not query_terms:
            return []

        # Get index metadata
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM index_metadata WHERE key = ?", ("total_documents",))
            result = cursor.fetchone()
            total_documents = int(result[0]) if result else 0

            cursor = conn.execute("SELECT value FROM index_metadata WHERE key = ?", ("average_document_length",))
            result = cursor.fetchone()
            avg_doc_length = float(result[0]) if result else 0

        if total_documents == 0:
            return []

        # Calculate BM25 scores
        document_scores = defaultdict(float)

        with sqlite3.connect(self.db_path) as conn:
            for term in query_terms:
                # Get term statistics
                cursor = conn.execute("SELECT document_frequency, posting_list FROM bm25_index WHERE term = ?", (term,))
                result = cursor.fetchone()

                if not result:
                    continue  # Term not in index

                doc_frequency, posting_list_json = result
                posting_list = json.loads(posting_list_json)

                # Calculate IDF
                idf = math.log((total_documents - doc_frequency + 0.5) / (doc_frequency + 0.5) + 1)

                # Score each document containing this term
                for doc_id, term_freq in posting_list:
                    # Get document length
                    cursor = conn.execute("SELECT document_length FROM document_stats WHERE document_id = ?", (doc_id,))
                    doc_length_result = cursor.fetchone()
                    if not doc_length_result:
                        continue

                    doc_length = doc_length_result[0]

                    # Apply language filter
                    if language:
                        cursor = conn.execute("SELECT language FROM document_stats WHERE document_id = ?", (doc_id,))
                        lang_result = cursor.fetchone()
                        if not lang_result or lang_result[0] != language:
                            continue

                    # Calculate BM25 score for this term in this document
                    normalized_tf = (term_freq * (self.k1 + 1)) / (
                        term_freq + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
                    )

                    document_scores[doc_id] += idf * normalized_tf

        # Sort by score and return top results
        sorted_results = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:max_results]

    def _tokenize_code(self, content: str, language: str) -> list[str]:
        """Tokenize code content for BM25 indexing."""
        tokenizer = CodeTokenizer()
        return tokenizer.tokenize(content, language)

    def _tokenize_query(self, query: str, language: str = None) -> list[str]:
        """Tokenize search query."""
        # Use same tokenization as code content
        return self._tokenize_code(query, language or "text")

    def _detect_language_from_path(self, file_path: str) -> str:
        """Detect language from file extension."""
        path = Path(file_path)
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
        }
        return extension_map.get(path.suffix.lower(), "text")

    async def update_index(self, added_chunks: list[CodeChunk], removed_doc_ids: list[str]) -> None:
        """Incrementally update BM25 index."""
        logger.info(f"Updating BM25 index: +{len(added_chunks)} chunks, -{len(removed_doc_ids)} docs")

        with sqlite3.connect(self.db_path) as conn:
            # Remove old documents
            for doc_id in removed_doc_ids:
                # Remove from document stats
                conn.execute("DELETE FROM document_stats WHERE document_id = ?", (doc_id,))

                # Update inverted index (remove doc from posting lists)
                cursor = conn.execute("SELECT term, posting_list FROM bm25_index")
                for term, posting_list_json in cursor.fetchall():
                    posting_list = json.loads(posting_list_json)
                    updated_list = [[did, freq] for did, freq in posting_list if did != doc_id]

                    if len(updated_list) != len(posting_list):
                        if updated_list:
                            conn.execute(
                                "UPDATE bm25_index SET document_frequency = ?, posting_list = ? WHERE term = ?",
                                (len(updated_list), json.dumps(updated_list), term),
                            )
                        else:
                            # Remove term if no documents contain it
                            conn.execute("DELETE FROM bm25_index WHERE term = ?", (term,))

        # Add new chunks to index
        if added_chunks:
            await self._add_chunks_to_index(added_chunks)

    async def _add_chunks_to_index(self, chunks: list[CodeChunk]) -> None:
        """Add new chunks to existing BM25 index."""
        with sqlite3.connect(self.db_path) as conn:
            for chunk in chunks:
                doc_id = f"{chunk.file_path}:{chunk.start_line}"

                # Tokenize and count terms
                terms = self._tokenize_code(chunk.content, chunk.language)
                term_counts = Counter(terms)
                doc_length = len(terms)

                # Store document stats
                conn.execute(
                    "INSERT OR REPLACE INTO document_stats "
                    "(document_id, document_length, file_path, language) "
                    "VALUES (?, ?, ?, ?)",
                    (doc_id, doc_length, chunk.file_path, chunk.language),
                )

                # Update inverted index
                for term, freq in term_counts.items():
                    # Check if term exists
                    cursor = conn.execute(
                        "SELECT document_frequency, posting_list FROM bm25_index WHERE term = ?", (term,)
                    )
                    result = cursor.fetchone()

                    if result:
                        # Update existing term
                        doc_frequency, posting_list_json = result
                        posting_list = json.loads(posting_list_json)

                        # Add or update document in posting list
                        updated = False
                        for i, (did, _f) in enumerate(posting_list):
                            if did == doc_id:
                                posting_list[i] = [doc_id, freq]
                                updated = True
                                break

                        if not updated:
                            posting_list.append([doc_id, freq])
                            doc_frequency += 1

                        conn.execute(
                            "UPDATE bm25_index SET document_frequency = ?, posting_list = ? WHERE term = ?",
                            (doc_frequency, json.dumps(posting_list), term),
                        )
                    else:
                        # New term
                        posting_list = [[doc_id, freq]]
                        conn.execute(
                            "INSERT INTO bm25_index (term, document_frequency, posting_list) VALUES (?, ?, ?)",
                            (term, 1, json.dumps(posting_list)),
                        )

            # Update metadata
            cursor = conn.execute("SELECT COUNT(*) FROM document_stats")
            total_docs = cursor.fetchone()[0]

            cursor = conn.execute("SELECT AVG(document_length) FROM document_stats")
            avg_length = cursor.fetchone()[0] or 0

            conn.execute(
                "INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?, ?)", ("total_documents", str(total_docs))
            )
            conn.execute(
                "INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?, ?)",
                ("average_document_length", str(avg_length)),
            )

    async def get_index_stats(self) -> dict[str, any]:
        """Get BM25 index statistics."""
        if not self.db_path.exists():
            return {"indexed": False}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM bm25_index")
            total_terms = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM document_stats")
            total_docs = cursor.fetchone()[0]

            cursor = conn.execute("SELECT AVG(document_length) FROM document_stats")
            avg_length = cursor.fetchone()[0] or 0

            return {
                "indexed": True,
                "total_terms": total_terms,
                "total_documents": total_docs,
                "average_document_length": avg_length,
                "index_size_kb": self.db_path.stat().st_size / 1024,
            }


class CodeTokenizer:
    """Advanced tokenizer for code content."""

    def __init__(self):
        # Code-specific stop words
        self.stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "if",
            "else",
            "elif",
            "while",
            "def",
            "class",
            "import",
            "from",
            "as",
            "try",
            "except",
            "finally",
            "pass",
            "break",
            "continue",
            "return",
            "function",
            "var",
            "let",
            "const",
            "new",
            "this",
            "super",
            "extends",
            "implements",
            "public",
            "private",
            "protected",
            "static",
            "abstract",
            "final",
            "override",
        }

        # Language-specific keywords to exclude
        self.language_keywords = {
            "python": {
                "def",
                "class",
                "if",
                "else",
                "elif",
                "while",
                "for",
                "try",
                "except",
                "finally",
                "with",
                "as",
                "import",
                "from",
            },
            "javascript": {
                "function",
                "var",
                "let",
                "const",
                "if",
                "else",
                "while",
                "for",
                "try",
                "catch",
                "finally",
                "class",
                "extends",
            },
            "typescript": {
                "function",
                "var",
                "let",
                "const",
                "if",
                "else",
                "while",
                "for",
                "try",
                "catch",
                "finally",
                "class",
                "extends",
                "interface",
                "type",
            },
            "java": {
                "public",
                "private",
                "protected",
                "static",
                "final",
                "abstract",
                "class",
                "interface",
                "extends",
                "implements",
                "if",
                "else",
                "while",
                "for",
                "try",
                "catch",
            },
            "go": {
                "func",
                "var",
                "const",
                "type",
                "struct",
                "interface",
                "if",
                "else",
                "for",
                "switch",
                "case",
                "default",
                "go",
                "defer",
            },
            "rust": {
                "fn",
                "let",
                "mut",
                "const",
                "struct",
                "enum",
                "trait",
                "impl",
                "if",
                "else",
                "while",
                "for",
                "match",
                "loop",
            },
        }

    def tokenize(self, content: str, language: str = None) -> list[str]:
        """Advanced code-aware tokenization."""
        # Remove comments and strings to focus on code structure
        cleaned_content = self._remove_comments_and_strings(content, language)

        # Split on various delimiters (preserve case for compound identifier splitting)
        tokens = re.split(r"[\s\(\)\[\]\{\},;\.:\+\-\*\/\=\<\>\!\&\|\^\%\#\@\$\~\?]+", cleaned_content)

        # Remove empty tokens
        tokens = [t for t in tokens if t and len(t) > 1]

        # Advanced token processing
        processed_tokens = []
        for token in tokens:
            # Split compound identifiers (preserving original case)
            subtokens = self._split_compound_identifier(token)
            # Convert to lowercase after compound splitting
            processed_tokens.extend([t.lower() for t in subtokens])

        # Filter tokens
        language_stops = self.language_keywords.get(language, set())
        final_tokens = []

        for token in processed_tokens:
            if (
                len(token) > 2
                and token not in self.stop_words
                and token not in language_stops
                and token.isalnum()
                and not token.isdigit()
            ):
                final_tokens.append(token)

        return final_tokens

    def _split_compound_identifier(self, identifier: str) -> list[str]:
        """Split camelCase, snake_case, and kebab-case identifiers."""
        tokens = []

        # Always include the original identifier
        tokens.append(identifier)

        # Split snake_case and kebab-case
        if "_" in identifier or "-" in identifier:
            parts = re.split(r"[_-]", identifier)
            tokens.extend([p for p in parts if p])

        # Split camelCase and PascalCase
        camel_parts = re.findall(r"[a-z]+|[A-Z][a-z]*|[A-Z]+(?=[A-Z][a-z]|\b)", identifier)
        if len(camel_parts) > 1:
            tokens.extend([part.lower() for part in camel_parts])

        return tokens

    def _remove_comments_and_strings(self, content: str, language: str = None) -> str:
        """Remove comments and string literals to focus on code structure."""
        try:
            if language == "python":
                # Remove Python comments and strings
                content = re.sub(r"#.*$", "", content, flags=re.MULTILINE)
                content = re.sub(r'""".*?"""', "", content, flags=re.DOTALL)
                content = re.sub(r"'''.*?'''", "", content, flags=re.DOTALL)
                content = re.sub(r'"[^"]*"', "", content)
                content = re.sub(r"'[^']*'", "", content)
            elif language in ["javascript", "typescript"]:
                # Remove JS/TS comments and strings
                content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
                content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
                content = re.sub(r'"[^"]*"', "", content)
                content = re.sub(r"'[^']*'", "", content)
                content = re.sub(r"`[^`]*`", "", content)
            elif language == "java":
                # Remove Java comments and strings
                content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
                content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
                content = re.sub(r'"[^"]*"', "", content)
            elif language == "go":
                # Remove Go comments
                content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
                content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
                content = re.sub(r'"[^"]*"', "", content)
                content = re.sub(r"`[^`]*`", "", content)
            elif language == "rust":
                # Remove Rust comments
                content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
                content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
                content = re.sub(r'"[^"]*"', "", content)

            return content
        except Exception:
            return content
