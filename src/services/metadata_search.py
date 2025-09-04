import json
import logging
import re
import sqlite3

from ..config import get_settings
from ..models.indexing_models import CodeChunk

logger = logging.getLogger(__name__)


class MetadataSearchEngine:
    def __init__(self):
        self.config = get_settings()
        self.db_path = self.config.VECTOR_DB_PATH

    async def search(self, query: str, language: str = None, max_results: int = 50) -> list[CodeChunk]:
        """Main metadata search entry point."""
        # Parse and route query to appropriate search method
        parsed_query = self._parse_metadata_query(query)

        if parsed_query["type"] == "function":
            return await self.search_functions(
                function_name=parsed_query.get("function_name"),
                return_type=parsed_query.get("return_type"),
                parameter_types=parsed_query.get("parameter_types"),
                language=language,
                max_results=max_results,
            )
        elif parsed_query["type"] == "class":
            return await self.search_classes(
                class_name=parsed_query.get("class_name"),
                inherits_from=parsed_query.get("inherits_from"),
                implements=parsed_query.get("implements"),
                language=language,
                max_results=max_results,
            )
        elif parsed_query["type"] == "import":
            return await self.search_by_imports(
                import_module=parsed_query.get("import_module"),
                uses_function=parsed_query.get("uses_function"),
                language=language,
                max_results=max_results,
            )
        else:
            return await self.search_general_metadata(query, language, max_results)

    async def search_functions(
        self,
        function_name: str = None,
        return_type: str = None,
        parameter_types: list[str] = None,
        language: str = None,
        max_results: int = 50,
    ) -> list[CodeChunk]:
        """Search for functions by metadata."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            # Build dynamic query
            where_clauses = []
            params = []

            if function_name:
                where_clauses.append("function_name LIKE ?")
                params.append(f"%{function_name}%")

            if return_type:
                where_clauses.append("return_type LIKE ?")
                params.append(f"%{return_type}%")

            if parameter_types:
                for param_type in parameter_types:
                    where_clauses.append("parameter_types LIKE ?")
                    params.append(f"%{param_type}%")

            if language:
                where_clauses.append("language = ?")
                params.append(language)

            # Only search function-like semantic types
            where_clauses.append(
                "semantic_type IN ('function_definition', 'async_function_definition', 'method_definition')"
            )

            query = f"""
                SELECT file_path, content, start_line, end_line, language, semantic_type,
                       function_signature, class_name, function_name, parameter_types, return_type,
                       inheritance_chain, import_statements, docstring, complexity_score,
                       dependencies, interfaces, decorators
                FROM embeddings_vec_metadata
                WHERE {" AND ".join(where_clauses)}
                ORDER BY complexity_score ASC
                LIMIT ?
            """

            params.append(max_results)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            chunks = []
            for row in rows:
                chunk = self._row_to_chunk(row)
                if chunk:
                    chunks.append(chunk)

            conn.close()
            logger.debug(f"Function search returned {len(chunks)} results")
            return chunks

        except Exception as e:
            logger.error(f"Error in function search: {e}")
            return []

    async def search_classes(
        self,
        class_name: str = None,
        inherits_from: str = None,
        implements: list[str] = None,
        language: str = None,
        max_results: int = 50,
    ) -> list[CodeChunk]:
        """Search for classes by metadata."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            where_clauses = []
            params = []

            if class_name:
                where_clauses.append("class_name LIKE ?")
                params.append(f"%{class_name}%")

            if inherits_from:
                where_clauses.append("inheritance_chain LIKE ?")
                params.append(f"%{inherits_from}%")

            if implements:
                for interface in implements:
                    where_clauses.append("interfaces LIKE ?")
                    params.append(f"%{interface}%")

            if language:
                where_clauses.append("language = ?")
                params.append(language)

            # Only search class-like semantic types
            where_clauses.append("semantic_type IN ('class_definition', 'class_declaration', 'interface_declaration')")

            query = f"""
                SELECT file_path, content, start_line, end_line, language, semantic_type,
                       function_signature, class_name, function_name, parameter_types, return_type,
                       inheritance_chain, import_statements, docstring, complexity_score,
                       dependencies, interfaces, decorators
                FROM embeddings_vec_metadata
                WHERE {" AND ".join(where_clauses)}
                LIMIT ?
            """

            params.append(max_results)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            chunks = []
            for row in rows:
                chunk = self._row_to_chunk(row)
                if chunk:
                    chunks.append(chunk)

            conn.close()
            logger.debug(f"Class search returned {len(chunks)} results")
            return chunks

        except Exception as e:
            logger.error(f"Error in class search: {e}")
            return []

    async def search_by_imports(
        self, import_module: str = None, uses_function: str = None, language: str = None, max_results: int = 50
    ) -> list[CodeChunk]:
        """Search code by import dependencies."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            where_clauses = []
            params = []

            if import_module:
                where_clauses.append("(import_statements LIKE ? OR dependencies LIKE ?)")
                params.extend([f"%{import_module}%", f"%{import_module}%"])

            if uses_function:
                where_clauses.append("content LIKE ?")
                params.append(f"%{uses_function}%")

            if language:
                where_clauses.append("language = ?")
                params.append(language)

            query = f"""
                SELECT file_path, content, start_line, end_line, language, semantic_type,
                       function_signature, class_name, function_name, parameter_types, return_type,
                       inheritance_chain, import_statements, docstring, complexity_score,
                       dependencies, interfaces, decorators
                FROM embeddings_vec_metadata
                WHERE {" AND ".join(where_clauses) if where_clauses else "1=1"}
                LIMIT ?
            """

            params.append(max_results)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            chunks = []
            for row in rows:
                chunk = self._row_to_chunk(row)
                if chunk:
                    chunks.append(chunk)

            conn.close()
            logger.debug(f"Import search returned {len(chunks)} results")
            return chunks

        except Exception as e:
            logger.error(f"Error in import search: {e}")
            return []

    async def search_general_metadata(self, query: str, language: str = None, max_results: int = 50) -> list[CodeChunk]:
        """General metadata search across all fields."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            # Search across multiple metadata fields
            where_clause = """
                (function_name LIKE ? OR class_name LIKE ? OR
                 function_signature LIKE ? OR docstring LIKE ? OR
                 return_type LIKE ? OR import_statements LIKE ?)
            """

            search_term = f"%{query}%"
            params = [search_term] * 6

            if language:
                where_clause += " AND language = ?"
                params.append(language)

            query_sql = f"""
                SELECT file_path, content, start_line, end_line, language, semantic_type,
                       function_signature, class_name, function_name, parameter_types, return_type,
                       inheritance_chain, import_statements, docstring, complexity_score,
                       dependencies, interfaces, decorators
                FROM embeddings_vec_metadata
                WHERE {where_clause}
                ORDER BY
                    CASE
                        WHEN function_name LIKE ? THEN 1
                        WHEN class_name LIKE ? THEN 2
                        WHEN function_signature LIKE ? THEN 3
                        ELSE 4
                    END
                LIMIT ?
            """

            # Add ranking parameters
            params.extend([search_term, search_term, search_term, max_results])

            cursor = conn.execute(query_sql, params)
            rows = cursor.fetchall()

            chunks = []
            for row in rows:
                chunk = self._row_to_chunk(row)
                if chunk:
                    chunks.append(chunk)

            conn.close()
            logger.debug(f"General metadata search returned {len(chunks)} results")
            return chunks

        except Exception as e:
            logger.error(f"Error in general metadata search: {e}")
            return []

    def _parse_metadata_query(self, query: str) -> dict[str, any]:
        """Parse natural language query to extract metadata search criteria."""
        query_lower = query.lower()
        parsed = {"type": "general"}

        # Function queries
        if re.search(r"function.*return.*(\w+)", query_lower):
            parsed["type"] = "function"
            return_match = re.search(r"return.*?(\w+)", query_lower)
            if return_match:
                parsed["return_type"] = return_match.group(1)

        elif re.search(r"function.*(\w+)", query_lower):
            parsed["type"] = "function"
            func_match = re.search(r"function.*?(\w+)", query_lower)
            if func_match:
                parsed["function_name"] = func_match.group(1)

        # Class queries
        elif re.search(r"class.*inherit.*(\w+)", query_lower):
            parsed["type"] = "class"
            inherit_match = re.search(r"inherit.*?(\w+)", query_lower)
            if inherit_match:
                parsed["inherits_from"] = inherit_match.group(1)

        elif re.search(r"class.*(\w+)", query_lower):
            parsed["type"] = "class"
            class_match = re.search(r"class.*?(\w+)", query_lower)
            if class_match:
                parsed["class_name"] = class_match.group(1)

        # Import queries
        elif re.search(r"import.*(\w+)", query_lower):
            parsed["type"] = "import"
            import_match = re.search(r"import.*?(\w+)", query_lower)
            if import_match:
                parsed["import_module"] = import_match.group(1)

        return parsed

    def _row_to_chunk(self, row) -> CodeChunk | None:
        """Convert database row to CodeChunk with metadata."""
        try:
            # Deserialize JSON metadata fields
            parameter_types = json.loads(row[9]) if row[9] else None
            inheritance_chain = json.loads(row[11]) if row[11] else None
            import_statements = json.loads(row[12]) if row[12] else None
            dependencies = json.loads(row[15]) if row[15] else None
            interfaces = json.loads(row[16]) if row[16] else None
            decorators = json.loads(row[17]) if row[17] else None

            return CodeChunk(
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
        except Exception as e:
            logger.debug(f"Error converting row to chunk: {e}")
            return None

    async def search_by_complexity(
        self, min_complexity: int = None, max_complexity: int = None, language: str = None
    ) -> list[CodeChunk]:
        """Search functions by complexity score."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            where_clauses = ["complexity_score IS NOT NULL"]
            params = []

            if min_complexity is not None:
                where_clauses.append("complexity_score >= ?")
                params.append(min_complexity)

            if max_complexity is not None:
                where_clauses.append("complexity_score <= ?")
                params.append(max_complexity)

            if language:
                where_clauses.append("language = ?")
                params.append(language)

            query = f"""
                SELECT file_path, content, start_line, end_line, language, semantic_type,
                       function_signature, class_name, function_name, parameter_types, return_type,
                       inheritance_chain, import_statements, docstring, complexity_score,
                       dependencies, interfaces, decorators
                FROM embeddings_vec_metadata
                WHERE {" AND ".join(where_clauses)}
                ORDER BY complexity_score DESC
                LIMIT 50
            """

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            chunks = []
            for row in rows:
                chunk = self._row_to_chunk(row)
                if chunk:
                    chunks.append(chunk)

            conn.close()
            return chunks

        except Exception as e:
            logger.error(f"Error in complexity search: {e}")
            return []

    async def search_similar_functions(self, signature: str, language: str = None) -> list[CodeChunk]:
        """Find functions with similar signatures."""
        try:
            # Extract key components from signature
            sig_tokens = self._tokenize_signature(signature)

            conn = sqlite3.connect(str(self.db_path))

            where_clauses = [
                "semantic_type IN ('function_definition', 'async_function_definition', 'method_definition')"
            ]
            params = []

            # Search for functions with similar signature components
            if sig_tokens:
                token_conditions = []
                for token in sig_tokens[:3]:  # Limit to top 3 tokens
                    token_conditions.append("function_signature LIKE ?")
                    params.append(f"%{token}%")

                if token_conditions:
                    where_clauses.append(f"({' OR '.join(token_conditions)})")

            if language:
                where_clauses.append("language = ?")
                params.append(language)

            query = f"""
                SELECT file_path, content, start_line, end_line, language, semantic_type,
                       function_signature, class_name, function_name, parameter_types, return_type,
                       inheritance_chain, import_statements, docstring, complexity_score,
                       dependencies, interfaces, decorators
                FROM embeddings_vec_metadata
                WHERE {" AND ".join(where_clauses)}
                LIMIT 50
            """

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            chunks = []
            for row in rows:
                chunk = self._row_to_chunk(row)
                if chunk:
                    chunks.append(chunk)

            conn.close()
            return chunks

        except Exception as e:
            logger.error(f"Error in similar function search: {e}")
            return []

    def _tokenize_signature(self, signature: str) -> list[str]:
        """Extract meaningful tokens from function signature."""
        # Remove common keywords and punctuation
        cleaned = re.sub(r"[(),\[\]{}<>]", " ", signature.lower())
        tokens = cleaned.split()

        # Filter out common keywords
        stop_words = {"def", "function", "async", "public", "private", "static", "void", "const", "let", "var"}
        meaningful_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

        return meaningful_tokens

    async def get_inheritance_tree(self, class_name: str, language: str = None) -> dict[str, any]:
        """Build complete inheritance hierarchy for a class."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            where_clauses = ["class_name = ?"]
            params = [class_name]

            if language:
                where_clauses.append("language = ?")
                params.append(language)

            query = f"""
                SELECT class_name, inheritance_chain, interfaces
                FROM embeddings_vec_metadata
                WHERE {" AND ".join(where_clauses)}
                AND semantic_type IN ('class_definition', 'class_declaration')
                LIMIT 1
            """

            cursor = conn.execute(query, params)
            row = cursor.fetchone()

            if not row:
                return {"class_name": class_name, "parents": [], "interfaces": []}

            inheritance_chain = json.loads(row[1]) if row[1] else []
            interfaces = json.loads(row[2]) if row[2] else []

            # Recursively find parent class information
            parent_info = {}
            for parent in inheritance_chain:
                parent_tree = await self.get_inheritance_tree(parent, language)
                parent_info[parent] = parent_tree

            conn.close()

            return {
                "class_name": class_name,
                "parents": inheritance_chain,
                "interfaces": interfaces,
                "parent_trees": parent_info,
            }

        except Exception as e:
            logger.error(f"Error building inheritance tree: {e}")
            return {"class_name": class_name, "parents": [], "interfaces": []}

    async def get_dependency_graph(self, file_path: str) -> dict[str, any]:
        """Build dependency relationships for a file."""
        try:
            conn = sqlite3.connect(str(self.db_path))

            # Get all chunks from the file
            cursor = conn.execute(
                """
                SELECT import_statements, dependencies, function_name, class_name
                FROM embeddings_vec_metadata
                WHERE file_path = ?
                """,
                (file_path,),
            )

            rows = cursor.fetchall()

            all_imports = set()
            all_dependencies = set()
            functions = set()
            classes = set()

            for row in rows:
                if row[0]:  # import_statements
                    imports = json.loads(row[0])
                    all_imports.update(imports)

                if row[1]:  # dependencies
                    deps = json.loads(row[1])
                    all_dependencies.update(deps)

                if row[2]:  # function_name
                    functions.add(row[2])

                if row[3]:  # class_name
                    classes.add(row[3])

            conn.close()

            return {
                "file_path": file_path,
                "imports": list(all_imports),
                "dependencies": list(all_dependencies),
                "functions": list(functions),
                "classes": list(classes),
            }

        except Exception as e:
            logger.error(f"Error building dependency graph: {e}")
            return {"file_path": file_path, "imports": [], "dependencies": [], "functions": [], "classes": []}
