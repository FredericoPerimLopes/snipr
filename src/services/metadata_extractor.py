import logging
from dataclasses import dataclass
from pathlib import Path

try:
    import tree_sitter_go as tsgo
    import tree_sitter_java as tsjava
    import tree_sitter_javascript as tsjs
    import tree_sitter_python as tspython
    import tree_sitter_rust as tsrust
    import tree_sitter_typescript as tsts
except ImportError:
    tspython = tsjs = tsts = tsgo = tsrust = tsjava = None

from tree_sitter import Language, Node, Parser

logger = logging.getLogger(__name__)


@dataclass
class ExtractedMetadata:
    semantic_type: str
    function_signature: str | None = None
    class_name: str | None = None
    function_name: str | None = None
    parameter_types: list[str] | None = None
    return_type: str | None = None
    inheritance_chain: list[str] | None = None
    import_statements: list[str] | None = None
    docstring: str | None = None
    complexity_score: int | None = None
    dependencies: list[str] | None = None
    interfaces: list[str] | None = None
    decorators: list[str] | None = None


class MetadataExtractor:
    def __init__(self):
        self.parsers: dict[str, Parser] = {}
        self.languages: dict[str, Language] = {}
        self._init_parsers()

    def _init_parsers(self) -> None:
        """Initialize Tree-sitter parsers for supported languages."""
        language_mappings = {
            "python": (tspython, "python"),
            "javascript": (tsjs, "javascript"),
            "typescript": (tsts, "typescript"),
            "go": (tsgo, "go"),
            "rust": (tsrust, "rust"),
            "java": (tsjava, "java"),
        }

        for lang_name, (module, _tree_name) in language_mappings.items():
            if module is None:
                continue

            try:
                language = Language(module.language())
                parser = Parser()
                parser.language = language

                self.languages[lang_name] = language
                self.parsers[lang_name] = parser
                logger.debug(f"Initialized metadata parser for {lang_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize {lang_name} metadata parser: {e}")

    async def extract_all_metadata(self, node: Node, content: str, language: str) -> ExtractedMetadata:
        """Extract comprehensive metadata from AST node."""
        metadata = ExtractedMetadata(semantic_type=node.type)

        if language == "python":
            await self._extract_python_metadata(node, content, metadata)
        elif language in ["javascript", "typescript"]:
            await self._extract_js_ts_metadata(node, content, metadata)
        elif language == "java":
            await self._extract_java_metadata(node, content, metadata)
        elif language == "go":
            await self._extract_go_metadata(node, content, metadata)
        elif language == "rust":
            await self._extract_rust_metadata(node, content, metadata)

        return metadata

    async def _extract_python_metadata(self, node: Node, content: str, metadata: ExtractedMetadata) -> None:
        """Extract Python-specific metadata using AST and Tree-sitter."""
        if node.type == "function_definition":
            metadata.function_name = self._get_node_text(node, "identifier", content)
            metadata.function_signature = self._extract_python_function_signature(node, content)
            metadata.parameter_types = self._extract_python_parameter_types(node, content)
            metadata.return_type = self._extract_python_return_type(node, content)
            metadata.decorators = self._extract_python_decorators(node, content)
            metadata.docstring = self._extract_python_docstring(node, content)
            metadata.complexity_score = self._calculate_complexity(node)

        elif node.type == "class_definition":
            metadata.class_name = self._get_node_text(node, "identifier", content)
            metadata.inheritance_chain = self._extract_python_inheritance(node, content)
            metadata.docstring = self._extract_python_docstring(node, content)

        # Extract import statements for any node type
        metadata.import_statements = self._extract_import_statements(content, "python")

    async def _extract_js_ts_metadata(self, node: Node, content: str, metadata: ExtractedMetadata) -> None:
        """Extract JavaScript/TypeScript-specific metadata."""
        if node.type in ["function_declaration", "method_definition", "arrow_function"]:
            metadata.function_name = self._get_node_text(node, "identifier", content)
            metadata.function_signature = self._extract_js_function_signature(node, content)
            metadata.parameter_types = self._extract_js_parameter_types(node, content)
            metadata.return_type = self._extract_js_return_type(node, content)
            metadata.complexity_score = self._calculate_complexity(node)

        elif node.type in ["class_declaration"]:
            metadata.class_name = self._get_node_text(node, "identifier", content)
            metadata.inheritance_chain = self._extract_js_inheritance(node, content)

        metadata.import_statements = self._extract_import_statements(content, "javascript")

    async def _extract_java_metadata(self, node: Node, content: str, metadata: ExtractedMetadata) -> None:
        """Extract Java-specific metadata."""
        if node.type == "method_declaration":
            metadata.function_name = self._get_node_text(node, "identifier", content)
            metadata.function_signature = self._extract_java_method_signature(node, content)
            metadata.parameter_types = self._extract_java_parameter_types(node, content)
            metadata.return_type = self._extract_java_return_type(node, content)

        elif node.type == "class_declaration":
            metadata.class_name = self._get_node_text(node, "identifier", content)
            metadata.inheritance_chain = self._extract_java_inheritance(node, content)
            metadata.interfaces = self._extract_java_interfaces(node, content)

        metadata.import_statements = self._extract_import_statements(content, "java")

    async def _extract_go_metadata(self, node: Node, content: str, metadata: ExtractedMetadata) -> None:
        """Extract Go-specific metadata."""
        if node.type == "function_declaration":
            metadata.function_name = self._get_node_text(node, "identifier", content)
            metadata.function_signature = self._extract_go_function_signature(node, content)
            metadata.parameter_types = self._extract_go_parameter_types(node, content)
            metadata.return_type = self._extract_go_return_type(node, content)

        elif node.type in ["type_declaration", "struct_type"]:
            metadata.class_name = self._get_node_text(node, "type_identifier", content)

        metadata.import_statements = self._extract_import_statements(content, "go")

    async def _extract_rust_metadata(self, node: Node, content: str, metadata: ExtractedMetadata) -> None:
        """Extract Rust-specific metadata."""
        if node.type == "function_item":
            metadata.function_name = self._get_node_text(node, "identifier", content)
            metadata.function_signature = self._extract_rust_function_signature(node, content)
            metadata.parameter_types = self._extract_rust_parameter_types(node, content)
            metadata.return_type = self._extract_rust_return_type(node, content)

        elif node.type in ["struct_item", "enum_item"]:
            metadata.class_name = self._get_node_text(node, "type_identifier", content)

        metadata.import_statements = self._extract_import_statements(content, "rust")

    def _get_node_text(self, node: Node, child_type: str, content: str) -> str | None:
        """Get text content of a specific child node type."""
        for child in node.children:
            if child.type == child_type:
                return content[child.start_byte : child.end_byte]
        return None

    def _extract_python_function_signature(self, node: Node, content: str) -> str | None:
        """Extract complete Python function signature."""
        try:
            sig_parts = []

            # Function name
            name = self._get_node_text(node, "identifier", content)
            if name:
                sig_parts.append(f"def {name}")

            # Parameters
            params = self._get_node_text(node, "parameters", content)
            if params:
                sig_parts.append(params)

            # Return type annotation
            for child in node.children:
                if child.type == "type":
                    return_type = content[child.start_byte : child.end_byte]
                    sig_parts.append(f" -> {return_type}")

            return "".join(sig_parts) if sig_parts else None
        except Exception:
            return None

    def _extract_python_parameter_types(self, node: Node, content: str) -> list[str] | None:
        """Extract parameter type annotations."""
        try:
            param_types = []
            for child in node.children:
                if child.type == "parameters":
                    for param_child in child.children:
                        if param_child.type == "typed_parameter":
                            # Look for type annotation
                            for annotation_child in param_child.children:
                                if annotation_child.type == "type":
                                    type_text = content[annotation_child.start_byte : annotation_child.end_byte]
                                    param_types.append(type_text)
            return param_types if param_types else None
        except Exception:
            return None

    def _extract_python_return_type(self, node: Node, content: str) -> str | None:
        """Extract return type annotation."""
        try:
            for child in node.children:
                if child.type == "type":
                    return content[child.start_byte : child.end_byte]
            return None
        except Exception:
            return None

    def _extract_python_decorators(self, node: Node, content: str) -> list[str] | None:
        """Extract decorator names."""
        try:
            decorators = []
            for child in node.children:
                if child.type == "decorator":
                    decorator_text = content[child.start_byte : child.end_byte]
                    decorators.append(decorator_text)
            return decorators if decorators else None
        except Exception:
            return None

    def _extract_python_docstring(self, node: Node, content: str) -> str | None:
        """Extract docstring from function or class."""
        try:
            # Look for string literal as first statement in function/class body
            for child in node.children:
                if child.type == "block":
                    for stmt_child in child.children:
                        if stmt_child.type == "expression_statement":
                            for expr_child in stmt_child.children:
                                if expr_child.type == "string":
                                    docstring = content[expr_child.start_byte : expr_child.end_byte]
                                    return docstring.strip().replace('"""', "").replace("'''", "")
            return None
        except Exception:
            return None

    def _extract_python_inheritance(self, node: Node, content: str) -> list[str] | None:
        """Extract base class names from class definition."""
        try:
            inheritance = []
            for child in node.children:
                if child.type == "argument_list":
                    for arg_child in child.children:
                        if arg_child.type == "identifier":
                            base_class = content[arg_child.start_byte : arg_child.end_byte]
                            inheritance.append(base_class)
            return inheritance if inheritance else None
        except Exception:
            return None

    def _extract_js_function_signature(self, node: Node, content: str) -> str | None:
        """Extract JavaScript/TypeScript function signature."""
        try:
            start_byte = node.start_byte
            end_byte = node.end_byte

            # Find the opening brace to get just the signature
            node_text = content[start_byte:end_byte]
            brace_index = node_text.find("{")
            if brace_index != -1:
                return node_text[:brace_index].strip()
            return node_text.strip()
        except Exception:
            return None

    def _extract_js_parameter_types(self, node: Node, content: str) -> list[str] | None:
        """Extract TypeScript parameter types."""
        if "typescript" not in str(type(node)):
            return None

        try:
            param_types = []
            for child in node.children:
                if child.type == "formal_parameters":
                    for param_child in child.children:
                        if param_child.type == "required_parameter":
                            # Look for type annotation
                            for annotation_child in param_child.children:
                                if annotation_child.type == "type_annotation":
                                    type_text = content[annotation_child.start_byte : annotation_child.end_byte]
                                    param_types.append(type_text)
            return param_types if param_types else None
        except Exception:
            return None

    def _extract_js_return_type(self, node: Node, content: str) -> str | None:
        """Extract TypeScript return type."""
        try:
            for child in node.children:
                if child.type == "type_annotation":
                    return content[child.start_byte : child.end_byte]
            return None
        except Exception:
            return None

    def _extract_js_inheritance(self, node: Node, content: str) -> list[str] | None:
        """Extract extends/implements from class."""
        try:
            inheritance = []
            for child in node.children:
                if child.type in ["class_heritage", "extends_clause"]:
                    for heritage_child in child.children:
                        if heritage_child.type == "identifier":
                            base_class = content[heritage_child.start_byte : heritage_child.end_byte]
                            inheritance.append(base_class)
            return inheritance if inheritance else None
        except Exception:
            return None

    def _extract_java_method_signature(self, node: Node, content: str) -> str | None:
        """Extract Java method signature."""
        try:
            sig_parts = []

            # Modifiers
            modifiers = []
            for child in node.children:
                if child.type == "modifiers":
                    modifiers_text = content[child.start_byte : child.end_byte]
                    modifiers.append(modifiers_text)

            # Return type
            return_type = None
            for child in node.children:
                if child.type in ["type_identifier", "primitive_type", "generic_type"]:
                    return_type = content[child.start_byte : child.end_byte]
                    break

            # Method name
            method_name = self._get_node_text(node, "identifier", content)

            # Parameters
            params = self._get_node_text(node, "formal_parameters", content)

            if modifiers:
                sig_parts.extend(modifiers)
            if return_type:
                sig_parts.append(return_type)
            if method_name:
                sig_parts.append(method_name)
            if params:
                sig_parts.append(params)

            return " ".join(sig_parts) if sig_parts else None
        except Exception:
            return None

    def _extract_java_parameter_types(self, node: Node, content: str) -> list[str] | None:
        """Extract Java parameter types."""
        try:
            param_types = []
            for child in node.children:
                if child.type == "formal_parameters":
                    for param_child in child.children:
                        if param_child.type == "formal_parameter":
                            # Get type from formal parameter
                            for type_child in param_child.children:
                                if type_child.type in ["type_identifier", "primitive_type", "generic_type"]:
                                    type_text = content[type_child.start_byte : type_child.end_byte]
                                    param_types.append(type_text)
                                    break
            return param_types if param_types else None
        except Exception:
            return None

    def _extract_java_return_type(self, node: Node, content: str) -> str | None:
        """Extract Java return type."""
        try:
            for child in node.children:
                if child.type in ["type_identifier", "primitive_type", "generic_type", "void_type"]:
                    return content[child.start_byte : child.end_byte]
            return None
        except Exception:
            return None

    def _extract_java_inheritance(self, node: Node, content: str) -> list[str] | None:
        """Extract Java inheritance (extends)."""
        try:
            inheritance = []
            for child in node.children:
                if child.type == "superclass":
                    for super_child in child.children:
                        if super_child.type == "type_identifier":
                            base_class = content[super_child.start_byte : super_child.end_byte]
                            inheritance.append(base_class)
            return inheritance if inheritance else None
        except Exception:
            return None

    def _extract_java_interfaces(self, node: Node, content: str) -> list[str] | None:
        """Extract Java interfaces (implements)."""
        try:
            interfaces = []
            for child in node.children:
                if child.type == "super_interfaces":
                    for interface_child in child.children:
                        if interface_child.type == "type_identifier":
                            interface = content[interface_child.start_byte : interface_child.end_byte]
                            interfaces.append(interface)
            return interfaces if interfaces else None
        except Exception:
            return None

    def _extract_go_function_signature(self, node: Node, content: str) -> str | None:
        """Extract Go function signature."""
        try:
            return content[node.start_byte : node.end_byte].split("{")[0].strip()
        except Exception:
            return None

    def _extract_go_parameter_types(self, node: Node, content: str) -> list[str] | None:
        """Extract Go parameter types."""
        try:
            param_types = []
            for child in node.children:
                if child.type == "parameter_list":
                    for param_child in child.children:
                        if param_child.type == "parameter_declaration":
                            # Get type from parameter
                            for type_child in param_child.children:
                                if type_child.type in ["type_identifier", "pointer_type", "slice_type"]:
                                    type_text = content[type_child.start_byte : type_child.end_byte]
                                    param_types.append(type_text)
            return param_types if param_types else None
        except Exception:
            return None

    def _extract_go_return_type(self, node: Node, content: str) -> str | None:
        """Extract Go return type."""
        try:
            for child in node.children:
                if child.type in ["type_identifier", "pointer_type", "slice_type"]:
                    return content[child.start_byte : child.end_byte]
            return None
        except Exception:
            return None

    def _extract_rust_function_signature(self, node: Node, content: str) -> str | None:
        """Extract Rust function signature."""
        try:
            return content[node.start_byte : node.end_byte].split("{")[0].strip()
        except Exception:
            return None

    def _extract_rust_parameter_types(self, node: Node, content: str) -> list[str] | None:
        """Extract Rust parameter types."""
        try:
            param_types = []
            for child in node.children:
                if child.type == "parameters":
                    for param_child in child.children:
                        if param_child.type == "parameter":
                            # Get type from parameter
                            for type_child in param_child.children:
                                if type_child.type in ["type_identifier", "primitive_type", "reference_type"]:
                                    type_text = content[type_child.start_byte : type_child.end_byte]
                                    param_types.append(type_text)
            return param_types if param_types else None
        except Exception:
            return None

    def _extract_rust_return_type(self, node: Node, content: str) -> str | None:
        """Extract Rust return type."""
        try:
            for child in node.children:
                if child.type in ["type_identifier", "primitive_type", "reference_type"]:
                    return content[child.start_byte : child.end_byte]
            return None
        except Exception:
            return None

    def _extract_import_statements(self, content: str, language: str) -> list[str] | None:
        """Extract import statements from file content."""
        try:
            imports = []
            lines = content.split("\n")

            if language == "python":
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("import ") or stripped.startswith("from "):
                        imports.append(stripped)
            elif language in ["javascript", "typescript"]:
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("import ") or stripped.startswith("const ") and "require(" in stripped:
                        imports.append(stripped)
            elif language == "java":
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("import "):
                        imports.append(stripped)
            elif language == "go":
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("import ") or (stripped.startswith('"') and "/" in stripped):
                        imports.append(stripped)
            elif language == "rust":
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("use "):
                        imports.append(stripped)

            return imports if imports else None
        except Exception:
            return None

    def _calculate_complexity(self, node: Node) -> int | None:
        """Calculate cyclomatic complexity for a function."""
        try:
            complexity = 1  # Base complexity

            # Count decision points
            decision_types = {
                "if_statement",
                "while_statement",
                "for_statement",
                "case_clause",
                "catch_clause",
                "conditional_expression",
                "logical_and",
                "logical_or",
                "switch_statement",
                "try_statement",
            }

            def count_decisions(n: Node) -> int:
                count = 0
                if n.type in decision_types:
                    count += 1
                for child in n.children:
                    count += count_decisions(child)
                return count

            complexity += count_decisions(node)
            return complexity
        except Exception:
            return None

    async def extract_file_level_metadata(self, file_path: str, content: str) -> dict[str, any]:
        """Extract file-level metadata including all imports and dependencies."""
        try:
            language = self._detect_language_from_path(file_path)
            if language not in self.parsers:
                return {}

            parser = self.parsers[language]
            tree = parser.parse(content.encode())

            file_metadata = {
                "file_path": file_path,
                "language": language,
                "imports": self._extract_import_statements(content, language),
                "dependencies": self._extract_dependencies(content, language),
                "total_functions": 0,
                "total_classes": 0,
                "total_lines": len(content.split("\n")),
            }

            # Count semantic constructs
            def count_constructs(node: Node):
                if node.type in ["function_definition", "async_function_definition", "method_definition"]:
                    file_metadata["total_functions"] += 1
                elif node.type in ["class_definition", "class_declaration"]:
                    file_metadata["total_classes"] += 1

                for child in node.children:
                    count_constructs(child)

            count_constructs(tree.root_node)
            return file_metadata

        except Exception as e:
            logger.error(f"Error extracting file metadata for {file_path}: {e}")
            return {}

    def _extract_dependencies(self, content: str, language: str) -> list[str] | None:
        """Extract external dependencies from imports."""
        try:
            dependencies = []
            imports = self._extract_import_statements(content, language) or []

            for import_stmt in imports:
                if language == "python":
                    if import_stmt.startswith("import "):
                        module = import_stmt.replace("import ", "").split(".")[0].split(" as ")[0]
                        dependencies.append(module)
                    elif import_stmt.startswith("from "):
                        module = import_stmt.split("from ")[1].split(" import")[0].split(".")[0]
                        dependencies.append(module)
                elif language in ["javascript", "typescript"]:
                    if "from " in import_stmt:
                        module = import_stmt.split("from ")[1].strip().replace("'", "").replace('"', "").rstrip(";")
                        dependencies.append(module)
                    elif "require(" in import_stmt:
                        start = import_stmt.find("require(") + 8
                        end = import_stmt.find(")", start)
                        if end > start:
                            module = import_stmt[start:end].replace("'", "").replace('"', "")
                            dependencies.append(module)

            return list(set(dependencies)) if dependencies else None
        except Exception:
            return None

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
