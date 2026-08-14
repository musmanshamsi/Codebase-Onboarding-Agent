"""Python AST symbol extractor using tree-sitter (FR-2, Algorithm 3.1)."""

from typing import List, Optional, Tuple
import tree_sitter
import tree_sitter_python as tspython

from codebase_agent.parser.models import (
    CallSite,
    FileParseResult,
    ImportStatement,
    Symbol,
    SymbolType,
)


class PythonExtractor:
    """Extracts symbols, import statements, and call sites from Python AST using tree-sitter."""

    def __init__(self):
        self.language = tree_sitter.Language(tspython.language())
        self.parser = tree_sitter.Parser(self.language)

    def parse_file(self, file_path: str, source_code: str) -> FileParseResult:
        """Parses source code into structured FileParseResult (FR-2.1 to FR-2.4)."""
        symbols: List[Symbol] = []
        imports: List[ImportStatement] = []
        call_sites: List[CallSite] = []

        try:
            tree = self.parser.parse(bytes(source_code, "utf-8"))
            root_node = tree.root_node

            if root_node.has_error and not root_node.children:
                return FileParseResult(
                    file_path=file_path,
                    language="python",
                    parse_status="failed",
                    error_message="Fatal syntax error: tree-sitter failed to build root AST node."
                )

            self._walk_ast(
                node=root_node,
                file_path=file_path,
                source_code=source_code,
                symbols=symbols,
                imports=imports,
                call_sites=call_sites,
                current_scope=None
            )

            return FileParseResult(
                file_path=file_path,
                language="python",
                symbols=symbols,
                imports=imports,
                call_sites=call_sites,
                parse_status="success"
            )
        except Exception as ex:
            return FileParseResult(
                file_path=file_path,
                language="python",
                parse_status="failed",
                error_message=f"Exception during tree-sitter parsing: {str(ex)}"
            )

    def _walk_ast(
        self,
        node: tree_sitter.Node,
        file_path: str,
        source_code: str,
        symbols: List[Symbol],
        imports: List[ImportStatement],
        call_sites: List[CallSite],
        current_scope: Optional[Symbol]
    ):
        """Recursive AST walk implementing Algorithm 3.1."""
        node_type = node.type

        # 1. Class Definitions
        if node_type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = self._node_text(name_node, source_code)
                symbol_id = f"{file_path}::{class_name}"
                class_symbol = Symbol(
                    id=symbol_id,
                    name=class_name,
                    type=SymbolType.CLASS,
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent_symbol=current_scope.name if current_scope else None,
                    language="python"
                )
                symbols.append(class_symbol)

                body_node = node.child_by_field_name("body")
                if body_node:
                    for child in body_node.children:
                        self._walk_ast(child, file_path, source_code, symbols, imports, call_sites, class_symbol)
                return

        # 2. Function & Method Definitions
        elif node_type in ("function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = self._node_text(name_node, source_code)
                is_method = current_scope is not None and current_scope.type == SymbolType.CLASS
                symbol_type = SymbolType.METHOD if is_method else SymbolType.FUNCTION

                symbol_id = f"{file_path}::{current_scope.name}.{func_name}" if is_method else f"{file_path}::{func_name}"

                func_symbol = Symbol(
                    id=symbol_id,
                    name=func_name,
                    type=symbol_type,
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent_symbol=current_scope.name if current_scope else None,
                    language="python"
                )
                symbols.append(func_symbol)

                body_node = node.child_by_field_name("body")
                if body_node:
                    for child in body_node.children:
                        self._walk_ast(child, file_path, source_code, symbols, imports, call_sites, func_symbol)
                return

        # 3. Import Statements
        elif node_type == "import_statement":
            imp = self._parse_import_statement(node, source_code)
            if imp:
                imports.extend(imp)

        elif node_type == "import_from_statement":
            imp = self._parse_import_from_statement(node, source_code)
            if imp:
                imports.append(imp)

        # 4. Call Expressions
        elif node_type == "call":
            call_site = self._parse_call_expression(node, source_code, current_scope)
            if call_site:
                call_sites.append(call_site)

        # Walk child AST nodes
        for child in node.children:
            self._walk_ast(child, file_path, source_code, symbols, imports, call_sites, current_scope)

    def _parse_import_statement(self, node: tree_sitter.Node, source_code: str) -> List[ImportStatement]:
        """Extracts imports from `import foo` or `import foo as f, bar`."""
        statements: List[ImportStatement] = []
        line_num = node.start_point[0] + 1

        for child in node.children:
            if child.type == "dotted_name":
                mod_name = self._node_text(child, source_code)
                statements.append(ImportStatement(
                    source_module=mod_name,
                    imported_symbols=[],
                    alias_map={},
                    is_relative=False,
                    level=0,
                    line_number=line_num
                ))
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node and alias_node:
                    mod_name = self._node_text(name_node, source_code)
                    alias_name = self._node_text(alias_node, source_code)
                    statements.append(ImportStatement(
                        source_module=mod_name,
                        imported_symbols=[],
                        alias_map={mod_name: alias_name},
                        is_relative=False,
                        level=0,
                        line_number=line_num
                    ))
        return statements

    def _parse_import_from_statement(self, node: tree_sitter.Node, source_code: str) -> Optional[ImportStatement]:
        """Extracts from ... import ... statements including relative imports."""
        line_num = node.start_point[0] + 1
        module_name = ""
        level = 0
        is_relative = False
        imported_symbols: List[str] = []
        alias_map: dict = {}

        for child in node.children:
            if child.type == "relative_import":
                text = self._node_text(child, source_code)
                dot_count = len(text) - len(text.lstrip("."))
                if dot_count > 0:
                    is_relative = True
                    level = dot_count
                    module_name = text.lstrip(".")
                else:
                    module_name = text

        in_imported_section = False
        for child in node.children:
            text = self._node_text(child, source_code)
            if text == "import":
                in_imported_section = True
                continue
            if not in_imported_section:
                if child.type == "dotted_name" and not is_relative:
                    module_name = text
                continue

            if child.type in ("dotted_name", "identifier"):
                imported_symbols.append(text)
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node and alias_node:
                    orig_name = self._node_text(name_node, source_code)
                    alias_name = self._node_text(alias_node, source_code)
                    imported_symbols.append(orig_name)
                    alias_map[orig_name] = alias_name
            elif child.type == "wildcard_import":
                imported_symbols.append("*")

        return ImportStatement(
            source_module=module_name,
            imported_symbols=imported_symbols,
            alias_map=alias_map,
            is_relative=is_relative,
            level=level,
            line_number=line_num
        )

    def _parse_call_expression(
        self, node: tree_sitter.Node, source_code: str, current_scope: Optional[Symbol]
    ) -> Optional[CallSite]:
        """Extracts function/method call details."""
        func_node = node.child_by_field_name("function")
        if not func_node:
            return None

        func_name = self._node_text(func_node, source_code)
        if "." in func_name:
            func_name = func_name.split(".")[-1]

        args_node = node.child_by_field_name("arguments")
        args_count = 0
        if args_node:
            args_count = sum(1 for c in args_node.children if c.type not in ("(", ")", ","))

        return CallSite(
            caller_symbol_id=current_scope.id if current_scope else None,
            function_name=func_name,
            line_number=node.start_point[0] + 1,
            args_count=args_count
        )

    @staticmethod
    def _node_text(node: tree_sitter.Node, source_code: str) -> str:
        """Extracts text slice from source code corresponding to AST node."""
        return source_code[node.start_byte:node.end_byte]
