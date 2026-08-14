"""Call-site resolution algorithm (Algorithm 3.2, FR-3.3)."""

from typing import Dict, Optional, Set, Tuple
from codebase_agent.parser.models import CallSite, FileParseResult, Symbol
from codebase_agent.parser.module_resolver import ModuleResolver


class CallResolver:
    """Resolves function/method call sites to target graph node IDs using static analysis."""

    @staticmethod
    def resolve_call(
        caller_file_path: str,
        call_site: CallSite,
        all_parse_results: Dict[str, FileParseResult],
        known_files: Set[str]
    ) -> Tuple[str, bool]:
        """
        Implements Algorithm 3.2 (resolve_call).

        Returns:
            Tuple[target_id, is_resolved]
            e.g., ("app/services/payment.py::charge_card", True)
            or ("print", False) if unresolved.
        """
        called_name = call_site.function_name
        caller_parse = all_parse_results.get(caller_file_path)

        if not caller_parse:
            return called_name, False

        # 1. Same-file definition priority
        for sym in caller_parse.symbols:
            if sym.name == called_name:
                return sym.id, True

        # 2. Import map resolution
        for imp in caller_parse.imports:
            target_symbol_name = None

            # Case A: Alias match e.g. from service import process_payment as pay (called_name = "pay")
            for orig_name, alias in imp.alias_map.items():
                if alias == called_name or orig_name == called_name:
                    target_symbol_name = orig_name
                    break

            # Case B: Direct imported symbol match e.g. from service import process_payment
            if not target_symbol_name and called_name in imp.imported_symbols:
                target_symbol_name = called_name

            if target_symbol_name:
                target_file = ModuleResolver.resolve_import_to_file(
                    calling_file_path=caller_file_path,
                    import_stmt=imp,
                    known_files=known_files
                )
                if target_file and target_file in all_parse_results:
                    target_parse = all_parse_results[target_file]
                    for sym in target_parse.symbols:
                        if sym.name == target_symbol_name:
                            return sym.id, True

            # Case C: Module-level import match e.g. import service.payment (called_name = "process_payment")
            if not target_symbol_name and imp.source_module:
                target_file = ModuleResolver.resolve_import_to_file(
                    calling_file_path=caller_file_path,
                    import_stmt=imp,
                    known_files=known_files
                )
                if target_file and target_file in all_parse_results:
                    target_parse = all_parse_results[target_file]
                    for sym in target_parse.symbols:
                        if sym.name == called_name:
                            return sym.id, True

        # 3. Global unique symbol fallback (if symbol name is unique across the entire repo)
        matching_symbols: List[Symbol] = []
        for file_path, parse_res in all_parse_results.items():
            for sym in parse_res.symbols:
                if sym.name == called_name:
                    matching_symbols.append(sym)

        if len(matching_symbols) == 1:
            return matching_symbols[0].id, True

        # 4. Cannot resolve statically (built-in, dynamic dispatch, etc.)
        return called_name, False
