"""Graph Builder component implementing Algorithm 3.3 (FR-3)."""

from typing import Dict, List, Set, Tuple, Optional
import networkx as nx

from codebase_agent.graph.resolver import CallResolver
from codebase_agent.parser.models import FileParseResult, SymbolType
from codebase_agent.parser.module_resolver import ModuleResolver


class GraphBuilder:
    """Constructs and queries directed NetworkX knowledge graph of codebase relationships."""

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.G = graph if graph is not None else nx.DiGraph()

    def build_graph(self, all_parse_results: Dict[str, FileParseResult]) -> nx.DiGraph:
        """
        Implements Algorithm 3.3 (build_graph - Two-Pass Construction).
        Pass 1: Add all file and symbol nodes + defined_in edges.
        Pass 2: Add imports, inherits_from, and call edges after all nodes exist.
        """
        self.G.clear()
        known_files: Set[str] = set(all_parse_results.keys())

        # --- PASS 1: Add all file nodes, symbol nodes, and defined_in edges ---
        for file_path, parse_res in all_parse_results.items():
            if parse_res.parse_status != "success":
                continue

            # Add File Node
            self.G.add_node(
                file_path,
                type="file",
                name=file_path,
                file_path=file_path,
                language=parse_res.language
            )

            # Add Symbol Nodes + defined_in edges
            for sym in parse_res.symbols:
                self.G.add_node(
                    sym.id,
                    type=sym.type.value,
                    name=sym.name,
                    file_path=sym.file_path,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                    language=sym.language
                )
                self.G.add_edge(
                    sym.id,
                    file_path,
                    relation="defined_in",
                    resolved=True
                )

        # --- PASS 2: Add relationship edges (imports, calls, inherits_from) ---
        for file_path, parse_res in all_parse_results.items():
            if parse_res.parse_status != "success":
                continue

            # 1. Imports Edges (File -> File)
            for imp in parse_res.imports:
                target_file = ModuleResolver.resolve_import_to_file(
                    calling_file_path=file_path,
                    import_stmt=imp,
                    known_files=known_files
                )
                if target_file and self.G.has_node(target_file):
                    self.G.add_edge(
                        file_path,
                        target_file,
                        relation="imports",
                        resolved=True
                    )

            # 2. Call Edges (Caller Symbol -> Target Symbol / Raw Target)
            for call in parse_res.call_sites:
                caller_id = call.caller_symbol_id or file_path
                target_id, resolved = CallResolver.resolve_call(
                    caller_file_path=file_path,
                    call_site=call,
                    all_parse_results=all_parse_results,
                    known_files=known_files
                )

                # Ensure target node exists even if unresolved
                if not self.G.has_node(target_id):
                    self.G.add_node(
                        target_id,
                        type="unresolved_call" if not resolved else "function",
                        name=call.function_name,
                        file_path="",
                        language=parse_res.language
                    )

                self.G.add_edge(
                    caller_id,
                    target_id,
                    relation="calls",
                    resolved=resolved
                )

        return self.G

    # --- Query Operations (FR-3.5) ---

    def get_callers(self, node_id: str) -> List[Tuple[str, dict]]:
        """Returns direct callers of a symbol node."""
        callers = []
        if not self.G.has_node(node_id):
            return callers

        for pred in self.G.predecessors(node_id):
            edge_data = self.G.get_edge_data(pred, node_id)
            if edge_data and edge_data.get("relation") == "calls":
                node_data = self.G.nodes[pred]
                callers.append((pred, node_data))
        return callers

    def get_callees(self, node_id: str) -> List[Tuple[str, dict]]:
        """Returns direct callees of a symbol node."""
        callees = []
        if not self.G.has_node(node_id):
            return callees

        for succ in self.G.successors(node_id):
            edge_data = self.G.get_edge_data(node_id, succ)
            if edge_data and edge_data.get("relation") == "calls":
                node_data = self.G.nodes[succ]
                callees.append((succ, node_data))
        return callees

    def get_imports(self, file_id: str) -> List[Tuple[str, dict]]:
        """Returns files imported by file_id."""
        imports = []
        if not self.G.has_node(file_id):
            return imports

        for succ in self.G.successors(file_id):
            edge_data = self.G.get_edge_data(file_id, succ)
            if edge_data and edge_data.get("relation") == "imports":
                node_data = self.G.nodes[succ]
                imports.append((succ, node_data))
        return imports

    def get_neighborhood(self, seed_node_ids: List[str], hops: int = 1) -> Set[str]:
        """
        Retrieves 1-hop (or K-hop) structural neighborhood for Graph Expansion (Algorithm 3.6).
        Pulls callers, callees, and importers of seed nodes.
        """
        expanded: Set[str] = set(seed_node_ids)
        frontier: List[str] = [n for n in seed_node_ids if self.G.has_node(n)]

        for _ in range(hops):
            next_frontier: List[str] = []
            for node_id in frontier:
                # Callers & Importers (predecessors)
                for pred in self.G.predecessors(node_id):
                    if pred not in expanded:
                        expanded.add(pred)
                        next_frontier.append(pred)

                # Callees & Imported Modules (successors)
                for succ in self.G.successors(node_id):
                    if succ not in expanded:
                        expanded.add(succ)
                        next_frontier.append(succ)
            frontier = next_frontier

        return expanded
