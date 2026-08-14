"""Graph persistence and disk serialization (Database Design Section 4, FR-3.4)."""

from pathlib import Path
import networkx as nx


class GraphStore:
    """Manages loading and saving NetworkX directed graph to GraphML files."""

    @staticmethod
    def save_graph(G: nx.DiGraph, output_path: Path):
        """Saves directed graph to GraphML format (Database Design Section 4)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure boolean and int attributes are properly typed before saving
        for u, v, data in G.edges(data=True):
            if "resolved" in data and isinstance(data["resolved"], bool):
                data["resolved"] = bool(data["resolved"])

        nx.write_graphml(G, str(output_path))

    @staticmethod
    def load_graph(input_path: Path) -> nx.DiGraph:
        """Loads directed graph from GraphML file."""
        if not input_path.exists():
            raise FileNotFoundError(f"GraphML file not found: {input_path}")
        
        G = nx.read_graphml(str(input_path))
        # Convert integer attributes back to int if needed
        for node, data in G.nodes(data=True):
            if "start_line" in data and data["start_line"] is not None:
                try:
                    data["start_line"] = int(data["start_line"])
                except ValueError:
                    pass
            if "end_line" in data and data["end_line"] is not None:
                try:
                    data["end_line"] = int(data["end_line"])
                except ValueError:
                    pass

        for u, v, data in G.edges(data=True):
            if "resolved" in data:
                if str(data["resolved"]).lower() in ("true", "1"):
                    data["resolved"] = True
                elif str(data["resolved"]).lower() in ("false", "0"):
                    data["resolved"] = False

        return G
