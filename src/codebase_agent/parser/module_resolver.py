"""Python module path resolver for call site and import resolution."""

from pathlib import Path
from typing import Optional, Set, Tuple
from codebase_agent.parser.models import ImportStatement


class ModuleResolver:
    """Resolves Python module import statements to repository relative file paths."""

    @staticmethod
    def resolve_import_to_file(
        calling_file_path: str,
        import_stmt: ImportStatement,
        known_files: Set[str]
    ) -> Optional[str]:
        """
        Resolves an ImportStatement from calling_file_path against known repo file paths.

        Returns relative file path in repo if matched, or None.
        """
        calling_path = Path(calling_file_path)

        if import_stmt.is_relative:
            # Relative import handling (e.g. from .service import payment or from ..models import order)
            # Level 1 = current package directory (calling_path.parent)
            # Level 2 = parent package directory (calling_path.parent.parent), etc.
            base_dir = calling_path.parent
            for _ in range(import_stmt.level - 1):
                if base_dir.parent != base_dir:
                    base_dir = base_dir.parent

            rel_module_path = import_stmt.source_module.replace(".", "/") if import_stmt.source_module else ""
            candidate_base = (base_dir / rel_module_path).as_posix() if rel_module_path else base_dir.as_posix()
        else:
            # Absolute import handling (e.g. app.services.payments)
            candidate_base = import_stmt.source_module.replace(".", "/")

        # Try exact file match with .py
        candidate_file = f"{candidate_base}.py" if not candidate_base.endswith(".py") else candidate_base
        candidate_file = candidate_file.lstrip("./")
        if candidate_file in known_files:
            return candidate_file

        # Try package __init__.py match
        candidate_init = f"{candidate_base.rstrip('/')}/__init__.py".lstrip("./")
        if candidate_init in known_files:
            return candidate_init

        return None
