"""Language-aware Parser component (Architecture Section 4.2, FR-2)."""

from pathlib import Path
from typing import Dict, Optional
from codebase_agent.parser.models import FileParseResult
from codebase_agent.parser.python_extractor import PythonExtractor


class Parser:
    """Coordinates tree-sitter AST parsing across source files with per-file fault isolation."""

    def __init__(self):
        self.extractors: Dict[str, object] = {
            "python": PythonExtractor()
        }

    def parse_file(self, file_path: str, repo_root: Optional[Path] = None) -> FileParseResult:
        """
        Parses a single file into FileParseResult (FR-2.1 - FR-2.4).
        Per-file parse errors are caught and returned in status without halting ingestion.
        """
        abs_path = (repo_root / file_path) if repo_root else Path(file_path)

        if not abs_path.exists():
            return FileParseResult(
                file_path=file_path,
                language="unknown",
                parse_status="failed",
                error_message=f"File not found: {abs_path}"
            )

        # Extension-based language resolution
        ext = abs_path.suffix.lower()
        language = "python" if ext == ".py" else "unknown"

        extractor = self.extractors.get(language)
        if not extractor:
            return FileParseResult(
                file_path=file_path,
                language=language,
                parse_status="failed",
                error_message=f"Unsupported language extractor for extension '{ext}'."
            )

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                source_code = f.read()

            return extractor.parse_file(file_path=file_path, source_code=source_code)
        except Exception as ex:
            return FileParseResult(
                file_path=file_path,
                language=language,
                parse_status="failed",
                error_message=f"Error reading or parsing file: {str(ex)}"
            )
