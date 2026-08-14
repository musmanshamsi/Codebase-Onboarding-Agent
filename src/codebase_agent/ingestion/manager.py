"""Ingestion Manager component (Architecture Section 4.1, FR-1)."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pathspec

try:
    import git
except ImportError:
    git = None

from codebase_agent.config import IngestionConfig
from codebase_agent.storage.metadata_cache import MetadataCache


EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sql": "sql",
}


class IngestionManager:
    """Discovers, filters, hashes, and detects language for repository source files."""

    def __init__(self, repo_target: str, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self.is_temp_clone = False
        self.repo_path = self._resolve_repo(repo_target)
        self.spec = pathspec.PathSpec.from_lines(
            "gitignore", self.config.exclude_patterns
        )

    def _resolve_repo(self, repo_target: str) -> Path:
        """Resolves target into local Path; clones if git URL (FR-1.1, FR-1.2)."""
        target_path = Path(repo_target).resolve()
        if target_path.exists() and target_path.is_dir():
            return target_path

        if repo_target.startswith(("http://", "https://", "git@")) or repo_target.endswith(".git"):
            if git is None:
                raise RuntimeError("GitPython is required to clone remote Git repositories.")
            temp_dir = Path(tempfile.mkdtemp(prefix="agent_repo_"))
            print(f"Cloning remote repository {repo_target} into {temp_dir}...")
            git.Repo.clone_from(repo_target, temp_dir)
            self.is_temp_clone = True
            return temp_dir

        raise ValueError(f"Repository target '{repo_target}' is neither a valid local directory nor a supported Git URL.")

    def cleanup(self):
        """Cleans up temporary directory if repo was cloned."""
        if self.is_temp_clone and self.repo_path.exists():
            shutil.rmtree(self.repo_path, ignore_errors=True)

    def is_binary_file(self, file_path: Path) -> bool:
        """Detects if file is binary by checking for null bytes in initial chunk (FR-1.5)."""
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:
                    return True
                # Check utf-8 decodability
                chunk.decode("utf-8")
                return False
        except (UnicodeDecodeError, OSError):
            return True

    def is_minified_file(self, file_path: Path) -> bool:
        """Detects minified files via line length heuristics (FR-1.5)."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(10)]
                lines = [l for l in lines if l]
                if not lines:
                    return False
                avg_length = sum(len(l) for l in lines) / len(lines)
                return avg_length > self.config.max_line_length_minified
        except OSError:
            return False

    def compute_content_hash(self, file_path: Path) -> str:
        """Computes SHA-256 content hash for a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def detect_language(self, file_path: Path) -> str:
        """Maps file extension to language tag (FR-1.4)."""
        ext = file_path.suffix.lower()
        return EXTENSION_LANGUAGE_MAP.get(ext, "unknown")

    def discover_files(self) -> List[Tuple[Path, str]]:
        """
        Walks directory tree, applies exclude patterns, size limits, binary/minified filters (FR-1.3 - FR-1.5).
        Returns list of (absolute_path, relative_path_str).
        """
        discovered: List[Tuple[Path, str]] = []

        for root, dirs, files in os.walk(self.repo_path):
            # Exclude directory matches early
            rel_root = Path(root).relative_to(self.repo_path)

            # Skip index directory
            if self.config.index_dir_name in rel_root.parts:
                dirs.clear()
                continue

            for file in files:
                abs_path = Path(root) / file
                rel_path = abs_path.relative_to(self.repo_path)
                rel_str = rel_path.as_posix()

                # Check pattern exclusions
                if self.spec.match_file(rel_str):
                    continue

                # Check extension filter (if supported_extensions configured)
                if self.config.supported_extensions and abs_path.suffix.lower() not in self.config.supported_extensions:
                    continue

                # Check size limit
                if abs_path.stat().st_size > self.config.max_file_size_bytes:
                    print(f"Skipping large file ({abs_path.stat().st_size} bytes): {rel_str}")
                    continue

                # Check binary
                if self.is_binary_file(abs_path):
                    continue

                # Check minified
                if self.is_minified_file(abs_path):
                    print(f"Skipping minified file: {rel_str}")
                    continue

                discovered.append((abs_path, rel_str))

        return discovered

    def compute_incremental_diff(
        self, discovered_files: List[Tuple[Path, str]], cache: MetadataCache
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        Implements Algorithm 3.8 (diff_for_reindex).
        Returns:
            to_process: List of dicts with file metadata (path, relative path, hash, language)
            to_delete: List of relative file paths that were removed from repo.
        """
        cached_states = cache.get_all_file_states()
        current_rel_paths: Set[str] = set()

        to_process: List[Dict[str, str]] = []

        for abs_path, rel_str in discovered_files:
            current_rel_paths.add(rel_str)
            curr_hash = self.compute_content_hash(abs_path)
            cached = cached_states.get(rel_str)

            if cached is None or cached["content_hash"] != curr_hash:
                to_process.append({
                    "abs_path": str(abs_path),
                    "rel_path": rel_str,
                    "content_hash": curr_hash,
                    "language": self.detect_language(abs_path)
                })

        to_delete = [
            cached_rel_path for cached_rel_path in cached_states
            if cached_rel_path not in current_rel_paths
        ]

        return to_process, to_delete
