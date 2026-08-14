"""Git Hook Installer component (FR-6.3)."""

import os
import stat
from pathlib import Path
from typing import Tuple


HOOK_SCRIPT_CONTENT = """#!/bin/sh
# Codebase Onboarding Agent - Automatic Incremental Re-Indexing Post-Commit Hook
echo "=== [Codebase Agent] Triggering Automatic Post-Commit Incremental Re-Index ==="
python -m codebase_agent.cli index --repo .
"""


class GitHookInstaller:
    """Installs and uninstalls Git post-commit hooks for automatic incremental re-indexing (FR-6.3)."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.git_dir = self.repo_root / ".git"
        self.hooks_dir = self.git_dir / "hooks"
        self.hook_file = self.hooks_dir / "post-commit"

    def is_git_repository(self) -> bool:
        """Checks if target directory is a valid Git repository."""
        return self.git_dir.exists() and self.git_dir.is_dir()

    def install_hook(self) -> Tuple[bool, str]:
        """Installs post-commit hook script in .git/hooks/post-commit."""
        if not self.is_git_repository():
            return False, f"Target directory '{self.repo_root}' is not a Git repository (.git folder missing)."

        try:
            self.hooks_dir.mkdir(parents=True, exist_ok=True)
            with open(self.hook_file, "w", encoding="utf-8", newline="\n") as f:
                f.write(HOOK_SCRIPT_CONTENT)

            # Make hook executable on POSIX systems
            if os.name == "posix":
                current_mode = os.stat(self.hook_file).st_mode
                os.chmod(self.hook_file, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

            return True, f"Successfully installed post-commit hook at: {self.hook_file}"
        except Exception as ex:
            return False, f"Failed to install Git hook: {str(ex)}"

    def uninstall_hook(self) -> Tuple[bool, str]:
        """Removes post-commit hook script if installed."""
        if not self.hook_file.exists():
            return True, f"Hook file '{self.hook_file}' is not installed."

        try:
            self.hook_file.unlink()
            return True, f"Successfully uninstalled Git post-commit hook."
        except Exception as ex:
            return False, f"Failed to uninstall Git hook: {str(ex)}"

    def check_status(self) -> bool:
        """Returns True if post-commit hook exists."""
        return self.hook_file.exists()
