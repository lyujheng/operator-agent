"""Repo tools: list_repo_files, search_file_content, read_file.

These tools operate on local clones of official repositories
(pytorch, torch_npu, vllm, sglang).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any


def list_repo_files(repo_path: str, directory: str = ".") -> list[str]:
    """List files in a repo directory.

    Args:
        repo_path: Root path of the repository.
        directory: Sub-directory to list (relative to repo_path). Defaults to root.

    Returns:
        List of relative file paths under *directory*.
    """
    target = Path(repo_path) / directory
    if not target.exists():
        return []

    result: list[str] = []
    for root, _dirs, files in os.walk(target):
        for f in files:
            full = Path(root) / f
            rel = str(full.relative_to(Path(repo_path)))
            result.append(rel)
    return sorted(result)


def search_file_content(
    repo_path: str,
    keyword: str,
    file_pattern: str = "*.py",
) -> list[dict[str, Any]]:
    """Search for *keyword* in repo files (grep-like).

    Returns matching lines with ±5 surrounding lines of context.

    Args:
        repo_path: Root path of the repository.
        keyword: Search keyword (case-insensitive).
        file_pattern: Glob pattern for files to search. Default ``"*.py"``.

    Returns:
        List of dicts with ``file_path``, ``line_number``, ``line_content``, ``context``.
    """
    repo = Path(repo_path)
    if not repo.exists():
        return []

    # Use grep for speed; fall back to Python scan if grep is unavailable.
    try:
        cmd = [
            "grep",
            "-rin",  # recursive, case-insensitive, line numbers
            "--include",
            file_pattern,
            "-n",
            keyword,
            str(repo),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = proc.stdout.strip().split("\n") if proc.stdout.strip() else []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Fallback: Python-based search.
        lines = _python_search(repo, keyword, file_pattern)

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()  # deduplicate

    for line in lines[:100]:  # cap at 100 matches
        match = re.match(r"^(.+?):(\d+):(.*)", line)
        if not match:
            continue
        filepath, lineno_str, content = match.groups()
        lineno = int(lineno_str)
        rel_path = str(Path(filepath).relative_to(repo))

        if (rel_path, lineno) in seen:
            continue
        seen.add((rel_path, lineno))

        # Fetch context (±5 lines).
        context = _get_context(filepath, lineno, radius=5)

        results.append(
            {
                "file_path": rel_path,
                "line_number": lineno,
                "line_content": content.strip(),
                "context": context,
            }
        )

    return results


def read_file(
    file_path: str,
    start_line: int = 1,
    max_lines: int = 200,
) -> dict[str, Any]:
    """Read file content with range control.

    Args:
        file_path: Absolute or relative path to the file.
        start_line: 1-indexed start line (default 1).
        max_lines: Maximum lines to return (default 200).

    Returns:
        Dict with ``content``, ``total_lines``, and ``range`` [start, end].
    """
    path = Path(file_path)
    if not path.exists():
        return {
            "content": f"File not found: {file_path}",
            "total_lines": 0,
            "range": [0, 0],
        }

    all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(all_lines)

    # Clamp to valid range (1-indexed).
    start_idx = max(0, start_line - 1)
    end_idx = min(total, start_idx + max_lines)
    selected = all_lines[start_idx:end_idx]

    return {
        "content": "\n".join(selected),
        "total_lines": total,
        "range": [start_idx + 1, end_idx],
    }


# ── Helpers ──────────────────────────────────────────────────────

def _get_context(filepath: str, lineno: int, radius: int = 5) -> str:
    """Read ±radius lines around *lineno* from *filepath*."""
    try:
        all_lines = Path(filepath).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    start = max(0, lineno - 1 - radius)
    end = min(len(all_lines), lineno + radius)
    context_lines = []
    for i, line in enumerate(all_lines[start:end], start=start + 1):
        marker = ">>>" if i == lineno else "   "
        context_lines.append(f"{marker} {i:>5}: {line}")
    return "\n".join(context_lines)


def _python_search(repo: Path, keyword: str, pattern: str) -> list[str]:
    """Pure-Python fallback when grep is not available."""
    results: list[str] = []
    glob_pattern = "**/" + pattern
    keyword_lower = keyword.lower()
    for filepath in repo.glob(glob_pattern):
        if not filepath.is_file():
            continue
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if keyword_lower in line.lower():
                results.append(f"{filepath}:{i}:{line}")
                if len(results) >= 100:
                    return results
    return results
