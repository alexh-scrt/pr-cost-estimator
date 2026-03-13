"""Diff analysis module for PR Cost Estimator.

Parses unified diffs to extract:

* Per-file line addition/removal counts
* File type (extension) distribution
* A simple cyclomatic-proxy complexity heuristic based on control-flow
  keywords present in added lines
* Overall summary statistics (total files, total lines changed)

Typical usage::

    from pr_cost_estimator.diff_analyzer import DiffAnalyzer

    analyzer = DiffAnalyzer()
    analysis = analyzer.analyze(diff_text)
    print(analysis.total_files_changed)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileDiff:
    """Statistics and metadata for a single file within a diff."""

    old_path: str
    """The original file path (before the change)."""

    new_path: str
    """The new file path (after the change, differs on renames)."""

    extension: str
    """Lower-case file extension including leading dot, e.g. '.py'; empty if none."""

    lines_added: int
    """Number of lines added (lines beginning with '+' in the hunk)."""

    lines_removed: int
    """Number of lines removed (lines beginning with '-' in the hunk)."""

    is_binary: bool
    """True when the diff header indicates a binary file."""

    is_renamed: bool
    """True when the file was renamed."""

    is_new_file: bool
    """True when the file is newly created."""

    is_deleted_file: bool
    """True when the file was deleted."""

    complexity_score: int
    """Simple complexity proxy: count of control-flow keywords in added lines."""

    @property
    def lines_changed(self) -> int:
        """Total lines changed (added + removed)."""
        return self.lines_added + self.lines_removed


@dataclass
class DiffAnalysis:
    """Aggregated analysis results for an entire diff."""

    raw_diff: str
    """The original raw unified diff string."""

    file_diffs: List[FileDiff] = field(default_factory=list)
    """Per-file breakdown."""

    @property
    def total_files_changed(self) -> int:
        """Number of files touched by the diff."""
        return len(self.file_diffs)

    @property
    def total_lines_added(self) -> int:
        """Sum of all added lines across all files."""
        return sum(f.lines_added for f in self.file_diffs)

    @property
    def total_lines_removed(self) -> int:
        """Sum of all removed lines across all files."""
        return sum(f.lines_removed for f in self.file_diffs)

    @property
    def total_lines_changed(self) -> int:
        """Sum of all changed lines (added + removed) across all files."""
        return self.total_lines_added + self.total_lines_removed

    @property
    def total_complexity_score(self) -> int:
        """Aggregate complexity score across all files."""
        return sum(f.complexity_score for f in self.file_diffs)

    @property
    def extension_distribution(self) -> Dict[str, int]:
        """Mapping of file extension to number of files with that extension."""
        dist: Dict[str, int] = {}
        for fd in self.file_diffs:
            ext = fd.extension or "(no extension)"
            dist[ext] = dist.get(ext, 0) + 1
        return dist

    @property
    def binary_file_count(self) -> int:
        """Number of binary files in the diff."""
        return sum(1 for f in self.file_diffs if f.is_binary)

    @property
    def non_binary_file_diffs(self) -> List[FileDiff]:
        """File diffs excluding binary files."""
        return [f for f in self.file_diffs if not f.is_binary]


# ---------------------------------------------------------------------------
# Complexity heuristic
# ---------------------------------------------------------------------------

# Control-flow keywords that suggest branching / loops (language-agnostic).
_COMPLEXITY_KEYWORDS: re.Pattern[str] = re.compile(
    r"\b(if|elif|else|for|while|switch|case|catch|except|finally|"
    r"&&|\|\||\?|=>|async|await|raise|throw|try)\b"
)


def _compute_complexity(added_lines: List[str]) -> int:
    """Count control-flow keyword occurrences across a list of added lines.

    This is a simple cyclomatic-complexity proxy: it counts the number of
    branching/looping constructs in newly added code, which correlates with
    the cognitive load an AI reviewer must handle.

    Args:
        added_lines: Lines starting with '+' (prefix stripped) from a hunk.

    Returns:
        Integer complexity score (≥ 0).
    """
    score = 0
    for line in added_lines:
        score += len(_COMPLEXITY_KEYWORDS.findall(line))
    return score


# ---------------------------------------------------------------------------
# File extension helper
# ---------------------------------------------------------------------------

def _get_extension(path: str) -> str:
    """Extract the lower-case extension from a file path.

    Args:
        path: File path string.

    Returns:
        Lower-case extension including leading dot, e.g. '.py'.
        Returns empty string if the file has no extension or path is
        '/dev/null'.
    """
    if not path or path == "/dev/null":
        return ""
    # Handle paths like 'a/src/foo.py'
    basename = path.split("/")[-1]
    if "." in basename:
        return "." + basename.rsplit(".", 1)[-1].lower()
    return ""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class DiffAnalyzer:
    """Parses unified diffs into structured :class:`DiffAnalysis` objects.

    The parser handles standard ``git diff`` output including:

    * Normal text diffs
    * Binary file notifications
    * File renames (``rename from`` / ``rename to``)
    * New-file and deleted-file markers
    * Multiple hunks per file
    """

    # Patterns for diff headers
    _DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
    _OLD_FILE = re.compile(r"^--- (.+)$")
    _NEW_FILE = re.compile(r"^\+\+\+ (.+)$")
    _BINARY = re.compile(r"^Binary files .+ and .+ differ$")
    _RENAME_FROM = re.compile(r"^rename from (.+)$")
    _RENAME_TO = re.compile(r"^rename to (.+)$")
    _NEW_FILE_MODE = re.compile(r"^new file mode")
    _DELETED_FILE_MODE = re.compile(r"^deleted file mode")
    _HUNK_HEADER = re.compile(r"^@@")

    def analyze(self, diff_text: str) -> DiffAnalysis:
        """Parse a unified diff string and return a DiffAnalysis.

        Args:
            diff_text: Raw unified diff content as a string.

        Returns:
            A populated :class:`DiffAnalysis` instance.
        """
        file_diffs: List[FileDiff] = []

        # Split into per-file chunks at each "diff --git" line
        chunks = self._split_into_file_chunks(diff_text)

        for chunk in chunks:
            fd = self._parse_file_chunk(chunk)
            if fd is not None:
                file_diffs.append(fd)

        return DiffAnalysis(raw_diff=diff_text, file_diffs=file_diffs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_into_file_chunks(self, diff_text: str) -> List[str]:
        """Split a full diff into per-file sections.

        Args:
            diff_text: Full unified diff string.

        Returns:
            List of per-file diff sections.
        """
        lines = diff_text.splitlines(keepends=True)
        chunks: List[str] = []
        current: List[str] = []

        for line in lines:
            if line.startswith("diff --git ") and current:
                chunks.append("".join(current))
                current = [line]
            else:
                current.append(line)

        if current:
            chunks.append("".join(current))

        return chunks

    def _parse_file_chunk(self, chunk: str) -> Optional[FileDiff]:
        """Parse a single file's diff chunk.

        Args:
            chunk: A string containing all lines for one file diff.

        Returns:
            A :class:`FileDiff` or None if the chunk cannot be parsed.
        """
        lines = chunk.splitlines()
        if not lines:
            return None

        # Must start with 'diff --git'
        header_match = self._DIFF_HEADER.match(lines[0])
        if not header_match:
            return None

        git_old = header_match.group(1)  # path after 'a/'
        git_new = header_match.group(2)  # path after 'b/'

        old_path = git_old
        new_path = git_new
        is_binary = False
        is_renamed = False
        is_new_file = False
        is_deleted_file = False
        lines_added = 0
        lines_removed = 0
        added_content: List[str] = []
        in_hunk = False

        for line in lines[1:]:
            if self._BINARY.match(line):
                is_binary = True
                continue
            if self._RENAME_FROM.match(line):
                is_renamed = True
                old_path = self._RENAME_FROM.match(line).group(1)  # type: ignore[union-attr]
                continue
            if self._RENAME_TO.match(line):
                new_path = self._RENAME_TO.match(line).group(1)  # type: ignore[union-attr]
                continue
            if self._NEW_FILE_MODE.match(line):
                is_new_file = True
                continue
            if self._DELETED_FILE_MODE.match(line):
                is_deleted_file = True
                continue
            if self._OLD_FILE.match(line):
                m = self._OLD_FILE.match(line)
                raw = m.group(1) if m else old_path  # type: ignore[union-attr]
                # Strip leading 'a/' prefix that git adds
                old_path = raw[2:] if raw.startswith("a/") else raw
                continue
            if self._NEW_FILE.match(line):
                m = self._NEW_FILE.match(line)
                raw = m.group(1) if m else new_path  # type: ignore[union-attr]
                new_path = raw[2:] if raw.startswith("b/") else raw
                continue
            if self._HUNK_HEADER.match(line):
                in_hunk = True
                continue

            if in_hunk:
                if line.startswith("+") and not line.startswith("+++"):
                    lines_added += 1
                    added_content.append(line[1:])  # strip leading '+'
                elif line.startswith("-") and not line.startswith("---"):
                    lines_removed += 1

        # For /dev/null paths use the other side
        if old_path == "/dev/null" or old_path == "dev/null":
            old_path = new_path
        if new_path == "/dev/null" or new_path == "dev/null":
            new_path = old_path

        extension = _get_extension(new_path)
        complexity = _compute_complexity(added_content) if not is_binary else 0

        return FileDiff(
            old_path=old_path,
            new_path=new_path,
            extension=extension,
            lines_added=lines_added,
            lines_removed=lines_removed,
            is_binary=is_binary,
            is_renamed=is_renamed,
            is_new_file=is_new_file,
            is_deleted_file=is_deleted_file,
            complexity_score=complexity,
        )
