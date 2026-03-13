"""Argparse-based CLI entry point for PR Cost Estimator.

Supports three input modes:

* ``--diff-file PATH``    — read a local unified diff file
* ``--branch BRANCH``    — compare current HEAD against a local git branch
* ``--pr-url URL``       — fetch a diff from a GitHub PR URL

Example usage::

    pr-cost-estimator --diff-file my.diff
    pr-cost-estimator --branch main
    pr-cost-estimator --pr-url https://github.com/owner/repo/pull/42
    pr-cost-estimator --pr-url https://github.com/owner/repo/pull/42 --json
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="pr-cost-estimator",
        description=(
            "Estimate the token and dollar cost of AI-based code review "
            "on a pull request before you commit to the expense."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pr-cost-estimator --diff-file changes.diff\n"
            "  pr-cost-estimator --branch main\n"
            "  pr-cost-estimator --pr-url https://github.com/owner/repo/pull/1\n"
            "  pr-cost-estimator --pr-url https://github.com/owner/repo/pull/1 --json\n"
        ),
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--diff-file",
        metavar="PATH",
        help="Path to a local unified diff file.",
    )
    input_group.add_argument(
        "--branch",
        metavar="BRANCH",
        help="Compare HEAD against this local git branch (e.g. main).",
    )
    input_group.add_argument(
        "--pr-url",
        metavar="URL",
        help="Full GitHub PR URL (e.g. https://github.com/owner/repo/pull/42).",
    )

    parser.add_argument(
        "--github-token",
        metavar="TOKEN",
        default=None,
        help=(
            "GitHub personal access token. Can also be set via the "
            "GITHUB_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--repo-path",
        metavar="PATH",
        default=".",
        help="Path to the local git repository (default: current directory).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="output_json",
        help="Output results as machine-readable JSON instead of a Rich table.",
    )
    parser.add_argument(
        "--output-file",
        metavar="PATH",
        default=None,
        help="Write output to this file instead of stdout.",
    )
    parser.add_argument(
        "--threshold",
        metavar="USD",
        type=float,
        default=5.0,
        help=(
            "Dollar cost threshold above which advisory suggestions are shown "
            "(default: 5.0)."
        ),
    )
    parser.add_argument(
        "--no-advice",
        action="store_true",
        default=False,
        help="Suppress advisory suggestions even when cost exceeds threshold.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    return parser


def _resolve_github_token(args: argparse.Namespace) -> Optional[str]:
    """Resolve the GitHub token from CLI args or environment variable.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The GitHub token string, or None if not provided.
    """
    import os

    return args.github_token or os.environ.get("GITHUB_TOKEN")


def _load_diff_from_file(path: str) -> str:
    """Read a unified diff from a local file.

    Args:
        path: Filesystem path to the diff file.

    Returns:
        Diff content as a string.

    Raises:
        SystemExit: If the file cannot be read.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError as exc:
        print(f"Error: Cannot read diff file '{path}': {exc}", file=sys.stderr)
        sys.exit(1)


def _load_diff_from_branch(repo_path: str, branch: str) -> str:
    """Generate a unified diff between HEAD and a local branch.

    Args:
        repo_path: Path to the local git repository.
        branch: Name of the branch to diff against.

    Returns:
        Unified diff content as a string.

    Raises:
        SystemExit: If the repository cannot be opened or the diff fails.
    """
    try:
        import git

        repo = git.Repo(repo_path, search_parent_directories=True)
        diff_output = repo.git.diff(branch, "HEAD")
        return diff_output
    except Exception as exc:  # noqa: BLE001
        print(f"Error: Failed to generate diff from branch '{branch}': {exc}", file=sys.stderr)
        sys.exit(1)


def _load_diff_from_github(pr_url: str, token: Optional[str]) -> str:
    """Fetch a diff from a GitHub PR URL via the REST API.

    Args:
        pr_url: Full GitHub PR URL.
        token: Optional GitHub API token.

    Returns:
        Unified diff content as a string.

    Raises:
        SystemExit: If the fetch fails.
    """
    try:
        from pr_cost_estimator.github_fetcher import GitHubFetcher

        fetcher = GitHubFetcher(token=token)
        return fetcher.fetch_diff(pr_url)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: Failed to fetch PR diff from GitHub: {exc}", file=sys.stderr)
        sys.exit(1)


def run(argv: list[str] | None = None) -> int:
    """Execute the CLI with the given argument list.

    Args:
        argv: List of CLI arguments; defaults to sys.argv[1:] if None.

    Returns:
        Integer exit code (0 for success, non-zero for errors).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Resolve diff text ---
    if args.diff_file:
        diff_text = _load_diff_from_file(args.diff_file)
    elif args.branch:
        diff_text = _load_diff_from_branch(args.repo_path, args.branch)
    else:
        token = _resolve_github_token(args)
        diff_text = _load_diff_from_github(args.pr_url, token)

    if not diff_text.strip():
        print("Warning: The diff is empty. Nothing to estimate.", file=sys.stderr)
        return 0

    # --- Analyse ---
    try:
        from pr_cost_estimator.diff_analyzer import DiffAnalyzer
        from pr_cost_estimator.token_counter import TokenCounter
        from pr_cost_estimator.cost_models import CostCalculator
        from pr_cost_estimator.reporter import Reporter
        from pr_cost_estimator.advisor import Advisor

        analyzer = DiffAnalyzer()
        analysis = analyzer.analyze(diff_text)

        counter = TokenCounter()
        token_counts = counter.count_for_all_models(diff_text)

        calculator = CostCalculator()
        estimates = calculator.estimate_all(analysis, token_counts)

        output_format = "json" if args.output_json else "table"
        reporter = Reporter()
        reporter.report(estimates, output_format=output_format, output_file=args.output_file)

        if not args.no_advice:
            advisor = Advisor(threshold_usd=args.threshold)
            suggestions = advisor.advise(estimates)
            if suggestions:
                advisor.print_advice(suggestions)

    except Exception as exc:  # noqa: BLE001
        print(f"Error: An unexpected error occurred: {exc}", file=sys.stderr)
        return 1

    return 0


def main() -> None:
    """CLI entry point installed by setuptools.

    Calls :func:`run` and exits with the returned code.
    """
    sys.exit(run())


if __name__ == "__main__":
    main()
