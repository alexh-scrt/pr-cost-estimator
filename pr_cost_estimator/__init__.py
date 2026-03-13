"""PR Cost Estimator — public API surface for programmatic use.

This package estimates the token and dollar cost of running AI-based
code review (Claude, GPT-4, Gemini, etc.) on a pull request before
you commit to the expense.

Typical usage::

    from pr_cost_estimator import analyze_diff, estimate_costs, generate_report

    diff_text = open("my.diff").read()
    analysis = analyze_diff(diff_text)
    estimates = estimate_costs(analysis)
    generate_report(estimates)

Programmatic model listing::

    from pr_cost_estimator import list_models

    for model in list_models():
        print(model.display_name, model.input_cost_per_million_tokens)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from pr_cost_estimator.cost_models import ModelCostEstimate, ModelPricing
    from pr_cost_estimator.diff_analyzer import DiffAnalysis

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "analyze_diff",
    "estimate_costs",
    "generate_report",
    "get_advice",
    "fetch_github_pr_diff",
    "list_models",
]


def analyze_diff(diff_text: str) -> "DiffAnalysis":
    """Parse a unified diff string and return a structured analysis.

    Args:
        diff_text: Raw unified diff content as a string.

    Returns:
        A :class:`~pr_cost_estimator.diff_analyzer.DiffAnalysis` dataclass
        containing file counts, line counts, file types, and complexity
        heuristics.
    """
    from pr_cost_estimator.diff_analyzer import DiffAnalyzer

    analyzer = DiffAnalyzer()
    return analyzer.analyze(diff_text)


def estimate_costs(
    analysis: "DiffAnalysis",
) -> List["ModelCostEstimate"]:
    """Compute per-model cost estimates from a diff analysis.

    Args:
        analysis: A :class:`~pr_cost_estimator.diff_analyzer.DiffAnalysis`
            object produced by :func:`analyze_diff`.

    Returns:
        A list of :class:`~pr_cost_estimator.cost_models.ModelCostEstimate`
        objects, one per supported AI model, sorted by ascending total cost.
    """
    from pr_cost_estimator.cost_models import CostCalculator
    from pr_cost_estimator.token_counter import TokenCounter

    counter = TokenCounter()
    calculator = CostCalculator()
    token_counts = counter.count_for_all_models(analysis.raw_diff)
    return calculator.estimate_all(analysis, token_counts)


def generate_report(
    estimates: List["ModelCostEstimate"],
    output_format: str = "table",
    output_file: Optional[str] = None,
) -> None:
    """Format and print cost estimates as a Rich table or JSON.

    Args:
        estimates: List of
            :class:`~pr_cost_estimator.cost_models.ModelCostEstimate`
            objects from :func:`estimate_costs`.
        output_format: Either ``"table"`` for Rich terminal output or
            ``"json"`` for machine-readable output.
        output_file: Optional file path to write output to; if ``None``,
            writes to stdout.

    Raises:
        ValueError: If ``output_format`` is not ``'table'`` or ``'json'``.
    """
    from pr_cost_estimator.reporter import Reporter

    reporter = Reporter()
    reporter.report(estimates, output_format=output_format, output_file=output_file)


def get_advice(
    estimates: List["ModelCostEstimate"],
    threshold_usd: float = 5.0,
) -> List[str]:
    """Return advisory messages when estimated cost exceeds a threshold.

    Args:
        estimates: List of
            :class:`~pr_cost_estimator.cost_models.ModelCostEstimate`
            objects from :func:`estimate_costs`.
        threshold_usd: Dollar threshold above which advice is triggered.
            Defaults to ``5.0``.

    Returns:
        A list of human-readable suggestion title strings, or an empty
        list if all estimates are below the threshold.
    """
    from pr_cost_estimator.advisor import Advisor

    advisor = Advisor(threshold_usd=threshold_usd)
    suggestions = advisor.advise(estimates)
    return [s.title for s in suggestions]


def fetch_github_pr_diff(
    pr_url: str,
    github_token: Optional[str] = None,
) -> str:
    """Fetch a pull request diff from the GitHub REST API.

    Args:
        pr_url: Full GitHub PR URL, e.g.
            ``"https://github.com/owner/repo/pull/123"``.
        github_token: Optional GitHub personal access token for private
            repositories or higher rate limits.

    Returns:
        Raw unified diff text as a string.

    Raises:
        ValueError: If the URL format is not a recognisable GitHub PR URL.
        requests.HTTPError: If the GitHub API returns a non-2xx response.
    """
    from pr_cost_estimator.github_fetcher import GitHubFetcher

    fetcher = GitHubFetcher(token=github_token)
    return fetcher.fetch_diff(pr_url)


def list_models() -> List["ModelPricing"]:
    """Return all supported AI models sorted by ascending input cost.

    Returns:
        List of :class:`~pr_cost_estimator.cost_models.ModelPricing` objects,
        cheapest first.
    """
    from pr_cost_estimator.cost_models import CostCalculator

    calc = CostCalculator()
    return calc.list_models()
