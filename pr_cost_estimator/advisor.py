"""Advisory module for PR Cost Estimator.

Generates human-readable suggestions when estimated AI review costs
exceed configurable thresholds. Suggestions aim to be actionable:

* Split the PR into smaller, focused changes
* Switch to a cheaper model for initial review
* Limit review scope to changed files only
* Remove large auto-generated or binary files from scope

Typical usage::

    from pr_cost_estimator.advisor import Advisor

    advisor = Advisor(threshold_usd=5.0)
    suggestions = advisor.advise(estimates)
    advisor.print_advice(suggestions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from pr_cost_estimator.cost_models import ModelCostEstimate


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Suggestion:
    """A single advisory suggestion.

    Attributes:
        title: Short one-line summary of the suggestion.
        detail: Longer explanation of why and how to act on the suggestion.
        severity: One of ``'info'``, ``'warning'``, or ``'critical'``.
        saving_estimate: Optional estimated saving, e.g. ``'~60% cost reduction'``.
    """

    title: str
    detail: str
    severity: str
    saving_estimate: Optional[str] = None


# ---------------------------------------------------------------------------
# Thresholds and constants
# ---------------------------------------------------------------------------

# Line-count thresholds for split-PR suggestions.
_HIGH_LINE_COUNT: int = 500
_VERY_HIGH_LINE_COUNT: int = 1_500

# File-count threshold for split-PR and scope-limiting suggestions.
_HIGH_FILE_COUNT: int = 20

# Complexity score threshold for a high-complexity warning.
_HIGH_COMPLEXITY: int = 100

# Model IDs considered "cheap" alternatives when suggesting a model switch.
_CHEAP_MODEL_IDS: frozenset[str] = frozenset({"claude-3-haiku"})


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


class Advisor:
    """Analyses cost estimates and diff metadata to produce actionable suggestions.

    The advisor triggers suggestions when:

    * Any model's estimated cost exceeds ``threshold_usd``
    * The diff spans an unusually large number of files or lines
    * The complexity score is high (many branching constructs)
    * The diff would exceed a model's context window

    Example::

        advisor = Advisor(threshold_usd=5.0)
        suggestions = advisor.advise(estimates)
        if suggestions:
            advisor.print_advice(suggestions)
    """

    def __init__(self, threshold_usd: float = 5.0) -> None:
        """Initialise the Advisor.

        Args:
            threshold_usd: Dollar threshold above which cost-based suggestions
                are emitted. Defaults to ``5.0``.
        """
        self.threshold_usd = threshold_usd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advise(
        self,
        estimates: List["ModelCostEstimate"],
    ) -> List[Suggestion]:
        """Generate suggestions based on cost estimates and diff metadata.

        Args:
            estimates: List of
                :class:`~pr_cost_estimator.cost_models.ModelCostEstimate`
                objects produced by the cost calculator.  The list may be
                empty, in which case an empty suggestion list is returned.

        Returns:
            List of :class:`Suggestion` objects ordered from most to least
            severe.  Returns an empty list when no suggestions are triggered.
        """
        if not estimates:
            return []

        suggestions: List[Suggestion] = []

        # Use the first (cheapest) estimate as the source of diff metadata;
        # all estimates share the same analysis metadata.
        ref = estimates[0]
        total_lines = ref.total_lines_changed
        total_files = ref.total_files_changed
        complexity = ref.complexity_score

        max_cost = max(e.total_cost_usd for e in estimates)
        min_cost = min(e.total_cost_usd for e in estimates)

        # ----------------------------------------------------------------
        # 1. Cost threshold exceeded
        # ----------------------------------------------------------------
        expensive = [e for e in estimates if e.total_cost_usd > self.threshold_usd]
        if expensive:
            most_exp = max(expensive, key=lambda e: e.total_cost_usd)
            suggestions.append(
                Suggestion(
                    title="Cost exceeds threshold",
                    detail=(
                        f"Estimated cost with {most_exp.model_name} is "
                        f"${most_exp.total_cost_usd:.2f}, which exceeds your "
                        f"${self.threshold_usd:.2f} threshold. "
                        f"Consider the suggestions below to reduce cost."
                    ),
                    severity="warning",
                )
            )

        # ----------------------------------------------------------------
        # 2. Suggest switching to a cheaper model
        # ----------------------------------------------------------------
        cheap_estimates = [e for e in estimates if e.model_id in _CHEAP_MODEL_IDS]
        pricey_estimates = [e for e in estimates if e.model_id not in _CHEAP_MODEL_IDS]
        if cheap_estimates and pricey_estimates and max_cost > self.threshold_usd:
            cheapest_alt = min(cheap_estimates, key=lambda e: e.total_cost_usd)
            savings_pct = 0.0
            if max_cost > 0:
                savings_pct = ((max_cost - cheapest_alt.total_cost_usd) / max_cost) * 100.0
            suggestions.append(
                Suggestion(
                    title=f"Switch to {cheapest_alt.model_name} for initial review",
                    detail=(
                        f"{cheapest_alt.model_name} costs approximately "
                        f"${cheapest_alt.total_cost_usd:.4f} for this PR, versus "
                        f"${max_cost:.2f} for the most expensive option. "
                        f"Use the cheaper model for a first-pass review and escalate "
                        f"only specific files to a premium model if needed."
                    ),
                    severity="info",
                    saving_estimate=f"~{savings_pct:.0f}% cost reduction",
                )
            )

        # ----------------------------------------------------------------
        # 3. Suggest splitting the PR
        # ----------------------------------------------------------------
        if total_lines > _HIGH_LINE_COUNT or total_files > _HIGH_FILE_COUNT:
            severity = "critical" if total_lines > _VERY_HIGH_LINE_COUNT else "warning"
            suggestions.append(
                Suggestion(
                    title="Consider splitting this PR",
                    detail=(
                        f"This PR changes {total_lines:,} lines across "
                        f"{total_files} file(s). "
                        f"Splitting it into 2–4 smaller, focused PRs would reduce "
                        f"the per-review token count proportionally, lower cost, and "
                        f"make reviews faster and more targeted."
                    ),
                    severity=severity,
                    saving_estimate="Proportional to the number of sub-PRs created",
                )
            )

        # ----------------------------------------------------------------
        # 4. High complexity warning
        # ----------------------------------------------------------------
        if complexity > _HIGH_COMPLEXITY:
            suggestions.append(
                Suggestion(
                    title="High complexity detected",
                    detail=(
                        f"The diff has a complexity score of {complexity}, indicating "
                        f"many branching constructs in the new code. AI reviewers may "
                        f"require more output tokens to thoroughly analyse this, "
                        f"increasing the overall cost. Simplifying logic or extracting "
                        f"helper functions could reduce both complexity and review cost."
                    ),
                    severity="info",
                )
            )

        # ----------------------------------------------------------------
        # 5. Context window overflow
        # ----------------------------------------------------------------
        overflows = [e for e in estimates if e.exceeds_context_window]
        if overflows:
            overflow_names = ", ".join(e.model_name for e in overflows)
            suggestions.append(
                Suggestion(
                    title="Context window exceeded for some models",
                    detail=(
                        f"The diff is too large to fit in the context window of: "
                        f"{overflow_names}. These models will either truncate the diff "
                        f"or fail entirely. Consider using a model with a larger context "
                        f"window (e.g. Claude 3.5 Sonnet: 200k tokens, "
                        f"Gemini 1.5 Pro: 2M tokens) or splitting the PR."
                    ),
                    severity="critical",
                )
            )

        # ----------------------------------------------------------------
        # 6. Limit review scope
        # ----------------------------------------------------------------
        if total_files > _HIGH_FILE_COUNT and max_cost > self.threshold_usd:
            suggestions.append(
                Suggestion(
                    title="Limit review scope to changed files only",
                    detail=(
                        f"If your AI review tool supports file filtering, configure it "
                        f"to review only the {total_files} changed files in this PR "
                        f"instead of scanning the full repository context. This can "
                        f"significantly reduce the number of tokens submitted."
                    ),
                    severity="info",
                    saving_estimate="Varies by tool configuration",
                )
            )

        return suggestions

    def print_advice(
        self,
        suggestions: List[Suggestion],
        use_rich: bool = True,
    ) -> None:
        """Print advisory suggestions to stdout.

        Attempts to use Rich for colourised terminal output; falls back to
        plain text if Rich is not importable.

        Args:
            suggestions: List of :class:`Suggestion` objects to display.
            use_rich: If ``True`` (default), attempt to use Rich for
                formatted output.
        """
        if not suggestions:
            return

        if use_rich:
            try:
                self._print_rich(suggestions)
                return
            except ImportError:
                pass

        self._print_plain(suggestions)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _print_rich(self, suggestions: List[Suggestion]) -> None:
        """Render suggestions with Rich formatting.

        Args:
            suggestions: List of :class:`Suggestion` objects to display.

        Raises:
            ImportError: If Rich is not installed.
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        console.print()

        severity_styles = {
            "info": "cyan",
            "warning": "yellow",
            "critical": "bold red",
        }
        severity_icons = {
            "info": "\u2139",   # ℹ
            "warning": "\u26a0",  # ⚠
            "critical": "\u2716",  # ✖
        }

        for suggestion in suggestions:
            style = severity_styles.get(suggestion.severity, "white")
            icon = severity_icons.get(suggestion.severity, "\u2022")
            title_text = Text(f"{icon}  {suggestion.title}", style=style)
            body = suggestion.detail
            if suggestion.saving_estimate:
                body += f"\n\n[dim]Potential saving: {suggestion.saving_estimate}[/dim]"
            console.print(
                Panel(
                    body,
                    title=title_text,
                    border_style=style,
                    padding=(0, 1),
                )
            )
        console.print()

    def _print_plain(self, suggestions: List[Suggestion]) -> None:
        """Render suggestions as plain text to stdout.

        Args:
            suggestions: List of :class:`Suggestion` objects to display.
        """
        print("\n=== Advisory Suggestions ===")
        for i, s in enumerate(suggestions, start=1):
            print(f"\n[{s.severity.upper()}] {i}. {s.title}")
            print(f"  {s.detail}")
            if s.saving_estimate:
                print(f"  Potential saving: {s.saving_estimate}")
        print()
