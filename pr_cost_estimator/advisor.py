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
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Suggestion:
    """A single advisory suggestion."""

    title: str
    """Short one-line summary of the suggestion."""

    detail: str
    """Longer explanation of why and how to act on the suggestion."""

    severity: str
    """One of 'info', 'warning', or 'critical'."""

    saving_estimate: Optional[str] = None
    """Optional estimated saving, e.g. '~60% cost reduction'."""


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_HIGH_LINE_COUNT = 500
_VERY_HIGH_LINE_COUNT = 1500
_HIGH_FILE_COUNT = 20
_HIGH_COMPLEXITY = 100
_CHEAP_MODELS = {"claude-3-haiku"}  # models suggested as alternatives


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------

class Advisor:
    """Analyses cost estimates and diff metadata to produce suggestions.

    The advisor triggers suggestions when:

    * Any model's estimated cost exceeds ``threshold_usd``
    * The diff spans an unusually large number of files or lines
    * The complexity score is high (many branching constructs)
    * The diff would exceed a model's context window
    """

    def __init__(self, threshold_usd: float = 5.0) -> None:
        """Initialise the Advisor.

        Args:
            threshold_usd: Dollar threshold above which cost-based
                suggestions are emitted.
        """
        self.threshold_usd = threshold_usd

    def advise(
        self,
        estimates: List["ModelCostEstimate"],  # type: ignore[name-defined]  # noqa: F821
    ) -> List[Suggestion]:
        """Generate suggestions based on cost estimates.

        Args:
            estimates: List of :class:`~pr_cost_estimator.cost_models.ModelCostEstimate`
                objects from the cost calculator.

        Returns:
            List of :class:`Suggestion` objects. Empty list if everything
            looks fine.
        """
        if not estimates:
            return []

        suggestions: List[Suggestion] = []

        # Use a representative estimate (cheapest model) for diff metadata
        ref = estimates[0]
        total_lines = ref.total_lines_changed
        total_files = ref.total_files_changed
        complexity = ref.complexity_score

        max_cost = max(e.total_cost_usd for e in estimates)
        min_cost = min(e.total_cost_usd for e in estimates)

        # --- Cost threshold check ---
        expensive_estimates = [e for e in estimates if e.total_cost_usd > self.threshold_usd]
        if expensive_estimates:
            most_expensive = max(expensive_estimates, key=lambda e: e.total_cost_usd)
            suggestions.append(
                Suggestion(
                    title="Cost exceeds threshold",
                    detail=(
                        f"Estimated cost with {most_expensive.model_name} is "
                        f"${most_expensive.total_cost_usd:.2f}, which exceeds your "
                        f"${self.threshold_usd:.2f} threshold. "
                        f"Consider the suggestions below to reduce cost."
                    ),
                    severity="warning",
                )
            )

        # --- Suggest cheaper model ---
        non_cheap = [e for e in estimates if e.model_id not in _CHEAP_MODELS]
        cheap = [e for e in estimates if e.model_id in _CHEAP_MODELS]
        if non_cheap and cheap and max_cost > self.threshold_usd:
            cheapest = min(cheap, key=lambda e: e.total_cost_usd)
            savings_pct = 0.0
            if max_cost > 0:
                savings_pct = ((max_cost - cheapest.total_cost_usd) / max_cost) * 100
            suggestions.append(
                Suggestion(
                    title=f"Switch to {cheapest.model_name} for initial review",
                    detail=(
                        f"{cheapest.model_name} costs ~${cheapest.total_cost_usd:.4f} "
                        f"for this PR versus ~${max_cost:.2f} for the most expensive "
                        f"option. Use the cheaper model for a first-pass review and "
                        f"escalate only specific files to a premium model."
                    ),
                    severity="info",
                    saving_estimate=f"~{savings_pct:.0f}% cost reduction",
                )
            )

        # --- Suggest splitting the PR ---
        if total_lines > _HIGH_LINE_COUNT or total_files > _HIGH_FILE_COUNT:
            severity = "critical" if total_lines > _VERY_HIGH_LINE_COUNT else "warning"
            suggestions.append(
                Suggestion(
                    title="Consider splitting this PR",
                    detail=(
                        f"This PR changes {total_lines} lines across {total_files} files. "
                        f"Splitting it into 2–4 smaller PRs would reduce the per-review "
                        f"token count proportionally, lower cost, and make reviews faster "
                        f"and more focused."
                    ),
                    severity=severity,
                    saving_estimate="Proportional to number of sub-PRs created",
                )
            )

        # --- High complexity warning ---
        if complexity > _HIGH_COMPLEXITY:
            suggestions.append(
                Suggestion(
                    title="High complexity detected",
                    detail=(
                        f"The diff has a complexity score of {complexity}, suggesting "
                        f"many branching constructs in the new code. AI reviewers may "
                        f"need more output tokens to thoroughly review this, increasing "
                        f"cost. Simplifying logic or extracting helper functions could "
                        f"reduce both complexity and review cost."
                    ),
                    severity="info",
                )
            )

        # --- Context window overflow ---
        overflows = [e for e in estimates if e.exceeds_context_window]
        if overflows:
            model_names = ", ".join(e.model_name for e in overflows)
            suggestions.append(
                Suggestion(
                    title="Context window exceeded for some models",
                    detail=(
                        f"The diff is too large to fit in the context window of: "
                        f"{model_names}. These models will either truncate the diff "
                        f"or fail. Consider using a model with a larger context window "
                        f"(e.g. Claude 3.5 Sonnet: 200k tokens, Gemini 1.5 Pro: 2M tokens) "
                        f"or splitting the PR."
                    ),
                    severity="critical",
                )
            )

        # --- Limit scope suggestion ---
        if total_files > _HIGH_FILE_COUNT and max_cost > self.threshold_usd:
            suggestions.append(
                Suggestion(
                    title="Limit review scope to changed files only",
                    detail=(
                        f"If your AI review tool supports file filtering, configure it "
                        f"to review only the {total_files} changed files in this PR "
                        f"instead of scanning the full repository context. This can "
                        f"significantly reduce the number of tokens sent."
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

        Args:
            suggestions: List of :class:`Suggestion` objects to display.
            use_rich: If True, use Rich for coloured terminal output;
                falls back to plain text on import error.
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
            suggestions: List of suggestions to display.
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
            "info": "ℹ",
            "warning": "⚠",
            "critical": "✖",
        }

        for suggestion in suggestions:
            style = severity_styles.get(suggestion.severity, "white")
            icon = severity_icons.get(suggestion.severity, "•")
            title = Text(f"{icon} {suggestion.title}", style=style)
            body = suggestion.detail
            if suggestion.saving_estimate:
                body += f"\n[dim]Potential saving: {suggestion.saving_estimate}[/dim]"
            console.print(
                Panel(body, title=title, border_style=style, padding=(0, 1))
            )

    def _print_plain(self, suggestions: List[Suggestion]) -> None:
        """Render suggestions as plain text.

        Args:
            suggestions: List of suggestions to display.
        """
        print("\n=== Advisory Suggestions ===")
        for i, s in enumerate(suggestions, start=1):
            print(f"\n[{s.severity.upper()}] {i}. {s.title}")
            print(f"  {s.detail}")
            if s.saving_estimate:
                print(f"  Potential saving: {s.saving_estimate}")
        print()
