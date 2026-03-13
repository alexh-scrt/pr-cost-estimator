"""Output formatting module for PR Cost Estimator.

Formats :class:`~pr_cost_estimator.cost_models.ModelCostEstimate` objects
as either:

* A Rich terminal table (default) with colour-coded cost columns
* Structured JSON for CI/CD consumption and downstream tooling

Typical usage::

    from pr_cost_estimator.reporter import Reporter

    reporter = Reporter()
    reporter.report(estimates, output_format="table")
    reporter.report(estimates, output_format="json", output_file="costs.json")
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import IO, Dict, List, Optional


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class Reporter:
    """Formats and outputs cost estimates as a table or JSON."""

    def report(
        self,
        estimates: List["ModelCostEstimate"],  # type: ignore[name-defined]  # noqa: F821
        output_format: str = "table",
        output_file: Optional[str] = None,
    ) -> None:
        """Format and output cost estimates.

        Args:
            estimates: List of
                :class:`~pr_cost_estimator.cost_models.ModelCostEstimate`
                objects to render.
            output_format: ``'table'`` for Rich terminal output, or
                ``'json'`` for machine-readable JSON.
            output_file: If provided, write output to this path instead
                of stdout.

        Raises:
            ValueError: If ``output_format`` is not ``'table'`` or ``'json'``.
        """
        if output_format not in ("table", "json"):
            raise ValueError(
                f"Unknown output format '{output_format}'. Use 'table' or 'json'."
            )

        if output_format == "json":
            self._report_json(estimates, output_file)
        else:
            self._report_table(estimates, output_file)

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------

    def to_dict(
        self,
        estimates: List["ModelCostEstimate"],  # type: ignore[name-defined]  # noqa: F821
    ) -> Dict:
        """Serialise estimates to a plain Python dict.

        Args:
            estimates: List of cost estimate objects.

        Returns:
            A dict suitable for JSON serialisation.
        """
        if not estimates:
            summary = {
                "total_files_changed": 0,
                "total_lines_changed": 0,
                "complexity_score": 0,
            }
        else:
            ref = estimates[0]
            summary = {
                "total_files_changed": ref.total_files_changed,
                "total_lines_changed": ref.total_lines_changed,
                "complexity_score": ref.complexity_score,
            }

        models = []
        for e in estimates:
            models.append(
                {
                    "model_id": e.model_id,
                    "model_name": e.model_name,
                    "provider": e.provider,
                    "input_tokens": e.input_tokens,
                    "output_tokens": e.output_tokens,
                    "total_tokens": e.total_tokens,
                    "input_cost_usd": round(e.input_cost_usd, 6),
                    "output_cost_usd": round(e.output_cost_usd, 6),
                    "total_cost_usd": round(e.total_cost_usd, 6),
                    "context_window_tokens": e.context_window_tokens,
                    "context_utilization_pct": round(e.context_utilization_pct, 2),
                    "exceeds_context_window": e.exceeds_context_window,
                }
            )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "models": models,
        }

    def _report_json(
        self,
        estimates: List["ModelCostEstimate"],  # type: ignore[name-defined]  # noqa: F821
        output_file: Optional[str],
    ) -> None:
        """Write JSON output.

        Args:
            estimates: List of cost estimate objects.
            output_file: Optional file path; defaults to stdout.
        """
        data = self.to_dict(estimates)
        json_text = json.dumps(data, indent=2)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as fh:
                fh.write(json_text)
                fh.write("\n")
        else:
            print(json_text)

    # ------------------------------------------------------------------
    # Rich table output
    # ------------------------------------------------------------------

    def _report_table(
        self,
        estimates: List["ModelCostEstimate"],  # type: ignore[name-defined]  # noqa: F821
        output_file: Optional[str],
    ) -> None:
        """Render a Rich table.

        Args:
            estimates: List of cost estimate objects.
            output_file: Optional file path; if provided the table is
                written as plain text (no ANSI codes).
        """
        try:
            self._render_rich_table(estimates, output_file)
        except ImportError:
            self._render_plain_table(estimates, output_file)

    def _render_rich_table(
        self,
        estimates: List["ModelCostEstimate"],  # type: ignore[name-defined]  # noqa: F821
        output_file: Optional[str],
    ) -> None:
        """Use Rich to render a formatted cost table.

        Args:
            estimates: List of cost estimate objects.
            output_file: Optional file path for output.
        """
        from rich.console import Console
        from rich.table import Table
        from rich import box

        table = Table(
            title="PR AI Review Cost Estimate",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title_style="bold white",
            padding=(0, 1),
        )

        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Provider", style="dim")
        table.add_column("Input Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")
        table.add_column("Input Cost", justify="right")
        table.add_column("Output Cost", justify="right")
        table.add_column("Total Cost", justify="right", style="bold")
        table.add_column("Context Use", justify="right")
        table.add_column("Flags", justify="center")

        for e in estimates:
            total_style = _cost_style(e.total_cost_usd)
            context_str = f"{e.context_utilization_pct:.1f}%"
            flags = "[red]OVERFLOW[/red]" if e.exceeds_context_window else "[green]OK[/green]"

            table.add_row(
                e.model_name,
                e.provider,
                f"{e.input_tokens:,}",
                f"{e.output_tokens:,}",
                f"${e.input_cost_usd:.4f}",
                f"${e.output_cost_usd:.4f}",
                f"[{total_style}]${e.total_cost_usd:.4f}[/{total_style}]",
                context_str,
                flags,
            )

        # Summary panel
        if estimates:
            ref = estimates[0]
            summary_lines = [
                f"Files changed: {ref.total_files_changed}",
                f"Lines changed: {ref.total_lines_changed:,}",
                f"Complexity score: {ref.complexity_score}",
            ]

        if output_file:
            with open(output_file, "w", encoding="utf-8") as fh:
                console = Console(file=fh, highlight=False, markup=False)
                console.print(table)
                if estimates:
                    console.print("  ".join(summary_lines))
        else:
            console = Console()
            console.print(table)
            if estimates:
                from rich.panel import Panel
                from rich.columns import Columns
                from rich.text import Text

                summary_text = "  ".join(summary_lines)
                console.print(
                    Panel(summary_text, title="Diff Summary", border_style="dim", padding=(0, 1))
                )

    def _render_plain_table(
        self,
        estimates: List["ModelCostEstimate"],  # type: ignore[name-defined]  # noqa: F821
        output_file: Optional[str],
    ) -> None:
        """Fallback plain-text table renderer.

        Args:
            estimates: List of cost estimate objects.
            output_file: Optional file path for output.
        """
        lines: List[str] = []
        lines.append("PR AI Review Cost Estimate")
        lines.append("=" * 80)
        header = (
            f"{'Model':<25} {'Input Tok':>10} {'Output Tok':>10} "
            f"{'Input $':>10} {'Output $':>10} {'Total $':>10} {'Context%':>8}"
        )
        lines.append(header)
        lines.append("-" * 80)

        for e in estimates:
            overflow = " [OVERFLOW]" if e.exceeds_context_window else ""
            row = (
                f"{e.model_name:<25} {e.input_tokens:>10,} {e.output_tokens:>10,} "
                f"${e.input_cost_usd:>9.4f} ${e.output_cost_usd:>9.4f} "
                f"${e.total_cost_usd:>9.4f} {e.context_utilization_pct:>7.1f}%"
                f"{overflow}"
            )
            lines.append(row)

        lines.append("=" * 80)
        if estimates:
            ref = estimates[0]
            lines.append(
                f"Files: {ref.total_files_changed}  "
                f"Lines: {ref.total_lines_changed:,}  "
                f"Complexity: {ref.complexity_score}"
            )

        output = "\n".join(lines) + "\n"

        if output_file:
            with open(output_file, "w", encoding="utf-8") as fh:
                fh.write(output)
        else:
            print(output)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cost_style(cost: float) -> str:
    """Return a Rich style name based on cost magnitude.

    Args:
        cost: Dollar cost value.

    Returns:
        Rich style string suitable for use in markup.
    """
    if cost < 1.0:
        return "green"
    if cost < 5.0:
        return "yellow"
    return "red"
