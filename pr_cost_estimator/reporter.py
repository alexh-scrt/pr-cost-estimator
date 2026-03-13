"""Output formatting module for PR Cost Estimator.

Formats :class:`~pr_cost_estimator.cost_models.ModelCostEstimate` objects
as either:

* A Rich terminal table (default) with colour-coded cost columns
* Structured JSON for CI/CD consumption and downstream tooling

The Rich table includes a per-model breakdown with input/output tokens,
individual costs, total cost, context window utilisation, and an overflow
flag.  A summary panel below the table shows diff-level metadata.

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
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from pr_cost_estimator.cost_models import ModelCostEstimate


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class Reporter:
    """Formats and outputs :class:`~pr_cost_estimator.cost_models.ModelCostEstimate`
    objects as a Rich table or JSON.

    The reporter is stateless; a single instance can be reused to produce
    multiple reports.
    """

    def report(
        self,
        estimates: List["ModelCostEstimate"],
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
            output_file: If provided, write output to this file path
                instead of stdout.

        Raises:
            ValueError: If ``output_format`` is not ``'table'`` or
                ``'json'``.
        """
        if output_format not in ("table", "json"):
            raise ValueError(
                f"Unknown output format '{output_format}'. "
                f"Supported formats: 'table', 'json'."
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
        estimates: List["ModelCostEstimate"],
    ) -> Dict:
        """Serialise a list of estimates to a plain Python dictionary.

        The resulting dict is suitable for direct JSON serialisation.  It
        includes a top-level ``generated_at`` timestamp, a ``summary``
        section with diff metadata, and a ``models`` list with one entry
        per estimate.

        Args:
            estimates: List of cost estimate objects to serialise.

        Returns:
            A dict with keys ``'generated_at'``, ``'summary'``, and
            ``'models'``.
        """
        if not estimates:
            summary: Dict = {
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
        estimates: List["ModelCostEstimate"],
        output_file: Optional[str],
    ) -> None:
        """Serialise estimates to JSON and write to file or stdout.

        Args:
            estimates: List of cost estimate objects.
            output_file: Optional path to write JSON output; uses stdout
                when ``None``.
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
    # Table output
    # ------------------------------------------------------------------

    def _report_table(
        self,
        estimates: List["ModelCostEstimate"],
        output_file: Optional[str],
    ) -> None:
        """Dispatch to the Rich table renderer, falling back to plain text.

        Args:
            estimates: List of cost estimate objects.
            output_file: Optional path for output.
        """
        try:
            self._render_rich_table(estimates, output_file)
        except ImportError:
            self._render_plain_table(estimates, output_file)

    def _render_rich_table(
        self,
        estimates: List["ModelCostEstimate"],
        output_file: Optional[str],
    ) -> None:
        """Render a fully formatted Rich cost table.

        The table has columns for model name, provider, token counts, per-type
        costs, total cost, context window utilisation, and an overflow flag.
        A summary panel below shows diff-level metadata.

        Args:
            estimates: List of cost estimate objects.
            output_file: Optional file path; if set the table is written as
                plain text with markup disabled so the file contains no ANSI
                escape codes.

        Raises:
            ImportError: If Rich is not installed.
        """
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        table = Table(
            title="\U0001f4b0  PR AI Review Cost Estimate",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title_style="bold white",
            padding=(0, 1),
            expand=False,
        )

        table.add_column("Model", style="cyan", no_wrap=True, min_width=20)
        table.add_column("Provider", style="dim", no_wrap=True)
        table.add_column("Input Tokens", justify="right", min_width=12)
        table.add_column("Output Tokens", justify="right", min_width=13)
        table.add_column("Input Cost", justify="right", min_width=10)
        table.add_column("Output Cost", justify="right", min_width=11)
        table.add_column("Total Cost", justify="right", style="bold", min_width=10)
        table.add_column("Context Use", justify="right", min_width=11)
        table.add_column("Status", justify="center", min_width=8)

        for e in estimates:
            total_style = _cost_style(e.total_cost_usd)
            context_str = f"{e.context_utilization_pct:.1f}%"

            if e.exceeds_context_window:
                status_str = "[bold red]OVERFLOW[/bold red]"
            else:
                status_str = "[green]OK[/green]"

            table.add_row(
                e.model_name,
                e.provider,
                f"{e.input_tokens:,}",
                f"{e.output_tokens:,}",
                f"${e.input_cost_usd:.4f}",
                f"${e.output_cost_usd:.4f}",
                f"[{total_style}]${e.total_cost_usd:.4f}[/{total_style}]",
                context_str,
                status_str,
            )

        # Build summary text
        summary_parts: List[str] = []
        if estimates:
            ref = estimates[0]
            summary_parts = [
                f"[bold]Files changed:[/bold] {ref.total_files_changed}",
                f"[bold]Lines changed:[/bold] {ref.total_lines_changed:,}",
                f"[bold]Complexity score:[/bold] {ref.complexity_score}",
            ]

        if output_file:
            with open(output_file, "w", encoding="utf-8") as fh:
                console = Console(file=fh, highlight=False, markup=False, width=120)
                console.print(table)
                if summary_parts:
                    # Strip markup for plain-text file output
                    plain_summary = (
                        f"Files changed: {estimates[0].total_files_changed}  "
                        f"Lines changed: {estimates[0].total_lines_changed:,}  "
                        f"Complexity score: {estimates[0].complexity_score}"
                    )
                    console.print(plain_summary)
        else:
            console = Console()
            console.print(table)
            if summary_parts:
                console.print(
                    Panel(
                        "  ".join(summary_parts),
                        title="Diff Summary",
                        border_style="dim",
                        padding=(0, 1),
                    )
                )

    def _render_plain_table(
        self,
        estimates: List["ModelCostEstimate"],
        output_file: Optional[str],
    ) -> None:
        """Fallback plain-text table renderer used when Rich is unavailable.

        Produces a fixed-width ASCII table with the same columns as the
        Rich renderer.

        Args:
            estimates: List of cost estimate objects.
            output_file: Optional file path for output; uses stdout when
                ``None``.
        """
        sep = "-" * 100
        header = (
            f"{'Model':<24} {'Provider':<12} "
            f"{'Input Tok':>11} {'Output Tok':>11} "
            f"{'Input $':>10} {'Output $':>10} {'Total $':>10} "
            f"{'Context%':>9} {'Status':>8}"
        )

        lines: List[str] = [
            "",
            "PR AI Review Cost Estimate",
            "=" * 100,
            header,
            sep,
        ]

        for e in estimates:
            overflow = " OVERFLOW" if e.exceeds_context_window else "      OK"
            row = (
                f"{e.model_name:<24} {e.provider:<12} "
                f"{e.input_tokens:>11,} {e.output_tokens:>11,} "
                f"${e.input_cost_usd:>9.4f} ${e.output_cost_usd:>9.4f} "
                f"${e.total_cost_usd:>9.4f} "
                f"{e.context_utilization_pct:>8.1f}%"
                f"{overflow}"
            )
            lines.append(row)

        lines.append("=" * 100)

        if estimates:
            ref = estimates[0]
            lines.append(
                f"Files changed: {ref.total_files_changed}  "
                f"Lines changed: {ref.total_lines_changed:,}  "
                f"Complexity score: {ref.complexity_score}"
            )

        lines.append("")
        output = "\n".join(lines)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as fh:
                fh.write(output)
                fh.write("\n")
        else:
            print(output)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _cost_style(cost: float) -> str:
    """Return a Rich style name based on the magnitude of a dollar cost.

    Costs below $1.00 are styled green (cheap), between $1.00 and $5.00
    are styled yellow (moderate), and $5.00 or above are styled red
    (expensive).

    Args:
        cost: Dollar cost value.

    Returns:
        A Rich-compatible style name string.
    """
    if cost < 1.0:
        return "green"
    if cost < 5.0:
        return "yellow"
    return "red"
