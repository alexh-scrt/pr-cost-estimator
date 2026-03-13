"""AI model pricing definitions and cost calculation logic.

Defines pricing data (input/output cost per million tokens) and context
window limits for each supported model, then computes estimated dollar
costs from token counts.

Pricing is sourced from publicly available provider pricing pages and
should be updated when providers change their rates.

Supported models:

* GPT-4o (OpenAI)
* GPT-4 (OpenAI)
* Claude 3.5 Sonnet (Anthropic)
* Claude 3 Haiku (Anthropic)
* Gemini 1.5 Pro (Google)

Typical usage::

    from pr_cost_estimator.cost_models import CostCalculator

    calculator = CostCalculator()
    estimates = calculator.estimate_all(diff_analysis, token_counts)
    for est in estimates:
        print(est.model_name, est.total_cost_usd)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Model pricing data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPricing:
    """Pricing specification for a single AI model."""

    model_id: str
    """Internal identifier used throughout the codebase."""

    display_name: str
    """Human-readable model name for reports."""

    provider: str
    """Provider name (e.g. 'OpenAI', 'Anthropic', 'Google')."""

    input_cost_per_million_tokens: float
    """USD cost per 1,000,000 input tokens."""

    output_cost_per_million_tokens: float
    """USD cost per 1,000,000 output tokens."""

    context_window_tokens: int
    """Maximum context window in tokens."""

    notes: str = ""
    """Optional notes about the pricing or model."""

    def input_cost(self, token_count: int) -> float:
        """Compute USD cost for a given number of input tokens.

        Args:
            token_count: Number of input tokens.

        Returns:
            Dollar cost as a float.
        """
        return (token_count / 1_000_000) * self.input_cost_per_million_tokens

    def output_cost(self, token_count: int) -> float:
        """Compute USD cost for a given number of output tokens.

        Args:
            token_count: Number of output tokens.

        Returns:
            Dollar cost as a float.
        """
        return (token_count / 1_000_000) * self.output_cost_per_million_tokens


# Pricing as of Q1 2025 — update when providers change rates.
MODEL_PRICING: Dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(
        model_id="gpt-4o",
        display_name="GPT-4o",
        provider="OpenAI",
        input_cost_per_million_tokens=2.50,
        output_cost_per_million_tokens=10.00,
        context_window_tokens=128_000,
        notes="Pricing as of 2024-11. Cached input may be cheaper.",
    ),
    "gpt-4": ModelPricing(
        model_id="gpt-4",
        display_name="GPT-4",
        provider="OpenAI",
        input_cost_per_million_tokens=30.00,
        output_cost_per_million_tokens=60.00,
        context_window_tokens=8_192,
        notes="Legacy GPT-4 (8k context). GPT-4 Turbo would be cheaper.",
    ),
    "claude-3-5-sonnet": ModelPricing(
        model_id="claude-3-5-sonnet",
        display_name="Claude 3.5 Sonnet",
        provider="Anthropic",
        input_cost_per_million_tokens=3.00,
        output_cost_per_million_tokens=15.00,
        context_window_tokens=200_000,
        notes="Claude 3.5 Sonnet pricing as of 2024-10.",
    ),
    "claude-3-haiku": ModelPricing(
        model_id="claude-3-haiku",
        display_name="Claude 3 Haiku",
        provider="Anthropic",
        input_cost_per_million_tokens=0.25,
        output_cost_per_million_tokens=1.25,
        context_window_tokens=200_000,
        notes="Anthropic's most affordable model as of 2024-10.",
    ),
    "gemini-1-5-pro": ModelPricing(
        model_id="gemini-1-5-pro",
        display_name="Gemini 1.5 Pro",
        provider="Google",
        input_cost_per_million_tokens=1.25,
        output_cost_per_million_tokens=5.00,
        context_window_tokens=2_000_000,
        notes="Pricing for prompts up to 128k tokens as of 2024-10.",
    ),
}


# ---------------------------------------------------------------------------
# Cost estimate result
# ---------------------------------------------------------------------------

@dataclass
class ModelCostEstimate:
    """Cost estimate result for a single AI model."""

    model_id: str
    """Internal model identifier."""

    model_name: str
    """Human-readable model name."""

    provider: str
    """Provider name."""

    input_tokens: int
    """Total input tokens (diff + system prompt)."""

    output_tokens: int
    """Estimated output tokens."""

    input_cost_usd: float
    """Dollar cost for input tokens."""

    output_cost_usd: float
    """Dollar cost for estimated output tokens."""

    context_window_tokens: int
    """Model's maximum context window."""

    exceeds_context_window: bool
    """True if total tokens exceed the model's context window."""

    total_files_changed: int = 0
    """Number of files changed (from diff analysis)."""

    total_lines_changed: int = 0
    """Total lines changed (from diff analysis)."""

    complexity_score: int = 0
    """Aggregate complexity score (from diff analysis)."""

    @property
    def total_cost_usd(self) -> float:
        """Combined input + output cost in USD."""
        return self.input_cost_usd + self.output_cost_usd

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def context_utilization_pct(self) -> float:
        """Percentage of the context window consumed by input tokens."""
        if self.context_window_tokens == 0:
            return 0.0
        return min(100.0, (self.input_tokens / self.context_window_tokens) * 100)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class CostCalculator:
    """Computes cost estimates from token counts and diff analysis.

    Combines token counts from :class:`~pr_cost_estimator.token_counter.TokenCounter`
    with pricing data from :data:`MODEL_PRICING` to produce
    :class:`ModelCostEstimate` objects.
    """

    def __init__(self, pricing: Optional[Dict[str, ModelPricing]] = None) -> None:
        """Initialise the calculator.

        Args:
            pricing: Optional custom pricing dict; defaults to
                :data:`MODEL_PRICING` if not provided.
        """
        self.pricing: Dict[str, ModelPricing] = pricing if pricing is not None else MODEL_PRICING

    def estimate(
        self,
        model_id: str,
        token_count: "TokenCount",  # type: ignore[name-defined]  # noqa: F821
        diff_analysis: Optional["DiffAnalysis"] = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> ModelCostEstimate:
        """Compute a cost estimate for a single model.

        Args:
            model_id: Model identifier matching a key in :data:`MODEL_PRICING`.
            token_count: A :class:`~pr_cost_estimator.token_counter.TokenCount`
                for the diff text.
            diff_analysis: Optional :class:`~pr_cost_estimator.diff_analyzer.DiffAnalysis`
                used to populate informational fields.

        Returns:
            A :class:`ModelCostEstimate` with full cost breakdown.

        Raises:
            ValueError: If ``model_id`` is not in the pricing table.
        """
        if model_id not in self.pricing:
            raise ValueError(
                f"No pricing data for model '{model_id}'. "
                f"Known models: {sorted(self.pricing)}"
            )

        mp = self.pricing[model_id]
        input_tokens = token_count.total_input_tokens
        output_tokens = token_count.estimated_output_tokens

        input_cost = mp.input_cost(input_tokens)
        output_cost = mp.output_cost(output_tokens)
        exceeds = (input_tokens + output_tokens) > mp.context_window_tokens

        return ModelCostEstimate(
            model_id=model_id,
            model_name=mp.display_name,
            provider=mp.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            context_window_tokens=mp.context_window_tokens,
            exceeds_context_window=exceeds,
            total_files_changed=diff_analysis.total_files_changed if diff_analysis else 0,
            total_lines_changed=diff_analysis.total_lines_changed if diff_analysis else 0,
            complexity_score=diff_analysis.total_complexity_score if diff_analysis else 0,
        )

    def estimate_all(
        self,
        diff_analysis: "DiffAnalysis",  # type: ignore[name-defined]  # noqa: F821
        token_counts: Dict[str, "TokenCount"],  # type: ignore[name-defined]  # noqa: F821
    ) -> List[ModelCostEstimate]:
        """Compute cost estimates for all models with available token counts.

        Args:
            diff_analysis: A :class:`~pr_cost_estimator.diff_analyzer.DiffAnalysis`
                for contextual metadata.
            token_counts: Dict mapping model_id ->
                :class:`~pr_cost_estimator.token_counter.TokenCount`.

        Returns:
            List of :class:`ModelCostEstimate` objects, one per model.
        """
        results: List[ModelCostEstimate] = []
        for model_id, tc in token_counts.items():
            if model_id in self.pricing:
                results.append(self.estimate(model_id, tc, diff_analysis))
        # Sort by ascending total cost for easy scanning
        results.sort(key=lambda e: e.total_cost_usd)
        return results

    def list_models(self) -> List[ModelPricing]:
        """Return all models in the pricing table, sorted by input cost.

        Returns:
            Sorted list of :class:`ModelPricing` objects.
        """
        return sorted(
            self.pricing.values(),
            key=lambda mp: mp.input_cost_per_million_tokens,
        )
