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

    from pr_cost_estimator.cost_models import CostCalculator, MODEL_PRICING

    calculator = CostCalculator()
    estimates = calculator.estimate_all(diff_analysis, token_counts)
    for est in estimates:
        print(est.model_name, f"${est.total_cost_usd:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from pr_cost_estimator.diff_analyzer import DiffAnalysis
    from pr_cost_estimator.token_counter import TokenCount


# ---------------------------------------------------------------------------
# Model pricing data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing:
    """Pricing specification for a single AI model.

    All costs are expressed in US dollars. Token limits reflect the
    model's maximum context window at the time of writing.

    Attributes:
        model_id: Internal identifier used throughout the codebase.
        display_name: Human-readable model name for reports.
        provider: Provider name (e.g. 'OpenAI', 'Anthropic', 'Google').
        input_cost_per_million_tokens: USD cost per 1,000,000 input tokens.
        output_cost_per_million_tokens: USD cost per 1,000,000 output tokens.
        context_window_tokens: Maximum context window in tokens.
        notes: Optional notes about the pricing or model.
    """

    model_id: str
    display_name: str
    provider: str
    input_cost_per_million_tokens: float
    output_cost_per_million_tokens: float
    context_window_tokens: int
    notes: str = ""

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

    def total_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute combined USD cost for input and output tokens.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Combined dollar cost as a float.
        """
        return self.input_cost(input_tokens) + self.output_cost(output_tokens)

    def fits_in_context(self, total_tokens: int) -> bool:
        """Return True if *total_tokens* fits within the context window.

        Args:
            total_tokens: Total token count (input + output combined).

        Returns:
            Boolean indicating whether the tokens fit.
        """
        return total_tokens <= self.context_window_tokens


# ---------------------------------------------------------------------------
# Pricing table — update when providers change rates
# Pricing as of Q1 2025.
# ---------------------------------------------------------------------------

MODEL_PRICING: Dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(
        model_id="gpt-4o",
        display_name="GPT-4o",
        provider="OpenAI",
        input_cost_per_million_tokens=2.50,
        output_cost_per_million_tokens=10.00,
        context_window_tokens=128_000,
        notes=(
            "OpenAI GPT-4o pricing as of 2024-11. "
            "Cached input tokens may be available at half price."
        ),
    ),
    "gpt-4": ModelPricing(
        model_id="gpt-4",
        display_name="GPT-4",
        provider="OpenAI",
        input_cost_per_million_tokens=30.00,
        output_cost_per_million_tokens=60.00,
        context_window_tokens=8_192,
        notes=(
            "Legacy GPT-4 (8k context) pricing. "
            "GPT-4 Turbo (128k) is available at lower rates."
        ),
    ),
    "claude-3-5-sonnet": ModelPricing(
        model_id="claude-3-5-sonnet",
        display_name="Claude 3.5 Sonnet",
        provider="Anthropic",
        input_cost_per_million_tokens=3.00,
        output_cost_per_million_tokens=15.00,
        context_window_tokens=200_000,
        notes=(
            "Anthropic Claude 3.5 Sonnet pricing as of 2024-10. "
            "Offers a 200k token context window."
        ),
    ),
    "claude-3-haiku": ModelPricing(
        model_id="claude-3-haiku",
        display_name="Claude 3 Haiku",
        provider="Anthropic",
        input_cost_per_million_tokens=0.25,
        output_cost_per_million_tokens=1.25,
        context_window_tokens=200_000,
        notes=(
            "Anthropic's most affordable production model as of 2024-10. "
            "Ideal for high-volume or cost-sensitive workflows."
        ),
    ),
    "gemini-1-5-pro": ModelPricing(
        model_id="gemini-1-5-pro",
        display_name="Gemini 1.5 Pro",
        provider="Google",
        input_cost_per_million_tokens=1.25,
        output_cost_per_million_tokens=5.00,
        context_window_tokens=2_000_000,
        notes=(
            "Google Gemini 1.5 Pro pricing for prompts ≤128k tokens as of 2024-10. "
            "Prompts >128k tokens are charged at $2.50/$10.00 per million."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Cost estimate result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelCostEstimate:
    """Cost estimate result for a single AI model.

    Combines token counts, pricing calculations, and diff metadata into
    a single result object that the reporter and advisor consume.

    Attributes:
        model_id: Internal model identifier matching a :data:`MODEL_PRICING` key.
        model_name: Human-readable model name (e.g. 'GPT-4o').
        provider: Provider name (e.g. 'OpenAI').
        input_tokens: Total input tokens (diff content + system prompt overhead).
        output_tokens: Estimated output tokens (model response length estimate).
        input_cost_usd: Dollar cost for the input tokens.
        output_cost_usd: Dollar cost for the estimated output tokens.
        context_window_tokens: Model's maximum context window size.
        exceeds_context_window: True when (input + output) tokens exceed the window.
        total_files_changed: Number of files changed (from diff analysis).
        total_lines_changed: Total lines changed (added + removed).
        complexity_score: Aggregate cyclomatic-proxy complexity score.
    """

    model_id: str
    model_name: str
    provider: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    context_window_tokens: int
    exceeds_context_window: bool
    total_files_changed: int = 0
    total_lines_changed: int = 0
    complexity_score: int = 0

    @property
    def total_cost_usd(self) -> float:
        """Combined input + output cost in USD.

        Returns:
            Sum of input and output costs.
        """
        return self.input_cost_usd + self.output_cost_usd

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output).

        Returns:
            Sum of input and output token counts.
        """
        return self.input_tokens + self.output_tokens

    @property
    def context_utilization_pct(self) -> float:
        """Percentage of the context window consumed by input tokens.

        Returns:
            Float percentage clamped to [0.0, 100.0].
        """
        if self.context_window_tokens == 0:
            return 0.0
        return min(100.0, (self.input_tokens / self.context_window_tokens) * 100.0)

    def to_dict(self) -> Dict:
        """Serialise this estimate to a plain Python dictionary.

        Returns:
            Dict suitable for JSON serialisation.
        """
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost_usd": round(self.input_cost_usd, 6),
            "output_cost_usd": round(self.output_cost_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "context_window_tokens": self.context_window_tokens,
            "context_utilization_pct": round(self.context_utilization_pct, 2),
            "exceeds_context_window": self.exceeds_context_window,
            "total_files_changed": self.total_files_changed,
            "total_lines_changed": self.total_lines_changed,
            "complexity_score": self.complexity_score,
        }


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class CostCalculator:
    """Computes cost estimates from token counts and diff analysis.

    Combines :class:`~pr_cost_estimator.token_counter.TokenCount` objects
    produced by the token counter with pricing data from :data:`MODEL_PRICING`
    to produce :class:`ModelCostEstimate` results.

    Example::

        from pr_cost_estimator.cost_models import CostCalculator
        from pr_cost_estimator.token_counter import TokenCounter
        from pr_cost_estimator.diff_analyzer import DiffAnalyzer

        analyzer = DiffAnalyzer()
        analysis = analyzer.analyze(diff_text)

        counter = TokenCounter()
        token_counts = counter.count_for_all_models(diff_text)

        calc = CostCalculator()
        estimates = calc.estimate_all(analysis, token_counts)
    """

    def __init__(self, pricing: Optional[Dict[str, ModelPricing]] = None) -> None:
        """Initialise the calculator.

        Args:
            pricing: Optional custom pricing dict mapping model_id ->
                :class:`ModelPricing`. Defaults to the global
                :data:`MODEL_PRICING` table if not provided.
        """
        self.pricing: Dict[str, ModelPricing] = (
            pricing if pricing is not None else MODEL_PRICING
        )

    def estimate(
        self,
        model_id: str,
        token_count: "TokenCount",
        diff_analysis: Optional["DiffAnalysis"] = None,
    ) -> ModelCostEstimate:
        """Compute a cost estimate for a single model.

        Args:
            model_id: Model identifier matching a key in :attr:`pricing`.
            token_count: A :class:`~pr_cost_estimator.token_counter.TokenCount`
                describing the diff's token usage for this model.
            diff_analysis: Optional :class:`~pr_cost_estimator.diff_analyzer.DiffAnalysis`
                used to populate informational metadata fields on the estimate.

        Returns:
            A fully populated :class:`ModelCostEstimate`.

        Raises:
            ValueError: If ``model_id`` is not present in the pricing table.
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

        # Pull diff metadata when available
        files_changed = 0
        lines_changed = 0
        complexity = 0
        if diff_analysis is not None:
            files_changed = diff_analysis.total_files_changed
            lines_changed = diff_analysis.total_lines_changed
            complexity = diff_analysis.total_complexity_score

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
            total_files_changed=files_changed,
            total_lines_changed=lines_changed,
            complexity_score=complexity,
        )

    def estimate_all(
        self,
        diff_analysis: "DiffAnalysis",
        token_counts: Dict[str, "TokenCount"],
    ) -> List[ModelCostEstimate]:
        """Compute cost estimates for all models present in both pricing and token counts.

        Models that appear in ``token_counts`` but lack a pricing entry are
        silently skipped, allowing partial pricing tables to be used.

        Args:
            diff_analysis: A :class:`~pr_cost_estimator.diff_analyzer.DiffAnalysis`
                providing contextual metadata (file count, line count, complexity).
            token_counts: Dict mapping model_id ->
                :class:`~pr_cost_estimator.token_counter.TokenCount`.

        Returns:
            List of :class:`ModelCostEstimate` objects sorted by ascending
            total cost (cheapest model first).
        """
        results: List[ModelCostEstimate] = []
        for model_id, tc in token_counts.items():
            if model_id in self.pricing:
                results.append(self.estimate(model_id, tc, diff_analysis))
        results.sort(key=lambda e: e.total_cost_usd)
        return results

    def list_models(self) -> List[ModelPricing]:
        """Return all models in the pricing table sorted by ascending input cost.

        Returns:
            Sorted list of :class:`ModelPricing` objects.
        """
        return sorted(
            self.pricing.values(),
            key=lambda mp: mp.input_cost_per_million_tokens,
        )

    def get_pricing(self, model_id: str) -> ModelPricing:
        """Retrieve the pricing entry for a specific model.

        Args:
            model_id: Model identifier to look up.

        Returns:
            The :class:`ModelPricing` for the requested model.

        Raises:
            ValueError: If ``model_id`` is not in the pricing table.
        """
        if model_id not in self.pricing:
            raise ValueError(
                f"No pricing data for model '{model_id}'. "
                f"Known models: {sorted(self.pricing)}"
            )
        return self.pricing[model_id]

    def cheapest_model(self) -> ModelPricing:
        """Return the model with the lowest input cost per million tokens.

        Returns:
            The :class:`ModelPricing` with the minimum input cost.

        Raises:
            ValueError: If the pricing table is empty.
        """
        if not self.pricing:
            raise ValueError("Pricing table is empty.")
        return min(
            self.pricing.values(),
            key=lambda mp: mp.input_cost_per_million_tokens,
        )

    def most_capable_model(self) -> ModelPricing:
        """Return the model with the largest context window.

        Returns:
            The :class:`ModelPricing` with the maximum context window.

        Raises:
            ValueError: If the pricing table is empty.
        """
        if not self.pricing:
            raise ValueError("Pricing table is empty.")
        return max(
            self.pricing.values(),
            key=lambda mp: mp.context_window_tokens,
        )
