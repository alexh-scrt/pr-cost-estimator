"""Unit tests for pr_cost_estimator.cost_models.

Covers:
- ModelPricing cost calculation math
- MODEL_PRICING table completeness and sanity checks
- CostCalculator.estimate() for each supported model
- CostCalculator.estimate_all() ordering and filtering
- ModelCostEstimate computed properties
- Edge cases: zero tokens, context window overflow, custom pricing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pytest

from pr_cost_estimator.cost_models import (
    MODEL_PRICING,
    CostCalculator,
    ModelCostEstimate,
    ModelPricing,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeTokenCount:
    """Minimal stand-in for TokenCount used in isolation tests."""

    model_id: str
    diff_tokens: int
    system_prompt_tokens: int
    estimated_output_tokens: int

    @property
    def total_input_tokens(self) -> int:
        return self.diff_tokens + self.system_prompt_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.estimated_output_tokens


@dataclass
class _FakeDiffAnalysis:
    """Minimal stand-in for DiffAnalysis."""

    raw_diff: str = ""
    total_files_changed: int = 3
    total_lines_changed: int = 150
    total_complexity_score: int = 20


def _make_token_count(
    model_id: str,
    diff_tokens: int = 1000,
    system_tokens: int = 512,
    output_tokens: int = 1024,
) -> _FakeTokenCount:
    return _FakeTokenCount(
        model_id=model_id,
        diff_tokens=diff_tokens,
        system_prompt_tokens=system_tokens,
        estimated_output_tokens=output_tokens,
    )


def _make_token_counts(
    diff_tokens: int = 1000,
    system_tokens: int = 512,
    output_tokens: int = 1024,
) -> Dict[str, _FakeTokenCount]:
    """Build a token_counts dict for all MODEL_PRICING keys."""
    return {
        mid: _make_token_count(mid, diff_tokens, system_tokens, output_tokens)
        for mid in MODEL_PRICING
    }


# ---------------------------------------------------------------------------
# ModelPricing tests
# ---------------------------------------------------------------------------


class TestModelPricing:
    """Tests for ModelPricing cost arithmetic."""

    def test_input_cost_zero_tokens(self) -> None:
        mp = MODEL_PRICING["gpt-4o"]
        assert mp.input_cost(0) == 0.0

    def test_output_cost_zero_tokens(self) -> None:
        mp = MODEL_PRICING["gpt-4o"]
        assert mp.output_cost(0) == 0.0

    def test_input_cost_one_million_tokens(self) -> None:
        mp = MODEL_PRICING["gpt-4o"]
        # 1M tokens at $2.50/M = $2.50
        assert mp.input_cost(1_000_000) == pytest.approx(2.50)

    def test_output_cost_one_million_tokens(self) -> None:
        mp = MODEL_PRICING["gpt-4o"]
        # 1M tokens at $10.00/M = $10.00
        assert mp.output_cost(1_000_000) == pytest.approx(10.00)

    def test_total_cost_combines_input_and_output(self) -> None:
        mp = MODEL_PRICING["gpt-4o"]
        in_cost = mp.input_cost(500_000)
        out_cost = mp.output_cost(100_000)
        assert mp.total_cost(500_000, 100_000) == pytest.approx(in_cost + out_cost)

    def test_input_cost_fractional(self) -> None:
        mp = MODEL_PRICING["claude-3-haiku"]
        # $0.25 per million => 1000 tokens = $0.00025
        expected = (1000 / 1_000_000) * 0.25
        assert mp.input_cost(1000) == pytest.approx(expected)

    def test_output_cost_fractional(self) -> None:
        mp = MODEL_PRICING["claude-3-haiku"]
        expected = (2000 / 1_000_000) * 1.25
        assert mp.output_cost(2000) == pytest.approx(expected)

    def test_fits_in_context_within_limit(self) -> None:
        mp = MODEL_PRICING["gpt-4o"]  # 128k window
        assert mp.fits_in_context(1000) is True

    def test_fits_in_context_at_limit(self) -> None:
        mp = MODEL_PRICING["gpt-4o"]
        assert mp.fits_in_context(mp.context_window_tokens) is True

    def test_fits_in_context_over_limit(self) -> None:
        mp = MODEL_PRICING["gpt-4o"]
        assert mp.fits_in_context(mp.context_window_tokens + 1) is False

    def test_frozen_dataclass_immutable(self) -> None:
        mp = MODEL_PRICING["gpt-4o"]
        with pytest.raises((AttributeError, TypeError)):
            mp.input_cost_per_million_tokens = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MODEL_PRICING table tests
# ---------------------------------------------------------------------------


class TestModelPricingTable:
    """Sanity checks for the MODEL_PRICING global table."""

    EXPECTED_MODEL_IDS = {
        "gpt-4o",
        "gpt-4",
        "claude-3-5-sonnet",
        "claude-3-haiku",
        "gemini-1-5-pro",
    }

    def test_all_expected_models_present(self) -> None:
        assert set(MODEL_PRICING.keys()) == self.EXPECTED_MODEL_IDS

    @pytest.mark.parametrize("model_id", list(EXPECTED_MODEL_IDS))
    def test_model_has_positive_input_cost(self, model_id: str) -> None:
        assert MODEL_PRICING[model_id].input_cost_per_million_tokens > 0

    @pytest.mark.parametrize("model_id", list(EXPECTED_MODEL_IDS))
    def test_model_has_positive_output_cost(self, model_id: str) -> None:
        assert MODEL_PRICING[model_id].output_cost_per_million_tokens > 0

    @pytest.mark.parametrize("model_id", list(EXPECTED_MODEL_IDS))
    def test_model_has_positive_context_window(self, model_id: str) -> None:
        assert MODEL_PRICING[model_id].context_window_tokens > 0

    @pytest.mark.parametrize("model_id", list(EXPECTED_MODEL_IDS))
    def test_model_has_display_name(self, model_id: str) -> None:
        assert MODEL_PRICING[model_id].display_name

    @pytest.mark.parametrize("model_id", list(EXPECTED_MODEL_IDS))
    def test_model_has_provider(self, model_id: str) -> None:
        assert MODEL_PRICING[model_id].provider

    def test_gpt4_context_smaller_than_gpt4o(self) -> None:
        # GPT-4 (8k) should have a smaller window than GPT-4o (128k)
        assert (
            MODEL_PRICING["gpt-4"].context_window_tokens
            < MODEL_PRICING["gpt-4o"].context_window_tokens
        )

    def test_gemini_largest_context_window(self) -> None:
        # Gemini 1.5 Pro has a 2M token window
        gemini_window = MODEL_PRICING["gemini-1-5-pro"].context_window_tokens
        for model_id, mp in MODEL_PRICING.items():
            if model_id != "gemini-1-5-pro":
                assert mp.context_window_tokens < gemini_window, (
                    f"{model_id} should have a smaller context window than Gemini 1.5 Pro"
                )

    def test_claude_haiku_cheapest_input(self) -> None:
        # Claude 3 Haiku should be the cheapest model by input cost
        haiku_cost = MODEL_PRICING["claude-3-haiku"].input_cost_per_million_tokens
        for model_id, mp in MODEL_PRICING.items():
            assert mp.input_cost_per_million_tokens >= haiku_cost, (
                f"{model_id} has lower input cost than Claude 3 Haiku"
            )

    def test_gpt4_most_expensive_input(self) -> None:
        # GPT-4 (legacy) should be the most expensive model by input cost
        gpt4_cost = MODEL_PRICING["gpt-4"].input_cost_per_million_tokens
        for model_id, mp in MODEL_PRICING.items():
            assert mp.input_cost_per_million_tokens <= gpt4_cost, (
                f"{model_id} has higher input cost than GPT-4"
            )

    def test_model_id_matches_dict_key(self) -> None:
        for key, mp in MODEL_PRICING.items():
            assert mp.model_id == key


# ---------------------------------------------------------------------------
# ModelCostEstimate property tests
# ---------------------------------------------------------------------------


class TestModelCostEstimate:
    """Tests for ModelCostEstimate computed properties."""

    def _make_estimate(
        self,
        input_tokens: int = 5000,
        output_tokens: int = 1000,
        input_cost: float = 0.0125,
        output_cost: float = 0.01,
        context_window: int = 128_000,
        exceeds: bool = False,
    ) -> ModelCostEstimate:
        return ModelCostEstimate(
            model_id="gpt-4o",
            model_name="GPT-4o",
            provider="OpenAI",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            context_window_tokens=context_window,
            exceeds_context_window=exceeds,
            total_files_changed=3,
            total_lines_changed=150,
            complexity_score=10,
        )

    def test_total_cost_usd(self) -> None:
        est = self._make_estimate(input_cost=0.01, output_cost=0.005)
        assert est.total_cost_usd == pytest.approx(0.015)

    def test_total_tokens(self) -> None:
        est = self._make_estimate(input_tokens=3000, output_tokens=500)
        assert est.total_tokens == 3500

    def test_context_utilization_pct_normal(self) -> None:
        est = self._make_estimate(input_tokens=12_800, context_window=128_000)
        assert est.context_utilization_pct == pytest.approx(10.0)

    def test_context_utilization_pct_zero_window(self) -> None:
        est = self._make_estimate(context_window=0)
        assert est.context_utilization_pct == 0.0

    def test_context_utilization_pct_clamped_at_100(self) -> None:
        # Input tokens larger than the window -> should clamp at 100%
        est = self._make_estimate(input_tokens=200_000, context_window=128_000)
        assert est.context_utilization_pct == 100.0

    def test_to_dict_contains_required_keys(self) -> None:
        est = self._make_estimate()
        d = est.to_dict()
        required_keys = {
            "model_id",
            "model_name",
            "provider",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "input_cost_usd",
            "output_cost_usd",
            "total_cost_usd",
            "context_window_tokens",
            "context_utilization_pct",
            "exceeds_context_window",
            "total_files_changed",
            "total_lines_changed",
            "complexity_score",
        }
        assert required_keys.issubset(d.keys())

    def test_to_dict_values_match_properties(self) -> None:
        est = self._make_estimate(input_cost=0.01, output_cost=0.005)
        d = est.to_dict()
        assert d["total_cost_usd"] == pytest.approx(0.015, rel=1e-5)
        assert d["total_tokens"] == est.total_tokens
        assert d["context_utilization_pct"] == pytest.approx(
            est.context_utilization_pct, rel=1e-3
        )

    def test_to_dict_costs_rounded_to_6_decimal_places(self) -> None:
        est = self._make_estimate(input_cost=0.00000012345678, output_cost=0.0)
        d = est.to_dict()
        # Should be rounded to 6 dp
        assert d["input_cost_usd"] == round(0.00000012345678, 6)


# ---------------------------------------------------------------------------
# CostCalculator.estimate() tests
# ---------------------------------------------------------------------------


class TestCostCalculatorEstimate:
    """Tests for CostCalculator.estimate()."""

    def setup_method(self) -> None:
        self.calc = CostCalculator()
        self.analysis = _FakeDiffAnalysis()

    def test_estimate_gpt4o(self) -> None:
        tc = _make_token_count("gpt-4o", diff_tokens=10_000, system_tokens=512, output_tokens=1024)
        est = self.calc.estimate("gpt-4o", tc, self.analysis)
        assert est.model_id == "gpt-4o"
        assert est.provider == "OpenAI"
        assert est.input_tokens == 10_512  # 10000 + 512
        assert est.output_tokens == 1024
        expected_input_cost = (10_512 / 1_000_000) * 2.50
        expected_output_cost = (1024 / 1_000_000) * 10.00
        assert est.input_cost_usd == pytest.approx(expected_input_cost)
        assert est.output_cost_usd == pytest.approx(expected_output_cost)

    def test_estimate_claude_haiku(self) -> None:
        tc = _make_token_count(
            "claude-3-haiku", diff_tokens=5000, system_tokens=512, output_tokens=1024
        )
        est = self.calc.estimate("claude-3-haiku", tc)
        assert est.model_id == "claude-3-haiku"
        assert est.provider == "Anthropic"
        expected_input_cost = (5512 / 1_000_000) * 0.25
        assert est.input_cost_usd == pytest.approx(expected_input_cost)

    def test_estimate_gemini_1_5_pro(self) -> None:
        tc = _make_token_count(
            "gemini-1-5-pro", diff_tokens=50_000, system_tokens=512, output_tokens=2048
        )
        est = self.calc.estimate("gemini-1-5-pro", tc, self.analysis)
        assert est.model_id == "gemini-1-5-pro"
        assert est.provider == "Google"
        assert est.context_window_tokens == 2_000_000
        assert est.exceeds_context_window is False

    def test_estimate_unknown_model_raises(self) -> None:
        tc = _make_token_count("nonexistent-model")
        with pytest.raises(ValueError, match="No pricing data"):
            self.calc.estimate("nonexistent-model", tc)

    def test_estimate_without_analysis_zero_metadata(self) -> None:
        tc = _make_token_count("gpt-4o")
        est = self.calc.estimate("gpt-4o", tc, diff_analysis=None)
        assert est.total_files_changed == 0
        assert est.total_lines_changed == 0
        assert est.complexity_score == 0

    def test_estimate_with_analysis_populates_metadata(self) -> None:
        tc = _make_token_count("gpt-4o")
        analysis = _FakeDiffAnalysis(
            total_files_changed=7,
            total_lines_changed=300,
            total_complexity_score=42,
        )
        est = self.calc.estimate("gpt-4o", tc, analysis)
        assert est.total_files_changed == 7
        assert est.total_lines_changed == 300
        assert est.complexity_score == 42

    def test_estimate_exceeds_context_window_when_total_exceeds_limit(self) -> None:
        # GPT-4 has an 8192 token window
        tc = _make_token_count(
            "gpt-4",
            diff_tokens=8000,
            system_tokens=512,
            output_tokens=1024,
        )
        # total = 8000 + 512 + 1024 = 9536 > 8192
        est = self.calc.estimate("gpt-4", tc)
        assert est.exceeds_context_window is True

    def test_estimate_does_not_exceed_context_window_within_limit(self) -> None:
        tc = _make_token_count(
            "gpt-4o",
            diff_tokens=100,
            system_tokens=100,
            output_tokens=100,
        )
        est = self.calc.estimate("gpt-4o", tc)
        assert est.exceeds_context_window is False

    def test_estimate_zero_tokens_cost_is_zero(self) -> None:
        tc = _make_token_count("gpt-4o", diff_tokens=0, system_tokens=0, output_tokens=0)
        est = self.calc.estimate("gpt-4o", tc)
        assert est.input_cost_usd == 0.0
        assert est.output_cost_usd == 0.0
        assert est.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# CostCalculator.estimate_all() tests
# ---------------------------------------------------------------------------


class TestCostCalculatorEstimateAll:
    """Tests for CostCalculator.estimate_all()."""

    def setup_method(self) -> None:
        self.calc = CostCalculator()
        self.analysis = _FakeDiffAnalysis()

    def test_returns_one_estimate_per_model(self) -> None:
        token_counts = _make_token_counts()
        estimates = self.calc.estimate_all(self.analysis, token_counts)
        assert len(estimates) == len(MODEL_PRICING)

    def test_sorted_by_ascending_cost(self) -> None:
        token_counts = _make_token_counts(diff_tokens=5000)
        estimates = self.calc.estimate_all(self.analysis, token_counts)
        costs = [e.total_cost_usd for e in estimates]
        assert costs == sorted(costs)

    def test_skips_models_not_in_pricing(self) -> None:
        token_counts = _make_token_counts()
        token_counts["some-unknown-model"] = _make_token_count("some-unknown-model")
        estimates = self.calc.estimate_all(self.analysis, token_counts)
        model_ids = [e.model_id for e in estimates]
        assert "some-unknown-model" not in model_ids

    def test_skips_models_not_in_token_counts(self) -> None:
        # Provide token counts for only one model
        token_counts = {"gpt-4o": _make_token_count("gpt-4o")}
        estimates = self.calc.estimate_all(self.analysis, token_counts)
        assert len(estimates) == 1
        assert estimates[0].model_id == "gpt-4o"

    def test_all_models_present_in_results(self) -> None:
        token_counts = _make_token_counts()
        estimates = self.calc.estimate_all(self.analysis, token_counts)
        result_ids = {e.model_id for e in estimates}
        assert result_ids == set(MODEL_PRICING.keys())

    def test_haiku_cheapest_in_results(self) -> None:
        token_counts = _make_token_counts(diff_tokens=5000)
        estimates = self.calc.estimate_all(self.analysis, token_counts)
        assert estimates[0].model_id == "claude-3-haiku"

    def test_empty_token_counts_returns_empty_list(self) -> None:
        estimates = self.calc.estimate_all(self.analysis, {})
        assert estimates == []


# ---------------------------------------------------------------------------
# CostCalculator helper method tests
# ---------------------------------------------------------------------------


class TestCostCalculatorHelpers:
    """Tests for list_models, get_pricing, cheapest_model, most_capable_model."""

    def setup_method(self) -> None:
        self.calc = CostCalculator()

    def test_list_models_returns_all_models(self) -> None:
        models = self.calc.list_models()
        assert len(models) == len(MODEL_PRICING)

    def test_list_models_sorted_by_input_cost(self) -> None:
        models = self.calc.list_models()
        costs = [m.input_cost_per_million_tokens for m in models]
        assert costs == sorted(costs)

    def test_get_pricing_known_model(self) -> None:
        mp = self.calc.get_pricing("gpt-4o")
        assert mp.model_id == "gpt-4o"
        assert mp.display_name == "GPT-4o"

    def test_get_pricing_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="No pricing data"):
            self.calc.get_pricing("gpt-99-ultra")

    def test_cheapest_model_is_haiku(self) -> None:
        cheapest = self.calc.cheapest_model()
        assert cheapest.model_id == "claude-3-haiku"

    def test_most_capable_model_is_gemini(self) -> None:
        most_cap = self.calc.most_capable_model()
        assert most_cap.model_id == "gemini-1-5-pro"

    def test_cheapest_model_empty_pricing_raises(self) -> None:
        calc = CostCalculator(pricing={})
        with pytest.raises(ValueError, match="empty"):
            calc.cheapest_model()

    def test_most_capable_model_empty_pricing_raises(self) -> None:
        calc = CostCalculator(pricing={})
        with pytest.raises(ValueError, match="empty"):
            calc.most_capable_model()

    def test_custom_pricing_overrides_defaults(self) -> None:
        custom = {
            "custom-model": ModelPricing(
                model_id="custom-model",
                display_name="Custom Model",
                provider="CustomCorp",
                input_cost_per_million_tokens=99.0,
                output_cost_per_million_tokens=199.0,
                context_window_tokens=4096,
            )
        }
        calc = CostCalculator(pricing=custom)
        assert "custom-model" in [m.model_id for m in calc.list_models()]
        assert "gpt-4o" not in [m.model_id for m in calc.list_models()]
