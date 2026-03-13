"""Token counting module for PR Cost Estimator.

Uses ``tiktoken`` to count tokens for diff content and estimated
system/context prompts for each supported AI model.

Because Anthropic and Google models do not have public tiktoken encodings,
we use a compatible encoding (cl100k_base) as a close approximation for
Claude and Gemini, which is accurate to within ~5%.

Typical usage::

    from pr_cost_estimator.token_counter import TokenCounter

    counter = TokenCounter()
    counts = counter.count_for_all_models(diff_text)
    # counts is a dict mapping model_id -> TokenCount
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import tiktoken


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Estimated token overhead for a typical AI code-review system prompt
# (instructions, persona, formatting rules, etc.).
_SYSTEM_PROMPT_TOKENS = 512

# Estimated token overhead for assistant response (summary, inline comments).
# Conservative estimate: a typical code review reply for a medium PR.
_ESTIMATED_OUTPUT_TOKENS = 1024

# Mapping from model family to tiktoken encoding name.
# cl100k_base is used by GPT-4, GPT-3.5-turbo, and as an approximation
# for Claude / Gemini (their tokenizers are similar in density).
_ENCODING_MAP: Dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "claude-3-5-sonnet": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
    "gemini-1-5-pro": "cl100k_base",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TokenCount:
    """Token breakdown for a single model."""

    model_id: str
    """Identifier for the model, e.g. ``'gpt-4o'``."""

    diff_tokens: int
    """Tokens consumed by the raw diff text."""

    system_prompt_tokens: int
    """Tokens consumed by the estimated system prompt."""

    estimated_output_tokens: int
    """Estimated tokens in the model's response."""

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens (diff + system prompt)."""
        return self.diff_tokens + self.system_prompt_tokens

    @property
    def total_tokens(self) -> int:
        """Grand total tokens (input + estimated output)."""
        return self.total_input_tokens + self.estimated_output_tokens


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------

class TokenCounter:
    """Counts tokens for diff content using tiktoken encodings.

    Maintains a small cache of loaded encodings to avoid repeated disk I/O
    when processing multiple models that share the same encoding.
    """

    def __init__(
        self,
        system_prompt_tokens: int = _SYSTEM_PROMPT_TOKENS,
        estimated_output_tokens: int = _ESTIMATED_OUTPUT_TOKENS,
    ) -> None:
        """Initialise the TokenCounter.

        Args:
            system_prompt_tokens: Overhead token count for the system/context
                prompt prepended before the diff.
            estimated_output_tokens: Conservative estimate for response length.
        """
        self.system_prompt_tokens = system_prompt_tokens
        self.estimated_output_tokens = estimated_output_tokens
        self._encoding_cache: Dict[str, tiktoken.Encoding] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_tokens(self, text: str, model_id: str) -> TokenCount:
        """Count tokens for ``text`` using the encoding for ``model_id``.

        Args:
            text: The text to tokenise (typically the raw diff).
            model_id: A model identifier key from :data:`_ENCODING_MAP`.

        Returns:
            A :class:`TokenCount` for the given model.

        Raises:
            ValueError: If ``model_id`` is not in the supported model list.
        """
        if model_id not in _ENCODING_MAP:
            raise ValueError(
                f"Unknown model id '{model_id}'. "
                f"Supported models: {sorted(_ENCODING_MAP)}"
            )

        encoding = self._get_encoding(model_id)
        diff_tokens = len(encoding.encode(text))

        return TokenCount(
            model_id=model_id,
            diff_tokens=diff_tokens,
            system_prompt_tokens=self.system_prompt_tokens,
            estimated_output_tokens=self.estimated_output_tokens,
        )

    def count_for_all_models(
        self, text: str
    ) -> Dict[str, TokenCount]:
        """Count tokens for ``text`` across all supported models.

        Args:
            text: The text to tokenise (typically the raw diff).

        Returns:
            A dict mapping model_id -> :class:`TokenCount`.
        """
        results: Dict[str, TokenCount] = {}
        for model_id in _ENCODING_MAP:
            results[model_id] = self.count_tokens(text, model_id)
        return results

    def supported_models(self) -> list[str]:
        """Return a list of supported model IDs.

        Returns:
            Sorted list of model identifier strings.
        """
        return sorted(_ENCODING_MAP.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_encoding(self, model_id: str) -> tiktoken.Encoding:
        """Retrieve (or load and cache) the tiktoken encoding for a model.

        Args:
            model_id: Model identifier whose encoding is needed.

        Returns:
            A :class:`tiktoken.Encoding` instance.
        """
        enc_name = _ENCODING_MAP[model_id]
        if enc_name not in self._encoding_cache:
            self._encoding_cache[enc_name] = tiktoken.get_encoding(enc_name)
        return self._encoding_cache[enc_name]
