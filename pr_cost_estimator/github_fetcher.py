"""GitHub REST API diff fetcher for PR Cost Estimator.

Fetches the unified diff of a GitHub pull request using the GitHub
REST API v3. Supports both public and private repositories when a
personal access token (PAT) or GitHub Actions ``GITHUB_TOKEN`` is
provided.

Typical usage::

    from pr_cost_estimator.github_fetcher import GitHubFetcher

    fetcher = GitHubFetcher(token="ghp_...")
    diff = fetcher.fetch_diff("https://github.com/owner/repo/pull/42")
"""

from __future__ import annotations

import re
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GITHUB_API_BASE = "https://api.github.com"
_PR_URL_PATTERN = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
)
_REQUEST_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class GitHubFetcher:
    """Fetches PR diffs from the GitHub REST API.

    Args:
        token: Optional GitHub personal access token or ``GITHUB_TOKEN``.
            Required for private repositories and recommended to avoid
            rate limiting (60 unauthenticated vs 5,000 authenticated
            requests/hour).
        api_base: GitHub API base URL; overridable for GitHub Enterprise.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        api_base: str = _GITHUB_API_BASE,
    ) -> None:
        """Initialise the fetcher.

        Args:
            token: Optional GitHub API token.
            api_base: Base URL for the GitHub REST API.
        """
        self.token = token
        self.api_base = api_base.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_diff(self, pr_url: str) -> str:
        """Fetch the unified diff for a GitHub pull request.

        Args:
            pr_url: Full GitHub PR URL, e.g.
                ``'https://github.com/owner/repo/pull/123'``.

        Returns:
            Raw unified diff text as a string.

        Raises:
            ValueError: If the URL does not match the expected GitHub
                PR URL pattern.
            requests.HTTPError: If the GitHub API returns a non-2xx
                HTTP status code.
            requests.Timeout: If the request exceeds the timeout.
            requests.ConnectionError: If a network error occurs.
        """
        owner, repo, pr_number = self._parse_pr_url(pr_url)
        api_url = f"{self.api_base}/repos/{owner}/{repo}/pulls/{pr_number}"
        diff_text = self._request_diff(api_url)
        return diff_text

    def fetch_pr_metadata(self, pr_url: str) -> dict:
        """Fetch PR metadata (title, body, author, labels) from the API.

        Args:
            pr_url: Full GitHub PR URL.

        Returns:
            A dict with PR metadata from the GitHub API JSON response.

        Raises:
            ValueError: If the URL cannot be parsed.
            requests.HTTPError: On non-2xx API response.
        """
        owner, repo, pr_number = self._parse_pr_url(pr_url)
        api_url = f"{self.api_base}/repos/{owner}/{repo}/pulls/{pr_number}"
        response = self._make_request(api_url, accept="application/vnd.github.v3+json")
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_pr_url(self, pr_url: str) -> tuple[str, str, str]:
        """Extract owner, repo, and PR number from a GitHub PR URL.

        Args:
            pr_url: Full GitHub PR URL string.

        Returns:
            Tuple of (owner, repo, pr_number) strings.

        Raises:
            ValueError: If the URL does not match the expected pattern.
        """
        match = _PR_URL_PATTERN.match(pr_url.strip())
        if not match:
            raise ValueError(
                f"Cannot parse GitHub PR URL: '{pr_url}'. "
                f"Expected format: https://github.com/OWNER/REPO/pull/NUMBER"
            )
        return match.group("owner"), match.group("repo"), match.group("number")

    def _build_headers(self, accept: str) -> dict[str, str]:
        """Build HTTP headers for the GitHub API request.

        Args:
            accept: The Accept header value for content negotiation.

        Returns:
            Dict of HTTP headers.
        """
        headers: dict[str, str] = {
            "Accept": accept,
            "User-Agent": "pr-cost-estimator/0.1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _make_request(
        self,
        url: str,
        accept: str = "application/vnd.github.v3.diff",
    ) -> requests.Response:
        """Execute an HTTP GET request against the GitHub API.

        Args:
            url: Full API endpoint URL.
            accept: Accept header value.

        Returns:
            The :class:`requests.Response` object.

        Raises:
            requests.Timeout: If the request exceeds the timeout.
            requests.ConnectionError: On network-level errors.
        """
        headers = self._build_headers(accept)
        response = requests.get(
            url,
            headers=headers,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        return response

    def _request_diff(self, api_url: str) -> str:
        """Request the diff content from a GitHub API pull endpoint.

        Args:
            api_url: Full URL for the pull endpoint
                (e.g. ``'https://api.github.com/repos/owner/repo/pulls/42'``).

        Returns:
            Raw unified diff text.

        Raises:
            requests.HTTPError: If the API returns an error status.
        """
        response = self._make_request(
            api_url,
            accept="application/vnd.github.v3.diff",
        )

        if response.status_code == 404:
            raise requests.HTTPError(
                f"PR not found (404). Check that the URL is correct and that "
                f"your token (if provided) has read access to the repository. "
                f"URL: {api_url}",
                response=response,
            )

        if response.status_code == 403:
            raise requests.HTTPError(
                f"Access denied (403). This may be a private repository. "
                f"Provide a GitHub token via --github-token or the GITHUB_TOKEN "
                f"environment variable.",
                response=response,
            )

        if response.status_code == 401:
            raise requests.HTTPError(
                f"Unauthorized (401). The provided GitHub token is invalid or expired.",
                response=response,
            )

        response.raise_for_status()
        return response.text
