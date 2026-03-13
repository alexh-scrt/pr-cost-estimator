"""GitHub REST API diff fetcher for PR Cost Estimator.

Fetches the unified diff of a GitHub pull request using the GitHub
REST API v3.  Supports both public and private repositories when a
personal access token (PAT) or GitHub Actions ``GITHUB_TOKEN`` is
provided.

The fetcher parses GitHub PR URLs in the form::

    https://github.com/OWNER/REPO/pull/NUMBER

and maps them to the corresponding ``/repos/{owner}/{repo}/pulls/{number}``
API endpoint, requesting the response in ``application/vnd.github.v3.diff``
content type to receive a raw unified diff.

Typical usage::

    from pr_cost_estimator.github_fetcher import GitHubFetcher

    fetcher = GitHubFetcher(token="ghp_...")
    diff = fetcher.fetch_diff("https://github.com/owner/repo/pull/42")
    print(diff[:200])  # first 200 chars of the unified diff

GitHub Enterprise::

    fetcher = GitHubFetcher(
        token="ghp_...",
        api_base="https://github.mycompany.com/api/v3",
    )
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import requests
import requests.exceptions


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GITHUB_API_BASE: str = "https://api.github.com"

# Pattern that matches public GitHub PR URLs and optionally trailing slashes
# or query strings / fragments.
_PR_URL_PATTERN: re.Pattern[str] = re.compile(
    r"https?://github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+)/"
    r"pull/(?P<number>\d+)"
    r"(?:[/?#].*)?"
    r"$"
)

# Timeout for all outgoing HTTP requests (seconds).
_REQUEST_TIMEOUT: int = 30

# User-Agent header sent with every request.
_USER_AGENT: str = "pr-cost-estimator/0.1.0 (https://github.com/example/pr_cost_estimator)"


# ---------------------------------------------------------------------------
# GitHubFetcher
# ---------------------------------------------------------------------------


class GitHubFetcher:
    """Fetches PR diffs from the GitHub REST API.

    Attributes:
        token: GitHub personal access token or ``GITHUB_TOKEN`` for
            authentication.  ``None`` for unauthenticated (rate-limited)
            requests.
        api_base: Base URL for the GitHub REST API.  Override for GitHub
            Enterprise Server deployments.

    Example::

        fetcher = GitHubFetcher(token="ghp_xxxxxxxxxxxx")
        diff_text = fetcher.fetch_diff(
            "https://github.com/psf/requests/pull/1"
        )
    """

    def __init__(
        self,
        token: Optional[str] = None,
        api_base: str = _GITHUB_API_BASE,
    ) -> None:
        """Initialise the fetcher.

        Args:
            token: Optional GitHub API token (PAT or ``GITHUB_TOKEN``).
                Providing a token raises the rate limit from 60 to 5,000
                requests per hour and enables access to private repositories.
            api_base: Base URL for the GitHub REST API endpoint.  Defaults
                to ``'https://api.github.com'``.
        """
        self.token = token
        self.api_base = api_base.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_diff(self, pr_url: str) -> str:
        """Fetch the unified diff for a GitHub pull request.

        Parses the PR URL to extract the owner, repository name, and PR
        number, then calls the GitHub REST API requesting the response in
        ``application/vnd.github.v3.diff`` format.

        Args:
            pr_url: Full GitHub PR URL, e.g.
                ``'https://github.com/owner/repo/pull/123'``.

        Returns:
            Raw unified diff text as a string.  May be an empty string for
            PRs with no file changes.

        Raises:
            ValueError: If ``pr_url`` does not match the expected GitHub PR
                URL pattern.
            requests.HTTPError: If the GitHub API returns a non-2xx HTTP
                status code (e.g. 401, 403, 404, 422).
            requests.Timeout: If the request exceeds
                :data:`_REQUEST_TIMEOUT` seconds.
            requests.ConnectionError: If a network-level error occurs.
        """
        owner, repo, number = self._parse_pr_url(pr_url)
        api_url = f"{self.api_base}/repos/{owner}/{repo}/pulls/{number}"
        return self._request_diff(api_url)

    def fetch_pr_metadata(self, pr_url: str) -> dict:
        """Fetch PR metadata (title, body, author, labels, state) from the API.

        Args:
            pr_url: Full GitHub PR URL.

        Returns:
            A dict with the raw JSON response from the GitHub API ``GET
            /repos/{owner}/{repo}/pulls/{number}`` endpoint.

        Raises:
            ValueError: If ``pr_url`` cannot be parsed.
            requests.HTTPError: On non-2xx API response.
            requests.Timeout: If the request times out.
            requests.ConnectionError: On network error.
        """
        owner, repo, number = self._parse_pr_url(pr_url)
        api_url = f"{self.api_base}/repos/{owner}/{repo}/pulls/{number}"
        response = self._make_request(
            api_url, accept="application/vnd.github.v3+json"
        )
        _raise_for_github_status(response)
        return response.json()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_pr_url(self, pr_url: str) -> Tuple[str, str, str]:
        """Extract owner, repository name, and PR number from a GitHub PR URL.

        Args:
            pr_url: Full GitHub PR URL string to parse.

        Returns:
            A three-tuple ``(owner, repo, pr_number)`` of strings.

        Raises:
            ValueError: If ``pr_url`` does not match the expected pattern.
        """
        cleaned = pr_url.strip()
        match = _PR_URL_PATTERN.match(cleaned)
        if not match:
            raise ValueError(
                f"Cannot parse GitHub PR URL: '{pr_url}'.\n"
                f"Expected format: https://github.com/OWNER/REPO/pull/NUMBER"
            )
        return (
            match.group("owner"),
            match.group("repo"),
            match.group("number"),
        )

    def _build_headers(self, accept: str) -> dict:
        """Build HTTP headers for a GitHub API request.

        Args:
            accept: The ``Accept`` header value for content negotiation.

        Returns:
            Dict of HTTP header name -> value strings.
        """
        headers: dict = {
            "Accept": accept,
            "User-Agent": _USER_AGENT,
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
        """Execute an authenticated GET request against the GitHub API.

        Args:
            url: Full API endpoint URL.
            accept: ``Accept`` header value for content negotiation.

        Returns:
            The raw :class:`requests.Response` object (status code not
            checked by this method).

        Raises:
            requests.Timeout: If the connection or read exceeds the timeout.
            requests.ConnectionError: On DNS / network-level failures.
        """
        headers = self._build_headers(accept)
        response = requests.get(
            url,
            headers=headers,
            timeout=_REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        return response

    def _request_diff(
        self,
        api_url: str,
    ) -> str:
        """Request and return the diff content from a GitHub pulls API endpoint.

        Provides human-readable error messages for common HTTP error codes
        (401, 403, 404) before falling through to ``raise_for_status`` for
        any other non-2xx responses.

        Args:
            api_url: Full GitHub API URL for the pull request resource,
                e.g. ``'https://api.github.com/repos/owner/repo/pulls/42'``.

        Returns:
            Raw unified diff text as a string.

        Raises:
            requests.HTTPError: If the API returns an error HTTP status.
        """
        response = self._make_request(
            api_url,
            accept="application/vnd.github.v3.diff",
        )
        _raise_for_github_status(response)
        return response.text


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _raise_for_github_status(response: requests.Response) -> None:
    """Raise :class:`requests.HTTPError` with a helpful message on failure.

    Provides specific guidance for the most common GitHub API error codes:

    * **401 Unauthorized** — token is missing or invalid.
    * **403 Forbidden** — rate-limited or insufficient permissions.
    * **404 Not Found** — the PR or repository does not exist (or is private
      without a token).
    * **422 Unprocessable Entity** — the PR number is syntactically valid but
      semantically rejected by the API.

    For all other non-2xx statuses the standard
    :meth:`requests.Response.raise_for_status` message is used.

    Args:
        response: The HTTP response to inspect.

    Raises:
        requests.HTTPError: If the response status code indicates an error.
    """
    if response.ok:
        return

    status = response.status_code
    url = response.url

    if status == 401:
        raise requests.HTTPError(
            "GitHub API returned 401 Unauthorized. "
            "The provided token is invalid or has expired. "
            "Generate a new token at https://github.com/settings/tokens.",
            response=response,
        )

    if status == 403:
        # Could be rate-limited or permission denied.
        remaining = response.headers.get("X-RateLimit-Remaining", "unknown")
        if remaining == "0":
            reset_ts = response.headers.get("X-RateLimit-Reset", "unknown")
            raise requests.HTTPError(
                f"GitHub API rate limit exceeded (403). "
                f"Rate limit resets at Unix timestamp {reset_ts}. "
                f"Provide a token via --github-token or GITHUB_TOKEN to increase the limit.",
                response=response,
            )
        raise requests.HTTPError(
            f"GitHub API returned 403 Forbidden for {url}. "
            f"This may be a private repository. "
            f"Provide a GitHub token with 'repo' scope via --github-token or GITHUB_TOKEN.",
            response=response,
        )

    if status == 404:
        raise requests.HTTPError(
            f"GitHub API returned 404 Not Found for {url}. "
            f"Check that the PR URL is correct and that the repository exists. "
            f"If the repository is private, provide a GitHub token via "
            f"--github-token or the GITHUB_TOKEN environment variable.",
            response=response,
        )

    if status == 422:
        raise requests.HTTPError(
            f"GitHub API returned 422 Unprocessable Entity for {url}. "
            f"The PR number may be invalid or the PR may have been closed "
            f"without any commits.",
            response=response,
        )

    # Fallback: use requests' built-in message for other error codes.
    response.raise_for_status()
