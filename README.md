# pr-cost-estimator
A CLI tool and GitHub Action that estimates the token and dollar cost of running AI-based code review (Claude, GPT-4, etc.) on a pull request before you commit to the expense. It analyzes diff size, file count, cyclomatic complexity, and context overhead to produce a detailed cost breakdown per model, flags expensive PRs with actionable suggestions
