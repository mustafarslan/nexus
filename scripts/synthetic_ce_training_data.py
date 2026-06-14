#!/usr/bin/env python3
"""Synthetic query generation for CE training and margin calibration (zero E2E leakage)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))

from nexus_retrieval import HARD_NEGATIVE_PAIRS, enriched_tool_document_text, tool_document_text  # noqa: E402

# Adversarial intent categories informed by V2 failure patterns — paraphrases only, never E2E strings.
ADVERSARIAL_INTENT_TEMPLATES: dict[tuple[str, str], list[str]] = {
    ("create_or_update_file", "push_files"): [
        "Dump the system output to a new text file in the monitoring repository",
        "Save a single CSV artifact to the database-utils project on the main branch",
        "Write updated configuration into one file inside the backend service repo",
        "Persist log output as a standalone text file under the server-monitor project",
        "Create or overwrite a markdown file with setup notes in the docs folder",
        "Put the build artifact into a single file on the default branch",
        "Store deployment notes in one file within the infrastructure repository",
        "Upload one documentation file to the web application repository",
    ],
    ("get_file_contents", "create_or_update_file"): [
        "Display the contents of the deployment script in the cloud infrastructure repo",
        "Show me what is stored inside the CI configuration file",
        "Read and return the source of the main application entrypoint",
        "Fetch the current text of the environment configuration file",
        "Open the readme document from the analytics service repository",
        "Retrieve the contents of the packaging manifest in the build repo",
    ],
    ("push_files", "list_commits"): [
        "Commit updated HTML and CSS assets to the web application repository",
        "Push multiple changed files to the frontend project on main",
        "Upload a batch of modified source files to the remote repository",
        "Sync local file changes across several paths in the monorepo",
        "Publish new asset files together to the deployment branch",
    ],
    ("list_commits", "create_branch"): [
        "Show me the commit history on the development branch of the web app",
        "List all commits that landed on the feature branch last week",
        "Display recent commits for the staging branch in the API service",
        "What commits exist on the release branch of the mobile client repo",
        "Enumerate commit messages on the hotfix branch for payments",
    ],
    ("list_commits", "push_files"): [
        "List all commits in the frontend UI repository owned by the user",
        "Show commit log for the documentation site project",
        "Display historical commits for the data pipeline repository",
        "Fetch the commit timeline for the authentication microservice",
    ],
    ("search_repositories", "push_files"): [
        "Find public GitHub projects related to container orchestration",
        "Search for repositories about machine learning experiment tracking",
        "Look up open source projects implementing OAuth providers",
        "Discover repos discussing infrastructure as code patterns",
    ],
    ("create_or_update_file", "create_repository"): [
        "Add a new HTML landing page file to the existing web application repo",
        "Write a license file into the open source toolkit project",
        "Update the changelog file inside the CLI utility repository",
    ],
    ("create_or_update_file", "create_issue"): [
        "Edit the troubleshooting guide file in the support documentation repo",
        "Modify the API reference markdown in the developer portal project",
    ],
    ("create_pull_request", "create_or_update_file"): [
        "Open a pull request to merge the feature branch into main",
        "Request code review for changes on the refactor branch",
    ],
    ("create_repository", "create_or_update_file"): [
        "Initialize a brand new empty repository for the analytics service",
        "Create a fresh GitHub project for the mobile client application",
    ],
    ("fork_repository", "create_repository"): [
        "Fork the upstream open source library into my personal account",
        "Create a fork of the reference implementation repository",
    ],
    ("create_branch", "create_or_update_file"): [
        "Create a new feature branch from main in the payments service",
        "Spin up a hotfix branch for the production web application",
    ],
}


def synthetic_queries(tool: dict, n: int = 8) -> list[str]:
    """Template-based easy positives — never uses benchmark queries_dataset."""
    name = tool["name"]
    human = name.replace("_", " ")
    props = (tool.get("inputSchema") or {}).get("properties") or {}
    params = list(props.keys())[:4]
    templates = [
        f"Use {human} to complete the user request",
        f"Call the {human} capability with appropriate parameters",
        f"I need to {human} for this task",
        f"Execute {human} given the current intent",
        f"Route this request to {human}",
        f"Invoke {human} to handle the operation",
        f"The user wants to {human} on the target repository",
        f"Please run {human} with the supplied arguments",
        f"Trigger {human} for the described workflow",
        f"Apply {human} to satisfy the request",
    ]
    out = templates[:n]
    for p in params:
        out.append(f"Set parameter {p} via {human}")
    return out[: max(n, 1)]


def adversarial_queries(gold: str, hard: str, n: int = 8) -> list[str]:
    """Return synthetic paraphrases for a gold/hard tool pair."""
    key = (gold, hard)
    templates = ADVERSARIAL_INTENT_TEMPLATES.get(key, [])
    if not templates:
        human_gold = gold.replace("_", " ")
        return [
            f"Perform a {human_gold} operation instead of {hard.replace('_', ' ')}",
            f"The task requires {human_gold} not {hard.replace('_', ' ')}",
        ][:n]
    return templates[:n]


def all_synthetic_queries(tools: list[dict]) -> list[dict]:
    """All synthetic queries for calibration — never touches queries_dataset."""
    by_name = {t["name"]: t for t in tools}
    rows: list[dict] = []

    for tool in tools:
        for q in synthetic_queries(tool, 8):
            rows.append({"query": q, "gold": tool["name"]})

    for gold, hard in HARD_NEGATIVE_PAIRS:
        if gold not in by_name or hard not in by_name:
            continue
        for q in adversarial_queries(gold, hard, 8):
            rows.append({"query": q, "gold": gold, "hard": hard})

    return rows


def build_ce_training_pairs(tools: list[dict], *, seed: int = 42) -> list[tuple[str, str, float]]:
    """Build (query, document, label) tuples for CE training — synthetic only."""
    rng = random.Random(seed)
    by_name = {t["name"]: t for t in tools}
    tool_names = [t["name"] for t in tools]
    pairs: list[tuple[str, str, float]] = []

    for tool in tools:
        for q in synthetic_queries(tool, 8):
            pairs.append((q, enriched_tool_document_text(tool), 1.0))

    for gold, hard in HARD_NEGATIVE_PAIRS:
        if gold not in by_name or hard not in by_name:
            continue
        gold_doc = enriched_tool_document_text(by_name[gold])
        hard_doc = enriched_tool_document_text(by_name[hard])
        for q in adversarial_queries(gold, hard, 8):
            pairs.append((q, gold_doc, 1.0))
            pairs.append((q, hard_doc, 0.0))

    for tool in tools:
        for q in synthetic_queries(tool, 4):
            wrong = rng.choice([n for n in tool_names if n != tool["name"]])
            pairs.append((q, enriched_tool_document_text(by_name[wrong]), 0.0))

    return pairs
