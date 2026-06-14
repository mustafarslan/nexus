#!/usr/bin/env python3
"""Fine-tune tool routing retriever with hard negatives + synthetic query pairs."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "test"))

from bench_routing_accuracy import load_first_10_tools, queries_dataset  # noqa: E402
from nexus_retrieval import tool_document_text, tokenize_lexical  # noqa: E402

# Confusion pairs mined from recall_miss_analysis.json (Jun 2026)
HARD_NEGATIVE_PAIRS = [
    ("create_or_update_file", "create_issue"),
    ("create_or_update_file", "push_files"),
    ("create_or_update_file", "create_repository"),
    ("create_pull_request", "create_branch"),
    ("create_pull_request", "push_files"),
    ("create_pull_request", "create_or_update_file"),
    ("create_repository", "create_or_update_file"),
    ("create_repository", "create_issue"),
    ("get_file_contents", "create_or_update_file"),
    ("get_file_contents", "push_files"),
    ("push_files", "create_issue"),
    ("push_files", "list_commits"),
    ("search_repositories", "push_files"),
    ("create_branch", "create_or_update_file"),
    ("list_commits", "push_files"),
    ("list_commits", "create_branch"),
]


def synthetic_queries(tool: dict, n: int = 5) -> list[str]:
    name = tool["name"]
    props = (tool.get("inputSchema") or {}).get("properties") or {}
    params = list(props.keys())[:4]
    templates = [
        f"Use {name} to complete the user request",
        f"Call {name} with appropriate parameters",
        f"I need to {name.replace('_', ' ')}",
        f"Execute {name} for this task",
        f"Route to {name} given the intent",
    ]
    out = templates[:n]
    for p in params:
        out.append(f"Set {p} via {name}")
    return out[: max(n, 1)]


def build_training_pairs(tools: list[dict]) -> list[dict]:
    by_name = {t["name"]: t for t in tools}
    rows: list[dict] = []

    for case in queries_dataset:
        if case["tool"] not in by_name:
            continue
        gold = case["tool"]
        doc = tool_document_text(by_name[gold])
        rows.append({"query": case["query"], "positive": doc, "negative": None})

    for gold, hard in HARD_NEGATIVE_PAIRS:
        if gold not in by_name or hard not in by_name:
            continue
        for q in synthetic_queries(by_name[gold], 3):
            rows.append({
                "query": q,
                "positive": tool_document_text(by_name[gold]),
                "negative": tool_document_text(by_name[hard]),
            })

    for tool in tools:
        for q in synthetic_queries(tool, 2):
            rows.append({"query": q, "positive": tool_document_text(tool), "negative": None})

    return rows


def export_contrastive_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def train(args) -> int:
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore
    except ImportError:
        print("Install sentence-transformers: pip install sentence-transformers", file=sys.stderr)
        return 1

    tools = load_first_10_tools()
    rows = build_training_pairs(tools)
    export_contrastive_jsonl(rows, Path(args.export_jsonl))

    model = SentenceTransformer(args.base_model)
    pair_rows = [row for row in rows if not row.get("negative")]
    triplet_rows = [row for row in rows if row.get("negative")]

    pair_examples = [InputExample(texts=[row["query"], row["positive"]]) for row in pair_rows]
    triplet_examples = [
        InputExample(texts=[row["query"], row["positive"], row["negative"]]) for row in triplet_rows
    ]

    random.shuffle(pair_examples)
    random.shuffle(triplet_examples)
    train_objectives = []
    if pair_examples:
        pair_loader = DataLoader(pair_examples, shuffle=True, batch_size=args.batch_size)
        train_objectives.append((pair_loader, losses.MultipleNegativesRankingLoss(model)))
    if triplet_examples:
        triplet_loader = DataLoader(triplet_examples, shuffle=True, batch_size=args.batch_size)
        train_objectives.append((triplet_loader, losses.TripletLoss(model)))

    total_examples = len(pair_examples) + len(triplet_examples)
    warmup = int(total_examples * args.epochs * 0.1 / max(args.batch_size, 1))
    model.fit(
        train_objectives=train_objectives,
        epochs=args.epochs,
        warmup_steps=max(warmup, 10),
        output_path=args.output_dir,
        show_progress_bar=True,
    )
    print(f"Saved fine-tuned model to {args.output_dir}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Train tool routing retriever.")
    p.add_argument("--base-model", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--output-dir", default="results/tool_retriever_finetuned")
    p.add_argument("--export-jsonl", default="results/tool_retriever_train.jsonl")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--export-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tools = load_first_10_tools()
    rows = build_training_pairs(tools)
    export_contrastive_jsonl(rows, Path(args.export_jsonl))
    print(f"Exported {len(rows)} training rows to {args.export_jsonl}")
    if args.export_only:
        return 0
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
