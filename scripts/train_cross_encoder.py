#!/usr/bin/env python3
"""Fine-tune cross-encoder reranker on synthetic tool pairs (zero E2E leakage)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_routing_accuracy import load_first_10_tools  # noqa: E402
from nexus_retrieval import DEFAULT_CROSS_ENCODER  # noqa: E402
from synthetic_ce_training_data import build_ce_training_pairs  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Fine-tune cross-encoder for tool routing (synthetic only).")
    p.add_argument("--base-model", default=DEFAULT_CROSS_ENCODER)
    p.add_argument("--output", default="results/tool_cross_encoder_finetuned_v3")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    try:
        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("pip install sentence-transformers torch") from exc

    tools = load_first_10_tools()
    raw = build_ce_training_pairs(tools)
    train = [InputExample(texts=[q, d], label=float(l)) for q, d, l in raw]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model = CrossEncoder(args.base_model, num_labels=1, max_length=512)
    loader = DataLoader(train, shuffle=True, batch_size=args.batch_size)
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=max(10, len(train) // args.batch_size),
        output_path=str(out),
    )
    model.save(str(out))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "base": args.base_model,
                "n_pairs": len(raw),
                "training": "synthetic_only",
                "no_queries_dataset": True,
            },
            indent=2,
        )
    )
    print(f"Wrote fine-tuned cross-encoder to {out} ({len(raw)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
