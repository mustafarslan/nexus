#!/usr/bin/env python3
"""Phase A fidelity sweep: does an isolated, RoPE-shifted KV splice match a full
prefill in the *dynamic-branch-point* regime the paper actually claims?

This is the gating experiment. It does NOT use the anchored/delta_pos=0 path as
evidence (that is a tautology — the schema KV is byte-identical to the reference
by construction). Instead it injects a schema block compiled *in isolation*
(base_pos=0) at a variable position P_start that sits *after realistic, non-empty
preceding context* (a system prompt + conversation), exactly as §2.2 of the paper
describes. delta_pos = P_start, so RoPE must shift the block, and the block's K/V
were computed without attending to the preceding context.

We report KL(ref‖splice) and top-1 agreement as a function of P_start. P_start=0
with empty preceding context is retained ONLY as a labeled byte-exact sanity
check, never as fidelity validation.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/external/llama.cpp/src"))
os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
os.environ["LLAMA_CPP_LIB"] = os.path.join(lib_dir, "libllama.dylib")

import llama_cpp  # noqa: E402
import nexus_fsm_ext  # noqa: E402
from benchmark_results import file_sha256, write_artifact  # noqa: E402

DEFAULT_MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
DEFAULT_SYSTEM_PROMPT = "You are a tool-using assistant. Use the provided tool schema to answer the user."

# Default tool corpus: 52 real schemas (github 26 + massive_20 20 + sqlite 6).
DEFAULT_CORPUS = [
    "test/schemas/github_tools.json",
    "test/schemas/massive_20_tools.json",
    "test/schemas/sqlite_tools.json",
]

# Intent-based query phrasings (distinct per index; see bench_splice_fidelity.py
# NOTE on the Phase A caveat — these are still derived from the schema, not a true
# discriminating BFCL/ToolBench set. They are adequate for the KL/top-1 curve,
# which does not depend on gold labels.
QUERY_TEMPLATES = [
    "A user asks: can you {intent}? Decide which tool to call.",
    "I need help with the following task: {intent}. Which tool handles this?",
    "Please {intent}. Select the appropriate tool and produce the call.",
    "Task: {intent}. Route this request to the correct tool.",
    "How would you {intent}? Pick the matching tool.",
]

# A long, generic multi-turn conversation used as *arbitrary preceding context*.
# It is deliberately unrelated to any specific tool so the schema block must be
# contextualized by content it never attended to at compile time.
_FILLER_TURNS = [
    "User: I'm building a small data pipeline and I keep running into timezone bugs. Any general advice?",
    "Assistant: Timezone bugs usually come from mixing naive and aware datetimes. Store everything in UTC, convert only at the display boundary, and never rely on the server's local clock for business logic.",
    "User: Makes sense. We also have a flaky integration test that fails about one run in twenty.",
    "Assistant: Flaky tests are often hidden ordering or timing assumptions. Try seeding randomness, pinning clocks, and isolating shared state between cases before reaching for retries.",
    "User: Good point. Separately, our log volume exploded last week and storage costs spiked.",
    "Assistant: Audit log levels first: a lot of teams ship debug logging to production by accident. Sampling high-frequency events and dropping redundant fields usually recovers most of the cost.",
    "User: We're also debating whether to add a cache in front of the database.",
    "Assistant: Caching helps read-heavy workloads but adds invalidation complexity. Measure your actual hit rate and tail latency before committing, and prefer a small TTL over manual invalidation when you can.",
    "User: Last thing, the team wants to standardize how we handle background jobs.",
    "Assistant: Pick one queue, make jobs idempotent, and always record an explicit status. Idempotency is what lets you retry safely when a worker dies mid-task.",
]


def _intent_from_description(description: str) -> str:
    desc = (description or "").strip().rstrip(".")
    if not desc:
        return "perform this operation"
    return desc[0].lower() + desc[1:]


def _normalize_input_schema(tool: dict) -> dict:
    return tool.get("inputSchema") or tool.get("input_schema") or {}


def load_corpus(paths: list[str], tool_limit: int | None) -> list[dict]:
    tools: list[dict] = []
    seen: set[str] = set()
    for p in paths:
        data = json.loads(Path(p).read_text())
        for tool in data.get("tools", []):
            name = tool.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            tools.append({
                "name": name,
                "description": tool.get("description", ""),
                "inputSchema": _normalize_input_schema(tool),
            })
    if tool_limit is not None:
        tools = tools[:tool_limit]
    return tools


def schema_text_for(tool: dict, max_chars: int) -> str:
    """Canonical schema serialization, optionally length-capped.

    Some corpus schemas (the synthetic massive_20 set) are ~12 KB / ~4k tokens,
    which is not representative of real MCP tool schemas (~100-400 tokens) and
    makes the sweep intractable. Capping the serialized length keeps block sizes
    realistic. The SAME capped text is used to compile the .atb and to build the
    reference, so the splice-vs-prefill comparison stays exact.
    """
    schema = {"name": tool["name"], "description": tool["description"], "inputSchema": tool["inputSchema"]}
    text = json.dumps(schema, sort_keys=True)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


def build_cases(tools: list[dict], query_limit: int) -> list[dict]:
    if query_limit > len(QUERY_TEMPLATES):
        raise ValueError(f"query_limit={query_limit} exceeds {len(QUERY_TEMPLATES)} templates.")
    cases = []
    for tool in tools:
        intent = _intent_from_description(tool["description"])
        for i in range(query_limit):
            cases.append({
                "id": f"{tool['name']}-{i}",
                "tool_name": tool["name"],
                "query": QUERY_TEMPLATES[i].format(intent=intent),
                "gold_continuation": tool["name"],
            })
    return cases


def tokenize(llm: llama_cpp.Llama, text: str) -> list[int]:
    return [int(t) for t in llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)]


def log_softmax(logits: np.ndarray) -> np.ndarray:
    import math
    x = logits.astype(np.float64)
    x = x - np.max(x)
    return x - math.log(float(np.exp(x).sum()))


def kl_ref_to_splice(ref_logits: np.ndarray, splice_logits: np.ndarray) -> float:
    ref_logp = log_softmax(ref_logits)
    splice_logp = log_softmax(splice_logits)
    ref_p = np.exp(ref_logp)
    return float(np.sum(ref_p * (ref_logp - splice_logp)))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def compile_isolated_atb(model_path: str, schema_path: Path, output_path: Path) -> None:
    """Compile a schema as a self-contained block (base_pos=0, no static anchor)."""
    cmd = [
        "./build/nexus_kv_compiler",
        "--model", model_path,
        "--schema", str(schema_path),
        "--output", str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_preceding_tokens(llm: llama_cpp.Llama, system_prompt: str, target_len: int) -> list[int]:
    """Return exactly target_len tokens of realistic preceding context.

    target_len == 0 -> empty (the labeled sanity point).
    Otherwise: system prompt + as much filler conversation as needed, sliced to
    exactly target_len tokens.
    """
    if target_len <= 0:
        return []
    text = system_prompt + "\n"
    idx = 0
    toks = tokenize(llm, text)
    while len(toks) < target_len:
        text += _FILLER_TURNS[idx % len(_FILLER_TURNS)] + "\n"
        idx += 1
        toks = tokenize(llm, text)
        if idx > 10000:
            break
    return toks[:target_len]


def reference_logits(ctx, preceding: list[int], schema: list[int], query: list[int]) -> np.ndarray:
    nexus_fsm_ext.clear_kv_cache(ctx)
    tokens = preceding + schema + query
    return np.array(nexus_fsm_ext.decode_tokens(ctx, tokens, 0, 0), dtype=np.float32)


def spliced_logits(ctx, cache, atb_path: Path, schema_len: int, preceding: list[int], query: list[int]) -> np.ndarray:
    p_start = len(preceding)
    nexus_fsm_ext.clear_kv_cache(ctx)
    if preceding:
        nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
    handle = cache.get_or_load(str(atb_path))
    # delta_pos = p_start - base_pos(=0); RoPE shift applied internally.
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
    return np.array(
        nexus_fsm_ext.decode_tokens(ctx, query, p_start + schema_len, 0),
        dtype=np.float32,
    )


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.array(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def finalize(records: list[dict], args, positions: list[int], n_tools: int, n_queries: int,
             model_hash: str, run_counts: dict) -> None:
    """Aggregate records into the by-position curve and write the artifact.

    Reads from the authoritative record list (loaded from the incremental JSONL),
    so it works even after a hard crash truncated the run.
    """
    by_position = {}
    for p in positions:
        rows = [r for r in records if r["p_start"] == p]
        kls = [r["kl_ref_to_splice"] for r in rows]
        agree = [1.0 if r["top1_agreement"] else 0.0 for r in rows]
        by_position[str(p)] = {
            "p_start": p,
            "is_sanity_check": (p == 0),
            "kl_ref_to_splice": summarize(kls),
            "top1_agreement_rate": float(np.mean(agree)) if agree else 0.0,
            "n": len(rows),
        }

    metrics = {
        "kl_ref_to_splice": {"unit": "nats", "raw_samples": [r["kl_ref_to_splice"] for r in records]},
        "top1_agreement": {"unit": "ratio", "raw_samples": [1.0 if r["top1_agreement"] else 0.0 for r in records]},
    }

    write_artifact(
        args.output,
        "bench_phaseA_fidelity",
        model_hash=model_hash,
        config={
            "model_path": args.model,
            "corpus": args.corpus,
            "n_tools": n_tools,
            "n_queries": n_queries,
            "positions": positions,
            "query_limit": args.query_limit,
            "max_schema_chars": args.max_schema_chars,
            "compile_mode": "isolated",
            "system_prompt": args.system_prompt,
            "n_ctx": args.n_ctx,
            "seed": args.seed,
            "metric_note": (
                "Isolated schema block (base_pos=0) spliced at variable P_start after realistic "
                "non-empty preceding context (delta_pos=P_start). KL & top-1 measured against full "
                "prefill of the same [preceding||schema||query]. P_start=0 is a labeled byte-exact "
                "sanity check only, NOT fidelity validation."
            ),
        },
        metrics=metrics,
        records=records,
        extra={"by_position": by_position, "run_counts": run_counts},
    )
    print(f"\nWrote artifact: {args.output}")
    print("\n=== Phase A fidelity curve (isolated mode) ===")
    print(f"{'P_start':>8} {'delta_pos':>10} {'KL mean':>10} {'KL p90':>10} {'KL max':>10} {'top1 agree':>11}  note")
    for p in positions:
        b = by_position[str(p)]
        klm = b["kl_ref_to_splice"]
        note = "SANITY (delta_pos=0)" if p == 0 else "claim regime"
        print(f"{p:>8} {p:>10} {klm.get('mean', 0):>10.4f} {klm.get('p90', 0):>10.4f} "
              f"{klm.get('max', 0):>10.4f} {b['top1_agreement_rate']:>11.3f}  {note}  (n={b['n']})")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Phase A splice-vs-prefill fidelity sweep over injection position.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--corpus", nargs="+", default=DEFAULT_CORPUS)
    parser.add_argument("--positions", default="0,64,256,1024", help="Comma-separated P_start values to sweep.")
    parser.add_argument("--tool-limit", type=int, default=None, help="Cap number of tools (default: all in corpus).")
    parser.add_argument("--query-limit", type=int, default=4)
    parser.add_argument("--max-schema-chars", type=int, default=1600,
                        help="Cap serialized schema length (~chars/3.3 tokens) for realistic, tractable blocks. 0 disables.")
    parser.add_argument("--output", default="results/bench_phaseA_fidelity.json")
    parser.add_argument("--workdir", default="results/phaseA_fidelity_work")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--reuse-atb", action="store_true", help="Skip compilation if a tool's .atb already exists in workdir.")
    parser.add_argument("--resume", action="store_true", help="Skip (case,position) pairs already present in the incremental JSONL.")
    parser.add_argument("--aggregate-only", action="store_true", help="Skip model load; aggregate the existing JSONL into the artifact and exit.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-batch", type=int, default=2048, help="Must exceed the longest prefill (max P_start + schema + query).")
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    positions = [int(x) for x in args.positions.split(",") if x.strip() != ""]
    jsonl_path = Path(args.output).with_suffix(".records.jsonl")

    if args.aggregate_only:
        records = load_jsonl(jsonl_path)
        tools = load_corpus(args.corpus, args.tool_limit)
        cases = build_cases(tools, args.query_limit)
        finalize(records, args, positions, len(tools), len(cases),
                 model_hash="", run_counts={"evaluated": len(records), "note": "aggregate-only"})
        return

    workdir = Path(args.workdir)
    if workdir.exists() and not args.keep_workdir and not args.reuse_atb:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    tools = load_corpus(args.corpus, args.tool_limit)
    cases = build_cases(tools, args.query_limit)
    print(f"Corpus: {len(tools)} tools, {len(cases)} queries, positions={positions}")

    # n_batch must cover the longest single-shot prefill (max P_start + schema +
    # query) to avoid the n_tokens_all <= n_batch assert, but no larger (a huge
    # n_batch inflates the compute buffer). Size it to the capped schema bound;
    # fall back to full context when schemas are uncapped. The skip logic below
    # is the backstop for anything still exceeding it.
    schema_tok_ub = (args.max_schema_chars // 2 + 128) if args.max_schema_chars else args.n_ctx
    needed_batch = max(positions) + schema_tok_ub + 128
    n_batch = min(args.n_ctx, max(args.n_batch, needed_batch))
    print(f"Using n_batch={n_batch} (n_ctx={args.n_ctx})")
    llm = llama_cpp.Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_batch=n_batch,
        n_ubatch=n_batch,
        n_gpu_layers=args.n_gpu_layers,
        logits_all=False,
        flash_attn=True,
        verbose=False,
    )
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(16 * 1024 * 1024 * 1024)

    # Compile one isolated .atb per unique tool (reused across queries/positions).
    atb_by_tool: dict[str, Path] = {}
    schema_len_by_tool: dict[str, int] = {}
    for i, tool in enumerate(tools):
        schema_text = schema_text_for(tool, args.max_schema_chars)
        schema_path = workdir / f"tool_{i}.schema.json"
        write_text(schema_path, schema_text)
        atb_path = workdir / f"tool_{i}.isolated.atb"
        if args.reuse_atb and atb_path.exists():
            print(f"Reusing existing .atb for {tool['name']} ({i + 1}/{len(tools)})")
        else:
            compile_isolated_atb(args.model, schema_path, atb_path)
            print(f"Compiled isolated .atb for {tool['name']} ({i + 1}/{len(tools)})")
        atb_by_tool[tool["name"]] = atb_path
        schema_len_by_tool[tool["name"]] = len(tokenize(llm, schema_text))

    # Precompute exact-length preceding contexts once per position.
    preceding_by_pos = {p: build_preceding_tokens(llm, args.system_prompt, p) for p in positions}

    records = []
    schema_text_cache: dict[str, list[int]] = {}
    for tool in tools:
        schema_text_cache[tool["name"]] = tokenize(llm, schema_text_for(tool, args.max_schema_chars))

    # Incremental persistence: every record is appended to JSONL and flushed
    # immediately, so a hard C-level crash (GGML_ASSERT / OOM) loses at most the
    # in-flight record. --resume skips (case,position) pairs already on disk.
    done_keys: set[tuple[str, int]] = set()
    if args.resume:
        records = load_jsonl(jsonl_path)
        done_keys = {(r["case_id"], r["p_start"]) for r in records}
        print(f"Resuming: {len(records)} records already on disk in {jsonl_path}")
    jsonl_f = open(jsonl_path, "a", buffering=1)  # line-buffered

    total = len(cases) * len(positions)
    done = 0
    skipped = 0
    failed = 0
    # Leave a margin below n_ctx for the spliced block + query; skip cases that
    # would overflow rather than crashing the whole run.
    max_prefill = args.n_ctx - 64
    for case in cases:
        name = case["tool_name"]
        schema_tokens = schema_text_cache[name]
        query_tokens = tokenize(llm, case["query"])
        atb_path = atb_by_tool[name]
        schema_len = schema_len_by_tool[name]
        for p in positions:
            done += 1
            if (case["id"], p) in done_keys:
                continue
            preceding = preceding_by_pos[p]
            prefill_len = len(preceding) + len(schema_tokens) + len(query_tokens)
            if prefill_len > max_prefill or (p + schema_len + len(query_tokens)) > max_prefill:
                skipped += 1
                print(f"  SKIP {case['id']} @P{p}: prefill {prefill_len} > {max_prefill}")
                continue
            try:
                ref = reference_logits(ctx, preceding, schema_tokens, query_tokens)
                splice = spliced_logits(ctx, cache, atb_path, schema_len, preceding, query_tokens)
            except Exception as exc:  # noqa: BLE001 - keep the run alive, record the failure
                failed += 1
                print(f"  FAIL {case['id']} @P{p}: {exc}")
                continue
            kl = kl_ref_to_splice(ref, splice)
            top1_ref = int(np.argmax(ref))
            top1_splice = int(np.argmax(splice))
            rec = {
                "case_id": case["id"],
                "tool_name": name,
                "p_start": p,
                "delta_pos": p,  # isolated mode: base_pos == 0
                "preceding_tokens": len(preceding),
                "schema_tokens": schema_len,
                "query_tokens": len(query_tokens),
                "kl_ref_to_splice": kl,
                "top1_reference": top1_ref,
                "top1_splice": top1_splice,
                "top1_agreement": top1_ref == top1_splice,
                "is_sanity_check": (p == 0),
            }
            records.append(rec)
            jsonl_f.write(json.dumps(rec) + "\n")
            jsonl_f.flush()
        print(f"  case {case['id']}: {done}/{total} (skipped={skipped}, failed={failed})")

    jsonl_f.close()
    # Re-read the full JSONL so aggregation reflects all completed work.
    all_records = load_jsonl(jsonl_path)
    finalize(all_records, args, positions, len(tools), len(cases),
             model_hash=file_sha256(args.model) if Path(args.model).exists() else "",
             run_counts={"total": total, "evaluated": len(all_records), "skipped": skipped, "failed": failed})

    if not args.keep_workdir and not args.reuse_atb:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
