#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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


DEFAULT_MODEL = "/Users/mustafarslan/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEFAULT_SYSTEM_PROMPT = "You are a tool-using assistant. Use the provided tool schema to answer the user."


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Nexus spliced KV logits against full-prefill reference logits.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cases", help="Optional JSONL cases with schema/query/gold_continuation fields")
    parser.add_argument("--output", default="results/bench_splice_fidelity.json")
    parser.add_argument("--workdir", default="results/splice_fidelity_work")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--compile-mode", choices=["isolated", "anchored", "both"], default="both")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--tool-limit", type=int, default=3)
    parser.add_argument("--query-limit", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


# Intent-based query phrasings. These deliberately avoid naming the tool so the
# query is a realistic user request rather than a tautology, and each template
# yields a *distinct* query so that query_limit produces real (not duplicated)
# cases. {intent} is filled from the tool's own description.
# NOTE (Phase A): these are still derived mechanically from the schema and the
# gold answer is the tool name. A proper discriminating function-calling set
# (BFCL / ToolBench) is the real Phase A fix; this only removes the N-inflation
# bug so the artifact is not misread as N=12.
QUERY_TEMPLATES = [
    "A user asks: can you {intent}? Decide which tool to call.",
    "I need help with the following task: {intent}. Which tool handles this?",
    "Please {intent}. Select the appropriate tool and produce the call.",
    "Task: {intent}. Route this request to the correct tool.",
    "How would you {intent}? Pick the matching tool.",
]


def _intent_from_description(description: str) -> str:
    desc = (description or "").strip().rstrip(".")
    if not desc:
        return "perform this operation"
    return desc[0].lower() + desc[1:]


def load_default_cases(tool_limit: int, query_limit: int) -> list[dict]:
    github_tools_path = Path("test/schemas/github_tools.json")
    data = json.loads(github_tools_path.read_text())
    tools = data.get("tools", [])[:tool_limit]
    if query_limit > len(QUERY_TEMPLATES):
        raise ValueError(
            f"query_limit={query_limit} exceeds {len(QUERY_TEMPLATES)} distinct query templates; "
            "add templates or lower query_limit to avoid duplicating queries."
        )
    cases = []
    for tool in tools:
        schema = {"name": tool["name"], "description": tool.get("description", ""), "inputSchema": tool.get("inputSchema", {})}
        intent = _intent_from_description(tool.get("description", ""))
        for i in range(query_limit):
            cases.append({
                "id": f"{tool['name']}-{i}",
                "schema": schema,
                "query": QUERY_TEMPLATES[i].format(intent=intent),
                "gold_continuation": tool["name"],
                "tool_name": tool["name"],
            })
    return cases


def load_cases(path: str | None, tool_limit: int, query_limit: int) -> list[dict]:
    if not path:
        return load_default_cases(tool_limit, query_limit)
    cases = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def tokenize(llm: llama_cpp.Llama, text: str) -> list[int]:
    return [int(t) for t in llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)]


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x)
    return x - math.log(float(np.exp(x).sum()))


def kl_ref_to_splice(ref_logits: np.ndarray, splice_logits: np.ndarray) -> float:
    ref_logp = log_softmax(ref_logits)
    splice_logp = log_softmax(splice_logits)
    ref_p = np.exp(ref_logp)
    return float(np.sum(ref_p * (ref_logp - splice_logp)))


def nll(logits: np.ndarray, token: int) -> float:
    logp = log_softmax(logits)
    return float(-logp[token])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def compile_atb(model_path: str, schema_path: Path, output_path: Path, static_anchor_path: Path | None) -> None:
    cmd = [
        "./build/nexus_kv_compiler",
        "--model",
        model_path,
        "--schema",
        str(schema_path),
        "--output",
        str(output_path),
    ]
    if static_anchor_path is not None:
        cmd.extend(["--static-anchor", str(static_anchor_path)])
    subprocess.run(cmd, check=True)


def reference_logits(ctx, system_tokens: list[int], schema_tokens: list[int], query_tokens: list[int], prefix_tokens: list[int]) -> np.ndarray:
    nexus_fsm_ext.clear_kv_cache(ctx)
    tokens = system_tokens + schema_tokens + query_tokens + prefix_tokens
    return np.array(nexus_fsm_ext.decode_tokens(ctx, tokens, 0, 0), dtype=np.float32)


def spliced_logits(
    ctx,
    cache,
    atb_path: Path,
    schema_len: int,
    system_tokens: list[int],
    query_tokens: list[int],
    prefix_tokens: list[int],
) -> np.ndarray:
    nexus_fsm_ext.clear_kv_cache(ctx)
    if system_tokens:
        nexus_fsm_ext.decode_tokens(ctx, system_tokens, 0, 0)
    handle = cache.get_or_load(str(atb_path))
    nexus_fsm_ext.inject_tool_page(ctx, handle, len(system_tokens), 0)
    return np.array(
        nexus_fsm_ext.decode_tokens(ctx, query_tokens + prefix_tokens, len(system_tokens) + schema_len, 0),
        dtype=np.float32,
    )


def evaluate_case(ctx, cache, case: dict, mode: str, atb_path: Path, schema_text: str, system_text: str) -> dict:
    llm_for_tokens = evaluate_case.llm_for_tokens
    system_tokens = tokenize(llm_for_tokens, system_text)
    schema_tokens = tokenize(llm_for_tokens, schema_text)
    query_tokens = tokenize(llm_for_tokens, case["query"])

    ref = reference_logits(ctx, system_tokens, schema_tokens, query_tokens, [])
    splice = spliced_logits(ctx, cache, atb_path, len(schema_tokens), system_tokens, query_tokens, [])
    top1_ref = int(np.argmax(ref))
    top1_splice = int(np.argmax(splice))
    kl = kl_ref_to_splice(ref, splice)

    gold_tokens = tokenize(llm_for_tokens, case.get("gold_continuation", ""))
    ref_nll = 0.0
    splice_nll = 0.0
    prefix: list[int] = []
    for token in gold_tokens:
        ref_step = reference_logits(ctx, system_tokens, schema_tokens, query_tokens, prefix)
        splice_step = spliced_logits(ctx, cache, atb_path, len(schema_tokens), system_tokens, query_tokens, prefix)
        ref_nll += nll(ref_step, token)
        splice_nll += nll(splice_step, token)
        prefix.append(token)

    token_count = max(1, len(gold_tokens))
    return {
        "case_id": case.get("id", ""),
        "tool_name": case.get("tool_name", ""),
        "compile_mode": mode,
        "schema_tokens": len(schema_tokens),
        "system_tokens": len(system_tokens),
        "query_tokens": len(query_tokens),
        "gold_tokens": len(gold_tokens),
        "top1_reference": top1_ref,
        "top1_splice": top1_splice,
        "top1_agreement": top1_ref == top1_splice,
        "kl_ref_to_splice": kl,
        "gold_ref_nll": ref_nll,
        "gold_splice_nll": splice_nll,
        "gold_nll_delta": splice_nll - ref_nll,
        "gold_ppl_delta": math.exp(splice_nll / token_count) - math.exp(ref_nll / token_count),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    workdir = Path(args.workdir)
    if workdir.exists() and not args.keep_workdir:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.cases, args.tool_limit, args.query_limit)
    modes = ["isolated", "anchored"] if args.compile_mode == "both" else [args.compile_mode]

    llm = llama_cpp.Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        logits_all=True,
        flash_attn=True,
        verbose=False,
    )
    evaluate_case.llm_for_tokens = llm
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(16 * 1024 * 1024 * 1024)

    records = []
    for idx, case in enumerate(cases):
        schema_text = json.dumps(case["schema"], sort_keys=True)
        schema_path = workdir / f"case_{idx}.schema.json"
        write_text(schema_path, schema_text)
        anchor_path = workdir / f"case_{idx}.anchor.txt"
        write_text(anchor_path, args.system_prompt)

        for mode in modes:
            atb_path = workdir / f"case_{idx}.{mode}.atb"
            compile_atb(args.model, schema_path, atb_path, anchor_path if mode == "anchored" else None)
            records.append(evaluate_case(ctx, cache, case, mode, atb_path, schema_text, args.system_prompt))
            print(f"Evaluated {case.get('id', idx)} [{mode}]")

    metrics = {
        "top1_agreement": {"unit": "ratio", "raw_samples": [1.0 if r["top1_agreement"] else 0.0 for r in records]},
        "kl_ref_to_splice": {"unit": "nats", "raw_samples": [r["kl_ref_to_splice"] for r in records]},
        "gold_nll_delta": {"unit": "nats", "raw_samples": [r["gold_nll_delta"] for r in records]},
        "gold_ppl_delta": {"unit": "ppl", "raw_samples": [r["gold_ppl_delta"] for r in records]},
    }

    write_artifact(
        args.output,
        "bench_splice_fidelity",
        model_hash=file_sha256(args.model) if Path(args.model).exists() else "",
        config={
            "model_path": args.model,
            "cases": args.cases or "default_github_tools",
            "compile_mode": args.compile_mode,
            "system_prompt": args.system_prompt,
            "tool_limit": args.tool_limit,
            "query_limit": args.query_limit,
            "n_ctx": args.n_ctx,
            "n_gpu_layers": args.n_gpu_layers,
            "seed": args.seed,
            "metric_note": "Compares next-token distributions after full prefill vs direct Nexus KV splice; gold continuation NLL is replayed token-by-token.",
        },
        metrics=metrics,
        records=records,
    )
    print(f"Wrote JSON artifact: {args.output}")

    if not args.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
