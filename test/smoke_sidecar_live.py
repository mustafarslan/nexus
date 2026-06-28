#!/usr/bin/env python3
"""Live smoke for the V1.1 Execution Sidecar (needs llama_cpp + GGUF).

Run: PYTHONPATH=build:src:test .venv/bin/python test/smoke_sidecar_live.py

Checks:
  V2  deep context routes via Path A inside the sidecar (deep_path_entered == 0)
      and execute_via_sidecar returns (tool_name, args_json) with args non-empty
      (this also exercises the refactored _generate_arguments live).
  HF3 rehydrate_main_context injects ONLY the assistant tool call into seq 0.
  REG generate_with_tool (shallow) still works after the _generate_arguments refactor.
"""
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/external/llama.cpp/src"))
os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
os.environ["LLAMA_CPP_LIB"] = os.path.join(lib_dir, "libllama.dylib")

import llama_cpp  # noqa: E402
from bench_routing_accuracy import queries_dataset, load_first_10_tools  # noqa: E402
from nexus_agent import NexusAgent  # noqa: E402
from nexus_retrieval import DEFAULT_EMBED_MODEL, DEFAULT_RERANK_MARGIN, embed_tool_document  # noqa: E402

MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
ATB_DIR = Path("results/phaseA_tool_match_work")

fails = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        fails.append(name)


def main():
    tools = load_first_10_tools()
    tool_names = {t["name"] for t in tools}

    llm_emb = llama_cpp.Llama(model_path=DEFAULT_EMBED_MODEL, embedding=True, verbose=False)
    llm = llama_cpp.Llama(model_path=MODEL, n_ctx=8192, n_batch=2048,
                          n_gpu_layers=999, flash_attn=True, verbose=False)

    agent = NexusAgent(llm, llm_emb, base_pos=256, max_splice_pos=256,
                       rerank_margin=DEFAULT_RERANK_MARGIN)
    for i, tool in enumerate(tools):
        emb = embed_tool_document(llm_emb, tool)
        digest = f"Tool Name: {tool['name']}. Description: {tool.get('description', '')}."
        agent.register_tool(i + 1, tool["name"], emb, digest,
                            str(ATB_DIR / f"tool_{i}.isolated.atb"),
                            schema=tool.get("inputSchema"))

    # Pick a query whose gold tool is in the first 10.
    case = next(c for c in queries_dataset if c["tool"] in tool_names)
    gold = case["tool"]

    # Build a DEEP conversation: a big tool dump + chatter, then the real query.
    big_dump = "{" + ", ".join(f'"row_{k}": "value_{k}"' for k in range(400)) + "}"
    messages = [
        {"role": "user", "content": "Earlier I was exploring the repo structure."},
        {"role": "assistant", "content": "Sure, here is what I found."},
        {"role": "tool", "name": "list_files", "content": big_dump},
        {"role": "assistant", "content": "The repo has many modules."},
        {"role": "user", "content": case["query"]},
    ]

    # V1.2 Hybrid: route via retrieval, args from compressed TEXT IR in main context.
    before = agent.deep_path_telemetry()["deep_path_entered"]
    tool_name, args_json, meta = agent.generate_via_hybrid(messages, max_tokens=128)
    after = agent.deep_path_telemetry()["deep_path_entered"]

    print(f"\n  gold={gold!r} pred={tool_name!r} ir={meta.get('ir')!r} "
          f"ir_tokens={meta.get('ir_tokens')} args={args_json!r}\n")

    check("Hybrid never enters the splice deep path", after == before,
          f"(entered {before}->{after})")
    check("Hybrid emits a compressed IR", bool(meta.get("ir")) and meta.get("ir_tokens", 0) > 0)
    check("Hybrid returns a tool name", bool(tool_name))
    check("Hybrid tool matches gold", tool_name == gold, f"(got {tool_name!r})")
    check("Hybrid produced args", isinstance(args_json, str) and len(args_json) > 0)

    # REG: legacy splice path still works.
    agent.llm.reset()
    rid, tname, gen = agent.generate_with_tool(case["query"], seq_id=0, max_tokens=64)
    check("REG generate_with_tool returns tool", rid != 0 and bool(tname), f"(tool={tname!r})")
    check("REG generate_with_tool produced args", isinstance(gen, str) and len(gen) > 0)

    print("\n" + ("ALL SMOKE CHECKS PASSED" if not fails else f"FAILED: {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
