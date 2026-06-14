"""Shared retrieval utilities for Nexus benchmarks (nomic-embed + hybrid BM25/RRF)."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

NOMIC_QUERY_PREFIX = "search_query: "
NOMIC_DOC_PREFIX = "search_document: "
DEFAULT_EMBED_MODEL = (
    "/Users/mustafarslan/.cache/huggingface/hub/models--nomic-ai--nomic-embed-text-v1.5-GGUF/"
    "snapshots/0188c9bf409793f810680a5a431e7b899c46104c/nomic-embed-text-v1.5.f16.gguf"
)
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

try:
    from nexus_calibration import CALIBRATED_MARGIN_THRESHOLD as DEFAULT_RERANK_MARGIN
    from nexus_calibration import CALIBRATED_AMBIGUOUS_MARGIN as AMBIGUOUS_RERANK_MARGIN
except ImportError:
    DEFAULT_RERANK_MARGIN = 0.028
    AMBIGUOUS_RERANK_MARGIN = 0.10

DEFAULT_MARGIN_THRESHOLD = DEFAULT_RERANK_MARGIN
RRF_K = 60

HARD_NEGATIVE_PAIRS: tuple[tuple[str, str], ...] = (
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
)

HARD_NEGATIVE_TUPLES: frozenset[tuple[str, str]] = frozenset(
    tuple(sorted(pair)) for pair in HARD_NEGATIVE_PAIRS
)


def _topk_tool_name(entry) -> str:
    if isinstance(entry, (tuple, list)):
        return str(entry[0])
    if isinstance(entry, dict):
        return str(entry.get("name", ""))
    return str(entry)


def effective_rerank_threshold(
    topk: list,
    base_threshold: float,
    tools_by_name: dict[str, dict] | None = None,
) -> float:
    """Return base P20 threshold; hard-pair disambiguation is handled by finetuned CE at gate time."""
    return base_threshold


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 0:
        return (v / n).astype(np.float32)
    return v.astype(np.float32)


def tokenize_lexical(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 1]


def fnv1a32(token: str) -> int:
    h = 2166136261
    for c in token.encode("utf-8"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def lexical_hashes(text: str) -> list[int]:
    return [fnv1a32(t) for t in tokenize_lexical(text)]


def embed_query(llm_emb, query: str) -> np.ndarray:
    """Embed a user query with the nomic search_query prefix."""
    raw = llm_emb.embed(NOMIC_QUERY_PREFIX + query)
    v = raw[0] if isinstance(raw[0], list) else raw
    return _normalize(np.array(v, dtype=np.float32))


def embed_query_token_matrix(llm_emb, query: str, max_tokens: int = 16,
                             hf_model=None, hf_tokenizer=None) -> tuple[np.ndarray, int]:
    """Row-major FP32 query token matrix [T*D] for ColBERT MaxSim (aligned with export)."""
    if hf_model is not None and hf_tokenizer is not None:
        try:
            import torch

            prefixed = NOMIC_QUERY_PREFIX + query
            enc = hf_tokenizer([prefixed], return_tensors="pt", truncation=True, max_length=max_tokens)
            device = next(hf_model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = hf_model(**enc).last_hidden_state[0].detach().cpu().numpy()
            tlen = min(hidden.shape[0], max_tokens)
            dim = hidden.shape[1]
            mat = np.zeros((tlen, dim), dtype=np.float32)
            mat[:tlen] = hidden[:tlen]
            for i in range(tlen):
                mat[i] = _normalize(mat[i])
            return mat.reshape(-1), tlen
        except Exception:
            pass

    model_path = getattr(llm_emb, "model_path", None)
    if model_path and Path(str(model_path)).exists():
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            st = SentenceTransformer(str(model_path))
            prefixed = NOMIC_QUERY_PREFIX + query
            enc = st.tokenizer([prefixed], return_tensors="pt", truncation=True, max_length=max_tokens)
            device = st.device
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = st[0].auto_model(**enc)
                hidden = out.last_hidden_state[0].detach().cpu().numpy()
            tlen = min(hidden.shape[0], max_tokens)
            dim = hidden.shape[1]
            mat = np.zeros((tlen, dim), dtype=np.float32)
            mat[:tlen] = hidden[:tlen]
            for i in range(tlen):
                mat[i] = _normalize(mat[i])
            return mat.reshape(-1), tlen
        except Exception:
            pass

    chunks = [c.strip() for c in re.split(r"[\s?.!,;]+", query) if c.strip()][:max_tokens]
    if not chunks:
        chunks = [query]
    rows = [embed_query(llm_emb, ch) for ch in chunks]
    dim = rows[0].shape[0]
    mat = np.stack(rows, axis=0).astype(np.float32)
    return mat.reshape(-1), len(chunks)


def tool_document_text(tool: dict[str, Any]) -> str:
    """Rich document string for retrieval (name + description + parameter names)."""
    name = tool.get("name", "")
    desc = tool.get("description") or tool.get("desc", "")
    schema = tool.get("inputSchema") or tool.get("input_schema") or {}
    props = schema.get("properties") or {}
    param_names = ", ".join(sorted(props.keys())) if props else ""
    parts = [f"Tool: {name}", f"Description: {desc}"]
    if param_names:
        parts.append(f"Parameters: {param_names}")
    return "\n".join(parts)


def _intent_signature_prefix(name: str, verb_prefix: str | None = None) -> str:
    vp = verb_prefix or (name[: name.find("_")] if "_" in name else name)
    return (
        f"This tool is named {name.replace('_', ' ')}. "
        f"Its primary action is to {vp.replace('_', ' ')}. "
        f"Description: "
    )


def enriched_tool_document_text(tool: dict[str, Any]) -> str:
    """Dense/CE document text with transient intent signature (not for BM25 or generative prompts)."""
    return _intent_signature_prefix(tool.get("name", "")) + tool_document_text(tool)


def embed_tool_document(llm_emb, tool: dict[str, Any]) -> np.ndarray:
    """Embed a tool schema document with the nomic search_document prefix and intent signature."""
    raw = llm_emb.embed(NOMIC_DOC_PREFIX + enriched_tool_document_text(tool))
    v = raw[0] if isinstance(raw[0], list) else raw
    return _normalize(np.array(v, dtype=np.float32))


def embed_tools(llm_emb, tools: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    embs = [embed_tool_document(llm_emb, t) for t in tools]
    names = [t["name"] for t in tools]
    return np.stack(embs), names


def build_lexical_index(tools: list[dict[str, Any]]) -> tuple[list[list[str]], dict[str, float], list[str]]:
    """BM25 corpus: token lists per tool, idf, tool names."""
    docs = [tokenize_lexical(tool_document_text(t)) for t in tools]
    names = [t["name"] for t in tools]
    df: Counter[str] = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1
    n_docs = max(len(docs), 1)
    idf = {term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
    return docs, idf, names


def bm25_score(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float], k1: float = 1.2, b: float = 0.75) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    avgdl = max(dl, 1.0)
    tf = Counter(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in tf:
            continue
        freq = tf[term]
        idf_t = idf.get(term, 0.0)
        denom = freq + k1 * (1.0 - b + b * dl / avgdl)
        score += idf_t * (freq * (k1 + 1.0)) / max(denom, 1e-8)
    return score


def bm25_rank(query: str, tools: list[dict[str, Any]], k: int) -> list[str]:
    docs, idf, names = build_lexical_index(tools)
    qtoks = tokenize_lexical(query)
    scores = [bm25_score(qtoks, doc, idf) for doc in docs]
    order = np.argsort(-np.asarray(scores, dtype=np.float64))[:k]
    return [names[i] for i in order]


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank, name in enumerate(ranked):
            scores[name] = scores.get(name, 0.0) + 1.0 / (k + rank + 1)
    return [name for name, _ in sorted(scores.items(), key=lambda x: -x[1])]


def retrieve_topk(
    query_emb: np.ndarray,
    tool_embs: np.ndarray,
    tool_names: list[str],
    k: int,
) -> list[str]:
    scores = tool_embs @ query_emb
    order = np.argsort(-scores)[:k]
    return [tool_names[i] for i in order]


def retrieve_hybrid(
    query: str,
    query_emb: np.ndarray,
    tool_embs: np.ndarray,
    tool_names: list[str],
    tools: list[dict[str, Any]],
    k: int,
) -> list[str]:
    """Dense + BM25 fused via RRF."""
    dense = retrieve_topk(query_emb, tool_embs, tool_names, max(k, 5))
    lexical = bm25_rank(query, tools, max(k, 5))
    fused = reciprocal_rank_fusion([dense, lexical])
    return fused[:k]


def dense_margin(query_emb: np.ndarray, tool_embs: np.ndarray) -> float:
    scores = tool_embs @ query_emb
    if scores.size < 2:
        return 1.0
    order = np.argsort(-scores)
    return float(scores[order[0]] - scores[order[1]])


def margin_percentile_threshold(margins: list[float], percentile: float) -> dict[str, float]:
    """Threshold at given percentile: CE fires when margin falls below this value."""
    if not margins:
        return {"threshold": DEFAULT_RERANK_MARGIN, "fire_rate": 0.0, "percentile": percentile}
    thr = float(np.percentile(margins, percentile))
    fire = sum(1 for m in margins if m < thr) / len(margins)
    return {"threshold": thr, "fire_rate": fire, "percentile": percentile}


def is_hard_negative_topk(topk: list) -> bool:
    if len(topk) < 2:
        return False
    name1 = _topk_tool_name(topk[0])
    name2 = _topk_tool_name(topk[1])
    return tuple(sorted([name1, name2])) in HARD_NEGATIVE_TUPLES


def _candidate_token_prefixes(candidates: list[str], llm) -> dict[str, list[list[int]]]:
    """Map each candidate tool name to its token-id prefix sequence."""
    out: dict[str, list[list[int]]] = {}
    for name in candidates:
        toks = [int(t) for t in llm.tokenize(name.encode("utf-8"), add_bos=False, special=False)]
        out[name] = [toks[: i + 1] for i in range(len(toks))]
    return out


def _allowed_next_tokens(
    partial: list[int],
    prefixes: dict[str, list[list[int]]],
    candidates: list[str],
) -> set[int]:
    allowed: set[int] = set()
    for name in candidates:
        seqs = prefixes.get(name, [])
        if len(partial) >= len(seqs):
            continue
        expected = seqs[len(partial)]
        if partial == expected[: len(partial)]:
            allowed.add(expected[len(partial)])
    return allowed


def rerank_with_llm_constrained(
    llm_rerank,
    query: str,
    candidates: list[str],
    tools_by_name: dict,
    *,
    doc_chars: int = 200,
    max_tokens: int = 32,
) -> tuple[str, float]:
    """Listwise rerank on an isolated context with FSM-style tool-name masking."""
    import time

    import nexus_fsm_ext  # only available when C++ extension is built

    lines = []
    for i, name in enumerate(candidates):
        t = tools_by_name[name]
        lines.append(f"{i + 1}. {name}: {tool_document_text(t)[:doc_chars]}")
    prompt = (
        "Select the single best tool for the user query. Reply with ONLY the exact tool name.\n\n"
        + "\n".join(lines)
        + f"\n\nQuery: {query}\nTool name:"
    )
    prefixes = _candidate_token_prefixes(candidates, llm_rerank)
    prompt_tokens = [int(t) for t in llm_rerank.tokenize(prompt.encode("utf-8"), add_bos=False, special=False)]
    t0 = time.perf_counter()
    ctx = llm_rerank._ctx.ctx
    nexus_fsm_ext.invalidate_sequence(ctx, 0, 0, -1)
    if not prompt_tokens:
        return candidates[0], 0.0
    logits_arr = np.array(nexus_fsm_ext.decode_tokens(ctx, prompt_tokens, 0, 0), dtype=np.float32)
    pos = len(prompt_tokens)
    decoded: list[int] = []
    for _ in range(max_tokens):
        if logits_arr.size == 0:
            break
        allowed = _allowed_next_tokens(decoded, prefixes, candidates)
        if not allowed:
            break
        masked = np.full(logits_arr.shape, -np.inf, dtype=np.float32)
        for tok in allowed:
            if 0 <= tok < logits_arr.size:
                masked[tok] = logits_arr[tok]
        pred = int(np.argmax(masked))
        if pred not in allowed:
            break
        decoded.append(pred)
        for name in candidates:
            name_toks = [int(t) for t in llm_rerank.tokenize(name.encode("utf-8"), add_bos=False, special=False)]
            if decoded == name_toks:
                return name, (time.perf_counter() - t0) * 1e6
        logits_arr = np.array(nexus_fsm_ext.decode_tokens(ctx, [pred], pos, 0), dtype=np.float32)
        pos += 1
    lat = (time.perf_counter() - t0) * 1e6
    if decoded:
        text = llm_rerank.detokenize(decoded).decode("utf-8", errors="replace").strip()
        for name in candidates:
            if name in text or text in name:
                return name, lat
    return candidates[0], lat


def rerank_with_llm(llm, query: str, candidates: list[str], tools_by_name: dict, *, doc_chars: int = 200) -> str:
    """14B listwise rerank (aligned with analyze_recall_misses.py)."""
    name, _ = rerank_with_llm_constrained(llm, query, candidates, tools_by_name, doc_chars=doc_chars)
    return name


def rerank_with_llm_isolated(
    llm_rerank,
    query: str,
    candidates: list[str],
    tools_by_name: dict,
    *,
    doc_chars: int = 200,
) -> tuple[str, float]:
    """Rerank on a dedicated llama context so generation KV is never polluted."""
    return rerank_with_llm_constrained(llm_rerank, query, candidates, tools_by_name, doc_chars=doc_chars)


class CrossEncoderReranker:
    """Sub-100ms cross-encoder reranker (lazy-loaded sentence-transformers)."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "CrossEncoderReranker requires sentence-transformers: pip install sentence-transformers"
                ) from exc
            self._model = CrossEncoder(self.model_name, max_length=512)

    @property
    def available(self) -> bool:
        try:
            self._load()
            return True
        except ImportError:
            return False

    def warmup(self) -> None:
        """Load model and run one predict to avoid cold-start latency in TTFT."""
        self._load()
        self._model.predict([("warmup query", "warmup document")])

    def rerank(self, query: str, candidates: list[str], tools_by_name: dict) -> tuple[str, float]:
        import time

        self._load()
        pairs = [
            (query, enriched_tool_document_text(tools_by_name[name])[:512]) for name in candidates
        ]
        t0 = time.perf_counter()
        scores = self._model.predict(pairs)
        lat_us = (time.perf_counter() - t0) * 1e6
        best = int(np.argmax(scores))
        return candidates[best], lat_us


def margin_gated_retrieve(
    query: str,
    query_emb: np.ndarray,
    tool_embs: np.ndarray,
    tool_names: list[str],
    tools: list[dict[str, Any]],
    *,
    k: int = 5,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    reranker: CrossEncoderReranker | None = None,
    tools_by_name: dict | None = None,
) -> tuple[list[str], str | None, float]:
    """Hybrid top-k; cross-encoder rerank only when dense margin is low."""
    topk = retrieve_hybrid(query, query_emb, tool_embs, tool_names, tools, k)
    margin = dense_margin(query_emb, tool_embs)
    threshold = effective_rerank_threshold(topk, margin_threshold, tools_by_name)
    if margin >= threshold or reranker is None or tools_by_name is None:
        return topk, None, 0.0
    reranked, lat = reranker.rerank(query, topk, tools_by_name)
    return [reranked] + [n for n in topk if n != reranked], reranked, lat


def calibrate_margin_threshold(
    margins: list[float],
    embed_hits: list[bool],
    *,
    target_fire_rate: float = 0.30,
    min_miss_coverage: float = 0.90,
) -> dict[str, float]:
    """Pick margin threshold: ~target_fire_rate with >=min_miss_coverage on embed misses."""
    if not margins:
        return {"threshold": DEFAULT_MARGIN_THRESHOLD, "fire_rate": 0.0, "miss_coverage": 0.0}
    misses = [i for i, hit in enumerate(embed_hits) if not hit]
    # Sweep unique margins plus epsilon steps between min and max.
    lo, hi = min(margins), max(margins)
    thresholds = sorted(set(margins) | {lo + (hi - lo) * i / 50.0 for i in range(51)})

    def stats(thr: float) -> dict[str, float]:
        fire = sum(1 for m in margins if m < thr) / len(margins)
        covered = (
            sum(1 for i in misses if margins[i] < thr) / len(misses) if misses else 1.0
        )
        return {"threshold": thr, "fire_rate": fire, "miss_coverage": covered}

    eligible = [stats(thr) for thr in thresholds if stats(thr)["miss_coverage"] >= min_miss_coverage]
    if eligible:
        return min(eligible, key=lambda s: abs(s["fire_rate"] - target_fire_rate))
    return min((stats(thr) for thr in thresholds), key=lambda s: -s["miss_coverage"])


def route_tool(
    query: str,
    query_emb: np.ndarray,
    tool_embs: np.ndarray,
    tool_names: list[str],
    tools: list[dict[str, Any]],
    tools_by_name: dict,
    *,
    k: int = 5,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    llm_rerank=None,
    cross_encoder: CrossEncoderReranker | None = None,
    rerank_mode: str = "cross_encoder",
) -> tuple[str, bool, float, float, list[str]]:
    """Router: hybrid top-k → optional rerank → single tool for decode/splice.

    Default reranker is cross-encoder; 14B listwise is offline ceiling only (rerank_mode='llm').

    Returns (selected_tool, rerank_used, rerank_us, dense_margin, topk).
    """
    topk = retrieve_hybrid(query, query_emb, tool_embs, tool_names, tools, k)
    margin = dense_margin(query_emb, tool_embs)
    threshold = effective_rerank_threshold(topk, margin_threshold, tools_by_name)
    if margin >= threshold:
        return topk[0], False, 0.0, margin, topk

    if rerank_mode == "cross_encoder" and cross_encoder is not None:
        selected, lat = cross_encoder.rerank(query, topk, tools_by_name)
        return selected, True, lat, margin, topk

    if rerank_mode == "llm" and llm_rerank is not None:
        selected, lat = rerank_with_llm_isolated(llm_rerank, query, topk, tools_by_name)
        return selected, True, lat, margin, topk

    return topk[0], False, 0.0, margin, topk


def int8_quantize_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.max(np.abs(x), axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-8)
    q = np.clip(np.round(x / scale * 127.0), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def fp32_scan(query: np.ndarray, embs: np.ndarray) -> np.ndarray:
    q = query / max(float(np.linalg.norm(query)), 1e-8)
    norms = np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-8)
    return (embs / norms) @ q


def int8_scan(
    query: np.ndarray,
    q_embs: np.ndarray,
    scales: np.ndarray,
    emb_norms: np.ndarray,
) -> np.ndarray:
    """INT8 dot-product with per-row dequantization and L2 normalization."""
    q = query / max(float(np.linalg.norm(query)), 1e-8)
    q_scale = max(float(np.abs(q).max()), 1e-8)
    q8 = np.clip(np.round(q / q_scale * 127.0), -127, 127).astype(np.int8)
    dots = (q_embs.astype(np.int32) @ q8.astype(np.int32)).astype(np.float64)
    dots = dots * scales.ravel() * q_scale / (127.0 * 127.0)
    emb_norms = np.maximum(emb_norms, 1e-8)
    return (dots / emb_norms).astype(np.float32)


def recall_at_k(scores: np.ndarray, gold_idx: int, k: int) -> float:
    top = np.argsort(-scores)[:k]
    return 1.0 if gold_idx in top else 0.0


def schema_text(tool: dict[str, Any]) -> str:
    return json.dumps(
        {
            "name": tool["name"],
            "description": tool.get("description") or tool.get("desc", ""),
            "inputSchema": tool.get("inputSchema") or tool.get("input_schema") or {},
        },
        sort_keys=True,
    )
