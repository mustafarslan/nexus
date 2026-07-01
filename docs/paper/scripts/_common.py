#!/usr/bin/env python3
"""Shared utilities for Nexus paper evaluation harness."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "docs" / "paper"
DATA_DIR = PAPER_ROOT / "data"
FIGURES_DIR = PAPER_ROOT / "figures"
TABLES_DIR = PAPER_ROOT / "tables"
RAW_DIR = DATA_DIR / "_raw"

DEFAULT_MODEL = (
    "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/"
    "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
)
DEFAULT_EMBED_MODEL = "models/nomic-embed-text-v1.5/nomic-embed-text-v1.5.Q8_0.gguf"


def setup_paths() -> None:
    """Append repo roots so frozen src/ and test/ modules import correctly."""
    os.chdir(REPO_ROOT)
    for p in (REPO_ROOT / "build", REPO_ROOT / "src", REPO_ROOT / "test"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    lib_dir = REPO_ROOT / "build" / "external" / "llama.cpp" / "src"
    os.environ.setdefault("LLAMA_CPP_LIB_PATH", str(lib_dir))
    os.environ.setdefault("LLAMA_CPP_LIB", str(lib_dir / "libllama.dylib"))


def ensure_dirs() -> None:
    for d in (DATA_DIR, FIGURES_DIR, TABLES_DIR, RAW_DIR):
        d.mkdir(parents=True, exist_ok=True)


def resolve_model() -> str:
    env = os.environ.get("NEXUS_MODEL", "").strip()
    if env and Path(env).exists():
        return env
    if Path(DEFAULT_MODEL).exists():
        return DEFAULT_MODEL
    try:
        out = subprocess.check_output(
            ["python3", str(REPO_ROOT / "scripts" / "resolve_ollama_gguf.py")],
            text=True,
            cwd=REPO_ROOT,
        ).strip()
        if out and Path(out).exists():
            return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    raise FileNotFoundError("Set NEXUS_MODEL to a valid GGUF path.")


def resolve_embed_model() -> str:
    env = os.environ.get("NEXUS_EMBED_MODEL", "").strip()
    if env and Path(env).exists():
        return env
    setup_paths()
    try:
        from nexus_retrieval import DEFAULT_EMBED_MODEL as NEXUS_EMBED  # noqa: WPS433

        if Path(NEXUS_EMBED).exists():
            return NEXUS_EMBED
    except ImportError:
        pass
    hf = (
        Path.home()
        / ".cache/huggingface/hub/models--nomic-ai--nomic-embed-text-v1.5-GGUF/snapshots"
    )
    if hf.is_dir():
        for gguf in hf.rglob("*.gguf"):
            return str(gguf)
    raise FileNotFoundError("Set NEXUS_EMBED_MODEL to a valid embedding GGUF path.")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT
        ).strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def hardware_str() -> str:
    import platform

    return f"{platform.system()} {platform.release()} {platform.machine()}"


def percentiles_us(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "n": 0}
    arr = np.array(samples, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "n": len(samples),
    }


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
    comment_lines: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if comment_lines:
            for line in comment_lines:
                f.write(f"# {line}\n")
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
