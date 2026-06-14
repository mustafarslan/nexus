#!/usr/bin/env python3
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any


def _capture(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def file_sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    values = sorted(samples)
    rank = p * (len(values) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    frac = rank - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _finite_samples(samples: list[float]) -> list[float]:
    return [float(v) for v in samples if math.isfinite(float(v))]


def summarize(samples: list[float]) -> dict[str, float | int]:
    finite = _finite_samples(samples)
    out: dict[str, float | int] = {
        "mean": statistics.fmean(finite) if finite else 0.0,
        "std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "p50": _percentile(finite, 0.50),
        "p90": _percentile(finite, 0.90),
        "p99": _percentile(finite, 0.99),
        "n": len(finite),
    }
    excluded = len(samples) - len(finite)
    if excluded:
        out["n_total"] = len(samples)
        out["n_excluded"] = excluded
    return out


def _artifact_sample_count(artifact: dict[str, Any]) -> int:
    cfg = artifact.get("config") or {}
    for key in ("n_cases", "n_queries", "iterations", "case_limit", "query_limit"):
        if key in cfg and cfg[key]:
            return int(cfg[key])
    records = artifact.get("records") or []
    if records:
        case_ids = {r.get("case_id") for r in records if "case_id" in r}
        if case_ids:
            return len(case_ids)
        arms = {r.get("arm") for r in records if "arm" in r}
        if arms:
            return len(records) // max(len(arms), 1)
        return len(records)
    metrics = artifact.get("metrics") or {}
    for metric in metrics.values():
        samples = metric.get("raw_samples") or []
        if samples:
            return len(samples)
    return 0


def resolve_output_path(
    output_path: str | os.PathLike[str],
    *,
    smoke: bool = False,
    tag: str | None = None,
    sample_count: int | None = None,
    force: bool = False,
) -> Path:
    path = Path(output_path)
    if smoke:
        path = Path("results/smoke") / path.name
    if tag:
        path = path.with_name(f"{path.stem}_{tag}{path.suffix}")

    if path.exists() and not force and sample_count is not None:
        try:
            old = json.loads(path.read_text())
            old_n = _artifact_sample_count(old)
            if old_n and sample_count < old_n:
                raise ValueError(
                    f"Refusing to overwrite {path} (existing n={old_n} > new n={sample_count}). "
                    "Use --force or a different --output/--tag."
                )
        except json.JSONDecodeError:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_artifact(
    output_path: str | os.PathLike[str],
    benchmark: str,
    *,
    model_hash: str = "",
    config: dict[str, Any] | None = None,
    metrics: dict[str, dict[str, Any]] | None = None,
    records: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
    smoke: bool = False,
    tag: str | None = None,
    force: bool = False,
) -> Path:
    cfg = dict(config or {})
    sample_count = cfg.get("n_cases") or cfg.get("n_queries") or cfg.get("iterations")
    if sample_count is None and records:
        sample_count = _artifact_sample_count({"config": cfg, "records": records, "metrics": metrics or {}})

    path = resolve_output_path(
        output_path,
        smoke=smoke,
        tag=tag,
        sample_count=int(sample_count) if sample_count is not None else None,
        force=force,
    )

    normalized_metrics: dict[str, dict[str, Any]] = {}
    for name, metric in (metrics or {}).items():
        samples = [float(v) for v in metric.get("raw_samples", [])]
        normalized_metrics[name] = {
            "unit": metric.get("unit", ""),
            "raw_samples": samples,
            "summary": metric.get("summary", summarize(samples)),
        }

    artifact = {
        "schema_version": 1,
        "benchmark": benchmark,
        "git_sha": _capture(["git", "rev-parse", "HEAD"]),
        "llama_cpp_sha": _capture(["git", "-C", "external/llama.cpp", "rev-parse", "HEAD"]),
        "model_hash": model_hash,
        "hardware": f"{platform.system()} {platform.release()} {platform.machine()}",
        "timestamp": _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "config": cfg,
        "metrics": normalized_metrics,
        "records": records or [],
    }
    for key, value in (extra or {}).items():
        if key not in artifact:
            artifact[key] = value

    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path
