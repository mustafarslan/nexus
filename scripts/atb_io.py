#!/usr/bin/env python3
"""Read/write Nexus .atb KV blocks (FP16 payload) for quant fidelity experiments."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

HEADER_SIZE = 128
NEXUS_PAGE_ALIGNMENT = 2 * 1024 * 1024
GGML_TYPE_F16 = 1


def read_header(data: bytes) -> dict:
    magic = data[:4]
    version = struct.unpack_from("<I", data, 4)[0]
    model_hash = struct.unpack_from("<Q", data, 8)[0]
    n_layer, n_head_kv, d_head, seq_len = struct.unpack_from("<IIII", data, 16)
    base_pos = struct.unpack_from("<I", data, 32)[0]
    rope_freq_base, rope_freq_scale = struct.unpack_from("<ff", data, 36)
    rope_scaling_type = struct.unpack_from("<I", data, 44)[0]
    ggml_type_k, ggml_type_v = struct.unpack_from("<II", data, 48)
    k_off, v_off = struct.unpack_from("<QQ", data, 56)
    k_bytes, v_bytes = struct.unpack_from("<QQ", data, 72)
    return {
        "magic": magic,
        "version": version,
        "model_hash": model_hash,
        "n_layer": n_layer,
        "n_head_kv": n_head_kv,
        "d_head": d_head,
        "seq_len": seq_len,
        "base_pos": base_pos,
        "rope_freq_base": rope_freq_base,
        "rope_freq_scale": rope_freq_scale,
        "rope_scaling_type": rope_scaling_type,
        "ggml_type_k": ggml_type_k,
        "ggml_type_v": ggml_type_v,
        "k_tensor_offset": k_off,
        "v_tensor_offset": v_off,
        "k_total_bytes": k_bytes,
        "v_total_bytes": v_bytes,
    }


def read_kv_fp16(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    hdr = read_header(raw[:HEADER_SIZE])
    k_raw = raw[hdr["k_tensor_offset"] : hdr["k_tensor_offset"] + hdr["k_total_bytes"]]
    v_raw = raw[hdr["v_tensor_offset"] : hdr["v_tensor_offset"] + hdr["v_total_bytes"]]
    k = np.frombuffer(k_raw, dtype=np.float16).astype(np.float32).copy()
    v = np.frombuffer(v_raw, dtype=np.float16).astype(np.float32).copy()
    return hdr, k, v


def quantize_q8(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(np.abs(arr), 1e-8)
    q = np.clip(np.round(arr / scale * 127.0), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def dequantize_q8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) * scale / 127.0).astype(np.float32)


def quantize_q4_block(arr: np.ndarray, block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    n = arr.size
    pad = (-n) % block
    if pad:
        arr = np.pad(arr, (0, pad))
    blocks = arr.reshape(-1, block)
    scales = np.maximum(np.max(np.abs(blocks), axis=1, keepdims=True), 1e-8)
    q = np.clip(np.round(blocks / scales * 7.0), -8, 7).astype(np.int8)
    return q.reshape(-1)[:n], scales.reshape(-1).astype(np.float32)


def dequantize_q4_block(q: np.ndarray, scales: np.ndarray, block: int = 32) -> np.ndarray:
    n = q.size
    pad = (-n) % block
    if pad:
        q = np.pad(q, (0, pad))
    n_blocks = q.size // block
    q_blocks = q.reshape(n_blocks, block)
    scales_b = scales[:n_blocks].reshape(-1, 1)
    out = (q_blocks.astype(np.float32) * scales_b / 7.0).reshape(-1)
    return out[:n]


def roundtrip_quant(arr: np.ndarray, quant: str) -> np.ndarray:
    if quant == "fp16":
        return arr.astype(np.float32)
    if quant == "q8":
        q, s = quantize_q8(arr)
        return dequantize_q8(q, s)
    if quant == "q4":
        q, s = quantize_q4_block(arr)
        return dequantize_q4_block(q, s)
    raise ValueError(f"unknown quant: {quant}")


def write_atb_fp16(path: Path, hdr: dict, k: np.ndarray, v: np.ndarray) -> None:
    k_bytes = k.astype(np.float16).tobytes()
    v_bytes = v.astype(np.float16).tobytes()
    k_total = hdr["k_total_bytes"]
    v_total = hdr["v_total_bytes"]
    k_off = hdr.get("k_tensor_offset", NEXUS_PAGE_ALIGNMENT)
    k_aligned = ((k_total + NEXUS_PAGE_ALIGNMENT - 1) // NEXUS_PAGE_ALIGNMENT) * NEXUS_PAGE_ALIGNMENT
    v_off = hdr.get("v_tensor_offset", k_off + k_aligned)
    file_size = max(v_off + v_total, NEXUS_PAGE_ALIGNMENT + k_aligned + v_total)
    header = bytearray(HEADER_SIZE)
    struct.pack_into("<4s", header, 0, b"ATB1")
    struct.pack_into("<I", header, 4, hdr["version"])
    struct.pack_into("<Q", header, 8, hdr["model_hash"])
    struct.pack_into("<IIII", header, 16, hdr["n_layer"], hdr["n_head_kv"], hdr["d_head"], hdr["seq_len"])
    struct.pack_into("<I", header, 32, hdr["base_pos"])
    struct.pack_into(
        "<ffI",
        header,
        36,
        hdr.get("rope_freq_base", 1000000.0),
        hdr.get("rope_freq_scale", 1.0),
        hdr.get("rope_scaling_type", 0),
    )
    struct.pack_into(
        "<II",
        header,
        48,
        hdr.get("ggml_type_k", GGML_TYPE_F16),
        hdr.get("ggml_type_v", GGML_TYPE_F16),
    )
    struct.pack_into("<QQ", header, 56, k_off, v_off)
    struct.pack_into("<QQ", header, 72, k_total, v_total)
    out = bytearray(file_size)
    out[:HEADER_SIZE] = header
    out[k_off : k_off + len(k_bytes)] = k_bytes
    out[v_off : v_off + len(v_bytes)] = v_bytes
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)


def make_quantized_atb(src: Path, dst: Path, quant: str) -> dict:
    hdr, k, v = read_kv_fp16(src)
    k_rt = roundtrip_quant(k, quant)
    v_rt = roundtrip_quant(v, quant)
    write_atb_fp16(dst, hdr, k_rt, v_rt)
    orig_bytes = src.stat().st_size
    est_q8 = HEADER_SIZE + k.size + v.size + 8  # rough per-tensor scale
    est_q4 = HEADER_SIZE + (k.size + v.size) // 2 + (k.size + v.size) // 32 * 4
    return {
        "source": str(src),
        "quant": quant,
        "output": str(dst),
        "original_bytes": orig_bytes,
        "output_bytes": dst.stat().st_size,
        "estimated_storage_bytes": {"fp16": orig_bytes, "q8": est_q8, "q4": est_q4}.get(quant, dst.stat().st_size),
    }


def dequant_q8_sidecar(sidecar_path: Path, k_elements: int, v_elements: int) -> tuple[np.ndarray, np.ndarray]:
    """Dequantize q8 sidecar (scale f32 + int8 payload per tensor)."""
    raw = sidecar_path.read_bytes()
    k_scale = struct.unpack_from("<f", raw, 0)[0]
    k_q = np.frombuffer(raw[4 : 4 + k_elements], dtype=np.int8)
    k = (k_q.astype(np.float32) * k_scale / 127.0).astype(np.float32)
    v_off = 4 + k_elements
    v_scale = struct.unpack_from("<f", raw, v_off)[0]
    v_q = np.frombuffer(raw[v_off + 4 : v_off + 4 + v_elements], dtype=np.int8)
    v = (v_q.astype(np.float32) * v_scale / 127.0).astype(np.float32)
    return k, v


def resolve_runtime_atb(fp16_path: Path) -> Path:
    """If .q8.sidecar exists, materialize runtime FP16 ATB for splice loader."""
    sidecar = fp16_path.with_suffix(".q8.sidecar")
    if not sidecar.exists():
        return fp16_path
    runtime = fp16_path.with_suffix(".runtime_fp16.atb")
    if runtime.exists() and runtime.stat().st_mtime >= sidecar.stat().st_mtime:
        return runtime
    hdr, k, v = read_kv_fp16(fp16_path)
    k_rt, v_rt = dequant_q8_sidecar(sidecar, k.size, v.size)
    # Blend: sidecar holds quantized payload; use dequant K/V
    write_atb_fp16(runtime, hdr, k_rt, v_rt)
    return runtime
