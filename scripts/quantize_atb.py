#!/usr/bin/env python3
"""I1: Quantize .atb KV tensors (FP16 -> INT8) and report storage + optional fidelity hook."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

HEADER_SIZE = 128
# ggml_type values from ggml.h (approx): F16=1, Q8_0=8
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8


def read_header(data: bytes) -> dict:
    magic, version = struct.unpack_from("<II", data, 0)
    n_layer, n_head_kv, d_head, seq_len = struct.unpack_from("<IIII", data, 16)
    base_pos = struct.unpack_from("<I", data, 32)[0]
    k_off, v_off = struct.unpack_from("<QQ", data, 56)
    k_bytes, v_bytes = struct.unpack_from("<QQ", data, 72)
    return {
        "magic": magic,
        "version": version,
        "n_layer": n_layer,
        "n_head_kv": n_head_kv,
        "d_head": d_head,
        "seq_len": seq_len,
        "base_pos": base_pos,
        "k_tensor_offset": k_off,
        "v_tensor_offset": v_off,
        "k_total_bytes": k_bytes,
        "v_total_bytes": v_bytes,
    }


def quantize_fp16_block(fp16_bytes: bytes) -> tuple[bytes, float]:
    arr = np.frombuffer(fp16_bytes, dtype=np.float16).astype(np.float32)
    scale = max(float(np.max(np.abs(arr))), 1e-8)
    q = np.clip(np.round(arr / scale * 127.0), -127, 127).astype(np.int8)
    # pack: scale (f32) + int8 payload
    return struct.pack("<f", scale) + q.tobytes(), scale


def quantize_atb(input_path: Path, output_path: Path, quant: str = "q8") -> dict:
    raw = input_path.read_bytes()
    hdr = read_header(raw[:HEADER_SIZE])
    k_raw = raw[hdr["k_tensor_offset"] : hdr["k_tensor_offset"] + hdr["k_total_bytes"]]
    v_raw = raw[hdr["v_tensor_offset"] : hdr["v_tensor_offset"] + hdr["v_total_bytes"]]

    if quant != "q8":
        raise ValueError("only q8 supported in this script")

    k_q, _ = quantize_fp16_block(k_raw)
    v_q, _ = quantize_fp16_block(v_raw)
    # Note: splicer currently expects FP16; this produces storage estimates + future format.
    # Write sidecar metadata for economics table until splicer supports Q8.
    sidecar = {
        "source": str(input_path),
        "original_bytes": len(raw),
        "fp16_kv_bytes": hdr["k_total_bytes"] + hdr["v_total_bytes"],
        "q8_payload_bytes": len(k_q) + len(v_q),
        "estimated_q8_file_bytes": HEADER_SIZE + len(k_q) + len(v_q),
        "compression_ratio": len(raw) / max(HEADER_SIZE + len(k_q) + len(v_q), 1),
        "note": "Q8 sidecar for storage economics; runtime splice still uses FP16 until loader updated",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(k_q + v_q)
    meta_path = output_path.with_suffix(".meta.json")
    import json
    meta_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def parse_args():
    p = argparse.ArgumentParser(description="Quantize ATB KV for storage economics (I1).")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--quant", default="q8", choices=["q8"])
    p.add_argument("--glob", default="", help="If set, quantize all matching ATBs under directory")
    return p.parse_args()


def main():
    args = parse_args()
    if args.glob:
        base = Path(args.input)
        results = []
        for atb in sorted(base.glob(args.glob)):
            out = Path(args.output) / (atb.stem + ".q8.sidecar")
            results.append(quantize_atb(atb, out))
        import json
        summary = {
            "n_files": len(results),
            "total_original_mb": sum(r["original_bytes"] for r in results) / 1e6,
            "total_q8_est_mb": sum(r["estimated_q8_file_bytes"] for r in results) / 1e6,
            "mean_compression_ratio": float(np.mean([r["compression_ratio"] for r in results])) if results else 0,
        }
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "storage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 0

    stats = quantize_atb(Path(args.input), Path(args.output), args.quant)
    import json
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
