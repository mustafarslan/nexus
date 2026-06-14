#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import sys
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Add src and build directories to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Force loading build dylib
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/external/llama.cpp/src'))
os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
os.environ["LLAMA_CPP_LIB"] = os.path.join(lib_dir, 'libllama.dylib')

import llama_cpp
from mcp import ClientSession
from mcp.client.sse import sse_client
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

def compile_subprocess(compiler_path, model_path, schema_path, output_path):
    """Run the compiler binary in a separate process."""
    import subprocess
    cmd = [
        compiler_path,
        "--model", model_path,
        "--schema", schema_path,
        "--output", output_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except Exception as e:
        return False, str(e)

async def main():
    parser = argparse.ArgumentParser(description="Nexus JIT Swarm Compiler.")
    parser.add_argument("--mcp-server", type=str, default="http://localhost:8765/sse", help="SSE MCP server endpoint.")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model for the compiler.")
    parser.add_argument("--embedding-model", type=str, required=True, help="Path to GGUF embedding model.")
    parser.add_argument("--output-dir", type=str, default="test/schemas/jit_bloat", help="Output directory for ATB files.")
    parser.add_argument("--manifest", type=str, default="test/schemas/slb_manifest.json", help="Path to save the SLB manifest.")
    args = parser.parse_args()

    compiler_path = "./build/nexus_kv_compiler"
    if not os.path.exists(compiler_path):
        console.print(f"[bold red]Error: Compiler binary not found at {compiler_path}. Make sure to build the C++ project first.[/bold red]")
        sys.exit(1)

    if not os.path.exists(args.model):
        console.print(f"[bold red]Error: Model not found at {args.model}[/bold red]")
        sys.exit(1)

    if not os.path.exists(args.embedding_model):
        console.print(f"[bold red]Error: Embedding model not found at {args.embedding_model}[/bold red]")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    console.print(f"[bold green]Connecting to live MCP server at {args.mcp_server}...[/bold green]")
    
    # 1. Connect to SSE server and fetch tools list
    tools = []
    try:
        async with sse_client(args.mcp_server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                console.print("[green]Session initialized with Megaserver.[/green]")
                list_result = await session.list_tools()
                
                # Extract tools list (pydantic to list of dicts)
                raw_tools = list_result.tools
                for i, t in enumerate(raw_tools, start=1):
                    tool_dict = {
                        "id": i,
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.inputSchema
                    }
                    tools.append(tool_dict)
    except Exception as e:
        console.print(f"[bold red]Failed to fetch tools from MCP Megaserver: {e}[/bold red]")
        sys.exit(1)

    num_tools = len(tools)
    console.print(f"Fetched [cyan]{num_tools}[/cyan] tools from Megaserver.")

    # 2. Write tool schemas as separate JSON files
    console.print("Writing JSON schemas to disk...")
    for t in tools:
        t_path = os.path.join(args.output_dir, f"{t['name']}.json")
        with open(t_path, "w") as f:
            json.dump(t, f, indent=2)
        t["schema_path"] = t_path
        t["atb_path"] = os.path.join(args.output_dir, f"{t['name']}.atb")

    # 3. Embedding Generation
    console.print(f"Loading embedding model from {args.embedding_model}...")
    llm_emb = llama_cpp.Llama(model_path=args.embedding_model, embedding=True, verbose=False)
    
    console.print("Extracting semantic embeddings for all tools...")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        task = progress.add_task("Generating embeddings...", total=num_tools)
        for t in tools:
            desc = t["description"]
            emb_raw = llm_emb.embed(desc)
            # Ensure shape matches dim
            emb = emb_raw[0] if (len(emb_raw) > 0 and isinstance(emb_raw[0], list)) else emb_raw
            emb = np.array(emb, dtype=np.float32)
            # Normalize embedding
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            t["embedding"] = emb.tolist()
            progress.advance(task)

    del llm_emb
    import gc
    gc.collect()

    # 4. Compilation Pipeline
    console.print("Starting compilation pipeline...")
    
    # Concurrently compile all schemas
    max_workers = min(os.cpu_count() or 4, 4)
    console.print(f"Compiling concurrently with {max_workers} processes...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for t in tools:
            futures.append(
                executor.submit(
                    compile_subprocess,
                    compiler_path,
                    args.model,
                    t["schema_path"],
                    t["atb_path"]
                )
            )
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
            task = progress.add_task("Compiling...", total=num_tools)
            for fut in futures:
                success, err = fut.result()
                if not success:
                    console.print(f"[yellow]Compilation failure warning: {err}[/yellow]")
                progress.advance(task)

    # 5. Output Manifest File
    manifest_data = {"tools": []}
    for t in tools:
        # Scent tokens generated based on tool ID (matching benchmark_mcp_scale.py logic)
        tid = t["id"]
        scent = [int((tid + i) % 150000) for i in range(5)]
        
        manifest_data["tools"].append({
            "id": tid,
            "name": t["name"],
            "description": t["description"],
            "atb_path": os.path.abspath(t["atb_path"]),
            "embedding": t["embedding"],
            "scent_tokens": scent,
            "schema": t["input_schema"]
        })

    with open(args.manifest, "w") as f:
        json.dump(manifest_data, f, indent=2)

    console.print(f"[bold green]Swarm compilation finished! Manifest saved to {args.manifest}[/bold green]")

if __name__ == "__main__":
    asyncio.run(main())
