#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import sys
import threading
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# Add src and build directories to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Force loading build dylib
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/external/llama.cpp/src'))
os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
os.environ["LLAMA_CPP_LIB"] = os.path.join(lib_dir, 'libllama.dylib')

import llama_cpp
import nexus_fsm_ext
from nexus_fsm_ext import ResourceExhaustedError
from nexus_agent import NexusAgent, NexusSpliceContext
from mcp import ClientSession
from mcp.client.sse import sse_client

app = FastAPI(title="Nexus OpenAI-Compatible API Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(ResourceExhaustedError)
async def resource_exhausted_handler(request: Request, exc: ResourceExhaustedError):
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc)},
        headers={"Retry-After": "1"}
    )

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

# Global structures
agent = None
manifest = None
mcp_server_url = "http://localhost:8765/sse"
_request_counter = 0
_request_counter_lock = threading.Lock()


def allocate_request_seq_id() -> int:
    global _request_counter
    with _request_counter_lock:
        _request_counter += 1
        idx = _request_counter
    return int(nexus_fsm_ext.allocate_request_seq(idx))

# Pydantic models for completions
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

@app.on_event("startup")
async def startup_event():
    global agent, manifest, mcp_server_url
    
    model_path = os.environ.get("NEXUS_MODEL")
    embedding_model_path = os.environ.get("NEXUS_EMBED_MODEL")
    manifest_path = os.environ.get("NEXUS_MANIFEST", "test/schemas/slb_manifest.json")
    mcp_server_url = os.environ.get("NEXUS_MCP_SERVER", "http://localhost:8765/sse")

    if not model_path or not embedding_model_path:
        # We will parse arguments if not set in environment
        print("WARNING: Model environment variables not set. Gateway will initialize when first request arrives or if args are supplied via runtime.")
        return

    print(f"Initializing Nexus Gateway with model: {model_path}")
    print(f"Embedding Model: {embedding_model_path}")
    print(f"Manifest Path: {manifest_path}")

    # Load manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Initialize llama-cpp models
    llm = llama_cpp.Llama(
        model_path=model_path,
        n_ctx=8192,
        n_seq_max=128,
        n_gpu_layers=999,
        embedding=False,
        logits_all=True,
        flash_attn=True,
        verbose=False
    )
    
    llm_emb = llama_cpp.Llama(
        model_path=embedding_model_path,
        embedding=True,
        verbose=False
    )

    agent = NexusAgent(llm, llm_emb, max_splice_pos=256)
    llm_rerank = llama_cpp.Llama(
        model_path=model_path, n_ctx=4096, n_batch=512, n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    agent.configure_reranker(llm_rerank=llm_rerank)
    agent.pin_hot_tools([t["id"] for t in manifest["tools"][:5]])

    # Register all tools from manifest
    print(f"Registering {len(manifest['tools'])} tools from manifest...")
    for tool in manifest["tools"]:
        agent.register_tool(
            tool_id=tool["id"],
            name=tool["name"],
            embedding=tool["embedding"],
            digest_text=tool.get("description") or tool.get("digest_text", tool["name"]),
            atb_path=tool["atb_path"],
            schema=tool["schema"]
        )
    print("Nexus engine initialization complete.")

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen2.5-0.5b-instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nexus"
            }
        ]
    }

async def execute_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Connect to live MCP server and run the tool."""
    try:
        async with sse_client(mcp_server_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print(f"Executing tool '{tool_name}' on live Megaserver...")
                call_result = await session.call_tool(tool_name, arguments)
                
                result_text = ""
                for block in call_result.content:
                    if hasattr(block, "text"):
                        result_text += block.text + "\n"
                    elif isinstance(block, dict) and "text" in block:
                        result_text += block["text"] + "\n"
                return result_text.strip()
    except Exception as e:
        return f"Error executing tool {tool_name} via MCP: {e}"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global agent, mcp_server_url
    if agent is None:
        raise HTTPException(status_code=500, detail="Gateway models are not initialized.")

    # Find the latest user query
    user_query = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_query = msg.content
            break
            
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found in the request payload.")

    # We run the synchronous Nexus search, hot-swap splice, GBNF argument generation in a thread
    seq_id = allocate_request_seq_id()
    
    print(f"Processing completions query: '{user_query}'")
    
    # 1. Execute agent generation
    t0 = time.perf_counter()
    resolved_id, tool_name, generated_json = await asyncio.to_thread(
        agent.generate_with_tool, user_query, seq_id, request.max_tokens or 512
    )
    t_gen = time.perf_counter() - t0
    print(f"Generation phase finished in {t_gen:.4f}s. Resolved tool ID: {resolved_id} ({tool_name})")

    if resolved_id == 0:
        # Fallback to standard chat completion
        print("No tool resolved. Falling back to base LLM chat completion...")
        prompt = ""
        for m in request.messages:
            prompt += f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        if request.stream:
            def stream_fallback():
                stream = agent.llm(prompt=prompt, max_tokens=request.max_tokens or 512, stream=True)
                for chunk in stream:
                    delta = chunk["choices"][0]["text"]
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': delta}, 'finish_reason': None}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream_fallback(), media_type="text/event-stream")
        else:
            res = agent.llm(prompt=prompt, max_tokens=request.max_tokens or 512)
            content = res["choices"][0]["text"]
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }]
            }

    # 2. A tool was resolved! We execute the tool on the Megaserver
    print(f"Parsed JSON arguments: {generated_json}")
    try:
        arguments = json.loads(generated_json)
    except Exception:
        arguments = {}

    # Run the live MCP tool call
    tool_output = await execute_mcp_tool(tool_name, arguments)
    print(f"MCP Tool Output: {tool_output}")

    # Build the final response integrating the tool result
    response_content = (
        f"**[Nexus Gateway Routing: Resolved Tool '{tool_name}']**\n"
        f"**Arguments**: `{json.dumps(arguments)}`\n"
        f"**Execution Logs**:\n```\n{tool_output}\n```"
    )

    if request.stream:
        async def stream_tool_response():
            # Simulate streaming of the formatted response
            words = response_content.split(" ")
            for w in words:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': w + ' '}, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_tool_response(), media_type="text/event-stream")
    else:
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "finish_reason": "stop"
            }]
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nexus OpenAI-Compatible Gateway.")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model.")
    parser.add_argument("--embedding-model", type=str, required=True, help="Path to GGUF embedding model.")
    parser.add_argument("--manifest", type=str, default="test/schemas/slb_manifest.json", help="Path to SLB manifest.")
    parser.add_argument("--mcp-server", type=str, default="http://localhost:8765/sse", help="SSE MCP server endpoint.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the gateway on.")
    args = parser.parse_args()

    os.environ["NEXUS_MODEL"] = args.model
    os.environ["NEXUS_EMBED_MODEL"] = args.embedding_model
    os.environ["NEXUS_MANIFEST"] = args.manifest
    os.environ["NEXUS_MCP_SERVER"] = args.mcp_server

    print(f"Starting Gateway FastAPI server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
