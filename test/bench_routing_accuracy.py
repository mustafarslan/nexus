#!/usr/bin/env python3
import time
import os
import sys
import json
import numpy as np
import ctypes
import gc
import subprocess
import shutil
import argparse
from pathlib import Path

# Add build and src directories to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Force llama-cpp-python to load our compiled libllama.dylib version
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/external/llama.cpp/src'))
os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
os.environ["LLAMA_CPP_LIB"] = os.path.join(lib_dir, 'libllama.dylib')

import llama_cpp
import nexus_fsm_ext
from benchmark_results import file_sha256, write_artifact

DEFAULT_MODEL = "/Users/mustafarslan/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf"
EMBED_MODEL = "/Users/mustafarslan/.cache/huggingface/hub/models--nomic-ai--nomic-embed-text-v1.5-GGUF/snapshots/0188c9bf409793f810680a5a431e7b899c46104c/nomic-embed-text-v1.5.f16.gguf"

# Define 100 queries mapping to 10 GitHub tools
queries_dataset = [
    # 1. create_or_update_file
    {"query": "Create a readme.md file in repository test-repo under user alice with content Hello World", "tool": "create_or_update_file"},
    {"query": "Update the file config.json in user bob's repo app-settings on branch dev", "tool": "create_or_update_file"},
    {"query": "Save a new python file script.py to the repo run-tasks", "tool": "create_or_update_file"},
    {"query": "Write the latest logs to log.txt in repo server-monitor", "tool": "create_or_update_file"},
    {"query": "Write a script named test.sh to repo auto-tests on dev branch", "tool": "create_or_update_file"},
    {"query": "Create file index.html in web-app repo of user charlie", "tool": "create_or_update_file"},
    {"query": "Modify readme.md to add setup instructions in project repo", "tool": "create_or_update_file"},
    {"query": "Upload data.csv to database-utils repository on main branch", "tool": "create_or_update_file"},
    {"query": "Add a license.txt file to repo open-source-project", "tool": "create_or_update_file"},
    {"query": "Update main.py file content in repo ml-inference", "tool": "create_or_update_file"},

    # 2. search_repositories
    {"query": "Search for repositories containing python web framework", "tool": "search_repositories"},
    {"query": "Find popular github repositories matching machine learning", "tool": "search_repositories"},
    {"query": "Search repositories with keyword rust-game-engine", "tool": "search_repositories"},
    {"query": "Search github projects about docker configurations", "tool": "search_repositories"},
    {"query": "Look for repositories with query react-native-navigation", "tool": "search_repositories"},
    {"query": "Find all repos containing LLM fine-tuning scripts", "tool": "search_repositories"},
    {"query": "Query repositories for microservices in go", "tool": "search_repositories"},
    {"query": "Search github for kubernetes operators", "tool": "search_repositories"},
    {"query": "Find repositories matching compiler design in c++", "tool": "search_repositories"},
    {"query": "Look up github repositories matching web scraper", "tool": "search_repositories"},

    # 3. create_repository
    {"query": "Create a new private repository named secrets-manager", "tool": "create_repository"},
    {"query": "Create a public github repo named personal-blog with README", "tool": "create_repository"},
    {"query": "Create repository web-crawler in my account", "tool": "create_repository"},
    {"query": "Initialize a repository named data-science-experiments", "tool": "create_repository"},
    {"query": "Make a new repository named task-manager-app", "tool": "create_repository"},
    {"query": "Create a private repository auth-service", "tool": "create_repository"},
    {"query": "Start a new github repo named api-gateway-node", "tool": "create_repository"},
    {"query": "Create repository chatbot-ui with default settings", "tool": "create_repository"},
    {"query": "Initialize a new repository named rust-cli-tool", "tool": "create_repository"},
    {"query": "Create a repository under my user called configuration-files", "tool": "create_repository"},

    # 4. get_file_contents
    {"query": "Get the content of readme.md in repo test-repo owned by alice", "tool": "get_file_contents"},
    {"query": "Read file package.json from bob's repo app-settings on branch main", "tool": "get_file_contents"},
    {"query": "Get the contents of src/main.cpp in repo game-engine", "tool": "get_file_contents"},
    {"query": "Retrieve config/database.yaml file from user db-admin's repo", "tool": "get_file_contents"},
    {"query": "Show me what is in scripts/deploy.sh in repository cloud-infra", "tool": "get_file_contents"},
    {"query": "Read the contents of docker-compose.yml from repo dev-env", "tool": "get_file_contents"},
    {"query": "Download the file requirements.txt from python-project repo", "tool": "get_file_contents"},
    {"query": "Retrieve file contents of index.js in repo frontend-ui", "tool": "get_file_contents"},
    {"query": "Read license file in repo open-source-project", "tool": "get_file_contents"},
    {"query": "Show file contents of src/lib.rs in rust-library repo", "tool": "get_file_contents"},

    # 5. push_files
    {"query": "Push src/app.py and tests/test_app.py to repo flask-app", "tool": "push_files"},
    {"query": "Commit files readme.md and license.md to repo project-init on branch main", "tool": "push_files"},
    {"query": "Push update to config.json and settings.yaml in user bob's repo", "tool": "push_files"},
    {"query": "Push script.py and requirements.txt to python-utils repository", "tool": "push_files"},
    {"query": "Commit new changes to index.html and style.css in web-app repo", "tool": "push_files"},
    {"query": "Push multiple files src/main.go and go.mod to go-service", "tool": "push_files"},
    {"query": "Push files to repository doc-site including docs/intro.md and docs/api.md", "tool": "push_files"},
    {"query": "Commit assets/logo.png and public/index.html to user charlie's repo", "tool": "push_files"},
    {"query": "Push updates for main.py and model.bin to repo model-server", "tool": "push_files"},
    {"query": "Push code files to dev branch of project-repo", "tool": "push_files"},

    # 6. create_issue
    {"query": "Create a new issue in repository test-repo with title Bug in login", "tool": "create_issue"},
    {"query": "Open an issue about performance degradation in user bob's repo", "tool": "create_issue"},
    {"query": "Create issue named Memory leak on startup in repo game-engine", "tool": "create_issue"},
    {"query": "Open issue report for broken link on homepage in web-site repo", "tool": "create_issue"},
    {"query": "Create an issue with title Feature request: dark mode in repo app", "tool": "create_issue"},
    {"query": "Open a bug report issue in repo database-connector", "tool": "create_issue"},
    {"query": "Create a new issue about security vulnerability in auth-service", "tool": "create_issue"},
    {"query": "Create issue Docker build failure in repository devops-tools", "tool": "create_issue"},
    {"query": "Open issue for API endpoint timeout in repo server-backend", "tool": "create_issue"},
    {"query": "Create an issue to update dependencies in repo library", "tool": "create_issue"},

    # 7. create_pull_request
    {"query": "Create a pull request in repo test-repo with title Add dark mode", "tool": "create_pull_request"},
    {"query": "Open a PR in user bob's repo from branch dev to branch main", "tool": "create_pull_request"},
    {"query": "Create pull request to merge feature-login into master in repo app", "tool": "create_pull_request"},
    {"query": "Open PR for hotfix-auth branch in repository auth-service", "tool": "create_pull_request"},
    {"query": "Create pull request named Update documentation in project repo", "tool": "create_pull_request"},
    {"query": "Open a PR to merge release-v1 into main in repo production", "tool": "create_pull_request"},
    {"query": "Create pull request to add test cases in repository core-engine", "tool": "create_pull_request"},
    {"query": "Open PR for branch bugfix-router in user charlie's repo", "tool": "create_pull_request"},
    {"query": "Create pull request to merge refactor-db to main branch", "tool": "create_pull_request"},
    {"query": "Open a PR titled Fix memory leak in repository native-app", "tool": "create_pull_request"},

    # 8. fork_repository
    {"query": "Fork the repository python-sdk to my account", "tool": "fork_repository"},
    {"query": "Fork repository status-monitor to organization dev-org", "tool": "fork_repository"},
    {"query": "Create a fork of user bob's repository app-settings", "tool": "fork_repository"},
    {"query": "Fork repo django/django to my profile", "tool": "fork_repository"},
    {"query": "Fork repository PyTorch to my personal workspace", "tool": "fork_repository"},
    {"query": "Create a fork of repo tensorflow/tensorflow", "tool": "fork_repository"},
    {"query": "Fork repository rust-lang/rust to my github account", "tool": "fork_repository"},
    {"query": "Fork repository nodejs/node to my account", "tool": "fork_repository"},
    {"query": "Create a fork of repository golang/go", "tool": "fork_repository"},
    {"query": "Fork repository helm/helm to my profile", "tool": "fork_repository"},

    # 9. create_branch
    {"query": "Create a new branch named feature-auth in repository test-repo", "tool": "create_branch"},
    {"query": "Create branch bugfix-login from branch main in user bob's repo", "tool": "create_branch"},
    {"query": "Make a branch named dev-setup in repo web-app", "tool": "create_branch"},
    {"query": "Create branch release-v2 from branch main in repo core", "tool": "create_branch"},
    {"query": "Create a new branch hotfix-crashes in repository server", "tool": "create_branch"},
    {"query": "Make branch refactor-models in repository ml-inference", "tool": "create_branch"},
    {"query": "Create branch patch-1 from branch main in user charlie's repo", "tool": "create_branch"},
    {"query": "Create branch feat-custom-themes in repository blog-theme", "tool": "create_branch"},
    {"query": "Create a branch named test-coverage in repo core-engine", "tool": "create_branch"},
    {"query": "Make branch docs-update from main in repo open-source-project", "tool": "create_branch"},

    # 10. list_commits
    {"query": "List the recent commits of branch main in repository test-repo", "tool": "list_commits"},
    {"query": "Get list of commits from user bob's repo app-settings", "tool": "list_commits"},
    {"query": "Show me the commits on dev branch in repo web-app", "tool": "list_commits"},
    {"query": "Retrieve commit history for branch main in repository server", "tool": "list_commits"},
    {"query": "List commits in repository game-engine on branch master", "tool": "list_commits"},
    {"query": "Get list of commits in repository core-engine", "tool": "list_commits"},
    {"query": "Show commits in user charlie's repo frontend-ui", "tool": "list_commits"},
    {"query": "List commits for branch bugfix-login in repo auth-service", "tool": "list_commits"},
    {"query": "Retrieve the commit list in repository ml-inference", "tool": "list_commits"},
    {"query": "List commits in repo database-connector on dev branch", "tool": "list_commits"},
]

def load_first_10_tools():
    json_path = "test/schemas/github_tools.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    tools = data.get("tools", [])
    
    selected_tools = []
    names = [
        "create_or_update_file",
        "search_repositories",
        "create_repository",
        "get_file_contents",
        "push_files",
        "create_issue",
        "create_pull_request",
        "fork_repository",
        "create_branch",
        "list_commits"
    ]
    for idx, name in enumerate(names):
        tool = next(t for t in tools if t["name"] == name)
        tool_copy = tool.copy()
        tool_copy["id"] = idx + 1
        tool_copy["desc"] = tool_copy["description"]
        selected_tools.append(tool_copy)
    return selected_tools

def compile_atb_files(tools, model_path):
    os.makedirs("test/schemas/jit_bloat", exist_ok=True)
    for tool in tools:
        name = tool["name"]
        json_path = f"test/schemas/jit_bloat/{name}.json"
        atb_path = f"test/schemas/jit_bloat/{name}.atb"
        
        # Write schema to temp json file
        schema_json = tool["inputSchema"] if "inputSchema" in tool else tool
        # Wrap it in standard schema object structure expected by compiler
        wrapper = {"inputSchema": schema_json}
        with open(json_path, "w") as f:
            json.dump(wrapper, f)
            
        # Compile to atb
        subprocess.run([
            "./build/nexus_kv_compiler",
            "--model", model_path,
            "--schema", json_path,
            "--output", atb_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tool["atb_path"] = atb_path

def parse_args():
    parser = argparse.ArgumentParser(description="Nexus routing accuracy benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GGUF chat/model path")
    parser.add_argument("--embed-model", default=EMBED_MODEL, help="GGUF embedding model path")
    parser.add_argument("--output", default="results/bench_routing_accuracy.json", help="JSON artifact path")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic seed for benchmark libraries")
    parser.add_argument("--auto-route-margin", type=float, default=0.10)
    parser.add_argument("--speculative-threshold", type=float, default=0.0)
    parser.add_argument("--speculative-margin", type=float, default=0.0)
    return parser.parse_args()

def run_evaluation(args):
    np.random.seed(args.seed)
    print("======================================================================")
    print("              NEXUS ROUTING ACCURACY COMPARATIVE BENCHMARK             ")
    print("======================================================================")
    
    # 1. Load the 10 tools and compile them
    tools = load_first_10_tools()
    print(f"Loaded 10 complex tools. Compiling to ATB...")
    compile_atb_files(tools, args.model)
    print("ATB compilation complete.")
    
    # 2. Instantiate embedding model
    print(f"Loading embedding model from {args.embed_model}...")
    llm_emb = llama_cpp.Llama(
        model_path=args.embed_model,
        embedding=True,
        verbose=False
    )
    emb_dim = llm_emb.n_embd()
    print(f"Embedding model loaded. Dimension: {emb_dim}")
    
    # 3. Instantiate LLM
    print(f"Loading Qwen model from {args.model}...")
    llm = llama_cpp.Llama(
        model_path=args.model,
        n_ctx=8192,
        n_gpu_layers=999,
        logits_all=True,
        flash_attn=True,
        verbose=False
    )
    ctx = llm._ctx.ctx
    print("Qwen model loaded successfully.")

    # 4. Setup SLB & FSM
    slb = nexus_fsm_ext.NexusSemanticSLB(emb_dim)
    fsm = nexus_fsm_ext.NexusRadixFSM()
    
    # Register tools
    for tool in tools:
        tid = tool["id"]
        tname = tool["name"]
        tdesc = tool["desc"]
        
        # Embed description
        emb_raw = llm_emb.embed(tdesc)
        emb = emb_raw[0] if (len(emb_raw) > 0 and isinstance(emb_raw[0], list)) else emb_raw
        emb = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        
        # Register in FSM
        route_tokens = [int(t) for t in llm.tokenize(tname.encode("utf-8"), add_bos=False, special=False)]
        fsm.add_route(tid, route_tokens)
        
        # Register in SLB
        digest_text = f"Tool Name: {tname}. Description: {tdesc}."
        digest_tokens = [int(t) for t in llm.tokenize(digest_text.encode("utf-8"), add_bos=False, special=False)]
        if len(digest_tokens) > 64:
            digest_tokens = digest_tokens[:64]
        slb.register_tool(tid, emb, digest_tokens)
        
    # 5. Initialize Orchestrator
    orchestrator = nexus_fsm_ext.NexusOrchestrator(
        ctx,
        slb,
        fsm,
        base_pos=256,
        speculative_threshold=args.speculative_threshold,
        speculative_margin=args.speculative_margin,
        auto_route_margin=args.auto_route_margin,
    )
    for tool in tools:
        orchestrator.register_tool_path(tool["id"], tool["atb_path"])
        
    # Evaluate System Prompt context
    system_prompt = "You are a helpful agent."
    sys_tokens = llm.tokenize(f"<|im_start|>system\n{system_prompt}<|im_end|>\n".encode("utf-8"), add_bos=False, special=False)
    sys_len = len(sys_tokens)
    
    # --- EVALUATE ---
    hits_a = 0
    hits_b = 0
    auto_route_count = 0
    fsm_fallback_count = 0
    total = len(queries_dataset)
    full_context_latencies_ms = []
    nexus_latencies_ms = []
    slb_latencies_ms = []
    records = []
    
    print(f"\nRunning comparative benchmark over {total} synthetic queries...")
    for idx, item in enumerate(queries_dataset):
        query = item["query"]
        target_tool = item["tool"]
        
        # (A) Scenario A: Standard Full-Context Schema Prefill
        llm.eval(sys_tokens)
        schemas_str = "\n".join([f"Tool {t['id']}: {t['name']}\nDescription: {t['desc']}" for t in tools])
        prompt_a = f"""You are a tool routing agent. Select the single most appropriate tool name from the list below that matches the user's intent.
You MUST output ONLY the name of the tool, with no other text, punctuation, explanation, or markdown.

Available Tools:
{schemas_str}

Query: {query}
Selected Tool Name:"""
        
        t0 = time.perf_counter()
        res_a = llm.create_completion(
            prompt=prompt_a,
            max_tokens=15,
            temperature=0.0
        )
        full_context_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        pred_a = res_a["choices"][0]["text"].strip()
        # Make comparison robust: check if target_tool is a clean substring or exact match
        is_hit_a = (target_tool in pred_a) or (pred_a in target_tool) or (pred_a.replace("`", "") == target_tool)
        if is_hit_a:
            hits_a += 1
            
        # Reset KV Cache for LLM
        llm.reset()
        
        # (B) Scenario B: Nexus SLB scan + FSM masking
        # Evaluate System Prompt once
        llm.eval(sys_tokens)
        query_tokens = [int(t) for t in llm.tokenize(query.encode("utf-8"), add_bos=False, special=False)]
        
        # Embed query
        query_emb_raw = llm_emb.embed(query)
        query_emb = query_emb_raw[0] if (len(query_emb_raw) > 0 and isinstance(query_emb_raw[0], list)) else query_emb_raw
        query_emb = np.array(query_emb, dtype=np.float32)
        q_norm = np.linalg.norm(query_emb)
        if q_norm > 0:
            query_emb = query_emb / q_norm
        
        # Track path distribution
        t_slb0 = time.perf_counter()
        matches = slb.search(query_emb, 3)
        slb_latencies_ms.append((time.perf_counter() - t_slb0) * 1000.0)
        is_auto_route = False
        if len(matches) >= 2:
            is_auto_route = (matches[0].score - matches[1].score) >= 0.10
        elif len(matches) > 0:
            is_auto_route = True
            
        if is_auto_route:
            auto_route_count += 1
        else:
            fsm_fallback_count += 1
            
        # Run route_and_splice
        t_nexus0 = time.perf_counter()
        resolved_id = orchestrator.route_and_splice(query_tokens, list(query_emb), sys_len, 0)
        nexus_latencies_ms.append((time.perf_counter() - t_nexus0) * 1000.0)
        
        # Map back to tool name
        pred_b = ""
        for tool in tools:
            if tool["id"] == resolved_id:
                pred_b = tool["name"]
                break
                
        is_hit_b = (pred_b == target_tool)
        if is_hit_b:
            hits_b += 1
        records.append({
            "idx": idx,
            "query": query,
            "target_tool": target_tool,
            "full_context_prediction": pred_a,
            "nexus_prediction": pred_b,
            "full_context_hit": is_hit_a,
            "nexus_hit": is_hit_b,
            "path": "auto_route" if is_auto_route else "fsm_fallback",
            "slb_top1_tool_id": matches[0].tool_id if matches else 0,
            "slb_top1_score": float(matches[0].score) if matches else 0.0,
            "slb_margin": float((matches[0].score - matches[1].score) if len(matches) >= 2 else 0.0),
        })
            
        # Clean up context for next iteration
        orchestrator.release_hazard(0)
        llm._ctx.kv_cache_seq_rm(0, sys_len, -1)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{total} queries...")
            
    accuracy_a = (hits_a / total) * 100
    accuracy_b = (hits_b / total) * 100
    auto_route_pct = (auto_route_count / total) * 100
    fsm_fallback_pct = (fsm_fallback_count / total) * 100
    
    print("\n======================================================================")
    print("                           BENCHMARK RESULTS                          ")
    print("======================================================================")
    print(f"Scenario A: Full-Context Schema Prefill Accuracy: {accuracy_a:.1f}% ({hits_a}/{total})")
    print(f"Scenario B: Nexus SLB Scan + FSM Masking Accuracy:  {accuracy_b:.1f}% ({hits_b}/{total})")
    print("----------------------------------------------------------------------")
    print(f"Auto-Route Fast Path Resolution:                    {auto_route_pct:.1f}% ({auto_route_count}/{total})")
    print(f"FSM Fallback Resolution:                            {fsm_fallback_pct:.1f}% ({fsm_fallback_count}/{total})")
    print("======================================================================")

    write_artifact(
        args.output,
        "bench_routing_accuracy",
        model_hash=file_sha256(args.model) if Path(args.model).exists() else "",
        config={
            "model_path": args.model,
            "embedding_model_path": args.embed_model,
            "embedding_model_hash": file_sha256(args.embed_model) if Path(args.embed_model).exists() else "",
            "seed": args.seed,
            "tool_count": len(tools),
            "query_count": total,
            "auto_route_margin": args.auto_route_margin,
            "speculative_threshold": args.speculative_threshold,
            "speculative_margin": args.speculative_margin,
        },
        metrics={
            "full_context_accuracy_pct": {"unit": "percent", "raw_samples": [accuracy_a]},
            "nexus_accuracy_pct": {"unit": "percent", "raw_samples": [accuracy_b]},
            "auto_route_pct": {"unit": "percent", "raw_samples": [auto_route_pct]},
            "fsm_fallback_pct": {"unit": "percent", "raw_samples": [fsm_fallback_pct]},
            "full_context_latency_ms": {"unit": "ms", "raw_samples": full_context_latencies_ms},
            "nexus_route_and_splice_latency_ms": {"unit": "ms", "raw_samples": nexus_latencies_ms},
            "slb_scan_latency_ms": {"unit": "ms", "raw_samples": slb_latencies_ms},
        },
        records=records,
    )
    print(f"Wrote JSON artifact: {args.output}")

    # Clean up directories and files robustly
    if os.path.exists("test/schemas/jit_bloat"):
        shutil.rmtree("test/schemas/jit_bloat")

if __name__ == "__main__":
    run_evaluation(parse_args())
