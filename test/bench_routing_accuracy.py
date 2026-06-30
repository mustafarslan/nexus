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
    return load_n_tools(10)

def load_n_tools(n: int):
    json_path = "test/schemas/github_tools.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    tools = data.get("tools", [])
    
    # 10 core target tools in their exact original order
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
    
    selected_tools = []
    for idx, name in enumerate(names):
        tool = next(t for t in tools if t["name"] == name)
        tool_copy = tool.copy()
        tool_copy["id"] = idx + 1
        tool_copy["desc"] = tool_copy["description"]
        selected_tools.append(tool_copy)
        
    if n <= 10:
        return selected_tools[:n]
        
    # Inject Semantic Blur tools at positions 11 and 12
    blur_tools = [
        {
            "name": "create_or_update_file_metadata",
            "description": "Create or update file metadata in a repository, such as custom headers, file attributes, labels, or version tags. Do NOT use for creating or editing file content.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Owner of the repo"},
                    "repo": {"type": "string", "description": "Repo name"},
                    "path": {"type": "string", "description": "File path"},
                    "metadata": {"type": "object", "description": "Metadata object"}
                },
                "required": ["owner", "repo", "path"]
            }
        },
        {
            "name": "get_repository_data",
            "description": "Retrieve generic data, settings, properties, or stats from a repository. Do NOT use for downloading the text/code content of files.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Owner of the repo"},
                    "repo": {"type": "string", "description": "Repo name"},
                    "include_stats": {"type": "boolean", "description": "Include repository stats"}
                },
                "required": ["owner", "repo"]
            }
        }
    ]
    
    added_names = set(names)
    current_id = 11
    
    for bt in blur_tools:
        bt_copy = bt.copy()
        bt_copy["id"] = current_id
        bt_copy["desc"] = bt_copy["description"]
        selected_tools.append(bt_copy)
        added_names.add(bt_copy["name"])
        current_id += 1
        if len(selected_tools) >= n:
            return selected_tools
            
    # Load other tools from github_tools.json
    for tool in tools:
        if tool["name"] not in added_names:
            tool_copy = tool.copy()
            tool_copy["id"] = current_id
            tool_copy["desc"] = tool_copy["description"]
            selected_tools.append(tool_copy)
            added_names.add(tool["name"])
            current_id += 1
            if len(selected_tools) >= n:
                return selected_tools
                
    # Sample from test/schemas/bloat/ if needed
    bloat_dir = Path("test/schemas/bloat")
    if bloat_dir.exists() and len(selected_tools) < n:
        bloat_files = sorted(list(bloat_dir.glob("*.json")))
        for bf in bloat_files:
            try:
                with open(bf, "r") as f:
                    tool_data = json.load(f)
                name = tool_data.get("name")
                if name not in added_names:
                    tool_copy = tool_data.copy()
                    tool_copy["id"] = current_id
                    tool_copy["desc"] = tool_data.get("description", "")
                    if "inputSchema" not in tool_copy and "input_schema" in tool_copy:
                        tool_copy["inputSchema"] = tool_copy["input_schema"]
                    selected_tools.append(tool_copy)
                    added_names.add(name)
                    current_id += 1
                    if len(selected_tools) >= n:
                        break
            except Exception as e:
                continue
                
    return selected_tools

def compile_atb_files(tools, model_path):
    os.makedirs("test/schemas/jit_bloat", exist_ok=True)
    
    # 1. Identify which tools need compilation
    to_compile = []
    for tool in tools:
        name = tool["name"]
        json_path = f"test/schemas/jit_bloat/{name}.json"
        atb_path = f"test/schemas/jit_bloat/{name}.atb"
        
        if os.path.exists(atb_path):
            tool["atb_path"] = atb_path
        else:
            to_compile.append((tool, json_path, atb_path))
            
    if not to_compile:
        return
        
    # 2. Write temp JSON files and prepare batch list
    batch_lines = []
    for tool, json_path, atb_path in to_compile:
        schema_json = tool.get("inputSchema") or tool.get("input_schema") or tool
        wrapper = {"inputSchema": schema_json}
        with open(json_path, "w") as f:
            json.dump(wrapper, f)
        batch_lines.append(f"{json_path},{atb_path}\n")
        
    batch_list_path = "test/schemas/jit_bloat/batch_list.txt"
    with open(batch_list_path, "w") as f:
        f.writelines(batch_lines)
        
    # 3. Launch compiler once (GPU-enabled)
    subprocess.run([
        "./build/nexus_kv_compiler",
        "--model", model_path,
        "--batch-list", batch_list_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 4. Update tools and clean up batch list
    for tool, json_path, atb_path in to_compile:
        tool["atb_path"] = atb_path
        
    if os.path.exists(batch_list_path):
        os.remove(batch_list_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Nexus routing accuracy benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GGUF chat/model path")
    parser.add_argument("--embed-model", default=EMBED_MODEL, help="GGUF embedding model path")
    parser.add_argument("--output", default="results/bench_routing_accuracy.json", help="JSON artifact path")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic seed for benchmark libraries")
    parser.add_argument("--auto-route-margin", type=float, default=0.10)
    parser.add_argument("--speculative-threshold", type=float, default=0.0)
    parser.add_argument("--speculative-margin", type=float, default=0.0)
    parser.add_argument("--tool-sizes", default="10", help="Space-separated list of tool registry sizes, e.g. '10 20 26'")
    parser.add_argument("--runs", type=int, default=1, help="Number of trials per tool size")
    return parser.parse_args()

def run_evaluation(args):
    sizes = [int(s) for s in args.tool_sizes.split()]
    print(f"Pre-compiling all tools up to size {max(sizes)} on GPU...")
    max_tools = load_n_tools(max(sizes))
    compile_atb_files(max_tools, args.model)
    print("Pre-compilation finished successfully.")
    
    global llama_cpp, nexus_fsm_ext
    import llama_cpp
    import nexus_fsm_ext
    
    # 1. Instantiate models once outside the loops
    print(f"Loading embedding model from {args.embed_model}...")
    llm_emb = llama_cpp.Llama(
        model_path=args.embed_model,
        embedding=True,
        verbose=False
    )
    emb_dim = llm_emb.n_embd()
    print(f"Embedding model loaded. Dimension: {emb_dim}")
    
    print(f"Loading Qwen model from {args.model}...")
    llm = llama_cpp.Llama(
        model_path=args.model,
        n_ctx=32768,  # Qwen2.5 supports 32k; 8192 overflows once the all-schemas anchor (sys_len) grows past N>=50
        n_gpu_layers=999,
        logits_all=True,
        flash_attn=True,
        verbose=False
    )
    ctx = llm._ctx.ctx
    print("Qwen model loaded successfully.")

    sizes = [int(s) for s in args.tool_sizes.split()]
    runs = args.runs
    
    overall_summary = {}

    print("======================================================================")
    print("              NEXUS ROUTING ACCURACY COMPARATIVE BENCHMARK             ")
    print("======================================================================")

    for size in sizes:
        print(f"\nEvaluating Registry Size N = {size} over {runs} runs...")
        
        runs_acc_a = []
        runs_acc_b = []
        runs_rec1 = []
        runs_rec3 = []
        runs_rec5 = []
        runs_auto_route = []
        runs_fsm_fallback = []
        runs_latency_a = []
        runs_latency_b = []
        runs_latency_slb = []
        runs_tokens_a = []
        runs_tokens_b = []
        
        for run_idx in range(runs):
            run_seed = args.seed + run_idx
            np.random.seed(run_seed)
            print(f"  Run {run_idx + 1}/{runs} (seed={run_seed})...")
            llm.reset()
            
            # Load tools
            tools = load_n_tools(size)
            compile_atb_files(tools, args.model)
            
            # Setup SLB & FSM
            slb = nexus_fsm_ext.NexusSemanticSLB(emb_dim)
            fsm = nexus_fsm_ext.NexusRadixFSM()
            
            # Register tools
            for tool in tools:
                tid = tool["id"]
                tname = tool["name"]
                tdesc = tool["desc"]
                
                # Embed document: nomic-embed-v1.5 requires the "search_document:" task
                # prefix, and the tool NAME carries dominant routing signal (probe: name+desc
                # lifts R@1 40%->74%, R@3 70%->95% at N=250). Mirror this on the query side.
                emb_raw = llm_emb.embed(f"search_document: {tname}: {tdesc}")
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
                
            # Initialize Orchestrator
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
                
            # System Prompt
            system_prompt = "You are a helpful agent."
            sys_tokens = llm.tokenize(f"<|im_start|>system\n{system_prompt}<|im_end|>\n".encode("utf-8"), add_bos=False, special=False)
            sys_len = len(sys_tokens)
            
            # Calculate raw schema tokens for Scenario A (Oracle)
            # Scenario A prompt contains all JSON schemas
            schemas_str = "\n".join([f"Tool {t['id']}: {t['name']}\nDescription: {t['desc']}" for t in tools])
            schemas_tokens = llm.tokenize(schemas_str.encode("utf-8"), add_bos=False, special=False)
            
            hits_a = 0
            hits_b = 0
            top1_recall_hits = 0
            top3_recall_hits = 0
            top5_recall_hits = 0
            auto_route_count = 0
            fsm_fallback_count = 0
            total = len(queries_dataset)
            
            lat_a = []
            lat_b = []
            lat_slb = []
            tokens_a_list = []
            tokens_b_list = []
            
            for idx, item in enumerate(queries_dataset):
                query = item["query"]
                target_tool = item["tool"]
                target_id = next((t["id"] for t in tools if t["name"] == target_tool), -1)
                
                # Check if target_tool is even in the active registry for this size
                # if not, skip this query for accurate metrics
                if target_id == -1:
                    total -= 1
                    continue
                
                query_tokens = llm.tokenize(query.encode("utf-8"), add_bos=False, special=False)
                
                # Token counts
                # Oracle: sys + schemas + query + prompt overhead
                tokens_a = sys_len + len(schemas_tokens) + len(query_tokens) + 50  # 50 approx formatting tokens
                tokens_a_list.append(tokens_a)
                
                # Nexus selected tool IR token count
                # Find matching target tool to compute its IR representation
                # Using a fallback length of 20 tokens if name doesn't match
                selected_tool_dict = next((t for t in tools if t["id"] == target_id), None)
                ir_len = 20
                if selected_tool_dict:
                    # Form signature IR
                    name = selected_tool_dict.get("name")
                    schema = selected_tool_dict.get("inputSchema") or selected_tool_dict.get("input_schema") or {}
                    props = schema.get("properties", {}) or {}
                    params = ", ".join(f"{p}: string" for p in props)
                    sig = f"{name}({params})"
                    ir_len = len(llm.tokenize(sig.encode("utf-8"), add_bos=False, special=False))
                    
                # Nexus: sys + query + selected tool IR
                tokens_b = sys_len + len(query_tokens) + ir_len + 15
                tokens_b_list.append(tokens_b)
                
                # (A) Scenario A
                if tokens_a < 7500:
                    llm.eval(sys_tokens)
                    prompt_a = f"""You are a tool routing agent. Select the single most appropriate tool name from the list below that matches the user's intent.
You MUST output ONLY the name of the tool, with no other text, punctuation, explanation, or markdown.

Available Tools:
{schemas_str}

Query: {query}
Selected Tool Name:"""
                    
                    t0 = time.perf_counter()
                    try:
                        res_a = llm.create_completion(
                            prompt=prompt_a,
                            max_tokens=15,
                            temperature=0.0
                        )
                        lat_a.append((time.perf_counter() - t0) * 1000.0)
                        pred_a = res_a["choices"][0]["text"].strip()
                        is_hit_a = (target_tool in pred_a) or (pred_a in target_tool) or (pred_a.replace("`", "") == target_tool)
                        if is_hit_a:
                            hits_a += 1
                    except Exception:
                        is_hit_a = False
                    llm.reset()
                else:
                    is_hit_a = False
                    pred_a = "N/A (Context Overflow)"
                
                # (B) Scenario B
                llm.eval(sys_tokens)
                query_tokens_list = [int(t) for t in query_tokens]
                
                # Embed query with nomic-embed-v1.5 "search_query:" task prefix (asymmetric
                # retrieval; must pair with the "search_document:" prefix used at registration).
                query_emb_raw = llm_emb.embed(f"search_query: {query}")
                query_emb = query_emb_raw[0] if (len(query_emb_raw) > 0 and isinstance(query_emb_raw[0], list)) else query_emb_raw
                query_emb = np.array(query_emb, dtype=np.float32)
                q_norm = np.linalg.norm(query_emb)
                if q_norm > 0:
                    query_emb = query_emb / q_norm
                
                # Track SLB search
                t_slb0 = time.perf_counter()
                matches = slb.search(query_emb, 5)  # Fetch up to 5 to measure Top-5 Recall
                lat_slb.append((time.perf_counter() - t_slb0) * 1000.0)
                
                # Measure SLB Recall
                match_ids = [m.tool_id for m in matches]
                if len(match_ids) > 0 and match_ids[0] == target_id:
                    top1_recall_hits += 1
                if target_id in match_ids[:3]:
                    top3_recall_hits += 1
                if target_id in match_ids[:5]:
                    top5_recall_hits += 1
                
                is_auto_route = False
                if len(matches) >= 2:
                    is_auto_route = (matches[0].score - matches[1].score) >= args.auto_route_margin
                elif len(matches) > 0:
                    is_auto_route = True
                    
                if is_auto_route:
                    auto_route_count += 1
                else:
                    fsm_fallback_count += 1
                    
                # Run route_and_splice
                t_nexus0 = time.perf_counter()
                resolved_id = orchestrator.route_and_splice(query_tokens_list, list(query_emb), sys_len, 0)
                lat_b.append((time.perf_counter() - t_nexus0) * 1000.0)
                
                # Map back
                pred_b = ""
                for tool in tools:
                    if tool["id"] == resolved_id:
                        pred_b = tool["name"]
                        break
                is_hit_b = (pred_b == target_tool)
                if is_hit_b:
                    hits_b += 1
                    
                # Clean up context
                orchestrator.release_hazard(0)
                llm._ctx.kv_cache_seq_rm(0, sys_len, -1)
                llm.reset()
                
            accuracy_a = (hits_a / total) * 100 if total > 0 else 0.0
            accuracy_b = (hits_b / total) * 100 if total > 0 else 0.0
            top1_recall = (top1_recall_hits / total) * 100 if total > 0 else 0.0
            top3_recall = (top3_recall_hits / total) * 100 if total > 0 else 0.0
            top5_recall = (top5_recall_hits / total) * 100 if total > 0 else 0.0
            auto_route_pct = (auto_route_count / total) * 100 if total > 0 else 0.0
            fsm_fallback_pct = (fsm_fallback_count / total) * 100 if total > 0 else 0.0
            
            runs_acc_a.append(accuracy_a)
            runs_acc_b.append(accuracy_b)
            runs_rec1.append(top1_recall)
            runs_rec3.append(top3_recall)
            runs_rec5.append(top5_recall)
            runs_auto_route.append(auto_route_pct)
            runs_fsm_fallback.append(fsm_fallback_pct)
            runs_latency_a.extend(lat_a)
            runs_latency_b.extend(lat_b)
            runs_latency_slb.extend(lat_slb)
            runs_tokens_a.extend(tokens_a_list)
            runs_tokens_b.extend(tokens_b_list)
            
            pass
                
        # Aggregate statistics
        import statistics
        def stats(vals):
            if not vals:
                return 0.0, 0.0
            return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)
            
        mean_acc_a, std_acc_a = stats(runs_acc_a)
        mean_acc_b, std_acc_b = stats(runs_acc_b)
        mean_rec1, std_rec1 = stats(runs_rec1)
        mean_rec3, std_rec3 = stats(runs_rec3)
        mean_rec5, std_rec5 = stats(runs_rec5)
        mean_auto, std_auto = stats(runs_auto_route)
        mean_fallback, std_fallback = stats(runs_fsm_fallback)
        
        avg_lat_a = statistics.mean(runs_latency_a) if runs_latency_a else 0.0
        avg_lat_b = statistics.mean(runs_latency_b) if runs_latency_b else 0.0
        avg_lat_slb = statistics.mean(runs_latency_slb) if runs_latency_slb else 0.0
        
        avg_tokens_a = statistics.mean(runs_tokens_a) if runs_tokens_a else 0.0
        avg_tokens_b = statistics.mean(runs_tokens_b) if runs_tokens_b else 0.0
        token_savings_pct = (1.0 - (avg_tokens_b / avg_tokens_a)) * 100.0 if avg_tokens_a > 0 else 0.0
        
        overall_summary[size] = {
            "mean_acc_a": mean_acc_a, "std_acc_a": std_acc_a,
            "mean_acc_b": mean_acc_b, "std_acc_b": std_acc_b,
            "mean_rec1": mean_rec1, "std_rec1": std_rec1,
            "mean_rec3": mean_rec3, "std_rec3": std_rec3,
            "mean_rec5": mean_rec5, "std_rec5": std_rec5,
            "mean_auto": mean_auto, "std_auto": std_auto,
            "mean_fallback": mean_fallback, "std_fallback": std_fallback,
            "avg_lat_a_ms": avg_lat_a,
            "avg_lat_b_ms": avg_lat_b,
            "avg_lat_slb_ms": avg_lat_slb,
            "avg_tokens_a": avg_tokens_a,
            "avg_tokens_b": avg_tokens_b,
            "token_savings_pct": token_savings_pct,
        }
        res = overall_summary[size]
        print(f"Results for N = {size}:")
        if res['avg_tokens_a'] < 7500:
            print(f"  Oracle Prefill Accuracy:    {mean_acc_a:.1f}% ± {std_acc_a:.1f}%")
        else:
            print(f"  Oracle Prefill Accuracy:    N/A (Context Overflow)")
        print(f"  Nexus Routing Accuracy:     {mean_acc_b:.1f}% ± {std_acc_b:.1f}%")
        print(f"  SLB Recall (Top-1/3/5):     {mean_rec1:.1f}% / {mean_rec3:.1f}% / {mean_rec5:.1f}%")
        print(f"  Auto-Route / Fallback %:    {mean_auto:.1f}% / {mean_fallback:.1f}%")
        if res['avg_tokens_a'] < 7500:
            print(f"  Avg Latency (Oracle / Nexus):{avg_lat_a:.1f} ms / {avg_lat_b:.1f} ms (SLB search: {avg_lat_slb:.1f} ms)")
        else:
            print(f"  Avg Latency (Oracle / Nexus):N/A / {avg_lat_b:.1f} ms (SLB search: {avg_lat_slb:.1f} ms)")
        print(f"  Avg Context Tokens (Oracle / Nexus): {avg_tokens_a:.1f} / {avg_tokens_b:.1f} ({token_savings_pct:.1f}% savings)")

    # Print final comparison table
    print("\n" + "="*80)
    print("                       ROUTING COMPARATIVE SCALING CURVE")
    print("="*80)
    print(f"{'Size (N)':<10} | {'Oracle Acc':<14} | {'Nexus Acc':<14} | {'SLB R@1/3/5':<18} | {'Latency Save':<14} | {'Token Save':<10}")
    print("-"*80)
    for size in sizes:
        res = overall_summary[size]
        acc_a_str = f"{res['mean_acc_a']:.1f}%" if res['avg_tokens_a'] < 7500 else "N/A (Overflow)"
        acc_b_str = f"{res['mean_acc_b']:.1f}%"
        recall_str = f"{res['mean_rec1']:.0f}/{res['mean_rec3']:.0f}/{res['mean_rec5']:.0f}%"
        lat_save = f"{(1.0 - res['avg_lat_b_ms']/res['avg_lat_a_ms'])*100.0:.1f}%" if (res['avg_lat_a_ms'] > 0 and res['avg_tokens_a'] < 7500) else "N/A"
        tok_save = f"{res['token_savings_pct']:.1f}%"
        print(f"{size:<10} | {acc_a_str:<14} | {acc_b_str:<14} | {recall_str:<18} | {lat_save:<14} | {tok_save:<10}")
    print("="*80)

    # Save to file
    write_artifact(
        args.output,
        "bench_routing_accuracy",
        model_hash=file_sha256(args.model) if Path(args.model).exists() else "",
        config={
            "model_path": args.model,
            "embedding_model_path": args.embed_model,
            "seed": args.seed,
            "runs": runs,
            "tool_sizes": sizes,
        },
        metrics=overall_summary,
        records=[],
    )
    print(f"Wrote JSON artifact: {args.output}")
    
    if os.path.exists("test/schemas/jit_bloat"):
        import shutil
        shutil.rmtree("test/schemas/jit_bloat")

if __name__ == "__main__":
    run_evaluation(parse_args())
