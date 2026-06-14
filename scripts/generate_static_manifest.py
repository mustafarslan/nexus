#!/usr/bin/env python3
import os
import json
import numpy as np
import sys

# Add scripts directory to Python path to import DOMAINS
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from generate_mcp_bloat import DOMAINS

def main():
    print("Generating static SLB manifest...")
    emb_path = "test/schemas/bloat_embeddings.npz"
    if not os.path.exists(emb_path):
        print(f"Error: Embeddings not found at {emb_path}. Please run generate_mcp_bloat.py first.")
        sys.exit(1)
        
    embeddings = np.load(emb_path)["embeddings"]
    
    tools = []
    for tool_id in range(1, 10001):
        domain_key, title, base_desc = DOMAINS[tool_id % len(DOMAINS)]
        tool_name = f"nexus_{domain_key}_api_{tool_id}"
        json_path = f"test/schemas/bloat/{tool_name}.json"
        
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, "r") as f:
            schema_data = json.load(f)
            
        atb_path = f"test/schemas/bloat/{tool_name}.atb"
        scent = [int((tool_id + i) % 150000) for i in range(5)]
        
        tools.append({
            "id": tool_id,
            "name": tool_name,
            "description": schema_data["description"],
            "atb_path": os.path.abspath(atb_path),
            "embedding": embeddings[tool_id - 1].tolist(),
            "scent_tokens": scent,
            "schema": schema_data["input_schema"]
        })
        
    manifest_data = {"tools": tools}
    with open("test/schemas/slb_manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Static SLB manifest written successfully with {len(tools)} tools.")

if __name__ == "__main__":
    main()
