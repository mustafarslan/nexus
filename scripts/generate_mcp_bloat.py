#!/usr/bin/env python3
import os
import json
import numpy as np
from multiprocessing import Pool

# List of domains and base descriptions to generate unique schemas
DOMAINS = [
    ("aws_ec2", "Amazon EC2 Cloud Controller", "Manages virtual server instances, security groups, volume attachments, VPC settings, and Elastic IP allocations."),
    ("jira", "Atlassian Jira Issue Tracker", "Creates tickets, updates statuses, transitions workflows, manages sprints, assigns tasks, and queries JQL reports."),
    ("github", "GitHub Repository Administrator", "Handles pull requests, branch protection rules, action runners, webhook configurations, repository secrets, and releases."),
    ("snowflake", "Snowflake Data Warehouse Engine", "Executes warehouse queries, schedules tasks, manages stages, copies data from S3, and configures role privileges."),
    ("kubernetes", "Kubernetes Cluster Operator", "Monitors pod lifecycles, deploys helm charts, auto-scales replica sets, configures configmaps, and manages ingress rules."),
    ("slack", "Slack Messaging Gateway", "Posts channel notifications, handles user groups, archives conversations, manages app shortcuts, and uploads files."),
    ("stripe", "Stripe Subscription & Billing", "Handles customer cards, processes webhooks, manages recurring subscriptions, refunds charges, and generates tax invoices."),
    ("elastic", "Elasticsearch Index Manager", "Creates search indexes, defines mapping analyzers, performs bulk inserts, runs aggregations, and monitors cluster health."),
    ("postgres", "PostgreSQL Database Admin", "Runs schema migrations, inspects table indices, monitors query locks, schedules backups, and scale replication pools."),
    ("redis", "Redis In-Memory Cache Store", "Manages key TTLs, configures eviction policies, scales cluster shards, runs pub/sub listener, and executes lua scripts.")
]

def generate_single_schema(tool_id):
    domain_key, title, base_desc = DOMAINS[tool_id % len(DOMAINS)]
    tool_name = f"nexus_{domain_key}_api_{tool_id}"
    
    properties = {
        "request_id": {
            "type": "string",
            "pattern": "^req_[a-zA-Z0-9]{16}$",
            "description": f"Unique correlation ID for tracing the {title} API call execution across downstream microservices."
        },
        "target_region": {
            "type": "string",
            "enum": ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "sa-east-1", "me-central-1"],
            "description": "Geographical region partition constraint for isolating target resource execution context."
        }
    }
    required = ["request_id", "target_region"]
    
    # Procedurally vary size from 2KB to 50KB by adjusting property counts and nesting
    # Determine target property counts based on tool_id to span the [2KB, 50KB] range evenly
    target_size_kb = 2.0 + (tool_id % 48) # Vary from 2KB to 50KB
    num_fields = int(target_size_kb * 1.8) # ~1.8 fields per KB
    
    for i in range(num_fields):
        p_name = f"nested_config_block_{i}"
        properties[p_name] = {
            "type": "object",
            "properties": {
                "active": {"type": "boolean", "default": True, "description": f"Enables execution pipeline feature {i} within the {title} scope."},
                "retries": {"type": "integer", "minimum": 0, "maximum": 5, "default": 2},
                "priority_class": {"type": "string", "enum": ["low", "normal", "high", "critical"]},
                "sub_details": {
                    "type": "object",
                    "properties": {
                        "spec_version": {"type": "string", "default": f"v{i}.0.1"},
                        "debug_trace": {"type": "boolean", "default": False},
                        "payload_limit_mb": {"type": "number", "default": 10.0 + i}
                    }
                }
            },
            "required": ["active", "priority_class"]
        }
        
    schema = {
        "name": tool_name,
        "description": f"{base_desc} instance ID {tool_id}. This API integrates multiple endpoints to synchronize state across nodes.",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }
    
    schema_str = json.dumps(schema, indent=2)
    return tool_name, schema, schema_str

def main():
    print("Generating 10,000 synthetic MCP tool schemas...")
    output_dir = "test/schemas/bloat"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate schemas concurrently
    with Pool() as pool:
        results = pool.map(generate_single_schema, range(1, 10001))
        
    tools_list = []
    
    for tool_id, (tool_name, schema, schema_str) in enumerate(results, start=1):
        # Verify schema size constraints
        size_bytes = len(schema_str)
        if size_bytes < 2048 or size_bytes > 51200:
            # Force size within limits if it goes out due to JSON encoding variations
            if size_bytes < 2048:
                # Add padding
                schema["description"] += " " + ("P" * (2048 - size_bytes))
            elif size_bytes > 51200:
                # Truncate properties
                keys = list(schema["input_schema"]["properties"].keys())
                while len(json.dumps(schema, indent=2)) > 51200 and len(keys) > 2:
                    k = keys.pop()
                    if k not in ["request_id", "target_region"]:
                        del schema["input_schema"]["properties"][k]
            schema_str = json.dumps(schema, indent=2)
            
        filepath = os.path.join(output_dir, f"{tool_name}.json")
        with open(filepath, "w") as f:
            f.write(schema_str)
            
        tools_list.append({
            "id": tool_id,
            "name": tool_name,
            "desc": schema["description"],
            "filepath": filepath
        })
        
    # Write metadata index file for fast loading in benchmarks
    with open("test/schemas/bloat_metadata.json", "w") as f:
        json.dump(tools_list, f, indent=2)
    print(f"Generated 10,000 JSON schemas in {output_dir}")
    
    # Generate 10,000 synthetic normalized FP32 embeddings of size 256
    print("Generating 10,000 synthetic normalized embeddings of dimension 256...")
    np.random.seed(42)
    embeddings = np.random.randn(10000, 256).astype(np.float32)
    
    # Normalize to unit length (L2 normalization)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    np.savez_compressed("test/schemas/bloat_embeddings.npz", embeddings=embeddings)
    print("Generated test/schemas/bloat_embeddings.npz successfully.")

if __name__ == "__main__":
    main()
