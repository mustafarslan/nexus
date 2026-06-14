#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
from typing import Any, List
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.sse import SseServerTransport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("live_mcp_server")

# 10 Domains to generate realistic tool sprawl (replicating scripts/generate_mcp_bloat.py)
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

def generate_tool_definition(tool_id: int) -> types.Tool:
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
    
    # Generate nesting and varying schema sizes (matches generate_mcp_bloat.py logic)
    target_size_kb = 2.0 + (tool_id % 48)
    num_fields = int(target_size_kb * 1.8)
    
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
        
    description = f"{base_desc} instance ID {tool_id}. This API integrates multiple endpoints to synchronize state across nodes."
    
    return types.Tool(
        name=tool_name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": properties,
            "required": required
        }
    )

def main():
    parser = argparse.ArgumentParser(description="Live Enterprise-Scale MCP Server using SSE transport.")
    parser.add_argument("--num-tools", type=int, default=1000, help="Total number of tools to generate (default: 1000).")
    parser.add_argument("--port", type=int, default=8765, help="Port to run the SSE server on (default: 8765).")
    args = parser.parse_args()

    # Pre-generate tools for fast lookup
    logger.info(f"Generating {args.num_tools} procedural MCP tools...")
    mcp_tools = [generate_tool_definition(i) for i in range(1, args.num_tools + 1)]
    logger.info(f"Procedural tools generation complete.")

    app = FastAPI(title="Nexus Enterprise Megaserver")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sse_transport = SseServerTransport(endpoint="/messages")
    mcp_server = Server("nexus-enterprise-megaserver")

    @mcp_server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        logger.info(f"Received tools/list request. Returning {len(mcp_tools)} tools.")
        return mcp_tools

    @mcp_server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> List[types.TextContent]:
        logger.info(f"Received tools/call request for tool '{name}' with arguments: {arguments}")
        # Validate that the tool actually exists
        tool_exists = any(t.name == name for t in mcp_tools)
        if not tool_exists:
            raise ValueError(f"Tool {name} not found on this server.")
            
        result_text = f"Success: Executed {name} with arguments: {json.dumps(arguments)}"
        return [types.TextContent(type="text", text=result_text)]

    class SseASGIApp:
        def __init__(self, fastapi_app, sse_transport, mcp_server):
            self.fastapi_app = fastapi_app
            self.sse_transport = sse_transport
            self.mcp_server = mcp_server

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                path = scope.get("path", "")
                if path == "/sse":
                    from mcp.server.lowlevel import NotificationOptions
                    logger.info("New client connecting to SSE endpoint")
                    async with self.sse_transport.connect_sse(scope, receive, send) as (read_stream, write_stream):
                        logger.info("SSE transport connected. Starting MCP server loop...")
                        await self.mcp_server.run(
                            read_stream,
                            write_stream,
                            InitializationOptions(
                                server_name="nexus-enterprise-megaserver",
                                server_version="1.0.0",
                                capabilities=self.mcp_server.get_capabilities(
                                    notification_options=NotificationOptions(),
                                    experimental_capabilities={}
                                )
                            )
                        )
                    return
                elif path == "/messages":
                    await self.sse_transport.handle_post_message(scope, receive, send)
                    return

            await self.fastapi_app(scope, receive, send)

    logger.info(f"Starting FastAPI/Uvicorn server on port {args.port}...")
    uvicorn.run(SseASGIApp(app, sse_transport, mcp_server), host="0.0.0.0", port=args.port, log_level="info")

if __name__ == "__main__":
    main()
