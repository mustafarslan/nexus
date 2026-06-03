#!/usr/bin/env python3
import asyncio
import json
import os
import sys
import subprocess
import shutil
import argparse
from rich.console import Console

console = Console()

# Predefined high-fidelity mock schemas for fallback when real servers fail or credentials/Node are missing
MOCK_SQLITE_SCHEMA = {
    "tools": [
        {
            "name": "query_schema",
            "description": "Get the schema for the database, including table names and column definitions.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "read_query",
            "description": "Execute a SELECT query to read data from the SQLite database. Returns rows as JSON objects.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL SELECT query to run."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "write_query",
            "description": "Execute an INSERT, UPDATE, or DELETE query to write data to the SQLite database. Returns modified rows count.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL write query to run."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "describe_table",
            "description": "Get schema information about a specific table, including columns, data types, and primary/foreign keys.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to describe."
                    }
                },
                "required": ["table_name"]
            }
        },
        {
            "name": "list_tables",
            "description": "List all tables in the SQLite database.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]
}

MOCK_GITHUB_SCHEMA = {
    "tools": [
        {
            "name": "search_repositories",
            "description": "Search for repositories on GitHub matching a specific query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g. 'nexus lang:python')."
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["stars", "forks", "help-wanted-issues", "updated"],
                        "description": "The sort field."
                    },
                    "order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort order (ascending or descending)."
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Number of results per page (max 100)."
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_file_contents",
            "description": "Retrieve the contents of a file from a GitHub repository.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "The owner of the repository."
                    },
                    "repo": {
                        "type": "string",
                        "description": "The repository name."
                    },
                    "path": {
                        "type": "string",
                        "description": "The file path within the repository."
                    },
                    "ref": {
                        "type": "string",
                        "description": "The branch, tag, or commit SHA."
                    }
                },
                "required": ["owner", "repo", "path"]
            }
        },
        {
            "name": "create_or_update_file",
            "description": "Create or update a file in a GitHub repository.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "The owner of the repository."
                    },
                    "repo": {
                        "type": "string",
                        "description": "The repository name."
                    },
                    "path": {
                        "type": "string",
                        "description": "The file path."
                    },
                    "content": {
                        "type": "string",
                        "description": "The file content as string."
                    },
                    "message": {
                        "type": "string",
                        "description": "The commit message."
                    },
                    "branch": {
                        "type": "string",
                        "description": "The branch name."
                    },
                    "sha": {
                        "type": "string",
                        "description": "The blob SHA of the file if updating."
                    }
                },
                "required": ["owner", "repo", "path", "content", "message"]
            }
        },
        {
            "name": "list_commits",
            "description": "List commits for a repository.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "The owner of the repository."
                    },
                    "repo": {
                        "type": "string",
                        "description": "The repository name."
                    },
                    "path": {
                        "type": "string",
                        "description": "Filter by commits containing this file path."
                    },
                    "author": {
                        "type": "string",
                        "description": "GitHub username or email."
                    },
                    "sha": {
                        "type": "string",
                        "description": "SHA or branch to start listing commits from."
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Number of results per page (max 100)."
                    }
                },
                "required": ["owner", "repo"]
            }
        },
        {
            "name": "list_issues",
            "description": "List issues for a repository.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "The owner of the repository."
                    },
                    "repo": {
                        "type": "string",
                        "description": "The repository name."
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "State of issues: open, closed, all."
                    },
                    "labels": {
                        "type": "string",
                        "description": "Comma-separated list of label names."
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Number of results per page (max 100)."
                    }
                },
                "required": ["owner", "repo"]
            }
        },
        {
            "name": "list_pull_requests",
            "description": "List pull requests for a repository.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "The owner of the repository."
                    },
                    "repo": {
                        "type": "string",
                        "description": "The repository name."
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "State of pull requests."
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Results per page."
                    }
                },
                "required": ["owner", "repo"]
            }
        }
    ]
}

def to_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "dict"):
        return obj.dict()
    elif isinstance(obj, list):
        return [to_dict(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {k: to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    else:
        return obj

async def fetch_from_server(name, command, args, env=None):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    
    console.print(f"[bold blue]Connecting to {name} MCP server via stdio...[/bold blue] ({command} {' '.join(args)})")
    server_params = StdioServerParameters(command=command, args=args, env=env)
    
    try:
        async with asyncio.timeout(10.0):
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    console.print(f"[green]Successfully initialized session with {name}[/green]")
                    tools_result = await session.list_tools()
                    tools_dict = to_dict(tools_result)
                    
                    if isinstance(tools_dict, dict) and "tools" in tools_dict:
                        return tools_dict
                    elif isinstance(tools_dict, list):
                        return {"tools": tools_dict}
                    else:
                        return {"tools": to_dict(tools_result.tools)}
    except Exception as e:
        console.print(f"[bold red]Failed to fetch real-world schema from {name}: {e}[/bold red]")
        return None

def compile_schema(schema_path, model_path, base_pos, threads):
    compiler_path = "./build/nexus_kv_compiler"
    if not os.path.exists(compiler_path):
        console.print(f"[yellow]Warning: {compiler_path} not found. Skipping compilation.[/yellow]")
        return False
    
    output_path = schema_path.replace(".json", ".atb")
    cmd = [
        compiler_path,
        "--model", model_path,
        "--schema", schema_path,
        "--output", output_path,
        "--base-pos", str(base_pos),
        "--threads", str(threads),
        "--verbose"
    ]
    console.print(f"[bold cyan]Compiling schema to ATB: {schema_path} -> {output_path}[/bold cyan]")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        console.print(f"[green]Successfully compiled ATB block.[/green]")
        if res.stdout:
            console.print(res.stdout)
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]KV Cache Compiler failed with code {e.returncode}:[/bold red]")
        console.print(e.stderr)
        return False

async def main():
    parser = argparse.ArgumentParser(description="Fetch real Model Context Protocol tool schemas.")
    parser.add_argument("--model", type=str, help="Optional GGUF model path to compile ATB files.")
    parser.add_argument("--base-pos", type=int, default=0, help="Starting RoPE position (default: 0).")
    parser.add_argument("--threads", type=int, default=-1, help="Threads for the compiler (default: -1).")
    parser.add_argument("--output-dir", type=str, default="test/schemas", help="Output directory (default: test/schemas).")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. SQLite Schema Fetch
    sqlite_db_path = os.path.join(args.output_dir, "test.db")
    sqlite_args = ["mcp-server-sqlite", "--db", sqlite_db_path]
    # Check if uvx/uv is available
    uv_path = shutil.which("uvx") or shutil.which("uv")
    if uv_path:
        sqlite_cmd = "uvx"
    else:
        sqlite_cmd = "npx"
        sqlite_args = ["-y", "mcp-server-sqlite", "--db", sqlite_db_path]
        
    sqlite_schema = await fetch_from_server("sqlite", sqlite_cmd, sqlite_args)
    if sqlite_schema is None:
        console.print(f"[yellow]Using high-fidelity mock SQLite schema fallback.[/yellow]")
        sqlite_schema = MOCK_SQLITE_SCHEMA
        
    sqlite_json_path = os.path.join(args.output_dir, "sqlite_tools.json")
    with open(sqlite_json_path, "w") as f:
        json.dump(sqlite_schema, f, indent=2)
    console.print(f"[green]Dumped SQLite schema to {sqlite_json_path}[/green]")
    
    # 2. GitHub Schema Fetch
    github_env = dict(os.environ)
    github_args = ["-y", "@modelcontextprotocol/server-github"]
    github_schema = await fetch_from_server("github", "npx", github_args, env=github_env)
    if github_schema is None:
        console.print(f"[yellow]Using high-fidelity mock GitHub schema fallback.[/yellow]")
        github_schema = MOCK_GITHUB_SCHEMA
        
    github_json_path = os.path.join(args.output_dir, "github_tools.json")
    with open(github_json_path, "w") as f:
        json.dump(github_schema, f, indent=2)
    console.print(f"[green]Dumped GitHub schema to {github_json_path}[/green]")
    
    # 3. Compilation Bonus
    if args.model:
        if not os.path.exists(args.model):
            console.print(f"[bold red]Model path does not exist: {args.model}[/bold red]")
            sys.exit(1)
        compile_schema(sqlite_json_path, args.model, args.base_pos, args.threads)
        compile_schema(github_json_path, args.model, args.base_pos, args.threads)

if __name__ == "__main__":
    asyncio.run(main())
