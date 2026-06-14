#!/usr/bin/env python3
import argparse
import json
import time
import httpx
from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Test client for the Nexus API Gateway.")
    parser.add_argument("--gateway", type=str, default="http://localhost:8000", help="Gateway base URL.")
    parser.add_argument("--query", type=str, default="Create a Jira ticket to track database replication locks in us-east-1 with normal priority.", help="User query matching a registered tool.")
    parser.add_argument("--stream", action="store_true", help="Enable streaming response.")
    args = parser.parse_args()

    client = httpx.Client(timeout=30.0)

    # 1. Fetch models
    console.print(f"[bold yellow]1. Fetching models from {args.gateway}/v1/models...[/bold yellow]")
    try:
        res = client.get(f"{args.gateway}/v1/models")
        if res.status_code == 200:
            console.print(Panel(json.dumps(res.json(), indent=2), title="Models List", border_style="green"))
        else:
            console.print(f"[bold red]Failed to fetch models: {res.status_code} - {res.text}[/bold red]")
            return
    except Exception as e:
        console.print(f"[bold red]Connection failed: {e}[/bold red]")
        return

    # 2. Call chat completions
    payload = {
        "model": "qwen2.5-0.5b-instruct",
        "messages": [
            {"role": "user", "content": args.query}
        ],
        "stream": args.stream,
        "temperature": 0.0
    }

    console.print(f"[bold yellow]2. Submitting completions query: '{args.query}' (stream={args.stream})...[/bold yellow]")
    t0 = time.perf_counter()
    try:
        if args.stream:
            with client.stream("POST", f"{args.gateway}/v1/chat/completions", json=payload) as r:
                if r.status_code != 200:
                    console.print(f"[bold red]Error: {r.status_code}[/bold red]")
                    return
                console.print("\n[bold green]Streaming Assistant response:[/bold green]")
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"].get("content", "")
                            print(delta, end="", flush=True)
                        except Exception:
                            pass
                print("\n")
        else:
            res = client.post(f"{args.gateway}/v1/chat/completions", json=payload)
            t_diff = time.perf_counter() - t0
            if res.status_code == 200:
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                console.print(Panel(content, title=f"Assistant Response (completed in {t_diff:.3f}s)", border_style="cyan"))
            else:
                console.print(f"[bold red]Completions query failed: {res.status_code} - {res.text}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Query failed: {e}[/bold red]")

if __name__ == "__main__":
    main()
