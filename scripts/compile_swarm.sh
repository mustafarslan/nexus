#!/bin/bash
set -e

# Swarm Compiler for Phase 32 MCP Stress Test

MODEL=""
SCHEMA_DIR="test/schemas/bloat"
COUNT=10000

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --schema) SCHEMA_DIR="$2"; shift ;;
        --count) COUNT="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$MODEL" ]; then
    echo "Error: Model path (--model) is required."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file does not exist at $MODEL"
    exit 1
fi

COMPILER="./build/nexus_kv_compiler"
VERIFIER="./build/verify_atb"

if [ ! -f "$COMPILER" ] || [ ! -f "$VERIFIER" ]; then
    echo "Error: C++ build artifacts not found. Please run: cmake -B build && cmake --build build"
    exit 1
fi

# Ensure schema directory exists
if [ ! -d "$SCHEMA_DIR" ]; then
    echo "Error: Schema directory does not exist: $SCHEMA_DIR"
    exit 1
fi

# Total files in directory
ALL_JSONS=($(find "$SCHEMA_DIR" -name "nexus_*.json" | sort))
TOTAL_SCHEMAS=${#ALL_JSONS[@]}

if [ "$TOTAL_SCHEMAS" -eq 0 ]; then
    echo "Error: No generated JSON schemas found in $SCHEMA_DIR. Please run scripts/generate_mcp_bloat.py first."
    exit 1
fi

# Limit to COUNT if specified
if [ "$COUNT" -lt "$TOTAL_SCHEMAS" ]; then
    TOTAL_SCHEMAS=$COUNT
fi

echo "Targeting compilation of $TOTAL_SCHEMAS schemas..."

echo "Concurrently compiling $TOTAL_SCHEMAS schemas using $(sysctl -n hw.ncpu) threads..."

# We write list of JSON files up to count in tool_id order
LIST_FILE="build/schemas_to_compile.txt"
rm -f "$LIST_FILE"
python3 -c '
import os, sys
domains = ["aws_ec2", "jira", "github", "snowflake", "kubernetes", "slack", "stripe", "elastic", "postgres", "redis"]
count = int(sys.argv[1])
schema_dir = sys.argv[2]
for tool_id in range(1, 10001):
    domain = domains[tool_id % len(domains)]
    tool_name = f"nexus_{domain}_api_{tool_id}"
    json_path = os.path.join(schema_dir, f"{tool_name}.json")
    if os.path.exists(json_path):
        print(json_path)
' "$TOTAL_SCHEMAS" "$SCHEMA_DIR" | head -n "$TOTAL_SCHEMAS" > "$LIST_FILE"


cat "$LIST_FILE" | xargs -P $(sysctl -n hw.ncpu) -I {} bash -c '
    json_file="$1"
    atb_file="${json_file%.json}.atb"
    if [ ! -f "$atb_file" ]; then
        ./build/nexus_kv_compiler --model "'"$MODEL"'" --schema "$json_file" --output "$atb_file" > /dev/null 2>&1
    fi
' _ {}

rm -f "$LIST_FILE"

# Verify one sample ATB
sample_atb="${ALL_JSONS[0]%.json}.atb"
$VERIFIER "$sample_atb"

echo "Compilation swarm completed successfully!"
