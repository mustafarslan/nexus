#!/bin/bash
set -e

# Project Nexus - MCP Tool KV Cache Batch Compiler Helper

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 --model <gguf_model_path> --schema <schema_json_or_dir> [--base-pos <int>] [--threads <int>]"
    echo "Example: $0 --model models/qwen2.5-7b-instruct.gguf --schema test/schemas"
    exit 1
fi

MODEL=""
SCHEMA=""
BASE_POS="0"
THREADS="-1"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --schema) SCHEMA="$2"; shift ;;
        --base-pos) BASE_POS="$2"; shift ;;
        --threads) THREADS="$2"; shift ;;
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

if [ -z "$SCHEMA" ]; then
    echo "Error: Schema path (--schema) is required."
    exit 1
fi

# Locate compiler binaries
COMPILER="./build/nexus_kv_compiler"
VERIFIER="./build/verify_atb"

if [ ! -f "$COMPILER" ] || [ ! -f "$VERIFIER" ]; then
    echo "Error: Build artifacts not found. Please compile the project first using:"
    echo "  cmake -B build && cmake --build build"
    exit 1
fi

compile_file() {
    local json_file=$1
    local output_atb="${json_file%.json}.atb"
    
    echo "=========================================================="
    echo "Compiling Schema: $json_file"
    echo "Output Block:     $output_atb"
    echo "----------------------------------------------------------"
    
    $COMPILER \
        --model "$MODEL" \
        --schema "$json_file" \
        --output "$output_atb" \
        --base-pos "$BASE_POS" \
        --threads "$THREADS" \
        --verbose
        
    echo "----------------------------------------------------------"
    echo "Verifying Output Block: $output_atb"
    $VERIFIER "$output_atb"
    echo "=========================================================="
    echo ""
}

if [ -f "$SCHEMA" ]; then
    compile_file "$SCHEMA"
elif [ -d "$SCHEMA" ]; then
    for f in "$SCHEMA"/*.json; do
        if [ -f "$f" ]; then
            compile_file "$f"
        fi
    done
else
    echo "Error: Schema path is neither a file nor a directory: $SCHEMA"
    exit 1
fi
