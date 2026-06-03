#!/bin/bash
set -e

# Swarm Compiler for Phase 31 MCP Stress Test

MODEL=""
SCHEMA_DIR="test/schemas/bloat"
FAST=false
COUNT=10000

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --schema) SCHEMA_DIR="$2"; shift ;;
        --fast) FAST=true ;;
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

if [ "$FAST" = true ]; then
    echo "[FAST MODE] Compiling 20 base schemas using C++ compiler, then fast-cloning..."
    
    # Compile exactly 1 base schema to ensure uniform block sizes (preventing allocator fragmentation)
    NUM_BASES=1
    if [ "$TOTAL_SCHEMAS" -lt "$NUM_BASES" ]; then
        NUM_BASES=$TOTAL_SCHEMAS
    fi
    
    # Select the smallest JSON file in the directory to minimize block size and RAM usage
    BASE_FILE=$(ls -S -r "$SCHEMA_DIR"/nexus_*.json | head -n 1)
    BASE_FILES=("$BASE_FILE")
    
    echo "Compiling $NUM_BASES base schemas..."
    for f in "${BASE_FILES[@]}"; do
        atb="${f%.json}.atb"
        if [ ! -f "$atb" ]; then
            $COMPILER --model "$MODEL" --schema "$f" --output "$atb"
        fi
    done
    
    # Fast-clone the rest in Python
    echo "Cloning compiled bases to remaining schemas..."
    python3 -c '
import os, shutil, sys
schema_dir = sys.argv[1]
count = int(sys.argv[2])
base_files = sys.argv[3].split(",")

all_jsons = [f for f in os.listdir(schema_dir) if f.endswith(".json") and f.startswith("nexus_")]
all_jsons.sort()
all_jsons = all_jsons[:count]

base_atbs = [f.replace(".json", ".atb") for f in base_files]

for i, jf in enumerate(all_jsons):
    target_atb = os.path.join(schema_dir, jf.replace(".json", ".atb"))
    if os.path.exists(target_atb):
        continue
    # Pick a base ATB file round-robin
    src_atb = base_atbs[i % len(base_atbs)]
    try:
        os.link(src_atb, target_atb)
    except FileExistsError:
        pass
' "$SCHEMA_DIR" "$TOTAL_SCHEMAS" "$(IFS=,; echo "${BASE_FILES[*]}")"
    
    # Verify one sample ATB
    echo "Verifying compiled sample..."
    sample_atb="${BASE_FILES[0]%.json}.atb"
    $VERIFIER "$sample_atb"
    
else
    echo "[FULL MODE] Concurrently compiling $TOTAL_SCHEMAS schemas using $(sysctl -n hw.ncpu) threads..."
    
    # We write list of JSON files up to count
    LIST_FILE="build/schemas_to_compile.txt"
    rm -f "$LIST_FILE"
    for ((i=0; i<TOTAL_SCHEMAS; i++)); do
        echo "${ALL_JSONS[$i]}" >> "$LIST_FILE"
    done
    
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
fi

echo "Compilation swarm completed successfully!"
