#!/usr/bin/env bash
# Project venv for cross-encoder router (sentence-transformers) and scale benchmarks.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Activate with: source .venv/bin/activate"
