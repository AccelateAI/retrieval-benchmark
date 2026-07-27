#!/usr/bin/env bash
# Creates a Python virtualenv (.venv) and installs requirements.txt.
# Usage: ./setup.sh   (or PYTHON_BIN=python3.12 ./setup.sh to force an interpreter)
set -euo pipefail
cd "$(dirname "$0")"

version_ok() {
  "$1" -c 'import sys; maj, min = sys.version_info[:2]; sys.exit(0 if (maj, min) >= (3, 11) and (maj, min) < (3, 14) else 1)' 2>/dev/null
}

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -n "$PYTHON_BIN" ]; then
  if ! version_ok "$PYTHON_BIN"; then
    echo "PYTHON_BIN=$PYTHON_BIN does not satisfy the required range (>=3.11,<3.14)." >&2
    exit 1
  fi
else
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && version_ok "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "No Python 3.11-3.13 interpreter found on PATH." >&2
  echo "Install one, e.g.: brew install python@3.13" >&2
  exit 1
fi

echo "Using $("$PYTHON_BIN" --version) at $(command -v "$PYTHON_BIN")"

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
  echo "Created .venv"
else
  echo ".venv already exists, reusing it"
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

cat <<'EOF'

Setup complete.

  Activate the venv:      source .venv/bin/activate
  Build the dataset:      python3 dataset.py
  Run the selftest:       python3 retrieval_benchmark.py --selftest
  Run the full benchmark: python3 retrieval_benchmark.py \
        --corpus data/corpus.jsonl \
        --queries data/queries.jsonl \
        --qrels data/qrels.tsv \
        --dense-model sentence-transformers/all-MiniLM-L6-v2 \
        --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --rerank-depth 100
EOF
