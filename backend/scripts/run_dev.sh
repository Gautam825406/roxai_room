#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
./.venv/bin/pip install -q -e ".[dev]"

if [ ! -f .env ]; then
    echo "No .env found -- copy .env.example to .env and fill in LiveKit credentials first." >&2
    exit 1
fi

./.venv/bin/python -m roxroom.main
