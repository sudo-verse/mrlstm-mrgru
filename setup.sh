#!/usr/bin/env bash
# One-shot setup: clone the MRINN dependency, build a Python 3.11 venv,
# install pinned requirements, then run the smoke test.
#
#   bash setup.sh
#
# TensorFlow 2.16.2 supports Python 3.10-3.11 only, so the venv is pinned to
# 3.11 regardless of the system interpreter. Uses `uv` if available (it can
# fetch 3.11 itself); otherwise falls back to whatever `python3.11` is on PATH.

set -euo pipefail
cd "$(dirname "$0")"

MRINN_DIR="external/MRINN"

if [ ! -d "$MRINN_DIR/library_imbalance" ]; then
    echo "==> cloning MRINN into $MRINN_DIR"
    mkdir -p external
    git clone --depth 1 https://github.com/runyao-yu/MRINN "$MRINN_DIR"
else
    echo "==> MRINN already present at $MRINN_DIR"
fi

if [ ! -d .venv ]; then
    if command -v uv >/dev/null 2>&1; then
        echo "==> creating .venv (uv, Python 3.11)"
        uv python install 3.11
        uv venv --python 3.11 .venv
        uv pip install --python .venv/bin/python -r requirements.txt
    elif command -v python3.11 >/dev/null 2>&1; then
        echo "==> creating .venv (python3.11)"
        python3.11 -m venv .venv
        .venv/bin/pip install --upgrade pip
        .venv/bin/pip install -r requirements.txt
    else
        echo "ERROR: need either 'uv' or 'python3.11' on PATH." >&2
        echo "TensorFlow 2.16.2 does not support Python 3.12+." >&2
        exit 1
    fi
else
    echo "==> .venv already exists"
fi

echo "==> smoke test"
.venv/bin/python verify.py
