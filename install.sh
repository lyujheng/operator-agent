#!/usr/bin/env bash
set -euo pipefail

# Usage: ./install.sh [--venv /path/to/venv]
# Builds both packages and installs them into the target Python environment.
# After installation, `operator-allocation` command is available globally.

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${1:-}"

# If --venv specified, activate it; otherwise use current Python.
if [[ "$VENV_DIR" == "--venv" ]]; then
    VENV_DIR="$2"
    echo "[1/4] Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "[2/4] Upgrading pip ..."
    pip install --upgrade pip
else
    echo "[1/4] Using current Python: $(which python3)"
    echo "[2/4] Skipping venv creation."
fi

echo "[3/4] Building packages ..."
cd "$PROJECT_ROOT"
pip install build
python -m build packages/agent_framework -o dist/
python -m build packages/operator_allocation -o dist/

echo "[4/4] Installing ..."
pip install dist/agent_framework-*.whl dist/operator_allocation-*.whl

echo ""
echo "Done. Run with:"
echo "  operator-allocation --config /path/to/config.toml --input /path/to/requirement.xlsx"
