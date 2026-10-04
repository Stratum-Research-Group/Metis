#!/usr/bin/env bash
# ==============================================================================
# Metis Sovereign Local AI Stack - Workspace Setup
# Bootstraps the uv workspace with Python 3.12, verifies member packages,
# and prepares the development environment.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "======================================================================"
echo " Metis Sovereign Local AI Stack: Workspace Setup"
echo "======================================================================"

# 1. Check for uv
if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] 'uv' is not installed or not in PATH."
    echo "Install uv via: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "[1/4] Found uv $(uv --version)"

# 2. Check or install Python 3.12 via uv
echo "[2/4] Ensuring Python 3.12 is available..."
uv python install 3.12

# 3. Synchronize workspace dependencies
SYNC_ARGS=("--python" "3.12")
if [[ "${1:-}" == "--all" || "${1:-}" == "--all-packages" ]]; then
    echo "[3/4] Synchronizing all workspace members (including heavy ML/training stacks)..."
    SYNC_ARGS+=("--all-packages")
else
    echo "[3/4] Synchronizing core control plane (Kybernetes, Hestia, Prosopon, Phylax, Krisis) & dev tools..."
fi
uv sync "${SYNC_ARGS[@]}"

# 4. Verify installed entrypoints & health
echo "[4/4] Verifying stack health with kybernetes doctor..."
uv run kybernetes doctor || true

echo -e "\n======================================================================"
echo " Metis workspace bootstrap complete!"
echo " Control plane entrypoints ready: kybernetes, hestia, prosopon, phylax, krisis"
echo " To run smoke tests:    uv run pytest tests/test_stack_smoke.py"
echo " To launch the stack:   ./scripts/run_stack.sh"
echo "======================================================================"
