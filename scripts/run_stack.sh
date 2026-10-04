#!/usr/bin/env bash
# ==============================================================================
# Metis Sovereign Local AI Stack Launcher
# Launch Sequence: Hestia (:8080) -> Kybernetes (:8081) -> Prosopon (:8000)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

HESTIA_PORT="${HESTIA_PORT:-8080}"
KYBERNETES_PORT="${KYBERNETES_PORT:-8081}"
PROSOPON_PORT="${PROSOPON_PORT:-8000}"

PIDS=()

cleanup() {
    echo -e "\n[Metis] Stopping all running stack processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo "[Metis] All services stopped cleanly."
}
trap cleanup EXIT INT TERM

wait_for_url() {
    local url="$1"
    local name="$2"
    local max_retries="${3:-30}"
    local count=0

    echo -n "[Metis] Waiting for $name at $url ... "
    while [ "$count" -lt "$max_retries" ]; do
        if curl -s -f -o /dev/null "$url" 2>/dev/null; then
            echo "READY"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done
    echo "TIMEOUT (proceeding with fallback/monitoring)"
    return 1
}

cd "${ROOT_DIR}"

echo "======================================================================"
echo " Metis Stack Startup: Sovereign Local AI"
echo " Ports: Hestia (:$HESTIA_PORT) | Kybernetes (:$KYBERNETES_PORT) | Prosopon (:$PROSOPON_PORT)"
echo "======================================================================"

# Step 1: Pre-flight doctor check
echo -e "\n[Step 1/4] Running pre-flight subsystem inspection..."
if [ -x "kybernetes/.venv/bin/python" ]; then
    "kybernetes/.venv/bin/python" -m kybernetes.cli doctor || true
elif command -v uv >/dev/null 2>&1; then
    uv run --package kybernetes kybernetes doctor || true
fi

# Step 2: Launch Hestia Model Server
echo -e "\n[Step 2/4] Starting Hestia Model Server on port ${HESTIA_PORT}..."
if [ -x "hestia/.venv/bin/hestia" ]; then
    hestia/.venv/bin/hestia serve --port "${HESTIA_PORT}" &
    PIDS+=($!)
elif command -v uv >/dev/null 2>&1; then
    uv run --package hestia hestia serve --port "${HESTIA_PORT}" &
    PIDS+=($!)
else
    echo "Hestia binary not found. Skipping live inference server."
fi
wait_for_url "http://127.0.0.1:${HESTIA_PORT}/v1/models" "Hestia" 5 || true

# Step 3: Launch Kybernetes Orchestrator
echo -e "\n[Step 3/4] Starting Kybernetes Orchestrator on port ${KYBERNETES_PORT}..."
if [ -x "kybernetes/.venv/bin/kybernetes" ]; then
    kybernetes/.venv/bin/kybernetes serve --host 127.0.0.1 --port "${KYBERNETES_PORT}" &
    PIDS+=($!)
elif command -v uv >/dev/null 2>&1; then
    uv run --package kybernetes kybernetes serve --host 127.0.0.1 --port "${KYBERNETES_PORT}" &
    PIDS+=($!)
fi
wait_for_url "http://127.0.0.1:${KYBERNETES_PORT}/v1/health" "Kybernetes" 10 || true

# Step 4: Launch Prosopon UI
echo -e "\n[Step 4/4] Starting Prosopon Web Interface on port ${PROSOPON_PORT}..."
echo "[Metis] Open http://127.0.0.1:${PROSOPON_PORT} in your browser."
if [ -x "prosopon/.venv/bin/prosopon" ]; then
    prosopon/.venv/bin/prosopon run --port "${PROSOPON_PORT}"
elif command -v uv >/dev/null 2>&1; then
    uv run --package prosopon prosopon run --port "${PROSOPON_PORT}"
else
    echo "Prosopon not found. Waiting for background processes."
    wait
fi
