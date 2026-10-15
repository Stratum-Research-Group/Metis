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

MOCK_MODE=false
for arg in "$@"; do
    case "$arg" in
        --mock)
            MOCK_MODE=true
            ;;
        --help|-h)
            echo "Usage: ./scripts/run_stack.sh [--mock]"
            echo "Starts Hestia (:$HESTIA_PORT), Kybernetes (:$KYBERNETES_PORT), and Prosopon (:$PROSOPON_PORT)."
            echo "Use --mock to run Prosopon in standalone mock-orchestrator mode."
            exit 0
            ;;
    esac
done

echo "======================================================================"
echo " Metis Stack Startup: Sovereign Local AI"
echo " Ports: Hestia (:$HESTIA_PORT) | Kybernetes (:$KYBERNETES_PORT) | Prosopon (:$PROSOPON_PORT)"
echo " Mode: $( [ "$MOCK_MODE" = "true" ] && echo "MOCK ORCHESTRATOR" || echo "LIVE CONNECTED STACK" )"
echo "======================================================================"

# Step 1: Pre-flight doctor check
echo -e "\n[Step 1/4] Running pre-flight subsystem inspection..."
if [ -x "kybernetes/.venv/bin/python" ]; then
    "kybernetes/.venv/bin/python" -m kybernetes.cli doctor || true
elif command -v uv >/dev/null 2>&1; then
    uv run kybernetes doctor || true
fi

# Step 2: Launch Hestia Model Server
echo -e "\n[Step 2/4] Starting Hestia Model Server on port ${HESTIA_PORT}..."
export HESTIA_SERVICE_TOKEN="${HESTIA_SERVICE_TOKEN:-$(openssl rand -hex 16 2>/dev/null || echo 'metis-local-hestia-token')}"
HESTIA_CMD=("serve" "--port" "${HESTIA_PORT}" "--config" "hestia/hestia.yaml" "--models" "hestia/models.yaml")
if [ -x "hestia/.venv/bin/hestia" ]; then
    hestia/.venv/bin/hestia "${HESTIA_CMD[@]}" &
    PIDS+=($!)
elif [ -x ".venv/bin/hestia" ]; then
    .venv/bin/hestia "${HESTIA_CMD[@]}" &
    PIDS+=($!)
elif command -v uv >/dev/null 2>&1; then
    uv run hestia "${HESTIA_CMD[@]}" &
    PIDS+=($!)
else
    echo "Hestia binary not found. Skipping live inference server."
fi
wait_for_url "http://127.0.0.1:${HESTIA_PORT}/v1/models" "Hestia" 5 || true

# Step 3: Launch Kybernetes Orchestrator
echo -e "\n[Step 3/4] Starting Kybernetes Orchestrator on port ${KYBERNETES_PORT}..."
export HESTIA_BASE_URL="http://127.0.0.1:${HESTIA_PORT}/v1"
export HESTIA_DEFAULT_MODEL="${HESTIA_DEFAULT_MODEL:-gemma-4-26b-a4b-abliterated}"
export KYBERNETES_ADMIN_TOKEN="${KYBERNETES_ADMIN_TOKEN:-$(openssl rand -hex 16 2>/dev/null || echo 'metis-local-admin-token')}"
export KYBERNETES_SERVICE_TOKEN="${KYBERNETES_SERVICE_TOKEN:-$(openssl rand -hex 16 2>/dev/null || echo 'metis-local-service-token')}"
KYB_CMD=("serve" "--host" "127.0.0.1" "--port" "${KYBERNETES_PORT}")
if [ -d ".stoma" ]; then
    echo "[Metis] Wiring live Stoma memory root (.stoma) into Kybernetes."
    KYB_CMD+=("--memory-root" ".stoma")
fi
if [ -x "kybernetes/.venv/bin/kybernetes" ]; then
    kybernetes/.venv/bin/kybernetes "${KYB_CMD[@]}" &
    PIDS+=($!)
elif [ -x ".venv/bin/kybernetes" ]; then
    .venv/bin/kybernetes "${KYB_CMD[@]}" &
    PIDS+=($!)
elif command -v uv >/dev/null 2>&1; then
    uv run kybernetes "${KYB_CMD[@]}" &
    PIDS+=($!)
fi
if ! wait_for_url "http://127.0.0.1:${KYBERNETES_PORT}/v1/health" "Kybernetes" 10; then
    echo "[ERROR] Kybernetes orchestrator failed to start on :${KYBERNETES_PORT}."
    exit 1
fi

# Step 4: Launch Prosopon UI
echo -e "\n[Step 4/4] Starting Prosopon Web Interface on port ${PROSOPON_PORT}..."
PROS_OPTS=("serve" "--port" "${PROSOPON_PORT}" "--kybernetes-url" "http://127.0.0.1:${KYBERNETES_PORT}/v1")
if [ "$MOCK_MODE" = "true" ]; then
    echo "[Metis] Running Prosopon in MOCK mode (--mock)."
    PROS_OPTS+=("--mock")
else
    echo "[Metis] Connecting Prosopon to live Kybernetes upstream at http://127.0.0.1:${KYBERNETES_PORT}/v1 (--no-mock)."
    PROS_OPTS+=("--no-mock")
fi

echo "[Metis] Open http://127.0.0.1:${PROSOPON_PORT} in your browser."
if [ -x "prosopon/.venv/bin/prosopon" ]; then
    prosopon/.venv/bin/prosopon "${PROS_OPTS[@]}"
elif [ -x ".venv/bin/prosopon" ]; then
    .venv/bin/prosopon "${PROS_OPTS[@]}"
elif command -v uv >/dev/null 2>&1; then
    uv run prosopon "${PROS_OPTS[@]}"
else
    echo "Prosopon not found. Waiting for background processes."
    wait
fi
