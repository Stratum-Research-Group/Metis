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

if [ -f "${ROOT_DIR}/deploy/metis.env" ]; then
    echo "[Metis] Loading environment configuration from deploy/metis.env..."
    set -a
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/deploy/metis.env"
    set +a
fi

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

generate_token() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 16
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "import secrets; print(secrets.token_hex(16))"
    else
        echo "metis-token-$(date +%s)-$RANDOM"
    fi
}

# Step 2: Launch Hestia Model Server
HESTIA_HOST="${HESTIA_HOST:-127.0.0.1}"
HESTIA_CONFIG="${HESTIA_CONFIG:-hestia/hestia.yaml}"
HESTIA_MODELS="${HESTIA_MODELS:-hestia/models.yaml}"
echo -e "\n[Step 2/4] Starting Hestia Model Server on ${HESTIA_HOST}:${HESTIA_PORT}..."
export HESTIA_SERVICE_TOKEN="${HESTIA_SERVICE_TOKEN:-$(generate_token)}"
HESTIA_CMD=("serve" "--host" "${HESTIA_HOST}" "--port" "${HESTIA_PORT}" "--config" "${HESTIA_CONFIG}" "--models" "${HESTIA_MODELS}")
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
wait_for_url "http://${HESTIA_HOST}:${HESTIA_PORT}/v1/models" "Hestia" 5 || true

# Step 3: Launch Kybernetes Orchestrator
KYBERNETES_HOST="${KYBERNETES_HOST:-127.0.0.1}"
KYBERNETES_DB_PATH="${KYBERNETES_DB_PATH:-kybernetes_budget.db}"
KYBERNETES_MEMORY_ROOT="${KYBERNETES_MEMORY_ROOT:-.stoma}"
echo -e "\n[Step 3/4] Starting Kybernetes Orchestrator on ${KYBERNETES_HOST}:${KYBERNETES_PORT}..."
export HESTIA_BASE_URL="http://${HESTIA_HOST}:${HESTIA_PORT}/v1"
export HESTIA_DEFAULT_MODEL="${HESTIA_DEFAULT_MODEL:-gemma-4-26b-a4b-abliterated}"
export KYBERNETES_ADMIN_TOKEN="${KYBERNETES_ADMIN_TOKEN:-$(generate_token)}"
export KYBERNETES_SERVICE_TOKEN="${KYBERNETES_SERVICE_TOKEN:-$(generate_token)}"
KYB_CMD=("serve" "--host" "${KYBERNETES_HOST}" "--port" "${KYBERNETES_PORT}" "--db-path" "${KYBERNETES_DB_PATH}")
if [ -n "${KYBERNETES_CONFIG:-}" ] && [ -f "$KYBERNETES_CONFIG" ]; then
    KYB_CMD+=("--config" "$KYBERNETES_CONFIG")
fi
# Tokens are passed securely via environment variables to avoid exposing credentials in /proc/PID/cmdline (Finding A13)
if [ -d "${KYBERNETES_MEMORY_ROOT}" ]; then
    echo "[Metis] Wiring live Stoma memory root (${KYBERNETES_MEMORY_ROOT}) into Kybernetes."
    KYB_CMD+=("--memory-root" "${KYBERNETES_MEMORY_ROOT}")
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
if ! wait_for_url "http://${KYBERNETES_HOST}:${KYBERNETES_PORT}/v1/health" "Kybernetes" 10; then
    echo "[ERROR] Kybernetes orchestrator failed to start on :${KYBERNETES_PORT}."
    exit 1
fi

# Step 4: Launch Prosopon UI
PROSOPON_HOST="${PROSOPON_HOST:-127.0.0.1}"
echo -e "\n[Step 4/4] Starting Prosopon Web Interface on port ${PROSOPON_PORT}..."
PROS_OPTS=("serve" "--host" "${PROSOPON_HOST}" "--port" "${PROSOPON_PORT}" "--kybernetes-url" "http://${KYBERNETES_HOST}:${KYBERNETES_PORT}/v1")
if [ "${PROSOPON_LAN:-false}" = "true" ]; then
    echo "[Metis] Enabling LAN 0.0.0.0 binding with mobile device pairing."
    PROS_OPTS+=("--lan")
fi
if [ "${PROSOPON_TLS:-false}" = "true" ]; then
    echo "[Metis] Enabling self-signed TLS for microphone access."
    PROS_OPTS+=("--tls")
fi
if [ -n "${PROSOPON_DATA_DIR:-}" ]; then
    PROS_OPTS+=("--data-dir" "${PROSOPON_DATA_DIR}")
fi
if [ "$MOCK_MODE" = "true" ]; then
    echo "[Metis] Running Prosopon in MOCK mode (--mock)."
    PROS_OPTS+=("--mock")
else
    echo "[Metis] Connecting Prosopon to live Kybernetes upstream at http://${KYBERNETES_HOST}:${KYBERNETES_PORT}/v1 (--no-mock)."
    PROS_OPTS+=("--no-mock")
fi

proto="http"
[ "${PROSOPON_TLS:-false}" = "true" ] && proto="https"
echo "[Metis] Open ${proto}://${PROSOPON_HOST}:${PROSOPON_PORT} in your browser."
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

