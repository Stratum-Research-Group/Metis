#!/usr/bin/env bash
# ==============================================================================
# Install Metis Systemd User Services
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

echo "=== Installing Metis Systemd User Services ==="
mkdir -p "${SYSTEMD_USER_DIR}"

# Copy service unit files
for s in metis-hestia.service metis-kybernetes.service metis-prosopon.service metis-stoma-watcher.service metis-paideia-retrainer.service; do
    echo "Installing ${s} -> ${SYSTEMD_USER_DIR}/${s}"
    cp "${SCRIPT_DIR}/${s}" "${SYSTEMD_USER_DIR}/${s}"
done

# Initialize default env file if absent
if [ ! -f "${ROOT_DIR}/deploy/metis.env" ]; then
    echo "Creating ${ROOT_DIR}/deploy/metis.env from template..."
    cp "${ROOT_DIR}/deploy/metis.env.example" "${ROOT_DIR}/deploy/metis.env"
    chmod 0600 "${ROOT_DIR}/deploy/metis.env"
fi

echo "Reloading systemd user daemon..."
systemctl --user daemon-reload

echo ""
echo "Installation complete!"
echo "To enable and start the complete Metis stack:"
echo "  systemctl --user enable --now metis-hestia metis-kybernetes metis-prosopon metis-stoma-watcher metis-paideia-retrainer"
echo ""
echo "To check service status:"
echo "  systemctl --user status metis-hestia metis-kybernetes metis-prosopon metis-stoma-watcher metis-paideia-retrainer"
echo ""
echo "To view live unified logs:"
echo "  journalctl --user -f -u metis-hestia -u metis-kybernetes -u metis-prosopon -u metis-paideia-retrainer"
