/**
 * Prosopon UI Component Renderers
 */
const UI = {
  renderMessage(msg, isStreaming = false) {
    const el = document.createElement("div");
    el.className = `message ${msg.role}`;
    el.id = `msg-${msg.message_id}`;

    let html = `<div class="message-content">${this.escapeHtml(msg.content)}</div>`;

    // Render citations if present
    if (msg.citations && msg.citations.length > 0) {
      html += `<div class="citations-container">`;
      for (const c of msg.citations) {
        html += `<span class="citation-badge" title="${this.escapeHtml(c.text)} (${this.escapeHtml(c.source_title)})">${c.citation_id}</span>`;
      }
      html += `</div>`;
    }

    // Render "Show Work" trace if present
    if (msg.route_summary) {
      const rs = msg.route_summary;
      html += `
        <button class="show-work-toggle" onclick="UI.toggleWorkTrace('${msg.message_id}')">Show Work (${rs.route || 'deliberation'})</button>
        <div id="trace-${msg.message_id}" class="route-trace-panel" style="display: none;">
          <div><strong>Route:</strong> ${rs.route || 'N/A'}</div>
          <div><strong>Verifier Status:</strong> ${rs.verifier_status || msg.status || 'N/A'}</div>
          <div><strong>Latency:</strong> ${rs.latency_ms ? rs.latency_ms.toFixed(1) + ' ms' : 'N/A'}</div>
          <div><strong>Components:</strong> ${(rs.components_used || []).join(' -> ')}</div>
        </div>
      `;
    }

    el.innerHTML = html;
    return el;
  },

  updateStreamingMessage(msgId, appendedText) {
    const el = document.getElementById(`msg-${msgId}`);
    if (el) {
      const contentEl = el.querySelector(".message-content");
      if (contentEl) {
        contentEl.textContent += appendedText;
      }
    }
  },

  renderApprovalCard(approval, onResolve) {
    const dock = document.getElementById("approval-dock");
    dock.innerHTML = "";

    const card = document.createElement("div");
    card.className = "approval-card";
    card.id = `app-${approval.approval_id}`;

    card.innerHTML = `
      <h4>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
        Phylax Security Approval Required
      </h4>
      <p><strong>Action:</strong> <code>${this.escapeHtml(approval.action)}</code> on <code>${this.escapeHtml(approval.target)}</code></p>
      <p><small>${this.escapeHtml(approval.policy_rule)}</small></p>
      <div class="approval-actions">
        <button id="btn-allow-${approval.approval_id}" class="btn primary-btn">Allow Once</button>
        <button id="btn-deny-${approval.approval_id}" class="btn secondary-btn">Deny</button>
      </div>
    `;

    dock.appendChild(card);

    document.getElementById(`btn-allow-${approval.approval_id}`).onclick = () => {
      dock.innerHTML = "";
      onResolve(approval.approval_id, "allow_once");
    };
    document.getElementById(`btn-deny-${approval.approval_id}`).onclick = () => {
      dock.innerHTML = "";
      onResolve(approval.approval_id, "deny");
    };
  },

  toggleWorkTrace(msgId) {
    const panel = document.getElementById(`trace-${msgId}`);
    if (panel) {
      panel.style.display = panel.style.display === "none" ? "block" : "none";
    }
  },

  escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  },
};
