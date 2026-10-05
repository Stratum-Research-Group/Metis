/**
 * Prosopon UI Component Renderers
 * Safely renders untrusted metadata, citations, and security approval cards using DOM APIs.
 */
const UI = {
  renderMessage(msg, isStreaming = false) {
    const el = document.createElement("div");
    el.className = `message ${this.escapeAttr(msg.role || "user")}`;
    el.id = `msg-${this.escapeAttr(msg.message_id || "")}`;

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    contentDiv.textContent = msg.content || "";
    el.appendChild(contentDiv);

    // Render citations if present
    if (msg.citations && msg.citations.length > 0) {
      const citContainer = document.createElement("div");
      citContainer.className = "citations-container";
      for (const c of msg.citations) {
        const badge = document.createElement("span");
        badge.className = "citation-badge";
        badge.textContent = c.citation_id || "citation";
        badge.title = `${c.text || ""} (${c.source_title || ""})`;
        citContainer.appendChild(badge);
      }
      el.appendChild(citContainer);
    }

    // Render "Show Work" trace if present
    if (msg.route_summary) {
      const rs = msg.route_summary;
      const routeName = rs.route || "deliberation";

      const toggleBtn = document.createElement("button");
      toggleBtn.className = "show-work-toggle";
      toggleBtn.textContent = `Show Work (${routeName})`;

      const tracePanel = document.createElement("div");
      tracePanel.id = `trace-${this.escapeAttr(msg.message_id || "")}`;
      tracePanel.className = "route-trace-panel";
      tracePanel.style.display = "none";

      const addRow = (label, value) => {
        const row = document.createElement("div");
        const strong = document.createElement("strong");
        strong.textContent = `${label}: `;
        row.appendChild(strong);
        row.appendChild(document.createTextNode(String(value)));
        tracePanel.appendChild(row);
      };

      addRow("Route", rs.route || "N/A");
      addRow("Verifier Status", rs.verifier_status || msg.status || "N/A");
      addRow("Latency", rs.latency_ms ? rs.latency_ms.toFixed(1) + " ms" : "N/A");
      addRow("Components", (rs.components_used || []).join(" -> ") || "N/A");

      toggleBtn.addEventListener("click", () => {
        tracePanel.style.display = tracePanel.style.display === "none" ? "block" : "none";
      });

      el.appendChild(toggleBtn);
      el.appendChild(tracePanel);
    }

    return el;
  },

  updateStreamingMessage(msgId, appendedText) {
    const el = document.getElementById(`msg-${this.escapeAttr(msgId)}`);
    if (el) {
      const contentEl = el.querySelector(".message-content");
      if (contentEl) {
        contentEl.textContent += appendedText;
      }
    }
  },

  renderApprovalCard(approval, onResolve) {
    const dock = document.getElementById("approval-dock");
    if (!dock) return;
    dock.innerHTML = "";

    const card = document.createElement("div");
    card.className = "approval-card";
    card.id = `app-${this.escapeAttr(approval.approval_id || "")}`;

    const title = document.createElement("h4");
    title.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
      Phylax Security Approval Required
    `;
    card.appendChild(title);

    const desc = document.createElement("p");
    const strongAction = document.createElement("strong");
    strongAction.textContent = "Action: ";
    const codeAction = document.createElement("code");
    codeAction.textContent = approval.action || "";
    const strongTarget = document.createElement("span");
    strongTarget.textContent = " on ";
    const codeTarget = document.createElement("code");
    codeTarget.textContent = approval.target || "";
    desc.appendChild(strongAction);
    desc.appendChild(codeAction);
    desc.appendChild(strongTarget);
    desc.appendChild(codeTarget);
    card.appendChild(desc);

    const ruleP = document.createElement("p");
    const smallRule = document.createElement("small");
    smallRule.textContent = approval.policy_rule || "";
    ruleP.appendChild(smallRule);
    card.appendChild(ruleP);

    const actionsDiv = document.createElement("div");
    actionsDiv.className = "approval-actions";

    const allowBtn = document.createElement("button");
    allowBtn.className = "btn primary-btn";
    allowBtn.textContent = "Allow Once";
    allowBtn.addEventListener("click", () => {
      dock.innerHTML = "";
      onResolve(approval.approval_id, "allow_once");
    });

    const denyBtn = document.createElement("button");
    denyBtn.className = "btn secondary-btn";
    denyBtn.textContent = "Deny";
    denyBtn.addEventListener("click", () => {
      dock.innerHTML = "";
      onResolve(approval.approval_id, "deny");
    });

    actionsDiv.appendChild(allowBtn);
    actionsDiv.appendChild(denyBtn);
    card.appendChild(actionsDiv);

    dock.appendChild(card);
  },

  toggleWorkTrace(msgId) {
    const panel = document.getElementById(`trace-${this.escapeAttr(msgId)}`);
    if (panel) {
      panel.style.display = panel.style.display === "none" ? "block" : "none";
    }
  },

  escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  },

  escapeAttr(str) {
    if (!str) return "";
    return String(str).replace(/[^a-zA-Z0-9_-]/g, "_");
  },
};
