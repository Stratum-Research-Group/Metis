/**
 * Prosopon Main Application Orchestrator
 */
class App {
  constructor() {
    this.currentSessionId = null;
    this.initElements();
    this.attachEventListeners();
    this.bootstrap();
  }

  initElements() {
    this.sidebar = document.getElementById("sidebar");
    this.sidebarOverlay = document.getElementById("sidebar-overlay");
    this.btnToggleSidebar = document.getElementById("btn-toggle-sidebar");
    this.btnNewSession = document.getElementById("btn-new-session");
    this.sessionList = document.getElementById("session-list");
    this.sessionTitle = document.getElementById("session-title");
    this.messageContainer = document.getElementById("message-container");
    this.chatForm = document.getElementById("chat-form");
    this.promptInput = document.getElementById("prompt-input");
    this.btnSend = document.getElementById("btn-send");
    this.btnPtt = document.getElementById("btn-ptt");

    // Views & Workspaces
    this.tabBtnChat = document.getElementById("tab-btn-chat");
    this.tabBtnEditor = document.getElementById("tab-btn-editor");
    this.tabBtnSquad = document.getElementById("tab-btn-squad");
    this.tabBtnDashboard = document.getElementById("tab-btn-dashboard");
    this.tabBtnSettings = document.getElementById("tab-btn-settings");
    this.tabBtnPinax = document.getElementById("tab-btn-pinax");

    this.actBtnChat = document.getElementById("act-btn-chat");
    this.actBtnEditor = document.getElementById("act-btn-editor");
    this.actBtnSquad = document.getElementById("act-btn-squad");
    this.actBtnPinax = document.getElementById("act-btn-pinax");
    this.actBtnDashboard = document.getElementById("act-btn-dashboard");
    this.actBtnSettings = document.getElementById("act-btn-settings");

    this.chatWorkspace = document.getElementById("chat-workspace");
    this.editorWorkspace = document.getElementById("editor-workspace");
    this.squadWorkspace = document.getElementById("squad-workspace");
    this.pinaxWorkspace = document.getElementById("pinax-workspace");
    this.dashboardWorkspace = document.getElementById("dashboard-workspace");
    this.settingsWorkspace = document.getElementById("settings-workspace");
    
    // Modals
    this.transcriptModal = document.getElementById("transcript-modal");
    this.voiceTranscriptText = document.getElementById("voice-transcript-text");
    this.btnConfirmVoice = document.getElementById("btn-confirm-voice");
    this.btnCancelVoice = document.getElementById("btn-cancel-voice");
    
    this.pairingModal = document.getElementById("pairing-modal");
    this.pairingPinInput = document.getElementById("pairing-pin-input");
    this.btnSubmitPairing = document.getElementById("btn-submit-pairing");
    this.btnClosePairing = document.getElementById("btn-close-pairing");
    this.btnPairingInfo = document.getElementById("btn-pairing-info");
    this.themeSelect = document.getElementById("theme-select");
  }

  attachEventListeners() {
    this.btnToggleSidebar.onclick = () => this.toggleSidebar();
    this.sidebarOverlay.onclick = () => this.closeSidebar();
    this.btnNewSession.onclick = () => this.createNewSession();

    // Theme Switching
    if (this.themeSelect) {
      const savedTheme = localStorage.getItem("metis_theme") || "theme-obsidian";
      this.setTheme(savedTheme);
      this.themeSelect.value = savedTheme;
      this.themeSelect.onchange = (e) => {
        this.setTheme(e.target.value);
      };
    }

    // Top View Switcher buttons
    const navViews = [
      { top: this.tabBtnChat, act: this.actBtnChat, id: "chat" },
      { top: this.tabBtnEditor, act: this.actBtnEditor, id: "editor" },
      { top: this.tabBtnSquad, act: this.actBtnSquad, id: "squad" },
      { top: this.tabBtnPinax, act: this.actBtnPinax, id: "pinax" },
      { top: this.tabBtnDashboard, act: this.actBtnDashboard, id: "dashboard" },
      { top: this.tabBtnSettings, act: this.actBtnSettings, id: "settings" },
    ];

    navViews.forEach((v) => {
      if (v.top) v.top.onclick = () => this.switchView(v.id);
      if (v.act) v.act.onclick = () => this.switchView(v.id);
    });

    this.chatForm.onsubmit = (e) => {
      e.preventDefault();
      this.handleSendText();
    };

    // Push to talk events
    this.btnPtt.onmousedown = () => this.startVoiceCapture();
    this.btnPtt.onmouseup = () => this.stopVoiceCapture();
    this.btnPtt.ontouchstart = (e) => { e.preventDefault(); this.startVoiceCapture(); };
    this.btnPtt.ontouchend = (e) => { e.preventDefault(); this.stopVoiceCapture(); };

    // Voice Modal
    this.btnConfirmVoice.onclick = () => {
      const q = this.voiceTranscriptText.value.trim();
      this.transcriptModal.classList.remove("active");
      if (q) this.sendQuery(q);
    };
    this.btnCancelVoice.onclick = () => this.transcriptModal.classList.remove("active");

    // Pairing Modal
    this.btnPairingInfo.onclick = () => this.pairingModal.classList.add("active");
    this.btnClosePairing.onclick = () => this.pairingModal.classList.remove("active");
    this.btnSubmitPairing.onclick = () => this.handlePairingSubmit();

    window.addEventListener("prosopon:unauthorized", () => {
      this.pairingModal.classList.add("active");
    });
  }

  async bootstrap() {
    await API.ensureAuthToken();
    if (window.Dashboard) {
      window.Dashboard.init();
    }
    if (window.MetisSettings) {
      window.MetisSettings.init();
    }
    if (window.MetisWorkbench) {
      window.MetisWorkbench.init();
    }
    if (window.MetisSquad) {
      window.MetisSquad.init();
    }
    if (window.MetisPinax) {
      window.MetisPinax.init();
    }
    try {
      const sessions = await API.listSessions();
      if (sessions.length > 0) {
        await this.selectSession(sessions[0].session_id);
      } else {
        await this.createNewSession();
      }
      this.renderSessionList(sessions);
    } catch (err) {
      console.warn("Bootstrap pairing check:", err);
    }
  }

  switchView(viewName) {
    const views = [
      { id: "chat", top: this.tabBtnChat, act: this.actBtnChat, ws: this.chatWorkspace },
      { id: "editor", top: this.tabBtnEditor, act: this.actBtnEditor, ws: this.editorWorkspace },
      { id: "squad", top: this.tabBtnSquad, act: this.actBtnSquad, ws: this.squadWorkspace },
      { id: "pinax", top: this.tabBtnPinax, act: this.actBtnPinax, ws: this.pinaxWorkspace },
      { id: "dashboard", top: this.tabBtnDashboard, act: this.actBtnDashboard, ws: this.dashboardWorkspace },
      { id: "settings", top: this.tabBtnSettings, act: this.actBtnSettings, ws: this.settingsWorkspace },
    ];

    views.forEach((v) => {
      const isTarget = v.id === viewName;
      if (v.top) {
        v.top.classList.toggle("active", isTarget);
        v.top.setAttribute("aria-selected", isTarget ? "true" : "false");
      }
      if (v.act) {
        v.act.classList.toggle("active", isTarget);
      }
      if (v.ws) {
        v.ws.classList.toggle("hidden", !isTarget);
      }
    });

    if (viewName === "dashboard" && window.Dashboard) {
      window.Dashboard.refresh();
    } else if (viewName === "settings" && window.MetisSettings) {
      window.MetisSettings.render();
    } else if (viewName === "editor" && window.MetisWorkbench) {
      window.MetisWorkbench.render();
    } else if (viewName === "squad" && window.MetisSquad) {
      window.MetisSquad.render();
    } else if (viewName === "pinax" && window.MetisPinax) {
      window.MetisPinax.render();
    }

    window.dispatchEvent(new CustomEvent("prosopon:tab-changed", { detail: { tab: viewName } }));
  }


  setTheme(themeName) {
    document.body.classList.remove("theme-obsidian", "theme-linear", "theme-amber", "theme-nordic");
    document.body.classList.add(themeName);
    localStorage.setItem("metis_theme", themeName);
  }

  toggleSidebar() {
    this.sidebar.classList.toggle("open");
    this.sidebarOverlay.classList.toggle("show");
  }

  closeSidebar() {
    this.sidebar.classList.remove("open");
    this.sidebarOverlay.classList.remove("show");
  }

  async renderSessionList(sessions) {
    this.sessionList.innerHTML = "";
    for (const s of sessions) {
      const li = document.createElement("li");
      li.className = `session-item ${s.session_id === this.currentSessionId ? "active" : ""}`;
      li.textContent = s.name;
      li.onclick = () => {
        this.selectSession(s.session_id);
        this.closeSidebar();
      };
      this.sessionList.appendChild(li);
    }
  }

  async selectSession(sessionId) {
    this.currentSessionId = sessionId;
    const session = await API.getSession(sessionId);
    if (!session) return;

    this.sessionTitle.textContent = session.name;
    this.messageContainer.innerHTML = "";
    for (const msg of session.messages) {
      this.messageContainer.appendChild(UI.renderMessage(msg));
    }
    this.scrollToBottom();
  }

  async createNewSession() {
    const s = await API.createSession();
    this.currentSessionId = s.session_id;
    this.sessionTitle.textContent = s.name;
    this.messageContainer.innerHTML = "";
    const list = await API.listSessions();
    this.renderSessionList(list);
    this.closeSidebar();
  }

  async handleSendText() {
    const text = this.promptInput.value.trim();
    if (!text) return;
    this.promptInput.value = "";
    await this.sendQuery(text);
  }

  async sendQuery(queryText) {
    if (!this.currentSessionId) await this.createNewSession();

    // 1. Render User Message
    const userMsg = {
      message_id: "user-" + Date.now(),
      role: "user",
      content: queryText,
    };
    this.messageContainer.appendChild(UI.renderMessage(userMsg));
    this.scrollToBottom();

    // 2. Placeholder Assistant Message
    const asstMsgId = "asst-" + Date.now();
    const asstMsg = {
      message_id: asstMsgId,
      role: "assistant",
      content: "",
      status: "draft",
    };
    const asstEl = UI.renderMessage(asstMsg, true);
    this.messageContainer.appendChild(asstEl);
    this.scrollToBottom();

    // 3. Connect to SSE Stream
    try {
      const response = await fetch(`/api/chat/sessions/${this.currentSessionId}/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${API.getAuthToken()}`,
        },
        body: JSON.stringify({ query: queryText }),
      });

      if (!response.ok) throw new Error("Stream request failed");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // Retain incomplete line

        for (const line of lines) {
          if (line.startsWith("data:")) {
            const dataStr = line.slice(5).trim();
            if (dataStr) {
              const event = JSON.parse(dataStr);
              this.handleStreamEvent(asstMsgId, event);
            }
          }
        }
      }
    } catch (err) {
      UI.updateStreamingMessage(asstMsgId, `\n[Error: ${err.message}]`);
    }
  }

  handleStreamEvent(msgId, event) {
    if (event.event_type === "state_changed") {
      const state = event.payload.to_state || event.payload.state || "planning";
      this.setAgentState(state, event.payload.reason || "");
    } else if (event.event_type === "delta" && event.payload.text) {
      UI.updateStreamingMessage(msgId, event.payload.text);
      this.scrollToBottom();
    } else if (event.event_type === "ask_approval") {
      this.setAgentState("awaiting_approval", event.payload.approval?.action || "Phylax security gate approval required");
      UI.renderApprovalCard(event.payload.approval, async (approvalId, decision) => {
        await API.resolveApproval(approvalId, decision);
        this.setAgentState("planning", "Approval resolved, continuing...");
      });
    } else if (event.event_type === "completed") {
      this.setAgentState("completed", "Task finished and verified");
      setTimeout(() => this.setAgentState("idle", "Ready"), 3000);
      // Reload session to populate final citations and work trace
      API.getSession(this.currentSessionId).then((session) => {
        const lastMsg = session.messages[session.messages.length - 1];
        if (lastMsg) {
          const oldEl = document.getElementById(`msg-${msgId}`);
          if (oldEl) {
            oldEl.replaceWith(UI.renderMessage(lastMsg));
          }
        }
      });
    } else if (event.event_type === "failed") {
      this.setAgentState("failed", event.payload.error || "Turn failed");
    }
  }

  setAgentState(state, reason = "") {
    const pill = document.getElementById("agent-state-pill");
    if (!pill) return;
    const dot = pill.querySelector(".state-dot");
    const text = pill.querySelector(".state-text");

    const normalized = state.toLowerCase();
    pill.className = `agent-state-pill state-${normalized}`;
    if (text) {
      text.textContent = normalized.replace("_", " ").toUpperCase();
    }
    if (reason) {
      pill.title = `Agent State: ${normalized.toUpperCase()} — ${reason}`;
    }

    if (normalized === "awaiting_approval") {
      this.triggerApprovalNotification(reason || "Phylax approval token required to proceed.");
    }
  }

  triggerApprovalNotification(body) {
    // 1. Synthesize subtle audio chime via native Web Audio API (no external asset needed)
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        const ctx = new AudioCtx();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5
        osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.15); // A5
        gain.gain.setValueAtTime(0.12, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.35);
      }
    } catch (_) {}

    // 2. Desktop notification via Web Notifications API
    if ("Notification" in window) {
      if (Notification.permission === "granted") {
        new Notification("Metis: Phylax Approval Required", {
          body: body,
          icon: "/static/favicon.ico",
        });
      } else if (Notification.permission !== "denied") {
        Notification.requestPermission().then((perm) => {
          if (perm === "granted") {
            new Notification("Metis: Phylax Approval Required", { body });
          }
        });
      }
    }
  }

  async startVoiceCapture() {
    this.btnPtt.classList.add("recording");
    try {
      await window.Voice.startRecording();
    } catch (err) {
      this.btnPtt.classList.remove("recording");
      alert(err.message);
    }
  }

  async stopVoiceCapture() {
    this.btnPtt.classList.remove("recording");
    const audioBlob = await window.Voice.stopRecording();
    if (!audioBlob) return;

    try {
      const res = await API.transcribeAudio(audioBlob);
      if (res && res.text) {
        this.voiceTranscriptText.value = res.text;
        this.transcriptModal.classList.add("active");
      }
    } catch (err) {
      alert("Transcription error: " + err.message);
    }
  }

  async handlePairingSubmit() {
    const pin = this.pairingPinInput.value.trim();
    if (pin.length !== 6) {
      alert("Please enter a 6-digit PIN");
      return;
    }
    try {
      const res = await API.pairDevice(pin);
      if (res.token) {
        API.setAuthToken(res.token);
        this.pairingModal.classList.remove("active");
        alert("Device paired successfully!");
        this.bootstrap();
      }
    } catch (err) {
      alert("Pairing failed: " + err.message);
    }
  }

  scrollToBottom() {
    this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.app = new App();
});
