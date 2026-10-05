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

    // Views
    this.tabBtnChat = document.getElementById("tab-btn-chat");
    this.tabBtnDashboard = document.getElementById("tab-btn-dashboard");
    this.chatWorkspace = document.getElementById("chat-workspace");
    this.dashboardWorkspace = document.getElementById("dashboard-workspace");
    
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
  }

  attachEventListeners() {
    this.btnToggleSidebar.onclick = () => this.toggleSidebar();
    this.sidebarOverlay.onclick = () => this.closeSidebar();
    this.btnNewSession.onclick = () => this.createNewSession();

    // View Switching
    if (this.tabBtnChat && this.tabBtnDashboard) {
      this.tabBtnChat.onclick = () => this.switchView("chat");
      this.tabBtnDashboard.onclick = () => this.switchView("dashboard");
    }

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
    if (viewName === "dashboard") {
      this.tabBtnChat.classList.remove("active");
      this.tabBtnChat.setAttribute("aria-selected", "false");
      this.tabBtnDashboard.classList.add("active");
      this.tabBtnDashboard.setAttribute("aria-selected", "true");
      this.chatWorkspace.classList.add("hidden");
      this.dashboardWorkspace.classList.remove("hidden");
      window.dispatchEvent(new CustomEvent("prosopon:tab-changed", { detail: { tab: "dashboard" } }));
    } else {
      this.tabBtnDashboard.classList.remove("active");
      this.tabBtnDashboard.setAttribute("aria-selected", "false");
      this.tabBtnChat.classList.add("active");
      this.tabBtnChat.setAttribute("aria-selected", "true");
      this.dashboardWorkspace.classList.add("hidden");
      this.chatWorkspace.classList.remove("hidden");
      window.dispatchEvent(new CustomEvent("prosopon:tab-changed", { detail: { tab: "chat" } }));
    }
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
    if (event.event_type === "delta" && event.payload.text) {
      UI.updateStreamingMessage(msgId, event.payload.text);
      this.scrollToBottom();
    } else if (event.event_type === "ask_approval") {
      UI.renderApprovalCard(event.payload.approval, async (approvalId, decision) => {
        await API.resolveApproval(approvalId, decision);
      });
    } else if (event.event_type === "completed") {
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
