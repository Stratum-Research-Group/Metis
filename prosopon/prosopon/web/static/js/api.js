/**
 * Prosopon API Client
 */
const API = {
  getAuthToken() {
    return localStorage.getItem("prosopon_token") || "";
  },

  setAuthToken(token) {
    if (token) {
      localStorage.setItem("prosopon_token", token);
    } else {
      localStorage.removeItem("prosopon_token");
    }
  },

  async request(endpoint, options = {}) {
    const headers = options.headers || {};
    const token = this.getAuthToken();
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    if (!(options.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
    }

    const res = await fetch(endpoint, { ...options, headers });
    if (res.status === 401) {
      window.dispatchEvent(new CustomEvent("prosopon:unauthorized"));
      throw new Error("Unauthorized: Device pairing required");
    }
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `HTTP ${res.status}`);
    }
    return res.json();
  },

  async listSessions() {
    return this.request("/api/sessions");
  },

  async createSession(name) {
    return this.request("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
  },

  async getSession(sessionId) {
    return this.request(`/api/sessions/${sessionId}`);
  },

  async deleteSession(sessionId) {
    return this.request(`/api/sessions/${sessionId}`, { method: "DELETE" });
  },

  async pairDevice(pin) {
    return this.request("/api/auth/pair", {
      method: "POST",
      body: JSON.stringify({ pin, device_name: navigator.userAgent }),
    });
  },

  async resolveApproval(approvalId, decision) {
    return this.request(`/api/approvals/${approvalId}/resolve`, {
      method: "POST",
      body: JSON.stringify({ decision }),
    });
  },

  async transcribeAudio(audioBlob) {
    const formData = new FormData();
    formData.append("file", audioBlob, "speech.webm");
    return this.request("/api/voice/transcribe", {
      method: "POST",
      body: formData,
    });
  },

  async synthesizeSpeech(text) {
    const res = await fetch("/api/voice/synthesize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${this.getAuthToken()}`,
      },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error("TTS failed");
    return res.blob();
  },
};
