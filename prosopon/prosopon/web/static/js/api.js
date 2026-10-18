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

  async ensureAuthToken() {
    const existing = this.getAuthToken();
    if (existing) return existing;

    // Check for single-use bootstrap token in URL parameters (from `prosopon open`)
    try {
      if (typeof window !== "undefined" && window.location) {
        const urlParams = new URLSearchParams(window.location.search);
        const bootstrapToken = urlParams.get("bootstrap_token") || urlParams.get("ticket");
        if (bootstrapToken) {
          const res = await fetch("/api/auth/exchange-ticket", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticket: bootstrapToken }),
          });
          if (res.ok) {
            const data = await res.json();
            if (data.token) {
              this.setAuthToken(data.token);
              urlParams.delete("bootstrap_token");
              urlParams.delete("ticket");
              const cleanSearch = urlParams.toString() ? `?${urlParams.toString()}` : "";
              window.history.replaceState({}, "", window.location.pathname + cleanSearch + window.location.hash);
              return data.token;
            }
          }
        }
      }
    } catch {
      // Offline or bootstrap exchange failed
    }
    return "";
  },

  async request(endpoint, options = {}) {
    const headers = options.headers || {};
    let token = this.getAuthToken();
    if (!token && !endpoint.startsWith("/api/auth/")) {
      token = await this.ensureAuthToken();
    }
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    if (!(options.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
    }

    const res = await fetch(endpoint, { ...options, headers });
    if (res.status === 401) {
      this.setAuthToken("");
      if (!options._retried && !endpoint.startsWith("/api/auth/")) {
        const fresh = await this.ensureAuthToken();
        if (fresh) {
          const retryOpts = { ...options, _retried: true };
          return this.request(endpoint, retryOpts);
        }
      }
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

  /* Unified Dashboard API */
  async getDashboardOverview() {
    return this.request("/api/dashboard/overview");
  },

  async getDashboardModules() {
    return this.request("/api/dashboard/modules");
  },

  async getDashboardModuleConfig(moduleId) {
    return this.request(`/api/dashboard/modules/${encodeURIComponent(moduleId)}/config`);
  },

  async getDashboardSessions(limit = 50, offset = 0) {
    return this.request(`/api/dashboard/sessions?limit=${limit}&offset=${offset}`);
  },

  async getDashboardTraces(limit = 50, offset = 0) {
    return this.request(`/api/dashboard/traces?limit=${limit}&offset=${offset}`);
  },

  async getDashboardAuditLog(limit = 50, offset = 0) {
    return this.request(`/api/dashboard/audit-log?limit=${limit}&offset=${offset}`);
  },

  async runDashboardProbe(componentId) {
    return this.request("/api/dashboard/probes/run", {
      method: "POST",
      body: JSON.stringify({ component_id: componentId }),
    });
  },

  async getDashboardGpuTelemetry() {
    return this.request("/api/dashboard/gpu-telemetry");
  },

  async validateDashboardModuleConfig(moduleId, proposed) {
    return this.request(`/api/dashboard/modules/${encodeURIComponent(moduleId)}/config/validate`, {
      method: "POST",
      body: JSON.stringify({ proposed }),
    });
  },

  async applyDashboardModuleConfig(moduleId, baseHash, patch, confirmToken = null) {
    return this.request(`/api/dashboard/modules/${encodeURIComponent(moduleId)}/config/apply`, {
      method: "POST",
      body: JSON.stringify({
        base_hash: baseHash,
        patch,
        confirm_token: confirmToken,
      }),
    });
  },

  async getDashboardModuleHistory(moduleId) {
    return this.request(`/api/dashboard/modules/${encodeURIComponent(moduleId)}/history`);
  },

  async rollbackDashboardModuleConfig(moduleId, changesetId) {
    return this.request(`/api/dashboard/modules/${encodeURIComponent(moduleId)}/config/rollback/${encodeURIComponent(changesetId)}`, {
      method: "POST",
    });
  },

  async loadDashboardModel(model, confirm = true) {
    return this.request("/api/dashboard/models/load", {
      method: "POST",
      body: JSON.stringify({ model, confirm }),
    });
  },

  async unloadDashboardModel(model, confirm = true) {
    return this.request("/api/dashboard/models/unload", {
      method: "POST",
      body: JSON.stringify({ model, confirm }),
    });
  },

  async getDokimeRuns(limit = 20) {
    return this.request(`/api/dashboard/batch/dokime/runs?limit=${limit}`);
  },

  async runDokimeBatch(payload) {
    return this.request("/api/dashboard/batch/dokime/run", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async getPaideiaRuns(limit = 20) {
    return this.request(`/api/dashboard/batch/paideia/runs?limit=${limit}`);
  },

  async runPaideiaBatch(payload) {
    return this.request("/api/dashboard/batch/paideia/run", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  /* Lochos Multi-Agent Fleet API */
  async getLochosSquad() {
    return this.request("/api/lochos/squad");
  },

  async getLochosQueue(status = null) {
    const url = status ? `/api/lochos/queue?status=${encodeURIComponent(status)}` : "/api/lochos/queue";
    return this.request(url);
  },

  async dispatchLochos(payload) {
    return this.request("/api/lochos/dispatch", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async resolveLochosAttention(payload) {
    return this.request("/api/lochos/resolve-attention", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  /* Workbench Real-Time API */
  async getWorkbenchTree() {
    return this.request("/api/workbench/tree");
  },

  async getWorkbenchFile(path) {
    return this.request(`/api/workbench/file?path=${encodeURIComponent(path)}`);
  },

  async getWorkbenchDiff() {
    return this.request("/api/workbench/diff");
  },

  /* Pinax Tool Catalogue API (Section 7) */
  async getPinaxTools(params = {}) {
    const qs = new URLSearchParams();
    if (params.query) qs.set("query", params.query);
    if (params.category) qs.set("category", params.category);
    if (params.risk_tier) qs.set("risk_tier", params.risk_tier);
    if (params.limit) qs.set("limit", params.limit);
    const queryStr = qs.toString();
    return this.request(`/api/pinax/tools${queryStr ? "?" + queryStr : ""}`);
  },

  async getPinaxToolDetail(toolId) {
    return this.request(`/api/pinax/tools/${encodeURIComponent(toolId)}`);
  },

  async getPinaxToolDocs(toolId, maxBytes = 4096) {
    return this.request(`/api/pinax/tools/${encodeURIComponent(toolId)}/docs?max_bytes=${maxBytes}`);
  },

  async triggerPinaxScan(pathDirs = null) {
    return this.request("/api/pinax/scan", {
      method: "POST",
      body: JSON.stringify(pathDirs ? { path_dirs: pathDirs } : {}),
    });
  },

  async evaluatePinaxInvocation(toolName, args = []) {
    return this.request("/api/pinax/evaluate", {
      method: "POST",
      body: JSON.stringify({ tool_name: toolName, args: args }),
    });
  },

  // --- Pyle Device & Access API ---
  async listDevices(includeRevoked = false) {
    return this.request(`/api/devices?all=${includeRevoked ? "true" : "false"}`);
  },

  async pairDevice(label = "Companion Device", preset = "companion", validitySeconds = 300) {
    return this.request("/api/devices/pair", {
      method: "POST",
      body: JSON.stringify({ label, preset, validity_seconds: validitySeconds }),
    });
  },

  async revokeDevice(deviceId, reason = "owner") {
    return this.request(`/api/devices/${encodeURIComponent(deviceId)}/revoke`, {
      method: "POST",
      body: JSON.stringify({ reason }),
    });
  },

  async panicLockdown(reason = "panic") {
    return this.request("/api/devices/lockdown", {
      method: "POST",
      body: JSON.stringify({ reason }),
    });
  },

  async getDeviceAudit(limit = 50) {
    return this.request(`/api/devices/audit?limit=${limit}`);
  },

  async getCurrentDevice() {
    return this.request("/api/devices/me");
  },
};

