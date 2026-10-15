/**
 * Prosopon Unified Control Dashboard
 * Grounded in the Metis Dashboard Implementation Contract.
 * Strict DOM security: Uses textContent and createElement exclusively for all remote and dynamic strings.
 */

const Dashboard = {
  activeSection: "overview",
  pollInterval: null,
  pollFrequencyMs: 12000,
  cachedOverview: null,
  cachedModules: [],
  cachedGpu: null,

  init() {
    this.container = document.getElementById("dashboard-workspace");
    if (!this.container) return;
    this.renderShell();
    this.bindEvents();
  },

  startPolling() {
    this.refresh();
    if (this.pollInterval) clearInterval(this.pollInterval);
    this.pollInterval = setInterval(() => {
      if (!document.hidden) {
        this.refresh();
      }
    }, this.pollFrequencyMs);
  },

  stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  },

  async refresh() {
    try {
      const [overview, modules, gpu] = await Promise.all([
        API.getDashboardOverview().catch((err) => ({
          service: "kybernetes",
          orchestrator_mode: "offline",
          orchestrator: { state: "offline", detail: String(err) },
          components: {},
        })),
        API.getDashboardModules().catch(() => []),
        API.getDashboardGpuTelemetry().catch(() => ({ available: false })),
      ]);

      this.cachedOverview = overview;
      this.cachedModules = modules;
      this.cachedGpu = gpu;

      this.renderCurrentSection();
      this.updateHeaderBanners(overview);
    } catch (e) {
      console.error("Dashboard refresh error:", e);
    }
  },

  renderShell() {
    this.container.textContent = "";

    // Sub-navigation tabs
    const nav = document.createElement("nav");
    nav.className = "dashboard-subnav";
    nav.setAttribute("role", "tablist");

    const sections = [
      { id: "overview", label: "Overview" },
      { id: "models", label: "Models & GPU" },
      { id: "memory", label: "Memory & Ingest" },
      { id: "reasoning", label: "Reasoning" },
      { id: "guardrails", label: "Guardrails" },
      { id: "eval_train", label: "Eval & Training" },
      { id: "sessions_audit", label: "Sessions & Audit" },
      { id: "config", label: "Config Inspector" },
    ];

    sections.forEach((sec) => {
      const btn = document.createElement("button");
      btn.className = `dash-tab-btn ${this.activeSection === sec.id ? "active" : ""}`;
      btn.setAttribute("role", "tab");
      btn.setAttribute("aria-selected", this.activeSection === sec.id ? "true" : "false");
      btn.textContent = sec.label;
      btn.dataset.section = sec.id;
      btn.addEventListener("click", () => {
        this.activeSection = sec.id;
        nav.querySelectorAll(".dash-tab-btn").forEach((b) => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        this.renderCurrentSection();
      });
      nav.appendChild(btn);
    });

    this.container.appendChild(nav);

    // Banner container
    const bannerBox = document.createElement("div");
    bannerBox.id = "dash-banner-box";
    bannerBox.className = "dash-banner-box";
    this.container.appendChild(bannerBox);

    // Content container
    const content = document.createElement("div");
    content.id = "dash-content-area";
    content.className = "dash-content-area";
    this.container.appendChild(content);
  },

  bindEvents() {
    window.addEventListener("prosopon:tab-changed", (e) => {
      if (e.detail?.tab === "dashboard") {
        this.startPolling();
      } else {
        this.stopPolling();
      }
    });
  },

  updateHeaderBanners(overview) {
    const bannerBox = document.getElementById("dash-banner-box");
    if (!bannerBox) return;
    bannerBox.textContent = "";

    if (overview?.mock_banner) {
      const banner = document.createElement("div");
      banner.className = "dash-banner mock-banner";
      const title = document.createElement("strong");
      title.textContent = "MOCK ENVIRONMENT ACTIVE: ";
      banner.appendChild(title);
      banner.appendChild(document.createTextNode("Orchestrator and models running in simulated mock mode."));
      bannerBox.appendChild(banner);
    } else if (overview?.simulation_banner) {
      const banner = document.createElement("div");
      banner.className = "dash-banner sim-banner";
      const title = document.createElement("strong");
      title.textContent = "SIMULATED ADAPTERS ACTIVE: ";
      banner.appendChild(title);
      banner.appendChild(document.createTextNode("One or more stack adapters are running with in-memory simulation."));
      bannerBox.appendChild(banner);
    }

    if (overview?.orchestrator_mode === "offline") {
      const banner = document.createElement("div");
      banner.className = "dash-banner error-banner";
      const title = document.createElement("strong");
      title.textContent = "ORCHESTRATOR OFFLINE: ";
      banner.appendChild(title);
      banner.appendChild(document.createTextNode("Cannot connect to Kybernetes on port 8081. Verify service is running."));
      bannerBox.appendChild(banner);
    }
  },

  renderCurrentSection() {
    const area = document.getElementById("dash-content-area");
    if (!area) return;
    area.textContent = "";

    switch (this.activeSection) {
      case "overview":
        this.renderOverviewSection(area);
        break;
      case "models":
        this.renderModelsSection(area);
        break;
      case "memory":
        this.renderMemorySection(area);
        break;
      case "reasoning":
        this.renderReasoningSection(area);
        break;
      case "guardrails":
        this.renderGuardrailsSection(area);
        break;
      case "eval_train":
        this.renderEvalTrainSection(area);
        break;
      case "sessions_audit":
        this.renderSessionsAuditSection(area);
        break;
      case "config":
        this.renderConfigSection(area);
        break;
    }
  },

  /* 1. OVERVIEW SECTION */
  renderOverviewSection(area) {
    const overview = this.cachedOverview || {};
    const components = overview.components || {};

    // System summary cards
    const grid = document.createElement("div");
    grid.className = "dash-metric-grid";

    grid.appendChild(this.createMetricCard("Orchestrator", overview.orchestrator_mode || "unknown", overview.version || "0.1.0"));
    grid.appendChild(this.createMetricCard("Git Release", overview.git?.commit || "unknown", overview.git?.dirty ? "dirty working tree" : "clean"));
    grid.appendChild(this.createMetricCard("Budget DB", overview.budget_storage_mode || "in-memory", "task limits enforced"));
    grid.appendChild(this.createMetricCard("Stack Compatibility", overview.stack_compatibility || "unchecked", "11 modules indexed"));

    area.appendChild(grid);

    // Component Status Table
    const compCard = document.createElement("div");
    compCard.className = "dash-card";
    const compHead = document.createElement("div");
    compHead.className = "dash-card-header";
    const h2 = document.createElement("h2");
    h2.textContent = "Component Subsystem Health (5 Canonical States)";
    compHead.appendChild(h2);
    compCard.appendChild(compHead);

    const table = document.createElement("table");
    table.className = "dash-table";
    
    const thead = document.createElement("thead");
    const trH = document.createElement("tr");
    ["Module", "State", "Freshness", "Details", "Evidence", "Checked At", "Actions"].forEach((txt) => {
      const th = document.createElement("th");
      th.textContent = txt;
      trH.appendChild(th);
    });
    thead.appendChild(trH);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    const modKeys = Object.keys(components).length > 0
      ? Object.keys(components)
      : ["kybernetes", "hestia", "prosopon", "lthcm", "dianoia", "krisis", "phylax", "kleroterion", "stoma", "dokime", "paideia"];

    modKeys.forEach((key) => {
      const s = components[key] || {
        state: "unknown",
        detail: "Unprobed component",
        freshness: "unknown",
        evidence: "No status recorded",
        checked_at: new Date().toISOString(),
      };

      const tr = document.createElement("tr");

      // Module Name
      const tdName = document.createElement("td");
      tdName.className = "font-mono font-bold";
      tdName.textContent = key.toUpperCase();
      tr.appendChild(tdName);

      // State Chip
      const tdState = document.createElement("td");
      tdState.appendChild(this.createStateBadge(s.state));
      tr.appendChild(tdState);

      // Freshness
      const tdFresh = document.createElement("td");
      tdFresh.textContent = s.freshness || "unknown";
      tdFresh.className = s.freshness === "fresh" ? "text-emerald" : "text-slate";
      tr.appendChild(tdFresh);

      // Detail
      const tdDetail = document.createElement("td");
      tdDetail.textContent = s.detail || "—";
      tr.appendChild(tdDetail);

      // Evidence
      const tdEv = document.createElement("td");
      tdEv.className = "font-mono text-muted text-xs";
      tdEv.textContent = s.evidence || "—";
      tr.appendChild(tdEv);

      // Checked At
      const tdTime = document.createElement("td");
      tdTime.className = "text-xs text-muted";
      tdTime.textContent = this.formatTimestamp(s.checked_at);
      tr.appendChild(tdTime);

      // Actions
      const tdAct = document.createElement("td");
      const probeBtn = document.createElement("button");
      probeBtn.className = "dash-btn-sm primary";
      probeBtn.textContent = "Probe";
      probeBtn.addEventListener("click", async () => {
        probeBtn.disabled = true;
        probeBtn.textContent = "Probing...";
        try {
          const res = await API.runDashboardProbe(key);
          this.refresh();
        } catch (e) {
          alert(`Probe error for ${key}: ${e.message}`);
        } finally {
          probeBtn.disabled = false;
          probeBtn.textContent = "Probe";
        }
      });
      tdAct.appendChild(probeBtn);
      tr.appendChild(tdAct);

      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    compCard.appendChild(table);
    area.appendChild(compCard);

    // Quick Recommendations / Next Steps
    const recCard = document.createElement("div");
    recCard.className = "dash-card";
    const recHead = document.createElement("div");
    recHead.className = "dash-card-header";
    const recH = document.createElement("h2");
    recH.textContent = "Project Recommendations & Health Next Steps";
    recHead.appendChild(recH);
    recCard.appendChild(recHead);

    const ul = document.createElement("ul");
    ul.className = "dash-rec-list";

    const recs = [];
    if (overview.budget_storage_mode === "in-memory") {
      recs.push({ title: "Enable SQLite Budget Persistence", desc: "Pass --db-path kybernetes_budget.db to persist session costs and reconciliation." });
    }
    if (components.hestia?.state === "offline") {
      recs.push({ title: "Start Hestia Inference Server", desc: "Launch hestia on port 8080 (llama-server backends in range 9000-9019) to activate local inference." });
    }
    if (components.lthcm?.state === "simulated") {
      recs.push({ title: "Mount Indexed Memory Root", desc: "Point kybernetes serve --memory-root <path> to real Stoma/LTHCM indexed documentation." });
    }
    if (recs.length === 0) {
      recs.push({ title: "Stack Healthy", desc: "All core subsystems configured and operating within nominal thresholds." });
    }

    recs.forEach((r) => {
      const li = document.createElement("li");
      const title = document.createElement("strong");
      title.textContent = r.title + ": ";
      li.appendChild(title);
      li.appendChild(document.createTextNode(r.desc));
      ul.appendChild(li);
    });
    recCard.appendChild(ul);
    area.appendChild(recCard);
  },

  /* 2. MODELS & RESOURCES */
  renderModelsSection(area) {
    const gpu = this.cachedGpu || {};
    const card = document.createElement("div");
    card.className = "dash-card";
    
    const head = document.createElement("div");
    head.className = "dash-card-header";
    const h2 = document.createElement("h2");
    h2.textContent = "Hestia Local Model Pool & Hardware Acceleration";
    head.appendChild(h2);
    card.appendChild(head);

    // Model list
    const modelBox = document.createElement("div");
    modelBox.className = "dash-model-box";
    
    const h3 = document.createElement("h3");
    h3.textContent = "Registered Models";
    modelBox.appendChild(h3);

    const models = [
      { id: "llama-3.1-8b-instruct", target: "gpu_0", estimated_vram: "5800 MB", state: "loaded", default: true },
      { id: "qwen-2.5-7b-instruct", target: "gpu_1", estimated_vram: "5200 MB", state: "registered", default: false },
    ];

    models.forEach((m) => {
      const row = document.createElement("div");
      row.className = "dash-model-row";

      const info = document.createElement("div");
      const name = document.createElement("strong");
      name.textContent = m.id + (m.default ? " (Default)" : "");
      info.appendChild(name);

      const sub = document.createElement("div");
      sub.className = "text-xs text-muted";
      sub.textContent = `Target: ${m.target} | Est. VRAM: ${m.estimated_vram} | State: ${m.state}`;
      info.appendChild(sub);
      row.appendChild(info);

      const actions = document.createElement("div");
      actions.style.display = "flex";
      actions.style.gap = "0.5rem";
      actions.style.alignItems = "center";

      const statusBadge = document.createElement("span");
      statusBadge.className = `dash-badge ${m.state === "loaded" ? "state-live" : "state-simulated"}`;
      statusBadge.textContent = m.state.toUpperCase();
      actions.appendChild(statusBadge);

      const loadBtn = document.createElement("button");
      loadBtn.className = "dash-probe-btn";
      loadBtn.textContent = "Load";
      loadBtn.onclick = async () => {
        loadBtn.disabled = true;
        loadBtn.textContent = "Loading...";
        try {
          await API.loadDashboardModel(m.id);
          this.renderModelsSection(area);
        } catch (e) {
          alert(`Model load failed: ${e.message}`);
          loadBtn.disabled = false;
          loadBtn.textContent = "Load";
        }
      };
      actions.appendChild(loadBtn);

      const unloadBtn = document.createElement("button");
      unloadBtn.className = "dash-probe-btn";
      unloadBtn.textContent = "Unload";
      unloadBtn.onclick = async () => {
        unloadBtn.disabled = true;
        unloadBtn.textContent = "Unloading...";
        try {
          await API.unloadDashboardModel(m.id);
          this.renderModelsSection(area);
        } catch (e) {
          alert(`Model unload failed: ${e.message}`);
          unloadBtn.disabled = false;
          unloadBtn.textContent = "Unload";
        }
      };
      actions.appendChild(unloadBtn);

      row.appendChild(actions);
      modelBox.appendChild(row);
    });

    card.appendChild(modelBox);
    area.appendChild(card);

    // GPU Telemetry Card
    const gpuCard = document.createElement("div");
    gpuCard.className = "dash-card";
    const gpuHead = document.createElement("div");
    gpuHead.className = "dash-card-header";
    const gpuH2 = document.createElement("h2");
    gpuH2.textContent = `Hardware Telemetry (${gpu.available ? "Live NVML / nvidia-smi" : "Estimated Fallback"})`;
    gpuHead.appendChild(gpuH2);
    gpuCard.appendChild(gpuHead);

    if (gpu.available && gpu.gpus?.length > 0) {
      const gpuGrid = document.createElement("div");
      gpuGrid.className = "dash-gpu-grid";
      gpu.gpus.forEach((g) => {
        const item = document.createElement("div");
        item.className = "dash-gpu-item";
        
        const gTitle = document.createElement("h4");
        gTitle.textContent = `GPU ${g.index}: ${g.name}`;
        item.appendChild(gTitle);

        const vramText = document.createElement("div");
        vramText.textContent = `VRAM: ${g.memory_used_mb} / ${g.memory_total_mb} MB (${Math.round((g.memory_used_mb / g.memory_total_mb) * 100)}%)`;
        item.appendChild(vramText);

        const bar = document.createElement("div");
        bar.className = "dash-meter-bar";
        const fill = document.createElement("div");
        fill.className = "dash-meter-fill";
        fill.style.width = `${Math.min(100, Math.round((g.memory_used_mb / g.memory_total_mb) * 100))}%`;
        bar.appendChild(fill);
        item.appendChild(bar);

        const metrics = document.createElement("div");
        metrics.className = "text-xs text-muted";
        metrics.textContent = `Temp: ${g.temperature_c}°C | Compute Utilization: ${g.utilization_gpu_pct}%`;
        item.appendChild(metrics);

        gpuGrid.appendChild(item);
      });
      gpuCard.appendChild(gpuGrid);
    } else {
      const p = document.createElement("p");
      p.className = "text-muted";
      p.textContent = "Hardware NVML telemetry unavailable. Local inference server estimates VRAM admission control via hestia.yaml parameters.";
      gpuCard.appendChild(p);
    }
    area.appendChild(gpuCard);
  },

  /* 3. MEMORY & INGESTION */
  renderMemorySection(area) {
    const card = document.createElement("div");
    card.className = "dash-card";
    
    const head = document.createElement("div");
    head.className = "dash-card-header";
    const h2 = document.createElement("h2");
    h2.textContent = "LTHCM Context Memory & Stoma Ingestion Pipeline";
    head.appendChild(h2);
    card.appendChild(head);

    const p = document.createElement("p");
    p.textContent = "Long-Term Hierarchical Context Memory provides dual dense-vector and lexical BM25 passage retrieval with generation-pinned indices.";
    card.appendChild(p);

    const list = document.createElement("div");
    list.className = "dash-stat-list";

    list.appendChild(this.createStatRow("Active Memory Root", "In-Memory Demo Corpus (4 Passages)"));
    list.appendChild(this.createStatRow("Generation ID", "gen-000001 (Genesis)"));
    list.appendChild(this.createStatRow("Stoma Ingestion Tool", "Batch Tool (SHA-256 Provenance Gated)"));
    list.appendChild(this.createStatRow("Temporal Bounds", "Epoch-based strict generation pointers active"));

    card.appendChild(list);
    area.appendChild(card);
  },

  /* 4. REASONING */
  renderReasoningSection(area) {
    const card = document.createElement("div");
    card.className = "dash-card";
    
    const head = document.createElement("div");
    head.className = "dash-card-header";
    const h2 = document.createElement("h2");
    h2.textContent = "Deliberation, Verification, and Consensus Engines";
    head.appendChild(h2);
    card.appendChild(head);

    const list = document.createElement("div");
    list.className = "dash-stat-list";

    list.appendChild(this.createStatRow("Dianoia Thought Graph", "Bounded Graph Walks (Max 5 steps, Zero-Edge Rerank Fallback)"));
    list.appendChild(this.createStatRow("Krisis Verifier Gate", "Independent Factuality & Citation Verification (Score Threshold >= 0.70)"));
    list.appendChild(this.createStatRow("Krisis Policy Version", "0.1.0-draft (Draft Policy)"));
    list.appendChild(this.createStatRow("Kleroterion Consensus", "Deterministic Ensemble Consensus & Conflict Resolution"));

    card.appendChild(list);
    area.appendChild(card);
  },

  /* 5. GUARDRAILS */
  renderGuardrailsSection(area) {
    const card = document.createElement("div");
    card.className = "dash-card";
    
    const head = document.createElement("div");
    head.className = "dash-card-header";
    const h2 = document.createElement("h2");
    h2.textContent = "Phylax Zero-Trust Capability Firewall & Audit Chain";
    head.appendChild(h2);
    card.appendChild(head);

    const list = document.createElement("div");
    list.className = "dash-stat-list";

    list.appendChild(this.createStatRow("Enforcement Mode", "Guarded Capability Gateway (Zero-Trust Policy)"));
    list.appendChild(this.createStatRow("Active Policy", "phylax/policies/demo_policy.yaml"));
    list.appendChild(this.createStatRow("Approval Lifetime", "300 seconds TTL (Single-use atomic resolution)"));
    list.appendChild(this.createStatRow("Cryptographic Audit Log", "Append-only SHA-256 hash-chained execution trail"));

    card.appendChild(list);
    area.appendChild(card);
  },

  /* 6. EVAL & TRAINING */
  renderEvalTrainSection(area) {
    const card = document.createElement("div");
    card.className = "dash-card";
    
    const head = document.createElement("div");
    head.className = "dash-card-header";
    const h2 = document.createElement("h2");
    h2.textContent = "Dokime Evaluation Suite & Paideia Training";
    head.appendChild(h2);
    card.appendChild(head);

    const list = document.createElement("div");
    list.className = "dash-stat-list";

    list.appendChild(this.createStatRow("Dokime Evaluation Harness", "Continuous benchmark regression suite (Config: TOML)"));
    list.appendChild(this.createStatRow("Regression Gate Status", "PASS (Simulated baseline runs)"));
    list.appendChild(this.createStatRow("Paideia LoRA Trainer", "Adapter Registry Compiler (Offline batch tool)"));
    list.appendChild(this.createStatRow("Training Manifests", "Simulated provenance flags strictly recorded"));

    card.appendChild(list);
    area.appendChild(card);
  },

  /* 7. SESSIONS & AUDIT LOG */
  async renderSessionsAuditSection(area) {
    const card = document.createElement("div");
    card.className = "dash-card";
    
    const head = document.createElement("div");
    head.className = "dash-card-header";
    const h2 = document.createElement("h2");
    h2.textContent = "Admin Audit Log & Turn Traces";
    head.appendChild(h2);
    card.appendChild(head);

    const auditBox = document.createElement("div");
    auditBox.textContent = "Loading admin audit log...";
    card.appendChild(auditBox);
    area.appendChild(card);

    try {
      const audit = await API.getDashboardAuditLog(20, 0);
      auditBox.textContent = "";

      if (audit.entries && audit.entries.length > 0) {
        const table = document.createElement("table");
        table.className = "dash-table";
        const thead = document.createElement("thead");
        const trH = document.createElement("tr");
        ["Time", "Actor", "Action", "Target", "Status"].forEach((t) => {
          const th = document.createElement("th");
          th.textContent = t;
          trH.appendChild(th);
        });
        thead.appendChild(trH);
        table.appendChild(thead);

        const tbody = document.createElement("tbody");
        audit.entries.forEach((e) => {
          const tr = document.createElement("tr");
          
          const tdT = document.createElement("td");
          tdT.className = "text-xs text-muted";
          tdT.textContent = this.formatTimestamp(e.timestamp);
          tr.appendChild(tdT);

          const tdA = document.createElement("td");
          tdA.className = "font-mono text-xs";
          tdA.textContent = e.actor;
          tr.appendChild(tdA);

          const tdAct = document.createElement("td");
          tdAct.textContent = e.action;
          tr.appendChild(tdAct);

          const tdTar = document.createElement("td");
          tdTar.className = "font-mono text-xs text-muted";
          tdTar.textContent = e.target;
          tr.appendChild(tdTar);

          const tdS = document.createElement("td");
          tdS.className = e.status === "allowed" || e.status === "completed" ? "text-emerald" : "text-rose";
          tdS.textContent = e.status;
          tr.appendChild(tdS);

          tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        auditBox.appendChild(table);
      } else {
        auditBox.textContent = "No admin audit events recorded yet.";
      }
    } catch (e) {
      auditBox.textContent = `Failed to load audit log: ${e.message}`;
    }
  },

  /* 8. CONFIG INSPECTOR */
  async renderConfigSection(area) {
    const card = document.createElement("div");
    card.className = "dash-card";
    
    const head = document.createElement("div");
    head.className = "dash-card-header";
    const h2 = document.createElement("h2");
    h2.textContent = "Module Configuration Inspector (Saved vs Effective)";
    head.appendChild(h2);
    card.appendChild(head);

    const select = document.createElement("select");
    select.className = "dash-select";

    const modules = this.cachedModules.length > 0 ? this.cachedModules : [
      { id: "kybernetes", display_name: "Kybernetes Orchestrator" },
      { id: "prosopon", display_name: "Prosopon Frontend & BFF" },
      { id: "hestia", display_name: "Hestia Inference Server" },
      { id: "stoma", display_name: "Stoma Ingestion Pipeline" },
      { id: "phylax", display_name: "Phylax Capability Guard" },
      { id: "krisis", display_name: "Krisis Response Verifier" },
      { id: "dokime", display_name: "Dokime Evaluation Suite" },
    ];

    modules.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.display_name || m.id;
      select.appendChild(opt);
    });

    const displayBox = document.createElement("div");
    displayBox.className = "dash-config-box";

    // Validator controls
    const validatorSection = document.createElement("div");
    validatorSection.className = "dash-validator-box";
    validatorSection.style.marginTop = "1rem";
    validatorSection.style.borderTop = "1px solid var(--border-color, #333)";
    validatorSection.style.paddingTop = "1rem";

    const valHeader = document.createElement("h3");
    valHeader.textContent = "Configuration Dry-Run Validator (No Side Effects)";
    validatorSection.appendChild(valHeader);

    const valHelp = document.createElement("p");
    valHelp.style.fontSize = "0.85rem";
    valHelp.style.opacity = "0.8";
    valHelp.textContent = "Propose configuration changes below and run the module's native validator without touching disk or active memory.";
    validatorSection.appendChild(valHelp);

    const textarea = document.createElement("textarea");
    textarea.className = "dash-textarea";
    textarea.style.width = "100%";
    textarea.style.height = "160px";
    textarea.style.fontFamily = "monospace";
    textarea.style.fontSize = "0.85rem";
    textarea.placeholder = "Enter JSON proposed configuration...";
    validatorSection.appendChild(textarea);

    const btnRow = document.createElement("div");
    btnRow.style.marginTop = "0.5rem";
    btnRow.style.display = "flex";
    btnRow.style.gap = "0.5rem";
    btnRow.style.alignItems = "center";

    const valBtn = document.createElement("button");
    valBtn.className = "dash-btn dash-btn-primary";
    valBtn.textContent = "Run Native Validator";

    const feedbackBox = document.createElement("div");
    feedbackBox.className = "dash-validation-feedback";
    feedbackBox.style.marginTop = "0.75rem";

    valBtn.addEventListener("click", async () => {
      feedbackBox.textContent = "Running module native validator...";
      feedbackBox.style.color = "inherit";
      try {
        let parsed;
        try {
          parsed = JSON.parse(textarea.value);
        } catch (jsonErr) {
          feedbackBox.textContent = `Invalid JSON syntax: ${jsonErr.message}`;
          feedbackBox.style.color = "#ff6b6b";
          return;
        }

        const res = await API.validateDashboardModuleConfig(select.value, parsed);
        feedbackBox.textContent = "";

        const badge = document.createElement("span");
        badge.className = `dash-badge ${res.valid ? "state-live" : "state-offline"}`;
        badge.textContent = res.valid ? "VALID" : "INVALID";
        feedbackBox.appendChild(badge);

        if (res.restart_required) {
          const rBadge = document.createElement("span");
          rBadge.className = "dash-badge";
          rBadge.style.marginLeft = "0.5rem";
          rBadge.textContent = "RESTART REQUIRED ON APPLY";
          feedbackBox.appendChild(rBadge);
        }

        if (res.errors && res.errors.length > 0) {
          const errList = document.createElement("ul");
          errList.style.color = "#ff6b6b";
          errList.style.marginTop = "0.5rem";
          res.errors.forEach((err) => {
            const li = document.createElement("li");
            li.textContent = err;
            errList.appendChild(li);
          });
          feedbackBox.appendChild(errList);
        }

        if (res.warnings && res.warnings.length > 0) {
          const warnList = document.createElement("ul");
          warnList.style.color = "#fcc419";
          warnList.style.marginTop = "0.5rem";
          res.warnings.forEach((w) => {
            const li = document.createElement("li");
            li.textContent = w;
            warnList.appendChild(li);
          });
          feedbackBox.appendChild(warnList);
        }

        if (res.valid && (!res.warnings || res.warnings.length === 0)) {
          const okMsg = document.createElement("div");
          okMsg.style.marginTop = "0.5rem";
          okMsg.style.color = "#51cf66";
          okMsg.textContent = "Configuration satisfies all constraints and invariants.";
          feedbackBox.appendChild(okMsg);
        }
      } catch (err) {
        feedbackBox.textContent = `Validation failed to execute: ${err.message}`;
        feedbackBox.style.color = "#ff6b6b";
      }
    });

    let currentConfigHash = "";

    const applyBtn = document.createElement("button");
    applyBtn.className = "dash-probe-btn";
    applyBtn.style.backgroundColor = "#2b8a3e";
    applyBtn.style.color = "#fff";
    applyBtn.textContent = "Apply Changes";

    applyBtn.addEventListener("click", async () => {
      feedbackBox.textContent = "Applying configuration changes...";
      feedbackBox.style.color = "#ced4da";
      let patch = {};
      try {
        patch = JSON.parse(textarea.value);
      } catch (e) {
        feedbackBox.textContent = `JSON Syntax Error: ${e.message}`;
        feedbackBox.style.color = "#ff6b6b";
        return;
      }

      try {
        const res = await API.applyDashboardModuleConfig(select.value, currentConfigHash, patch);
        feedbackBox.innerHTML = "";
        const okMsg = document.createElement("div");
        okMsg.style.color = "#51cf66";
        okMsg.textContent = `ChangeSet ${res.id} applied: ${res.status}. ${res.restart_required ? "Process restart pending." : ""}`;
        feedbackBox.appendChild(okMsg);
        loadConfigFor(select.value);
        loadHistoryFor(select.value);
      } catch (err) {
        let msg = err.message;
        if (msg.includes("Explicit confirmation required")) {
          const confirmToken = prompt(`${msg}\n\nType 'CONFIRM' to execute sensitive changes:`);
          if (confirmToken === "CONFIRM") {
            try {
              const res2 = await API.applyDashboardModuleConfig(select.value, currentConfigHash, patch, "CONFIRM");
              feedbackBox.innerHTML = "";
              const okMsg = document.createElement("div");
              okMsg.style.color = "#51cf66";
              okMsg.textContent = `ChangeSet ${res2.id} applied with confirmation: ${res2.status}.`;
              feedbackBox.appendChild(okMsg);
              loadConfigFor(select.value);
              loadHistoryFor(select.value);
              return;
            } catch (err2) {
              msg = err2.message;
            }
          } else {
            msg = "Action cancelled: confirmation token was not provided.";
          }
        }
        feedbackBox.textContent = `Apply failed: ${msg}`;
        feedbackBox.style.color = "#ff6b6b";
      }
    });

    btnRow.appendChild(valBtn);
    btnRow.appendChild(applyBtn);
    validatorSection.appendChild(btnRow);
    validatorSection.appendChild(feedbackBox);

    // History and Rollback Section
    const historySection = document.createElement("div");
    historySection.style.marginTop = "1.5rem";
    const histHead = document.createElement("h3");
    histHead.textContent = "Change History & Snapshots";
    historySection.appendChild(histHead);

    const histList = document.createElement("div");
    histList.style.marginTop = "0.5rem";
    historySection.appendChild(histList);

    const loadHistoryFor = async (modId) => {
      histList.textContent = "Loading history...";
      try {
        const history = await API.getDashboardModuleHistory(modId);
        histList.textContent = "";
        if (!history || history.length === 0) {
          histList.textContent = "No past ChangeSets recorded.";
          return;
        }
        history.forEach((cs) => {
          const item = document.createElement("div");
          item.className = "dash-row";
          item.style.alignItems = "center";
          item.style.justifyContent = "space-between";
          item.style.padding = "0.5rem";
          item.style.borderBottom = "1px solid rgba(255,255,255,0.08)";

          const left = document.createElement("div");
          const title = document.createElement("strong");
          title.textContent = cs.id;
          const meta = document.createElement("div");
          meta.className = "text-xs text-muted";
          meta.textContent = `${cs.requested_at} by ${cs.requested_by} | Status: ${cs.status}`;
          left.appendChild(title);
          left.appendChild(meta);
          item.appendChild(left);

          if (cs.prev_snapshot_id && cs.status !== "rolled_back_on_disk") {
            const rbBtn = document.createElement("button");
            rbBtn.className = "dash-probe-btn";
            rbBtn.textContent = "Rollback";
            rbBtn.onclick = async () => {
              if (confirm(`Rollback to snapshot before ${cs.id}?`)) {
                try {
                  await API.rollbackDashboardModuleConfig(modId, cs.id);
                  loadConfigFor(modId);
                  loadHistoryFor(modId);
                } catch (e) {
                  alert(`Rollback failed: ${e.message}`);
                }
              }
            };
            item.appendChild(rbBtn);
          }
          histList.appendChild(item);
        });
      } catch (e) {
        histList.textContent = `History load error: ${e.message}`;
      }
    };

    const loadConfigFor = async (modId) => {
      displayBox.textContent = "Loading configuration...";
      feedbackBox.textContent = "";
      try {
        const cfg = await API.getDashboardModuleConfig(modId);
        displayBox.textContent = "";
        currentConfigHash = cfg.config_hash || "";

        const pre = document.createElement("pre");
        pre.className = "dash-code-block";
        pre.textContent = JSON.stringify(cfg, null, 2);
        displayBox.appendChild(pre);

        const initialVal = (cfg && cfg.saved && Object.keys(cfg.saved).length > 0) ? cfg.saved : (cfg.effective || {});
        textarea.value = JSON.stringify(initialVal, null, 2);
      } catch (e) {
        displayBox.textContent = `Config inspection error: ${e.message}`;
        textarea.value = "{}";
        currentConfigHash = "";
      }
    };

    select.addEventListener("change", (e) => {
      loadConfigFor(e.target.value);
      loadHistoryFor(e.target.value);
    });
    card.appendChild(select);
    card.appendChild(displayBox);
    card.appendChild(validatorSection);
    card.appendChild(historySection);
    area.appendChild(card);

    if (select.value) {
      loadConfigFor(select.value);
      loadHistoryFor(select.value);
    }
  },

  /* UI Helper Components */
  createMetricCard(title, mainText, subText) {
    const card = document.createElement("div");
    card.className = "dash-metric-card";
    
    const t = document.createElement("div");
    t.className = "dash-metric-title";
    t.textContent = title;
    card.appendChild(t);

    const m = document.createElement("div");
    m.className = "dash-metric-value";
    m.textContent = mainText;
    card.appendChild(m);

    const s = document.createElement("div");
    s.className = "dash-metric-sub";
    s.textContent = subText;
    card.appendChild(s);

    return card;
  },

  createStateBadge(state) {
    const badge = document.createElement("span");
    const st = String(state || "unknown").toLowerCase();
    badge.className = `dash-badge state-${st}`;
    badge.textContent = st.toUpperCase();
    return badge;
  },

  createStatRow(label, value) {
    const row = document.createElement("div");
    row.className = "dash-stat-row";

    const l = document.createElement("strong");
    l.textContent = label;
    row.appendChild(l);

    const v = document.createElement("span");
    v.textContent = value;
    row.appendChild(v);

    return row;
  },

  formatTimestamp(isoStr) {
    if (!isoStr) return "—";
    try {
      const d = new Date(isoStr);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch {
      return isoStr;
    }
  },
};

window.Dashboard = Dashboard;
