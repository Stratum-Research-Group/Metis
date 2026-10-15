/**
 * Metis Antigravity & VS Code Styled Settings Engine
 * Provides a categorized, searchable, rich settings panel for all Metis modules.
 */

const MetisSettings = {
  activeCategory: "hestia",
  searchQuery: "",
  modulesData: {},
  categoryConfigs: {},

  categories: [
    { id: "general", label: "General & Workbench", icon: "⚙️", moduleId: null },
    { id: "hestia", label: "Hestia (Local Models)", icon: "🧠", moduleId: "hestia" },
    { id: "kybernetes", label: "Kybernetes (Engine & Budget)", icon: "⚡", moduleId: "kybernetes" },
    { id: "phylax", label: "Phylax (Guardrails & Policies)", icon: "🛡️", moduleId: "phylax" },
    { id: "memory", label: "LTHCM & Stoma (Memory)", icon: "🌐", moduleId: "lthcm" },
    { id: "voice", label: "Voice & Audio (Piper/Whisper)", icon: "🎙️", moduleId: "prosopon" },
  ],

  // Default editable schemas when module config is loaded or offline
  settingDefinitions: {
    general: [
      {
        key: "general.theme",
        label: "Workbench Theme",
        description: "Visual aesthetic applied across the entire Metis workspace.",
        type: "select",
        options: [
          { value: "theme-obsidian", label: "Obsidian Cyber (Antigravity Default)" },
          { value: "theme-linear", label: "Linear Minimalist" },
          { value: "theme-amber", label: "Amber Cyberdeck" },
          { value: "theme-nordic", label: "Nordic Frost" },
        ],
        default: "theme-obsidian",
      },
      {
        key: "general.editorMinimap",
        label: "Editor Minimap",
        description: "Controls whether the code editor renders an interactive code minimap on the right.",
        type: "boolean",
        default: true,
      },
      {
        key: "general.fontSize",
        label: "Editor Font Size",
        description: "Font size in pixels for code editor and terminal outputs.",
        type: "number",
        default: 13,
        min: 10,
        max: 24,
      },
      {
        key: "general.showBreadcrumbs",
        label: "Workbench Breadcrumbs",
        description: "Display file navigation hierarchy and active document paths in top bar.",
        type: "boolean",
        default: true,
      },
    ],
    hestia: [
      {
        key: "hestia.active_model",
        label: "Active Coding Model",
        description: "Primary LLM checkpoint loaded by local Hestia server for reasoning and code generation.",
        type: "select",
        options: [
          { value: "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF", label: "Qwen 2.5 Coder 32B (Recommended)" },
          { value: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF", label: "Qwen 2.5 Coder 14B (Fast / Low VRAM)" },
          { value: "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", label: "DeepSeek Coder V2 Lite" },
          { value: "meta-llama/Llama-3.1-8B-Instruct", label: "Llama 3.1 8B Instruct" },
        ],
        default: "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
      },
      {
        key: "hestia.context_length",
        label: "Max Context Length (Tokens)",
        description: "Maximum tokens allocated per generation turn.",
        type: "number",
        default: 32768,
        min: 4096,
        max: 131072,
        step: 4096,
      },
      {
        key: "hestia.vram_budget_mb",
        label: "VRAM Admission Budget (MB)",
        description: "Maximum GPU memory allocated before rejecting model load.",
        type: "number",
        default: 24000,
        min: 4000,
        max: 96000,
        step: 1000,
      },
      {
        key: "hestia.idle_unload_timeout_seconds",
        label: "Idle Model Unload Timeout",
        description: "Seconds of inactivity before releasing model weights from GPU memory (0 to disable).",
        type: "number",
        default: 600,
        min: 0,
        max: 3600,
      },
    ],
    kybernetes: [
      {
        key: "kybernetes.kalman_budget_alpha",
        label: "Kalman Filter Budget Alpha",
        description: "Smoothing weight for online execution cost tracking and token rate estimation.",
        type: "range",
        default: 0.85,
        min: 0.1,
        max: 0.99,
        step: 0.05,
      },
      {
        key: "kybernetes.tool_timeout_seconds",
        label: "Tool Execution Timeout",
        description: "Maximum duration allowed for safe sandboxed command and file operations.",
        type: "number",
        default: 30,
        min: 5,
        max: 120,
      },
      {
        key: "kybernetes.max_tree_depth",
        label: "Dianoia Max Reasoning Depth",
        description: "Maximum deliberative graph search expansions before forced answer synthesis.",
        type: "number",
        default: 8,
        min: 1,
        max: 20,
      },
      {
        key: "kybernetes.enable_milestone_checkpoints",
        label: "Odyssey Milestone Checkpointing",
        description: "Automatically commit working-tree git milestones before mutating code actions.",
        type: "boolean",
        default: true,
      },
    ],
    phylax: [
      {
        key: "phylax.default_deny",
        label: "Strict Default Deny Mode",
        description: "Block any capability or path not explicitly whitelisted in Phylax YAML policies.",
        type: "boolean",
        default: true,
      },
      {
        key: "phylax.approval_ttl_seconds",
        label: "Single-Use Approval Token TTL",
        description: "Seconds before an unconsumed interactive Phylax approval token expires.",
        type: "number",
        default: 180,
        min: 30,
        max: 600,
      },
      {
        key: "phylax.auto_allow_reads",
        label: "Pre-Approve Safe Reads & AST Search",
        description: "Allow read_file_range and file_search without interactive prompt cards.",
        type: "boolean",
        default: true,
      },
      {
        key: "phylax.audit_hash_chain",
        label: "Cryptographic Audit Hash-Chain",
        description: "Maintain immutable SHA-256 state chain of all executed tools and approvals.",
        type: "boolean",
        default: true,
      },
    ],
    memory: [
      {
        key: "lthcm.light_cone_horizon_seconds",
        label: "Lorentzian Light-Cone Horizon (s)",
        description: "Temporal cutoff horizon for causal memory retrieval.",
        type: "number",
        default: 86400,
        min: 3600,
        max: 604800,
      },
      {
        key: "lthcm.poincare_manifold_dimension",
        label: "Poincaré Hyperbolic Dimension",
        description: "Latent manifold dimension for taxonomic hierarchical code embeddings.",
        type: "number",
        default: 384,
        min: 64,
        max: 1024,
      },
      {
        key: "stoma.max_chunk_tokens",
        label: "Stoma AST Chunk Token Bound",
        description: "Strict token boundary enforced on code syntax slices (must be <= 254).",
        type: "number",
        default: 254,
        min: 64,
        max: 254,
      },
    ],
    voice: [
      {
        key: "prosopon.whisper_model",
        label: "Whisper STT Model Checkpoint",
        description: "Local speech-to-text model weight file utilized for voice transcription.",
        type: "select",
        options: [
          { value: "ggml-base.en.bin", label: "ggml-base.en (Fast / Responsive)" },
          { value: "ggml-small.en.bin", label: "ggml-small.en (Higher Accuracy)" },
          { value: "ggml-medium.en.bin", label: "ggml-medium.en (High Fidelity)" },
        ],
        default: "ggml-base.en.bin",
      },
      {
        key: "prosopon.piper_voice",
        label: "Piper TTS Sovereign Voice Model",
        description: "Local neural voice model used for synthesized audio waveform playback.",
        type: "select",
        options: [
          { value: "en_US-lessac-medium", label: "Lessac (Clear Studio)" },
          { value: "en_US-amy-medium", label: "Amy (Natural Conversational)" },
          { value: "en_GB-southern_english_female-low", label: "Southern English" },
        ],
        default: "en_US-lessac-medium",
      },
      {
        key: "prosopon.audio_sample_rate",
        label: "Audio Sample Rate (Hz)",
        description: "Resampling frequency for local ffmpeg voice pipeline.",
        type: "number",
        default: 16000,
        min: 16000,
        max: 48000,
        step: 8000,
      },
    ],
  },

  init() {
    this.container = document.getElementById("settings-workspace");
    if (!this.container) return;
    this.loadPersistedValues();
    this.render();
  },

  loadPersistedValues() {
    try {
      const stored = localStorage.getItem("metis_settings_overrides");
      if (stored) {
        this.overrides = JSON.parse(stored);
      } else {
        this.overrides = {};
      }
    } catch {
      this.overrides = {};
    }
  },

  saveValue(key, value) {
    this.overrides[key] = value;
    localStorage.setItem("metis_settings_overrides", JSON.stringify(this.overrides));

    // Handle theme immediately if general.theme changed
    if (key === "general.theme" && window.App) {
      window.App.setTheme(value);
      const themeSelect = document.getElementById("theme-select");
      if (themeSelect) themeSelect.value = value;
    }
  },

  getValue(setting) {
    if (this.overrides[setting.key] !== undefined) {
      return this.overrides[setting.key];
    }
    return setting.default;
  },

  render() {
    this.container.textContent = "";

    const wrap = document.createElement("div");
    wrap.className = "settings-workbench";

    // Top Header with Search
    const header = document.createElement("header");
    header.className = "settings-header";

    const titleGroup = document.createElement("div");
    titleGroup.className = "settings-title-group";
    const h2 = document.createElement("h2");
    h2.textContent = "Settings";
    const sub = document.createElement("span");
    sub.className = "settings-subtitle";
    sub.textContent = "Sovereign Workspace Configuration";
    titleGroup.appendChild(h2);
    titleGroup.appendChild(sub);

    const searchBox = document.createElement("div");
    searchBox.className = "settings-search-wrap";
    const searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.className = "settings-search-input";
    searchInput.placeholder = "Search settings (e.g. model, vram, kalman, budget, timeout)...";
    searchInput.value = this.searchQuery;
    searchInput.addEventListener("input", (e) => {
      this.searchQuery = e.target.value.toLowerCase().trim();
      this.renderBody(body);
    });
    searchBox.appendChild(searchInput);

    header.appendChild(titleGroup);
    header.appendChild(searchBox);
    wrap.appendChild(header);

    // Body (Category Nav + Setting Cards List)
    const body = document.createElement("div");
    body.className = "settings-body";

    this.renderBody(body);
    wrap.appendChild(body);

    this.container.appendChild(wrap);
  },

  renderBody(body) {
    body.textContent = "";

    // Left Navigation Tree
    const nav = document.createElement("nav");
    nav.className = "settings-nav-sidebar";
    nav.setAttribute("role", "tablist");

    this.categories.forEach((cat) => {
      const btn = document.createElement("button");
      btn.className = `settings-nav-item ${this.activeCategory === cat.id ? "active" : ""}`;
      btn.setAttribute("role", "tab");
      btn.setAttribute("aria-selected", this.activeCategory === cat.id ? "true" : "false");

      const icon = document.createElement("span");
      icon.className = "cat-icon";
      icon.textContent = cat.icon;

      const label = document.createElement("span");
      label.className = "cat-label";
      label.textContent = cat.label;

      btn.appendChild(icon);
      btn.appendChild(label);

      btn.addEventListener("click", () => {
        this.activeCategory = cat.id;
        nav.querySelectorAll(".settings-nav-item").forEach((b) => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        this.renderCards(cardsPanel);
      });

      nav.appendChild(btn);
    });

    body.appendChild(nav);

    // Right Cards Panel
    const cardsPanel = document.createElement("div");
    cardsPanel.className = "settings-cards-panel";
    this.renderCards(cardsPanel);
    body.appendChild(cardsPanel);
  },

  renderCards(panel) {
    panel.textContent = "";

    const currentCat = this.categories.find((c) => c.id === this.activeCategory) || this.categories[0];

    // Breadcrumb title
    const groupHeader = document.createElement("div");
    groupHeader.className = "settings-group-header";
    const groupTitle = document.createElement("h3");
    groupTitle.textContent = currentCat.label;
    const groupDesc = document.createElement("p");
    groupDesc.textContent = `Configure parameters and operational policies for ${currentCat.label}.`;
    groupHeader.appendChild(groupTitle);
    groupHeader.appendChild(groupDesc);
    panel.appendChild(groupHeader);

    // Filter settings based on active category & search query
    let settingsToRender = [];
    if (this.searchQuery) {
      // Search across ALL categories
      for (const [catKey, list] of Object.entries(this.settingDefinitions)) {
        list.forEach((s) => {
          if (
            s.label.toLowerCase().includes(this.searchQuery) ||
            s.description.toLowerCase().includes(this.searchQuery) ||
            s.key.toLowerCase().includes(this.searchQuery)
          ) {
            settingsToRender.push(s);
          }
        });
      }
    } else {
      settingsToRender = this.settingDefinitions[this.activeCategory] || [];
    }

    if (settingsToRender.length === 0) {
      const empty = document.createElement("div");
      empty.className = "settings-empty-state";
      empty.textContent = `No settings found matching "${this.searchQuery}"`;
      panel.appendChild(empty);
      return;
    }

    settingsToRender.forEach((setting) => {
      const card = this.createSettingCard(setting);
      panel.appendChild(card);
    });

    // Action Bar at bottom
    const bottomBar = document.createElement("div");
    bottomBar.className = "settings-action-bar";

    const saveMsg = document.createElement("span");
    saveMsg.className = "save-status-msg";
    saveMsg.id = "save-status-msg";
    saveMsg.textContent = "Settings saved automatically to local sovereign vault";

    const resetBtn = document.createElement("button");
    resetBtn.className = "btn secondary-btn btn-sm";
    resetBtn.textContent = "Reset Category Defaults";
    resetBtn.addEventListener("click", () => {
      settingsToRender.forEach((s) => {
        delete this.overrides[s.key];
      });
      localStorage.setItem("metis_settings_overrides", JSON.stringify(this.overrides));
      this.renderCards(panel);
    });

    bottomBar.appendChild(saveMsg);
    bottomBar.appendChild(resetBtn);
    panel.appendChild(bottomBar);
  },

  createSettingCard(setting) {
    const card = document.createElement("div");
    card.className = "setting-card";

    const info = document.createElement("div");
    info.className = "setting-info";

    const label = document.createElement("div");
    label.className = "setting-label";
    label.textContent = setting.label;

    const keyBadge = document.createElement("span");
    keyBadge.className = "setting-key-badge";
    keyBadge.textContent = setting.key;
    label.appendChild(keyBadge);

    const desc = document.createElement("div");
    desc.className = "setting-desc";
    desc.textContent = setting.description;

    info.appendChild(label);
    info.appendChild(desc);
    card.appendChild(info);

    const controlWrap = document.createElement("div");
    controlWrap.className = "setting-control-wrap";

    const currentVal = this.getValue(setting);

    if (setting.type === "boolean") {
      const toggle = document.createElement("label");
      toggle.className = "antigravity-toggle";

      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(currentVal);
      input.addEventListener("change", (e) => {
        this.saveValue(setting.key, e.target.checked);
        this.flashSaved();
      });

      const slider = document.createElement("span");
      slider.className = "slider";

      toggle.appendChild(input);
      toggle.appendChild(slider);
      controlWrap.appendChild(toggle);
    } else if (setting.type === "select") {
      const select = document.createElement("select");
      select.className = "setting-select";

      setting.options.forEach((opt) => {
        const option = document.createElement("option");
        option.value = opt.value;
        option.textContent = opt.label;
        if (opt.value === currentVal) option.selected = true;
        select.appendChild(option);
      });

      select.addEventListener("change", (e) => {
        this.saveValue(setting.key, e.target.value);
        this.flashSaved();
      });

      controlWrap.appendChild(select);
    } else if (setting.type === "number") {
      const input = document.createElement("input");
      input.type = "number";
      input.className = "setting-input number-input";
      input.value = currentVal;
      if (setting.min !== undefined) input.min = setting.min;
      if (setting.max !== undefined) input.max = setting.max;
      if (setting.step !== undefined) input.step = setting.step;

      input.addEventListener("change", (e) => {
        const num = parseFloat(e.target.value);
        if (!isNaN(num)) {
          this.saveValue(setting.key, num);
          this.flashSaved();
        }
      });

      controlWrap.appendChild(input);
    } else if (setting.type === "range") {
      const wrap = document.createElement("div");
      wrap.className = "range-wrap";

      const input = document.createElement("input");
      input.type = "range";
      input.className = "setting-range";
      input.value = currentVal;
      input.min = setting.min || 0;
      input.max = setting.max || 1;
      input.step = setting.step || 0.1;

      const display = document.createElement("span");
      display.className = "range-value-display";
      display.textContent = String(currentVal);

      input.addEventListener("input", (e) => {
        display.textContent = e.target.value;
      });

      input.addEventListener("change", (e) => {
        const val = parseFloat(e.target.value);
        this.saveValue(setting.key, val);
        this.flashSaved();
      });

      wrap.appendChild(input);
      wrap.appendChild(display);
      controlWrap.appendChild(wrap);
    }

    card.appendChild(controlWrap);
    return card;
  },

  flashSaved() {
    const el = document.getElementById("save-status-msg");
    if (!el) return;
    el.textContent = "✓ Settings saved & applied";
    el.classList.add("saved-flash");
    setTimeout(() => {
      el.textContent = "Settings saved automatically to local sovereign vault";
      el.classList.remove("saved-flash");
    }, 2000);
  },
};

window.MetisSettings = MetisSettings;
