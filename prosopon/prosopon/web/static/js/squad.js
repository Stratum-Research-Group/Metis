/**
 * Metis Lochos Squad & Multi-Agent Fleet Manager
 * Provides team-level agent visualization, seat configuration, and context switching without tmux.
 */

const MetisSquad = {
  activeSeatId: "coder",
  rigData: {
    name: "metis-sovereign-squad",
    summary: "Autonomous engineering squad with specialized build and review seats",
    version: "0.2",
    seats: [
      {
        id: "coder",
        label: "Autonomous Coding Agent",
        role: "Primary code implementation, AST navigation, and exact-anchor editing",
        model: "qwen2.5-coder-32b",
        temperature: 0.2,
        effort: "high",
        permission_policy: "strict",
        worktree_branch: "squad/coder",
        state: "mutating",
        skills: ["python", "refactoring", "git", "ast-mapping"],
        baton: {
          task_id: "task-09f18a",
          mission: "release-0.5.8",
          slice_id: "OPR.0.5.8.14",
          title: "Implement Pi 4-Tool Core Contract and Sandboxing",
        },
        system_prompt: "You are the primary Metis sovereign coding agent. You implement changes using the 4 atomic Pi tools (read_range, write_file, edit_block, sandboxed_bash). All mutations are gated by Phylax single-use cryptographic tokens.",
      },
      {
        id: "reviewer",
        label: "Independent Review & Verification Agent",
        role: "Zero-trust diff auditing, Dokime test verification, and security checks",
        model: "qwen2.5-coder-32b",
        temperature: 0.0,
        effort: "medium",
        permission_policy: "audit_only",
        worktree_branch: "squad/reviewer",
        state: "idle",
        skills: ["security", "code-review", "pytest", "dokime"],
        baton: null,
        system_prompt: "You are the independent Metis auditor. You review proposed patches for security violations, regression risks, and test coverage before release gates are cleared.",
      },
      {
        id: "architect",
        label: "System Architect & Deliberation Guide",
        role: "Dianoia graph traversal, API contract design, and architectural governance",
        model: "qwen2.5-coder-32b",
        temperature: 0.3,
        effort: "high",
        permission_policy: "audit_only",
        worktree_branch: "squad/architect",
        state: "idle",
        skills: ["architecture", "dianoia", "api-design"],
        baton: null,
        system_prompt: "You guide architectural decomposition and evaluate multi-module boundaries across Metis subsystems.",
      },
      {
        id: "tester",
        label: "Dokime Paired Scenario Runner",
        role: "Seeded-regression test authoring and benchmark execution",
        model: "qwen2.5-coder-32b",
        temperature: 0.1,
        effort: "medium",
        permission_policy: "strict",
        worktree_branch: "squad/tester",
        state: "idle",
        skills: ["pytest", "benchmark", "dokime-runner"],
        baton: null,
        system_prompt: "You author Dokime paired seeded-regression scenarios ensuring every failure condition is tested with explicit pass/fail checks.",
      },
    ],
    attention_required: [
      {
        event_id: 1,
        task_id: "task-09f18a",
        seat_id: "coder@metis-sovereign-squad",
        reason: "phylax_approval_required",
        details: { tool: "edit_file_block", target: "kybernetes/tools/registry.py" },
        created_at: Date.now() - 120000,
        task_title: "Implement Pi 4-Tool Core Contract and Sandboxing",
      },
    ],
  },

  async init() {
    this.container = document.getElementById("squad-workspace");
    if (!this.container) return;
    this.render();
    await this.fetchLiveSquad();
  },

  async fetchLiveSquad() {
    try {
      if (window.API && window.API.getLochosSquad) {
        const live = await window.API.getLochosSquad();
        if (live && live.seats) {
          this.rigData.name = live.rig_name || this.rigData.name;
          this.rigData.version = live.version || this.rigData.version;
          this.rigData.summary = live.summary || this.rigData.summary;
          if (live.seats.length > 0) {
            this.rigData.seats = live.seats.map((s) => ({
              ...s,
              baton: s.current_task || s.baton || null,
            }));
          }
          if (live.attention_required) {
            this.rigData.attention_required = live.attention_required;
          }
          this.render();
        }
      }
    } catch {
      // Local offline fallback retains starter state
    }
  },

  async resolveAttention(taskId, seatId) {
    try {
      if (window.API && window.API.resolveLochosAttention) {
        await window.API.resolveLochosAttention({ task_id: taskId, seat_id: seatId, resolution: "approved" });
        await this.fetchLiveSquad();
      }
    } catch (err) {
      console.warn("Resolve attention error:", err);
    }
  },

  selectSeat(seatId) {
    this.activeSeatId = seatId;
    this.render();
  },

  openInAgentSession(seatId) {
    this.activeSeatId = seatId;
    const seat = this.rigData.seats.find((s) => s.id === seatId);
    if (!seat) return;

    // Switch view to chat/agent workspace
    if (window.app && window.app.switchView) {
      window.app.switchView("chat");
    }

    // Update session title and agent state indicator
    const titleEl = document.getElementById("session-title");
    if (titleEl) {
      titleEl.textContent = `${seat.label} (${seat.id}@${this.rigData.name})`;
    }
    if (window.app && window.app.updateAgentState) {
      window.app.updateAgentState(seat.state);
    }
  },

  render() {
    if (!this.container) return;
    const selected = this.rigData.seats.find((s) => s.id === this.activeSeatId) || this.rigData.seats[0];

    this.container.innerHTML = `
      <div class="squad-layout">
        <!-- Column 1: Fleet Seats List -->
        <div class="squad-col squad-seats-col">
          <div class="squad-panel-header">
            <div>
              <h3>Squad Fleet</h3>
              <p class="text-secondary">${this.rigData.name} (v${this.rigData.version})</p>
            </div>
            <span class="badge ${this.rigData.attention_required.length > 0 ? 'badge-warning' : 'badge-success'}">
              ${this.rigData.attention_required.length} Attention
            </span>
          </div>

          <div class="squad-seats-list">
            ${this.rigData.seats.map((seat) => `
              <div 
                class="squad-seat-card ${seat.id === this.activeSeatId ? 'selected' : ''}" 
                onclick="MetisSquad.selectSeat('${seat.id}')"
              >
                <div class="seat-card-top">
                  <span class="seat-name">${seat.label}</span>
                  <span class="agent-state-pill state-${seat.state}">
                    <span class="state-dot"></span>
                    <span class="state-text">${seat.state.toUpperCase()}</span>
                  </span>
                </div>
                <div class="seat-card-address">${seat.id}@${this.rigData.name}</div>
                <div class="seat-card-meta">
                  <span>${seat.model}</span>
                  <span>Branch: ${seat.worktree_branch}</span>
                </div>
                ${seat.baton ? `
                  <div class="seat-card-baton">
                    <span class="baton-label">Active Baton:</span> ${seat.baton.title}
                  </div>
                ` : `
                  <div class="seat-card-idle text-secondary">No active baton (idle)</div>
                `}
              </div>
            `).join("")}
          </div>
        </div>

        <!-- Column 2: Selected Seat Settings & Context -->
        <div class="squad-col squad-details-col">
          <div class="squad-panel-header">
            <div>
              <h3>Seat Configuration</h3>
              <p class="text-secondary">Selected: ${selected.id}@${this.rigData.name}</p>
            </div>
            <button class="btn primary-btn" onclick="MetisSquad.openInAgentSession('${selected.id}')">
              Open in Agent Pane ->
            </button>
          </div>

          <div class="squad-seat-details">
            <div class="settings-group">
              <label class="settings-label">Agent Label & ID</label>
              <div class="settings-value-row">
                <input type="text" class="input-text" value="${selected.label}" readonly />
                <span class="badge">${selected.id}</span>
              </div>
            </div>

            <div class="settings-group">
              <label class="settings-label">Role Definition</label>
              <textarea class="textarea-field" rows="2" readonly>${selected.role}</textarea>
            </div>

            <div class="settings-row-2">
              <div class="settings-group">
                <label class="settings-label">Local Model (Hestia Inference)</label>
                <input type="text" class="input-text" value="${selected.model}" readonly />
              </div>
              <div class="settings-group">
                <label class="settings-label">Temperature / Sampling</label>
                <input type="text" class="input-text" value="${selected.temperature}" readonly />
              </div>
            </div>

            <div class="settings-row-2">
              <div class="settings-group">
                <label class="settings-label">Phylax Permission Policy</label>
                <span class="badge ${selected.permission_policy === 'strict' ? 'badge-danger' : 'badge-info'}">
                  ${selected.permission_policy.toUpperCase()}
                </span>
              </div>
              <div class="settings-group">
                <label class="settings-label">Git Worktree Sandbox</label>
                <span class="font-mono text-secondary">${selected.worktree_branch}</span>
              </div>
            </div>

            <div class="settings-group">
              <label class="settings-label">Loaded Skills (.metis/skills/)</label>
              <div class="skills-pill-wrap">
                ${selected.skills.map((s) => `<span class="skill-pill">${s}</span>`).join("")}
              </div>
            </div>

            <div class="settings-group">
              <label class="settings-label">System Persona Prompt</label>
              <textarea class="textarea-field font-mono" rows="4" readonly>${selected.system_prompt}</textarea>
            </div>
          </div>
        </div>

        <!-- Column 3: Work Queue & Attention Reconciler -->
        <div class="squad-col squad-queue-col">
          <div class="squad-panel-header">
            <div>
              <h3>Lochos Work Queue</h3>
              <p class="text-secondary">Transactional SQLite (.metis/lochos.sqlite)</p>
            </div>
          </div>

          <div class="queue-section">
            <h4>Attention Required (${this.rigData.attention_required.length})</h4>
            ${this.rigData.attention_required.length === 0 ? `
              <div class="empty-state">No seats blocked.</div>
            ` : this.rigData.attention_required.map((attn) => `
              <div class="attention-card">
                <div class="attention-header">
                  <span class="badge badge-warning">BLOCKED</span>
                  <span class="seat-badge">${attn.seat_id}</span>
                </div>
                <div class="attention-body">
                  <strong>${attn.task_title}</strong>
                  <p class="text-secondary">Reason: ${attn.reason}</p>
                  <p class="font-mono text-secondary">Target: ${attn.details.target || 'N/A'}</p>
                </div>
                <div class="attention-actions">
                  <button class="btn primary-btn btn-sm" onclick="MetisSquad.resolveAttention('${attn.task_id}', '${attn.seat_id}')">
                    Approve & Resume
                  </button>
                  <button class="btn secondary-btn btn-sm" onclick="MetisSquad.openInAgentSession('${selected.id}')">
                    Inspect in Agent
                  </button>
                </div>
              </div>
            `).join("")}
          </div>

          <div class="queue-section" style="margin-top: 1.5rem;">
            <h4>Baton Queue Status</h4>
            <div class="queue-item-card">
              <div class="queue-card-top">
                <span class="badge badge-info">IN PROGRESS</span>
                <span class="font-mono">#task-09f18a</span>
              </div>
              <div class="queue-card-title">Implement Pi 4-Tool Core Contract and Sandboxing</div>
              <div class="queue-card-owner">Owner: coder@${this.rigData.name}</div>
            </div>
          </div>
        </div>
      </div>
    `;
  },
};

window.MetisSquad = MetisSquad;
