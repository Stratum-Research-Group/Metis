/**
 * Metis Lochos Squad & Multi-Agent Fleet Manager
 * Provides team-level agent visualization, seat configuration, and context switching without tmux.
 */

const MetisSquad = {
  activeSeatId: "coder",
  queueTasks: [],
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
    this.connectWebSocket();
  },

  connectWebSocket() {
    if (this.ws) {
      try { this.ws.close(); } catch {}
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const token = typeof API !== "undefined" && API.getAuthToken ? API.getAuthToken() : (localStorage.getItem("prosopon_token") || "");
    const tokenParam = token ? `?token=${encodeURIComponent(token)}` : "";
    const wsUrl = `${protocol}//${host}/api/lochos/ws${tokenParam}`;

    try {
      this.ws = new WebSocket(wsUrl);
      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "squad_state" && msg.data) {
            this.updateFromSquadState(msg.data);
          } else if (
            msg.type === "task_dispatched" ||
            msg.type === "baton_transferred" ||
            msg.type === "task_completed" ||
            msg.type === "task_failed" ||
            msg.type === "attention_resolved"
          ) {
            this.fetchLiveSquad();
          }
        } catch (e) {
          console.warn("Lochos WS message parse error:", e);
        }
      };
      this.ws.onclose = () => {
        setTimeout(() => this.connectWebSocket(), 5000);
      };
      this.ws.onerror = () => {
        try { this.ws.close(); } catch {}
      };
    } catch (err) {
      console.warn("Lochos WS connection error:", err);
    }
  },

  updateFromSquadState(live) {
    if (!live || !live.seats) return;
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
  },

  async fetchLiveSquad() {
    try {
      if (window.API && window.API.getLochosSquad) {
        const live = await window.API.getLochosSquad();
        if (live && live.seats) {
          this.updateFromSquadState(live);
        }
      }
      if (window.API && window.API.getLochosQueue) {
        const queue = await window.API.getLochosQueue();
        if (Array.isArray(queue)) {
          this.queueTasks = queue;
          this.render();
        }
      }
    } catch {
      // Local offline fallback retains starter state
    }
  },

  async completeTask(taskId, currentOwner) {
    const seatId = currentOwner ? currentOwner.split("@")[0] : this.activeSeatId;
    if (!window.confirm(`Sign off and mark task #${taskId} complete by seat ${seatId}?`)) return;
    try {
      if (window.API && window.API.completeLochosTask) {
        await window.API.completeLochosTask({ task_id: taskId, seat_id: seatId, resolution: "completed" });
        alert(`Task #${taskId} marked completed!`);
        await this.fetchLiveSquad();
      }
    } catch (err) {
      alert(`Complete failed: ${err.message || err}`);
    }
  },

  openPipelineModal() {
    const prompt = window.prompt("Enter autonomous pipeline prompt/goal:", "Audit and implement test suite verification");
    if (!prompt || !prompt.trim()) return;
    const mission = window.prompt("Enter mission tag (optional):", "sovereign-mission") || "sovereign-mission";
    const sliceId = window.prompt("Enter slice ID (optional):", "001") || "001";
    this.runPipeline({ prompt: prompt.trim(), mission: mission.trim(), slice_id: sliceId.trim() });
  },

  async runPipeline(payload) {
    try {
      if (window.API && window.API.runLochosPipeline) {
        const res = await window.API.runLochosPipeline(payload);
        alert(`Autonomous pipeline initiated: Task #${res.task_id || 'dispatched'}`);
        await this.fetchLiveSquad();
      }
    } catch (err) {
      alert(`Pipeline launch failed: ${err.message || err}`);
    }
  },

  openDispatchModal() {
    const title = window.prompt("Enter task title:", "New mission task");
    if (!title || !title.trim()) return;
    const targetSeat = window.prompt("Target seat (coder, reviewer, architect, tester):", "coder") || "coder";
    const mission = window.prompt("Mission name:", "release-0.5.8") || "release-0.5.8";
    const sliceId = window.prompt("Slice ID:", "001") || "001";
    this.dispatchTask({ title: title.trim(), target_seat: targetSeat.trim(), mission: mission.trim(), slice_id: sliceId.trim() });
  },

  async dispatchTask(payload) {
    try {
      if (window.API && window.API.dispatchLochos) {
        const res = await window.API.dispatchLochos(payload);
        alert(`Task #${res.task_id || 'dispatched'} assigned to ${res.assigned_to || payload.target_seat}`);
        await this.fetchLiveSquad();
      }
    } catch (err) {
      alert(`Dispatch failed: ${err.message || err}`);
    }
  },

  openTransferModal(taskId, currentOwner) {
    const fromSeat = currentOwner ? currentOwner.split("@")[0] : this.activeSeatId;
    const toSeat = window.prompt(`Transfer baton for #${taskId} from ${fromSeat} to seat:`, "reviewer");
    if (!toSeat || !toSeat.trim()) return;
    const note = window.prompt("Transfer handoff note (optional):", "Passing work for independent verification");
    this.transferBaton({ task_id: taskId, from_seat: fromSeat, to_seat: toSeat.trim(), note: note ? note.trim() : undefined });
  },

  async transferBaton(payload) {
    try {
      if (window.API && window.API.transferLochosBaton) {
        await window.API.transferLochosBaton(payload);
        alert(`Baton #${payload.task_id} transferred to ${payload.to_seat}!`);
        await this.fetchLiveSquad();
      }
    } catch (err) {
      alert(`Transfer failed: ${err.message || err}`);
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
            <div style="display: flex; gap: 0.4rem; align-items: center;">
              <button class="btn primary-btn btn-sm" onclick="MetisSquad.openPipelineModal()">Run Pipeline</button>
              <button class="btn secondary-btn btn-sm" onclick="MetisSquad.openDispatchModal()">+ Dispatch</button>
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
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
              <h4>Baton Queue Status (${this.queueTasks.length})</h4>
              <button class="btn secondary-btn btn-sm" onclick="MetisSquad.fetchLiveSquad()" style="padding: 2px 6px; font-size: 10px;">↻ Refresh</button>
            </div>
            ${this.queueTasks.length === 0 ? `
              <div class="queue-item-card">
                <div class="text-secondary" style="font-size: 0.8rem; text-align: center; padding: 0.5rem 0;">No active tasks in queue. Click "+ Dispatch" or "Run Pipeline".</div>
              </div>
            ` : this.queueTasks.map((task) => `
              <div class="queue-item-card" style="margin-bottom: 0.6rem;">
                <div class="queue-card-top">
                  <span class="badge ${task.status === 'in_progress' ? 'badge-info' : task.status === 'completed' ? 'badge-success' : task.status === 'blocked' ? 'badge-danger' : 'badge-secondary'}">
                    ${(task.status || 'PENDING').toUpperCase()}
                  </span>
                  <span class="font-mono">#${task.task_id}</span>
                </div>
                <div class="queue-card-title">${task.title}</div>
                <div class="queue-card-owner">
                  <span>Owner: ${task.baton_owner || 'Unassigned'}</span>
                  ${task.mission ? ` • <span class="font-mono">${task.mission}</span>` : ''}
                </div>
                ${task.status === 'in_progress' ? `
                  <div style="margin-top: 0.5rem; display: flex; gap: 0.4rem;">
                    <button class="btn primary-btn btn-sm" onclick="MetisSquad.completeTask('${task.task_id}', '${task.baton_owner}')">
                      ✓ Complete
                    </button>
                    <button class="btn secondary-btn btn-sm" onclick="MetisSquad.openTransferModal('${task.task_id}', '${task.baton_owner}')">
                      Transfer
                    </button>
                  </div>
                ` : ''}
              </div>
            `).join("")}
          </div>
        </div>
      </div>
    `;
  },
};

window.MetisSquad = MetisSquad;
