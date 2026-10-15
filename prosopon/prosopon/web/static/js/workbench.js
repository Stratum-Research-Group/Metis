/**
 * Metis Antigravity & VS Code Styled Workbench & Diff Review Engine
 * Provides multi-tab code viewing, file exploration, and Phylax-guarded side-by-side diff review.
 */

const MetisWorkbench = {
  activeFile: "kybernetes/kybernetes/tools/registry.py",
  openTabs: [
    { path: "kybernetes/kybernetes/tools/registry.py", name: "registry.py", modified: false },
    { path: "phylax/policies/default_policy.yaml", name: "default_policy.yaml", modified: false },
    { path: "DIFF_PREVIEW", name: "Proposed Diff [Agent]", isDiff: true, modified: true },
  ],
  activeDiff: {
    targetFile: "kybernetes/kybernetes/tools/registry.py",
    approvalId: "appr-09f18a",
    oldLines: [
      "        calc = SafeCalculator()",
      "        self.register(",
      "            calc.get_definition(),",
      "            lambda args: {\"result\": calc.evaluate(args.get(\"expression\", \"\"))},",
      "        )",
    ],
    newLines: [
      "        calc = SafeCalculator()",
      "        self.register(",
      "            calc.get_definition(),",
      "            lambda args: {\"result\": calc.evaluate(args.get(\"expression\", \"\"))},",
      "        )",
      "        # Pi 4-Tool Core Contract for Autonomous Coding",
      "        rrange = SafeReadRange()",
      "        self.register(rrange.get_definition(), ...)",
      "        eblock = SafeEditBlock()",
      "        self.register(eblock.get_definition(), ...)",
    ],
  },

  mockFileTree: [
    {
      name: "kybernetes",
      type: "dir",
      children: [
        { name: "tools/read_file_range.py", type: "file" },
        { name: "tools/edit_file_block.py", type: "file" },
        { name: "tools/write_file.py", type: "file" },
        { name: "tools/sandboxed_bash.py", type: "file" },
        { name: "tools/registry.py", type: "file" },
      ],
    },
    {
      name: "phylax",
      type: "dir",
      children: [
        { name: "policies/default_policy.yaml", type: "file" },
        { name: "gateway/gateway.py", type: "file" },
      ],
    },
    {
      name: "harness",
      type: "dir",
      children: [
        { name: "hardness.md", type: "file" },
        { name: "harness.md", type: "file" },
      ],
    },
    {
      name: "prosopon",
      type: "dir",
      children: [
        { name: "server.py", type: "file" },
        { name: "web/static/js/settings.js", type: "file" },
        { name: "web/static/js/workbench.js", type: "file" },
      ],
    },
  ],

  init() {
    this.container = document.getElementById("editor-workspace");
    if (!this.container) return;
    this.render();
  },

  render() {
    this.container.textContent = "";

    const workbench = document.createElement("div");
    workbench.className = "vscode-workbench";

    // Left sub-sidebar: Explorer
    const explorer = document.createElement("aside");
    explorer.className = "workbench-explorer";

    const expHeader = document.createElement("div");
    expHeader.className = "explorer-header";
    expHeader.textContent = "EXPLORER: METIS WORKSPACE";
    explorer.appendChild(expHeader);

    const treeContainer = document.createElement("div");
    treeContainer.className = "explorer-tree";
    this.renderFileTree(treeContainer);
    explorer.appendChild(treeContainer);

    workbench.appendChild(explorer);

    // Center Editor Area
    const editorArea = document.createElement("div");
    editorArea.className = "workbench-editor-area";

    // Tab Bar
    const tabBar = document.createElement("div");
    tabBar.className = "editor-tab-bar";
    this.renderTabs(tabBar, editorArea);
    editorArea.appendChild(tabBar);

    // Active Tab Content (Code or Diff)
    const editorContent = document.createElement("div");
    editorContent.className = "editor-content-viewport";
    this.renderActiveViewport(editorContent);
    editorArea.appendChild(editorContent);

    workbench.appendChild(editorArea);
    this.container.appendChild(workbench);
  },

  renderTabs(tabBar, editorArea) {
    tabBar.textContent = "";
    this.openTabs.forEach((tab) => {
      const tabEl = document.createElement("div");
      tabEl.className = `editor-tab ${this.activeFile === tab.path ? "active" : ""} ${tab.isDiff ? "diff-tab" : ""}`;

      const icon = document.createElement("span");
      icon.className = "tab-icon";
      icon.textContent = tab.isDiff ? "±" : "📄";
      tabEl.appendChild(icon);

      const title = document.createElement("span");
      title.className = "tab-title";
      title.textContent = tab.name;
      tabEl.appendChild(title);

      if (tab.modified) {
        const dot = document.createElement("span");
        dot.className = "tab-modified-dot";
        dot.textContent = "●";
        tabEl.appendChild(dot);
      }

      tabEl.addEventListener("click", () => {
        this.activeFile = tab.path;
        this.render();
      });

      tabBar.appendChild(tabEl);
    });
  },

  renderFileTree(container) {
    container.textContent = "";
    this.mockFileTree.forEach((node) => {
      const dirHeader = document.createElement("div");
      dirHeader.className = "tree-dir-header";
      dirHeader.textContent = `▾ ${node.name}`;
      container.appendChild(dirHeader);

      node.children.forEach((child) => {
        const fileItem = document.createElement("div");
        fileItem.className = `tree-file-item ${this.activeFile.endsWith(child.name) ? "selected" : ""}`;
        fileItem.textContent = `  ${child.name}`;
        fileItem.addEventListener("click", () => {
          const fullPath = `${node.name}/${child.name}`;
          this.openFile(fullPath, child.name);
        });
        container.appendChild(fileItem);
      });
    });
  },

  openFile(path, name) {
    let existing = this.openTabs.find((t) => t.path === path);
    if (!existing) {
      this.openTabs.push({ path, name, modified: false });
    }
    this.activeFile = path;
    this.render();
  },

  renderActiveViewport(viewport) {
    viewport.textContent = "";

    const activeTab = this.openTabs.find((t) => t.path === this.activeFile);
    if (activeTab && activeTab.isDiff) {
      this.renderDiffViewer(viewport);
    } else {
      this.renderCodeViewer(viewport);
    }
  },

  renderDiffViewer(viewport) {
    const diffWrap = document.createElement("div");
    diffWrap.className = "diff-viewer-wrap";

    // Phylax Action Banner
    const banner = document.createElement("div");
    banner.className = "phylax-diff-banner";

    const bannerInfo = document.createElement("div");
    bannerInfo.className = "banner-info";
    bannerInfo.innerHTML = `<strong>🛡️ Phylax Action-Bound Verification</strong>: Agent proposes patch to <code>${this.activeDiff.targetFile}</code>`;

    const bannerActions = document.createElement("div");
    bannerActions.className = "banner-actions";

    const btnApprove = document.createElement("button");
    btnApprove.className = "btn primary-btn btn-sm";
    btnApprove.textContent = "✓ Approve & Apply (Consume Token)";
    btnApprove.addEventListener("click", () => {
      this.handleApproveDiff();
    });

    const btnReject = document.createElement("button");
    btnReject.className = "btn secondary-btn btn-sm";
    btnReject.textContent = "✗ Reject & Backtrack";
    btnReject.addEventListener("click", () => {
      this.handleRejectDiff();
    });

    bannerActions.appendChild(btnApprove);
    bannerActions.appendChild(btnReject);
    banner.appendChild(bannerInfo);
    banner.appendChild(bannerActions);
    diffWrap.appendChild(banner);

    // Split View (Before vs After)
    const splitGrid = document.createElement("div");
    splitGrid.className = "diff-split-grid";

    // Left: Original
    const leftPane = document.createElement("div");
    leftPane.className = "diff-pane diff-left";
    const leftTitle = document.createElement("div");
    leftTitle.className = "diff-pane-title";
    leftTitle.textContent = "Original Code";
    leftPane.appendChild(leftTitle);

    const leftLines = document.createElement("div");
    leftLines.className = "diff-lines";
    this.activeDiff.oldLines.forEach((line, idx) => {
      const row = document.createElement("div");
      row.className = "diff-row";
      row.innerHTML = `<span class="diff-num">${idx + 20}</span><span class="diff-code">${line}</span>`;
      leftLines.appendChild(row);
    });
    leftPane.appendChild(leftLines);

    // Right: Proposed Replacement
    const rightPane = document.createElement("div");
    rightPane.className = "diff-pane diff-right";
    const rightTitle = document.createElement("div");
    rightTitle.className = "diff-pane-title";
    rightTitle.textContent = "Proposed Replacement (Verified by Krisis)";
    rightPane.appendChild(rightTitle);

    const rightLines = document.createElement("div");
    rightLines.className = "diff-lines";
    this.activeDiff.newLines.forEach((line, idx) => {
      const row = document.createElement("div");
      const isAdded = idx >= 5;
      row.className = `diff-row ${isAdded ? "diff-added" : ""}`;
      row.innerHTML = `<span class="diff-num">${idx + 20}</span><span class="diff-code">${line}</span>`;
      rightLines.appendChild(row);
    });
    rightPane.appendChild(rightLines);

    splitGrid.appendChild(leftPane);
    splitGrid.appendChild(rightPane);
    diffWrap.appendChild(splitGrid);

    viewport.appendChild(diffWrap);
  },

  renderCodeViewer(viewport) {
    const codeWrap = document.createElement("div");
    codeWrap.className = "code-viewer-wrap";

    const pathBar = document.createElement("div");
    pathBar.className = "code-path-bar";
    pathBar.textContent = `Workspace > ${this.activeFile}`;
    codeWrap.appendChild(pathBar);

    const linesContainer = document.createElement("div");
    linesContainer.className = "code-editor-lines";

    const sampleLines = [
      "# Metis Sovereign Stack Component",
      '"""Vetted declarative tool specifications with capability guardrails."""',
      "",
      "import time",
      "from typing import Any, Callable, Dict, Optional",
      "",
      "from kybernetes.contracts.enums import ToolSideEffectClass",
      "from kybernetes.contracts.tool_spec import ToolDefinition, ToolResult",
      "from kybernetes.tools.edit_file_block import SafeEditBlock",
      "from kybernetes.tools.read_file_range import SafeReadRange",
      "from kybernetes.tools.sandboxed_bash import SafeSandboxedBash",
      "from kybernetes.tools.write_file import SafeWriteFile",
      "",
      "class ToolRegistry:",
      '    """Central registry of vetted tools with capability sandboxing."""',
      "    def __init__(self):",
      "        self._tools = {}",
      "        self._register_default_tools()",
    ];

    sampleLines.forEach((line, idx) => {
      const row = document.createElement("div");
      row.className = "editor-row";
      row.innerHTML = `<span class="line-num">${idx + 1}</span><span class="line-code">${line}</span>`;
      linesContainer.appendChild(row);
    });

    codeWrap.appendChild(linesContainer);
    viewport.appendChild(codeWrap);
  },

  handleApproveDiff() {
    const banner = document.querySelector(".phylax-diff-banner");
    if (banner) {
      banner.innerHTML = `<strong style="color: var(--accent-emerald);">✓ Patch Committed Successfully</strong>. Phylax single-use token consumed. Working tree updated.`;
    }
  },

  handleRejectDiff() {
    const banner = document.querySelector(".phylax-diff-banner");
    if (banner) {
      banner.innerHTML = `<strong style="color: var(--accent-rose);">✗ Patch Rejected</strong>. Dianoia triggered backtrack to parent hypothesis node.`;
    }
  },
};

window.MetisWorkbench = MetisWorkbench;
