"""Terminal-native interactive Coding Agent Runner for the Metis Sovereign Stack (MSS).

Enables autonomous coding, tool execution, terminal diff inspection, Phylax interactive
approvals, Odyssey milestone checkpoints, and Hermes skill learning directly from the CLI
without requiring a web browser.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import click

try:
    from harness.skill_learner import DynamicSkillLearner
except ImportError:
    project_root = Path(__file__).resolve().parents[3]
    if (project_root / "harness").exists() and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from harness.skill_learner import DynamicSkillLearner
    except ImportError:
        DynamicSkillLearner = None
from kybernetes.adapters.hestia import HestiaAdapter
from kybernetes.contracts.enums import AgentSemanticState, ToolSideEffectClass, TurnOutcomeStatus
from kybernetes.contracts.tool_spec import ToolCallRequest, ToolResult
from kybernetes.engine.checkpoint import GitMilestoneCheckpointer
from kybernetes.engine.coding_agent import CodingAgentHarness, CodingSessionResult, CodingTurnState
from kybernetes.engine.state import AgentStateMachine
from kybernetes.tools.registry import ToolRegistry


SYSTEM_PROMPT_TEMPLATE = """You are the Metis Sovereign Coding Agent (MSS), a precise, autonomous software engineering agent.
Your workspace is at: {workspace_root}
Current Git branch: {git_branch}

You operate strictly within the Pi 4-Tool Core Contract:
1. `read_file_range(path, start_line, end_line)`: Read 1-indexed lines with line counts and hashline anchors.
2. `edit_file_block(path, search_block, replace_block)`: Replace exact code blocks. Match search_block exactly.
3. `write_file(path, content, overwrite)`: Atomically create or overwrite a file.
4. `sandboxed_bash(command)`: Run a non-interactive shell command (tests, linters, git status).
5. `file_search(pattern, directory)`: Search for filenames or paths matching pattern.

Relevant Learned Skills for this task:
{learned_skills}

Instructions:
- To inspect code before editing, ALWAYS call `read_file_range` or `file_search`.
- For code modifications, make minimal, surgical changes using `edit_file_block`.
- When you want to call a tool, output a JSON tool_call block:
```tool_call
{{
  "tool_name": "tool_id",
  "arguments": {{ ... }}
}}
```
- Only call ONE tool per turn.
- When you have completed the task and verified your work, output:
```done
Task complete: <brief summary of what you did and verified>
```
"""


class TerminalCodingAgent:
    """CLI / Terminal runner for the Metis Sovereign Coding Agent."""

    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        hestia_adapter: Optional[HestiaAdapter] = None,
        max_steps: int = 20,
        auto_approve: bool = False,
        model_name: Optional[str] = None,
        test_command: Optional[str] = None,
        temperature: float = 0.15,
        base_url: Optional[str] = None,
    ):
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.base_url = base_url
        self.hestia = hestia_adapter or HestiaAdapter(base_url=base_url)
        self.max_steps = max_steps
        self.auto_approve = auto_approve
        self.model_name = model_name or self.hestia.default_model
        self.test_command = test_command
        self.temperature = temperature

        self.tool_registry = ToolRegistry(
            allowed_roots=[self.workspace_root],
            enforce_approvals=not auto_approve,
        )
        self.checkpointer = GitMilestoneCheckpointer(self.workspace_root)
        self.skill_learner = DynamicSkillLearner(self.workspace_root) if DynamicSkillLearner else None
        self.state_machine = AgentStateMachine()

        self.harness = CodingAgentHarness(
            workspace_root=self.workspace_root,
            tool_registry=self.tool_registry,
            checkpointer=self.checkpointer,
            max_steps=max_steps,
            state_machine=self.state_machine,
        )

    def _get_git_branch(self) -> str:
        try:
            import subprocess
            out = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.workspace_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out or "main"
        except Exception:
            return "main"

    def _format_diff_preview(self, path: str, search_block: str, replace_block: str) -> str:
        """Render a colorized terminal diff preview for edit_file_block."""
        target_file = (self.workspace_root / path).resolve()
        if not target_file.exists():
            return f"\n{click.style('--- New File / Missing File: ' + path, fg='yellow')}\n"

        search_lines = search_block.splitlines(keepends=True)
        replace_lines = replace_block.splitlines(keepends=True)

        diff = difflib.unified_diff(
            search_lines,
            replace_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )

        formatted: List[str] = []
        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                formatted.append(click.style(line, fg="green"))
            elif line.startswith("-") and not line.startswith("---"):
                formatted.append(click.style(line, fg="red"))
            elif line.startswith("@"):
                formatted.append(click.style(line, fg="cyan"))
            else:
                formatted.append(line)

        return "\n".join(formatted) if formatted else click.style(f"Replacing text in {path}", fg="yellow")

    def _extract_action(self, response_text: str) -> Tuple[Optional[Dict[str, Any]], bool, str]:
        """Parse tool_call or done block from model response."""
        # Check for done marker
        done_match = re.search(r"```done\s*(.*?)\s*```", response_text, re.DOTALL)
        if done_match:
            return None, True, done_match.group(1).strip()

        # Check for tool_call block
        tool_match = re.search(r"```(?:tool_call|json)\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if tool_match:
            try:
                data = json.loads(tool_match.group(1))
                if "tool_name" in data:
                    return data, False, ""
            except json.JSONDecodeError:
                pass

        # Fallback regex search for json object containing tool_name
        json_obj_match = re.search(r'\{\s*"tool_name"\s*:\s*"([^"]+)",\s*"arguments"\s*:\s*(\{.*?\})\s*\}', response_text, re.DOTALL)
        if json_obj_match:
            try:
                tool_name = json_obj_match.group(1)
                arguments = json.loads(json_obj_match.group(2))
                return {"tool_name": tool_name, "arguments": arguments}, False, ""
            except json.JSONDecodeError:
                pass

        # If model indicates completion in natural language without tool call
        if any(term in response_text.lower() for term in ["task is complete", "task completed", "i have finished", "all steps complete"]):
            return None, True, response_text.strip()

        return None, False, response_text.strip()

    def run(self, goal: str) -> CodingSessionResult:
        """Run the sovereign coding agent loop in the terminal."""
        return asyncio.run(self._run_async(goal))

    async def _run_async(self, goal: str) -> CodingSessionResult:
        click.echo("=" * 72)
        click.echo(click.style(" Metis Sovereign Coding Agent (MSS) — Terminal Harness", bold=True, fg="cyan"))
        click.echo(f" Workspace: {self.workspace_root}")
        click.echo(f" Model:     {self.model_name}")
        click.echo(f" Goal:      {click.style(goal, bold=True)}")
        click.echo("=" * 72 + "\n")

        # 1. Retrieve dynamic skills
        if self.skill_learner:
            skills = self.skill_learner.find_relevant_skills(goal)
            skills_text = (
                "\n".join([f"- **{s.name}** ({s.domain}): {s.description}" for s in skills])
                if skills else "No prior domain skills found in .metis/skills."
            )
        else:
            skills_text = "No dynamic skill learner available."

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            workspace_root=self.workspace_root,
            git_branch=self._get_git_branch(),
            learned_skills=skills_text,
        )

        history: List[Dict[str, str]] = []
        turn_history: List[CodingTurnState] = []
        backtracks = 0
        active_checkpoint: Optional[str] = None

        # Take initial milestone checkpoint
        initial_cp = self.checkpointer.create_checkpoint(f"Initial baseline: {goal[:50]}")
        active_checkpoint = initial_cp
        self.state_machine.transition(
            AgentSemanticState.PLANNING,
            reason="Starting MSS autonomous loop",
            metadata={"goal": goal},
        )

        current_prompt = f"Goal: {goal}\n\nPlease inspect the workspace and begin taking appropriate tool actions."

        for step in range(1, self.max_steps + 1):
            click.echo(click.style(f"\n[Step {step}/{self.max_steps}] Generating next action...", fg="blue", bold=True))

            # Append user / tool observation turn to conversation history
            history.append({"role": "user", "content": current_prompt})

            # Format active conversation prompt (windowed to last 10 turns to preserve token limits)
            full_prompt = "\n\n".join([
                f"### {item['role'].upper()}:\n{item['content']}"
                for item in history[-10:]
            ])

            response_chunks: List[str] = []
            click.echo(click.style("MSS Reasoning: ", fg="cyan"), nl=False)
            async for chunk in self.hestia.stream_chat(
                prompt=full_prompt,
                system_prompt=system_prompt,
                model=self.model_name,
                temperature=self.temperature,
            ):
                delta = chunk.get("delta", "")
                if delta:
                    response_chunks.append(delta)
                    click.echo(delta, nl=False)

            click.echo("")
            full_response = "".join(response_chunks).strip()
            history.append({"role": "assistant", "content": full_response})

            action, is_done, done_message = self._extract_action(full_response)
            if is_done:
                click.echo(click.style(f"\n[✓] {done_message or 'Task completed by agent.'}", fg="green", bold=True))
                # Synthesize Hermes Skill on successful resolution
                synth_id = None
                if self.skill_learner:
                    skill_id = self.skill_learner.synthesize_skill(
                        name=goal[:40],
                        domain="coding",
                        description=f"Solved: {goal}",
                        instructions=done_message or full_response[:300],
                        verification_command=self.test_command,
                    )
                    synth_id = skill_id.id
                    click.echo(click.style(f"[Hermes Skill Learned] Saved to .metis/skills/{skill_id.id}/SKILL.md", fg="magenta"))
                self.state_machine.transition(AgentSemanticState.COMPLETED, reason="Task completed successfully")
                return CodingSessionResult(
                    session_id=f"mss-term-{int(time.time())}",
                    success=True,
                    status=TurnOutcomeStatus.VERIFIED,
                    total_steps=step,
                    history=turn_history,
                    final_message=done_message or "Task complete",
                    skill_synthesized=synth_id,
                    checkpoints_taken=1,
                    backtracks_executed=backtracks,
                    final_state=AgentSemanticState.COMPLETED,
                    state_transitions=len(self.state_machine.history),
                )

            if not action:
                click.echo(click.style("[!] No clear tool action parsed. Prompting agent for clarification.", fg="yellow"))
                current_prompt = "No tool action was parsed. Please provide a ```tool_call block or a ```done block."
                continue

            tool_name = action.get("tool_name", "")
            args = action.get("arguments", {})
            click.echo(click.style(f"-> Proposed Tool: {tool_name}", fg="yellow", bold=True))

            is_mutating = tool_name in ("edit_file_block", "write_file", "sandboxed_bash")

            # Terminal Diff & Inspection
            if tool_name == "edit_file_block":
                diff_preview = self._format_diff_preview(
                    args.get("path", ""),
                    args.get("search_block", ""),
                    args.get("replace_block", ""),
                )
                click.echo(f"\n{diff_preview}\n")
            elif tool_name == "write_file":
                click.echo(click.style(f"  Writing file: {args.get('path')} ({len(args.get('content', ''))} characters)", fg="yellow"))
            elif tool_name == "sandboxed_bash":
                click.echo(click.style(f"  Command: $ {args.get('command')}", fg="yellow"))

            # Interactive Phylax Approval Gate
            approval_token: Optional[str] = None
            if is_mutating and not self.auto_approve:
                self.state_machine.transition(
                    AgentSemanticState.AWAITING_APPROVAL,
                    reason=f"Phylax gating approval for {tool_name}",
                    metadata={"tool": tool_name, "args": args},
                )
                approved = click.confirm(
                    click.style(f"[Phylax Gate] Allow execution of '{tool_name}'?", bold=True, fg="yellow"),
                    default=True,
                )
                if not approved:
                    click.echo(click.style("[-] Tool execution rejected by operator.", fg="red"))
                    current_prompt = f"Tool '{tool_name}' was rejected by the operator. Please propose an alternative approach."
                    continue

                # Obtain single-use approval token
                _, app_req = self.tool_registry.phylax_adapter.is_action_permitted(
                    capability=tool_name,
                    target=args.get("path", tool_name),
                    arguments=args,
                )
                if app_req:
                    self.tool_registry.phylax_adapter.resolve_approval(app_req["approval_id"], "allow_once")
                    approval_token = app_req["approval_id"]

            elif is_mutating and self.auto_approve:
                _, app_req = self.tool_registry.phylax_adapter.is_action_permitted(
                    capability=tool_name,
                    target=args.get("path", tool_name),
                    arguments=args,
                )
                if app_req:
                    self.tool_registry.phylax_adapter.resolve_approval(app_req["approval_id"], "allow_once")
                    approval_token = app_req["approval_id"]

            # Milestone checkpoint before mutation
            if is_mutating:
                active_checkpoint = self.checkpointer.create_checkpoint(f"Pre-{tool_name}-step-{step}")
                self.state_machine.transition(AgentSemanticState.MUTATING, reason=f"Executing {tool_name}")

            # Execute Tool
            tool_req = ToolCallRequest(tool_id=tool_name, arguments=args, approval_id=approval_token)
            result = self.tool_registry.execute(tool_req)

            status_color = "green" if result.success else "red"
            status_text = "SUCCESS" if result.success else "FAILED"
            click.echo(click.style(f"<- Result [{status_text}]:", fg=status_color))

            output_summary = ""
            if result.output is not None:
                out_str = str(result.output)
                if len(out_str) > 1000:
                    h = hashlib.sha256(out_str.encode("utf-8")).hexdigest()[:16]
                    artifacts_dir = self.workspace_root / ".metis" / "artifacts"
                    try:
                        artifacts_dir.mkdir(parents=True, exist_ok=True)
                        (artifacts_dir / f"tool_out_{h}.txt").write_text(out_str, encoding="utf-8")
                        out_str = (
                            out_str[:500]
                            + f"\n... [truncated {len(out_str) - 1000} bytes; full output preserved in .metis/artifacts/tool_out_{h}.txt] ...\n"
                            + out_str[-500:]
                        )
                    except Exception:
                        out_str = out_str[:500] + f"\n... [truncated {len(out_str) - 1000} bytes] ...\n" + out_str[-500:]
                click.echo(out_str)
                output_summary = out_str
            if result.error:
                click.echo(click.style(f"Error: {result.error}", fg="red"))
                output_summary += f"\nError: {result.error}"

            # Append structured turn to turn_history
            turn_history.append(
                CodingTurnState(
                    step=step,
                    tool_name=tool_name,
                    arguments=args,
                    result=result,
                    passed_validation=result.success,
                    error=result.error,
                    semantic_state=self.state_machine.current_state,
                )
            )

            # Optional Verification Hook
            if is_mutating and self.test_command and result.success:
                click.echo(click.style(f"-> Running Verification: `{self.test_command}`...", fg="cyan"))
                self.state_machine.transition(AgentSemanticState.VERIFYING, reason="Running verification test command")
                from kybernetes.tools.sandboxed_bash import SafeSandboxedBash
                runner = SafeSandboxedBash(allowed_roots=[self.workspace_root])
                v_res = runner.execute(self.test_command, cwd=str(self.workspace_root), timeout_seconds=60)

                if v_res["exit_code"] != 0:
                    click.echo(click.style(f"[✗] Verification failed (exit code {v_res['exit_code']})! Rolling back checkpoint...", fg="red", bold=True))
                    self.checkpointer.rollback(active_checkpoint)
                    backtracks += 1
                    current_prompt = (
                        f"Step {step} resulted in test failure with command `{self.test_command}`:\n"
                        f"{v_res['stderr'] or v_res['stdout']}\n"
                        "Working tree has been rolled back. Please analyze the error and try a different approach."
                    )
                    continue
                else:
                    click.echo(click.style("[✓] Verification passed cleanly!", fg="green"))
                    self.checkpointer.commit_milestone(f"Verified milestone: {goal[:40]} (step {step})")

            current_prompt = f"Tool '{tool_name}' executed with result:\n{output_summary}\n\nWhat is your next step?"

        click.echo(click.style(f"\n[!] Maximum steps ({self.max_steps}) reached without full completion.", fg="yellow"))
        return CodingSessionResult(
            session_id=f"mss-term-{int(time.time())}",
            success=False,
            status=TurnOutcomeStatus.FAILED,
            total_steps=self.max_steps,
            history=turn_history,
            final_message="Maximum steps reached",
            backtracks_executed=backtracks,
            final_state=self.state_machine.current_state,
        )
