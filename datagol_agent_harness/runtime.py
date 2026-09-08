"""Agent Runtime: full lifecycle management for agents.

The runtime wraps the Agent, not the other way around. The Agent is a pure
agentic loop; the runtime adds lifecycle concerns: initialization, checkpointing,
pause/resume, cleanup, and resource tracking.

You can use Agent without Runtime for simple cases, or wrap it in Runtime
for production-grade lifecycle management.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from .core import Agent
from .hooks import HookContext, HookEvent, HookManager, MiddlewarePipeline
from .memory import ConversationMemory, PersistentMemory
from .permissions import PermissionManager
from .providers import LLMProvider
from .sandbox import Sandbox
from .tools import ToolRegistry
from .types import (
    AgentConfig,
    CheckpointData,
    RuntimeConfig,
    RuntimeState,
    RuntimeStatus,
    SessionState,
    TokenUsage,
)


class AgentRuntime:
    """Full agent lifecycle management: init, execute, checkpoint, pause/resume, cleanup.

    Usage:
        runtime = AgentRuntime(
            agent_config=AgentConfig(system_prompt="You are helpful."),
            runtime_config=RuntimeConfig(storage_dir=".sessions"),
        )

        session_id = await runtime.start()
        result = await runtime.execute("Hello!")
        await runtime.checkpoint()
        await runtime.stop()

        # Later: resume
        runtime2 = AgentRuntime(...)
        await runtime2.resume(session_id)
        result = await runtime2.execute("Continue our conversation.")
    """

    def __init__(
        self,
        agent_config: AgentConfig | None = None,
        runtime_config: RuntimeConfig | None = None,
        tool_registrar: Callable[[ToolRegistry], None] | None = None,
        sandbox: Sandbox | None = None,
        hooks: HookManager | None = None,
        middleware: MiddlewarePipeline | None = None,
        permissions: PermissionManager | None = None,
        provider: LLMProvider | None = None,
        extensions: list[Any] | None = None,
    ) -> None:
        self.agent_config = agent_config or AgentConfig()
        self.runtime_config = runtime_config or RuntimeConfig()
        self.tool_registrar = tool_registrar
        self.sandbox = sandbox
        self.hooks = hooks or HookManager()
        self.middleware = middleware
        self.permissions = permissions
        self.provider = provider
        self.extensions = extensions

        self._agent: Agent | None = None
        self._session_id: str = ""
        self._state: RuntimeState = RuntimeState.INITIALIZING
        self._started_at: float = 0.0
        self._iteration_count: int = 0
        self._checkpoints: list[str] = []
        self._storage = PersistentMemory(self.runtime_config.storage_dir)

    # ── Phase 1: Initialization ──────────────────────────────────────────

    async def start(self, session_id: str | None = None) -> str:
        """Initialize the runtime and create a new agent session.

        Returns the session_id for tracking.
        """
        self._session_id = session_id or str(uuid.uuid4())
        self._started_at = time.time()
        self._state = RuntimeState.INITIALIZING

        # Create session directory
        session_dir = os.path.join(self.runtime_config.storage_dir, self._session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Create the Agent with all dependencies
        self._agent = Agent(
            config=self.agent_config,
            provider=self.provider,
            tools=ToolRegistry(),
            memory=ConversationMemory(max_result_chars=self.agent_config.max_result_chars),
            permissions=self.permissions or PermissionManager(),
            hooks=self.hooks,
            middleware=self.middleware or MiddlewarePipeline(),
            sandbox=self.sandbox,
            extensions=self.extensions,
        )
        self._agent._session_id = self._session_id

        # Register tools
        if self.tool_registrar:
            self.tool_registrar(self._agent.tools)

        self._state = RuntimeState.RUNNING

        await self.hooks.emit(
            HookEvent.AGENT_START,
            HookContext(
                event=HookEvent.AGENT_START,
                agent=self._agent,
                data={"session_id": self._session_id, "phase": "runtime_start"},
            ),
        )

        return self._session_id

    # ── Phase 2: Execution ───────────────────────────────────────────────

    async def execute(self, user_message: str) -> str:
        """Execute a user message within the runtime context.

        Wraps Agent.run() with resource tracking and auto-checkpointing.
        """
        if self._agent is None:
            raise RuntimeError("Runtime not started. Call start() first.")
        if self._state != RuntimeState.RUNNING:
            raise RuntimeError(f"Runtime is {self._state.value}, not running.")

        # Check session duration
        if self.runtime_config.max_session_duration_seconds:
            elapsed = time.time() - self._started_at
            if elapsed > self.runtime_config.max_session_duration_seconds:
                self._state = RuntimeState.ERROR
                raise RuntimeError(
                    f"Session duration ({elapsed:.0f}s) exceeds limit "
                    f"({self.runtime_config.max_session_duration_seconds:.0f}s)"
                )

        start_iteration = self._agent.guardrails.iteration_count

        try:
            result = await self._agent.run(user_message)
        except Exception as e:
            self._state = RuntimeState.ERROR
            await self.hooks.emit(
                HookEvent.ERROR,
                HookContext(event=HookEvent.ERROR, agent=self._agent, data={"error": str(e)}),
            )
            raise

        new_iterations = self._agent.guardrails.iteration_count - start_iteration
        self._iteration_count += new_iterations

        # Auto-checkpoint at configured intervals
        if (
            self.runtime_config.checkpoint_interval > 0
            and self._iteration_count % self.runtime_config.checkpoint_interval == 0
        ):
            await self.checkpoint()

        return result

    # ── Phase 3: Checkpointing ───────────────────────────────────────────

    async def checkpoint(self) -> CheckpointData:
        """Snapshot current state for pause/resume and crash recovery."""
        if self._agent is None:
            raise RuntimeError("No agent to checkpoint.")

        timestamp = datetime.now(timezone.utc).isoformat()
        session_state = SessionState(
            session_id=self._session_id,
            messages=self._agent.memory.get_messages(),
            total_usage=self._agent.guardrails.total_usage,
            created_at=timestamp,
            updated_at=timestamp,
        )

        checkpoint = CheckpointData(
            session_state=session_state,
            agent_config_dict={
                "model": self.agent_config.model,
                "max_tokens": self.agent_config.max_tokens,
                "max_iterations": self.agent_config.max_iterations,
                "system_prompt": self.agent_config.system_prompt,
                "temperature": self.agent_config.temperature,
            },
            metadata={
                "runtime_state": self._state.value,
                "iteration_count": self._iteration_count,
                "uptime_seconds": time.time() - self._started_at,
            },
            timestamp=timestamp,
            iteration_count=self._iteration_count,
        )

        # Save to disk
        session_dir = os.path.join(self.runtime_config.storage_dir, self._session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Use a sortable filename
        safe_ts = timestamp.replace(":", "-").replace("+", "_")
        checkpoint_file = os.path.join(session_dir, f"checkpoint_{safe_ts}.json")

        data = {
            "session_state": {
                "session_id": session_state.session_id,
                "messages": session_state.messages,
                "total_usage": {
                    "input_tokens": session_state.total_usage.input_tokens,
                    "output_tokens": session_state.total_usage.output_tokens,
                    "cache_creation_input_tokens": session_state.total_usage.cache_creation_input_tokens,
                    "cache_read_input_tokens": session_state.total_usage.cache_read_input_tokens,
                },
                "created_at": session_state.created_at,
                "updated_at": session_state.updated_at,
            },
            "agent_config": checkpoint.agent_config_dict,
            "metadata": checkpoint.metadata,
            "timestamp": checkpoint.timestamp,
            "iteration_count": checkpoint.iteration_count,
        }

        with open(checkpoint_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self._checkpoints.append(checkpoint_file)

        # Prune old checkpoints
        max_cp = self.runtime_config.max_checkpoints
        if len(self._checkpoints) > max_cp:
            to_remove = self._checkpoints[:-max_cp]
            for path in to_remove:
                try:
                    os.remove(path)
                except OSError:
                    pass
            self._checkpoints = self._checkpoints[-max_cp:]

        await self.hooks.emit(
            HookEvent.CHECKPOINT,
            HookContext(
                event=HookEvent.CHECKPOINT,
                agent=self._agent,
                data={"checkpoint_file": checkpoint_file},
            ),
        )

        return checkpoint

    # ── Phase 4: Pause / Resume ──────────────────────────────────────────

    async def pause(self) -> str:
        """Save checkpoint, release resources, mark session as paused.

        Returns the session_id for later resume.
        """
        if self._agent is None:
            raise RuntimeError("No agent to pause.")

        await self.checkpoint()
        self._state = RuntimeState.PAUSED

        # Release the API client
        await self._agent.client.close()

        return self._session_id

    async def resume(self, session_id: str) -> str:
        """Load latest checkpoint and resume execution.

        The agent picks up where it left off — same conversation history.
        """
        self._session_id = session_id
        self._started_at = time.time()

        # Find latest checkpoint
        session_dir = os.path.join(self.runtime_config.storage_dir, session_id)
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"No session directory found: {session_dir}")

        checkpoint_files = sorted(
            [f for f in os.listdir(session_dir) if f.startswith("checkpoint_") and f.endswith(".json")]
        )
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {session_dir}")

        latest = os.path.join(session_dir, checkpoint_files[-1])
        with open(latest) as f:
            data = json.load(f)

        # Restore agent
        session_data = data["session_state"]
        usage_data = session_data.get("total_usage", {})

        self._agent = Agent(
            config=self.agent_config,
            provider=self.provider,
            tools=ToolRegistry(),
            memory=ConversationMemory(max_result_chars=self.agent_config.max_result_chars),
            permissions=self.permissions or PermissionManager(),
            hooks=self.hooks,
            middleware=self.middleware or MiddlewarePipeline(),
            sandbox=self.sandbox,
            extensions=self.extensions,
        )
        self._agent._session_id = session_id

        # Restore conversation history
        self._agent.memory.set_messages(session_data.get("messages", []))

        # Restore usage
        self._agent.guardrails._total_usage = TokenUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0),
        )

        # Restore iteration count
        self._iteration_count = data.get("iteration_count", 0)
        self._agent.guardrails._iteration_count = self._iteration_count

        # Register tools
        if self.tool_registrar:
            self.tool_registrar(self._agent.tools)

        self._state = RuntimeState.RUNNING

        await self.hooks.emit(
            HookEvent.AGENT_START,
            HookContext(
                event=HookEvent.AGENT_START,
                agent=self._agent,
                data={"session_id": session_id, "phase": "runtime_resume"},
            ),
        )

        return session_id

    # ── Phase 5: Cleanup ─────────────────────────────────────────────────

    async def stop(self) -> RuntimeStatus:
        """Save final checkpoint, close connections, record final stats."""
        if self._agent is not None and self._state == RuntimeState.RUNNING:
            try:
                await self.checkpoint()
            except Exception:
                pass

            try:
                await self._agent.client.close()
            except Exception:
                pass

        status = self.get_status()
        self._state = RuntimeState.STOPPED

        await self.hooks.emit(
            HookEvent.AGENT_END,
            HookContext(
                event=HookEvent.AGENT_END,
                agent=self._agent,
                data={"phase": "runtime_stop", "status": status},
            ),
        )

        return status

    # ── Status ───────────────────────────────────────────────────────────

    def get_status(self) -> RuntimeStatus:
        uptime = time.time() - self._started_at if self._started_at else 0.0
        usage = self._agent.guardrails.total_usage if self._agent else TokenUsage()

        return RuntimeStatus(
            session_id=self._session_id,
            state=self._state,
            started_at=datetime.fromtimestamp(self._started_at, tz=timezone.utc).isoformat()
            if self._started_at
            else "",
            uptime_seconds=uptime,
            iterations_completed=self._iteration_count,
            total_usage=usage,
            checkpoints=list(self._checkpoints),
        )

    @property
    def agent(self) -> Agent | None:
        return self._agent

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def state(self) -> RuntimeState:
        return self._state
