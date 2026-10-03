"""HarnessX — composable Python agents, tools, and durable execution."""

import importlib.util as _importlib_util
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError, version as _dist_version

try:
    __version__ = _dist_version("harnessx")
except _PackageNotFoundError:  # bare checkout without an installed distribution
    __version__ = "0.0.0"

from .core import Agent
from .errors import (
    ConfigurationError,
    HarnessError,
    ResolutionError,
    RunAwaitingInput,
    RunCancelled, RunTruncated,
    RunError,
    RunFailed,
    RuntimeStateError,
    TransientToolError,
    UnknownExecutionKey,
)
from .subagents import SubAgent
from .extensions import Extension, ExtensionContext, LangSmithExtension, ResultSpillExtension
from .mcp import MCPManager, MCPServerConfig, MCPToolInfo
from .hooks import HookContext, HookEvent, HookManager, Middleware, MiddlewarePipeline, Registration
from .memory import (
    AgentMemory,
    ConversationMemory,
    InMemoryVectorStore,
    LongTermMemory,
    PersistentMemory,
    VectorMemoryStore,
)
from .messages import Message, ContentBlock
from .permissions import CliPermissionManager, CostLimitError, GuardrailsEngine, MaxIterationsError, PermissionManager
from .providers import AnthropicProvider, FallbackProvider, LLMProvider, make_provider, register_provider
from .runtime import AgentRuntime, RunHandle
from .recorder import IncidentRecorder, ExportPolicy, BundleLimits, IncidentError, VerificationReport, Playback, export_incident
from ._journal import RecordingError
from .backends.temporal import TemporalBackend, RedisEvents
from .artifacts import S3ArtifactStore
from .backends import SQLiteBackend, PostgresBackend, SchemaError, StorageError, SessionBusyError, LeaseLostError
from .sandbox import Sandbox
from .skills import Skill, SkillManager
from .execution import RunEvent, RunEventType, RunResult, RunStatus, RunStream, RunFailure, PendingTool, ToolApprovalRequired, ToolExecutionContext, current_tool_context
from .registry import AgentRef, AgentRegistry, agents
from .tools import ToolNotFoundError, ToolRegistry, normalize_tool_registry
from .types import (
    DEFAULT_TIMEOUT_SECONDS,
    AgentConfig,
    CheckpointData,
    Fallback,
    Limits,
    LoopGuard,
    PermissionLevel,
    PromptCacheHint,
    PromptCachePolicy,
    ProviderResponse,
    ReplayPolicy,
    RetryPolicy,
    RuntimeConfig,
    RuntimeState,
    RuntimeStatus,
    SandboxConfig,
    SandboxResult,
    SessionState,
    StopReason,
    StreamChunk,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolPolicy,
    ToolResult,
    ToolRetry,
)

__all__ = [
    "__version__",
    # Agent and configuration
    "Agent", "SubAgent", "AgentConfig", "Limits", "LoopGuard", "RetryPolicy", "PromptCachePolicy", "ToolPolicy", "ToolRetry",
    "PromptCacheHint", "DEFAULT_TIMEOUT_SECONDS", "StopReason", "ProviderResponse", "StreamChunk",
    # Providers
    "LLMProvider", "AnthropicProvider", "FallbackProvider", "Fallback", "make_provider", "register_provider",
    # Tools
    "ToolRegistry", "ToolDefinition", "ToolCall", "ToolResult", "ToolNotFoundError", "PermissionLevel",
    "ReplayPolicy", "ToolExecutionContext", "ToolApprovalRequired", "current_tool_context", "normalize_tool_registry",
    # Runs, results, streaming
    "RunEvent", "RunEventType", "RunResult", "RunStatus", "RunStream", "RunFailure", "PendingTool", "TokenUsage",
    # Errors
    "HarnessError", "ConfigurationError", "RuntimeStateError", "ResolutionError", "UnknownExecutionKey",
    "RunError", "RunFailed", "RunAwaitingInput", "RunCancelled", "RunTruncated", "TransientToolError",
    "MaxIterationsError", "CostLimitError",
    # Memory and sessions
    "ConversationMemory", "Message", "ContentBlock", "PersistentMemory", "LongTermMemory", "AgentMemory",
    "VectorMemoryStore", "InMemoryVectorStore", "SessionState",
    # Permissions and guardrails
    "PermissionManager", "CliPermissionManager", "GuardrailsEngine",
    # Hooks, middleware, extensions
    "HookManager", "HookEvent", "HookContext", "Registration", "Middleware", "MiddlewarePipeline",
    "Extension", "ExtensionContext", "LangSmithExtension", "ResultSpillExtension",
    # Skills
    "Skill", "SkillManager",
    # MCP
    "MCPManager", "MCPServerConfig", "MCPToolInfo",
    # Sandbox
    "Sandbox", "SandboxConfig", "SandboxResult",
    # Durable execution (also importable from harnessx.durable)
    "AgentRuntime", "RunHandle", "AgentRef", "AgentRegistry", "agents",
    "RuntimeConfig", "RuntimeState", "RuntimeStatus", "CheckpointData",
    "SQLiteBackend", "PostgresBackend", "TemporalBackend", "RedisEvents", "S3ArtifactStore",
    "SchemaError", "StorageError", "SessionBusyError", "LeaseLostError",
    # Flight recorder
    "IncidentRecorder", "ExportPolicy", "BundleLimits", "IncidentError",
    "VerificationReport", "Playback", "RecordingError", "export_incident",
    # Evals (lazy; require the langsmith extra)
    "evaluate_agent", "evaluate_agent_async",
]

# Keep optional evaluation imports (and their environment setup) out of core startup.

if _importlib_util.find_spec("langsmith") is None:
    __all__.remove("evaluate_agent")
    __all__.remove("evaluate_agent_async")


def __getattr__(name):
    if name in ("evaluate_agent", "evaluate_agent_async"):
        from . import evals
        value = getattr(evals, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
