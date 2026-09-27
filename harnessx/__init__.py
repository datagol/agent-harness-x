"""HarnessX — composable Python agents, tools, and durable execution."""

import importlib.util as _importlib_util
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError, version as _dist_version

try:
    __version__ = _dist_version("harnessx")
except _PackageNotFoundError:  # bare checkout without an installed distribution
    __version__ = "0.0.0"

from .core import Agent
from .subagents import SubAgent
from .extensions import Extension, ExtensionContext, LangSmithExtension, ResultSpillExtension
from .mcp import MCPManager, MCPServerConfig, MCPToolInfo
from .hooks import HookContext, HookEvent, HookManager, Middleware, MiddlewarePipeline
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
from .providers import AnthropicProvider, LLMProvider, make_provider
from .runtime import AgentRuntime, RunHandle
from .recorder import IncidentRecorder, ExportPolicy, BundleLimits, IncidentError, VerificationReport, Playback, export_incident
from ._journal import RecordingError
from .backends.temporal import TemporalBackend, RedisEvents
from .artifacts import S3ArtifactStore
from .backends import SQLiteBackend, PostgresBackend, SchemaError, StorageError, SessionBusyError, LeaseLostError
from .sandbox import Sandbox
from .skills import Skill, SkillManager
from .execution import RunEvent, RunEventType, RunResult, RunStatus, RunStream, RunFailure, PendingTool, ReplayPolicy, current_tool_context
from .registry import AgentRef, AgentRegistry, agents
from .tools import ToolNotFoundError, ToolRegistry, normalize_tool_registry
from .types import (
    AgentConfig,
    CheckpointData,
    PermissionLevel,
    PromptCacheHint,
    PromptCachePolicy,
    ProviderResponse,
    Role,
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
    ToolResult,
)

__all__ = [
    "__version__",
    # Core
    "Agent",
    "SubAgent",
    "AgentConfig",
    "PromptCachePolicy",
    "PromptCacheHint",
    "Role",
    "StopReason",
    "ProviderResponse",
    "StreamChunk",
    # Tools
    "ToolRegistry",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ToolNotFoundError",
    "PermissionLevel",
    "normalize_tool_registry",
    # Memory
    "ConversationMemory",
    "Message", "ContentBlock",
    "PersistentMemory",
    "LongTermMemory",
    "AgentMemory",
    "VectorMemoryStore",
    "InMemoryVectorStore",
    "SessionState",
    "TokenUsage",
    # Permissions
    "PermissionManager",
    "CliPermissionManager",
    "GuardrailsEngine",
    "MaxIterationsError",
    "CostLimitError",
    # Hooks
    "HookManager",
    "HookEvent",
    "HookContext",
    "Middleware",
    "MiddlewarePipeline",
    # Streaming
    "RunEvent", "RunEventType", "RunResult", "RunStatus", "RunStream",
    "RunFailure", "PendingTool",
    "ReplayPolicy", "current_tool_context", "AgentRef", "AgentRegistry", "agents",
    # Sandbox
    "Sandbox",
    "SandboxConfig",
    "SandboxResult",
    # Runtime
    "AgentRuntime", "RunHandle", "TemporalBackend", "RedisEvents", "S3ArtifactStore", "SQLiteBackend", "PostgresBackend",
    "SchemaError", "StorageError", "SessionBusyError", "LeaseLostError",
    "IncidentRecorder", "ExportPolicy", "BundleLimits", "IncidentError",
    "VerificationReport", "Playback", "RecordingError",
    "export_incident",
    "RuntimeConfig",
    "RuntimeState",
    "RuntimeStatus",
    "CheckpointData",
    # MCP
    "MCPManager",
    "MCPServerConfig",
    "MCPToolInfo",
    # Skills
    "Skill",
    "SkillManager",
    # Providers
    "LLMProvider",
    "AnthropicProvider",
    "make_provider",
    # Extensions
    "Extension",
    "ExtensionContext",
    "LangSmithExtension",
    "ResultSpillExtension",
    # Evals
    "evaluate_agent",
]

# Keep optional evaluation imports (and their environment setup) out of core startup.

if _importlib_util.find_spec("langsmith") is None:
    __all__.remove("evaluate_agent")


def __getattr__(name):
    if name == "evaluate_agent":
        from .evals import evaluate_agent
        globals()[name] = evaluate_agent
        return evaluate_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
