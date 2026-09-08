"""DIY Agent Harness — demonstrating the 9 pillars of LLM agent infrastructure."""

from .core import Agent
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
from .permissions import CostLimitError, GuardrailsEngine, MaxIterationsError, PermissionManager
from .providers import AnthropicProvider, LLMProvider, make_provider
from .runtime import AgentRuntime
from .sandbox import Sandbox
from .skills import Skill, SkillManager
from .streaming import StreamEvent, StreamEventType, StreamingAgent
from .tools import ToolNotFoundError, ToolRegistry
from .types import (
    AgentConfig,
    CheckpointData,
    PermissionLevel,
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
    # Core
    "Agent",
    "AgentConfig",
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
    # Memory
    "ConversationMemory",
    "PersistentMemory",
    "LongTermMemory",
    "AgentMemory",
    "VectorMemoryStore",
    "InMemoryVectorStore",
    "SessionState",
    "TokenUsage",
    # Permissions
    "PermissionManager",
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
    "StreamingAgent",
    "StreamEvent",
    "StreamEventType",
    # Sandbox
    "Sandbox",
    "SandboxConfig",
    "SandboxResult",
    # Runtime
    "AgentRuntime",
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

try:
    from .evals import evaluate_agent
except ImportError:
    pass
