"""Extension system for the agent harness.

An Extension is a self-contained unit of pluggable behaviour that wires
itself into an agent at construction time. Extensions can register tools,
subscribe to hooks, install middleware, mutate the system prompt, and
publish per-session cleanup via teardown().

Usage:
    from harnessx import Agent, AgentConfig
    from harnessx.extensions import ResultSpillExtension

    agent = Agent(
        config=AgentConfig(...),
        extensions=[ResultSpillExtension(threshold_bytes=32_000)],
    )
    try:
        await agent.run("...")
    finally:
        await agent.aclose()  # fires teardown() on each extension
"""

from .base import Extension, ExtensionContext

__all__ = ["Extension", "ExtensionContext"]

try:
    from .result_spill import ResultSpillExtension  # noqa: F401
    __all__.append("ResultSpillExtension")
except ImportError:
    pass

try:
    from .langsmith import LangSmithExtension  # noqa: F401
    __all__.append("LangSmithExtension")
except ImportError:
    pass

