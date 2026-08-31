"""Built-in tools for the agent harness."""

from .bash import register_bash_tools
from .filesystem import register_filesystem_tools
from .memory import register_memory_tools
from .web import register_web_tools


def register_all_tools(registry, sandbox=None, long_term=None, agent_memory=None, vector_store=None) -> None:
    """Convenience: register all built-in tools."""
    register_filesystem_tools(registry)
    register_bash_tools(registry, sandbox=sandbox)
    register_web_tools(registry)
    register_memory_tools(
        registry,
        long_term=long_term,
        agent_memory=agent_memory,
        vector_store=vector_store,
    )
