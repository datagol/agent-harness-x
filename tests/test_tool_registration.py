"""Unit tests for the enhanced built-in tool registration flow."""

from __future__ import annotations

import os
import tempfile
import unittest

from harnessx import (
    Agent,
    AgentConfig,
    PermissionLevel,
    Agent,
    ToolRegistry,
    normalize_tool_registry,
)
from harnessx.builtin import (
    fetch_url,
    generate_file,
    list_directory,
    read_file,
    recall_memories,
    register_all_tools,
    register_bash_tools,
    register_filesystem_tools,
    register_memory_tools,
    register_web_tools,
    run_bash,
    save_memory,
    write_file,
)
from harnessx.types import ToolCall


def sample_tool(text: str, count: int = 1) -> str:
    """Repeats text multiple times.

    This multi-line description contains important instructions
    that should be fully preserved for the language model.

    Args:
        text: Input text string to repeat.
        count: Number of repetitions.
    """
    return text * count


class TestEnhancedToolRegistration(unittest.IsolatedAsyncioTestCase):
    def test_standalone_imports(self):
        """Verify all built-in tools are directly importable and callable."""
        self.assertTrue(callable(read_file))
        self.assertTrue(callable(write_file))
        self.assertTrue(callable(list_directory))
        self.assertTrue(callable(generate_file))
        self.assertTrue(callable(run_bash))
        self.assertTrue(callable(fetch_url))
        self.assertTrue(callable(save_memory))
        self.assertTrue(callable(recall_memories))

    def test_register_tool_direct_call(self):
        """ToolRegistry.register_tool should register plain functions without @decorator."""
        reg = ToolRegistry()
        tool_def = reg.register_tool(sample_tool, permission=PermissionLevel.ALLOW)
        self.assertEqual(tool_def.name, "sample_tool")
        self.assertEqual(tool_def.permission_level, PermissionLevel.ALLOW)
        self.assertTrue(reg.has_tool("sample_tool"))

        # Verify multi-line docstring preservation
        self.assertIn("This multi-line description contains important instructions", tool_def.description)
        # Verify Args: was not leaked into description
        self.assertNotIn("Args:", tool_def.description)

        # Verify schema parameters
        schema = tool_def.input_schema
        self.assertEqual(schema["type"], "object")
        self.assertIn("text", schema["properties"])
        self.assertIn("count", schema["properties"])
        self.assertEqual(schema["properties"]["text"]["description"], "Input text string to repeat.")

    def test_load_builtin_bundles(self):
        """ToolRegistry.load_builtin should support selective bundle loading."""
        reg = ToolRegistry()
        reg.load_builtin("filesystem", include=["read_file", "list_directory"])
        self.assertEqual(sorted(reg.list_tools()), ["list_directory", "read_file"])

        # Add bash with permission override
        reg.load_builtin("bash", permission=PermissionLevel.ALLOW)
        self.assertTrue(reg.has_tool("run_bash"))
        self.assertEqual(reg.get_tool("run_bash").permission_level, PermissionLevel.ALLOW)

        # Unregister works
        self.assertTrue(reg.unregister("run_bash"))
        self.assertFalse(reg.has_tool("run_bash"))

    def test_load_builtin_single_tool(self):
        """ToolRegistry.load_builtin can load a single tool by name."""
        reg = ToolRegistry()
        loaded = reg.load_builtin("read_file")
        self.assertEqual(loaded, ["read_file"])
        self.assertTrue(reg.has_tool("read_file"))

    def test_load_builtin_all(self):
        """ToolRegistry.load_builtin('all') should load all built-in tools."""
        reg = ToolRegistry()
        reg.load_builtin("all", permission=PermissionLevel.ALLOW)
        tools = reg.list_tools()
        self.assertIn("read_file", tools)
        self.assertIn("write_file", tools)
        self.assertIn("list_directory", tools)
        self.assertIn("run_bash", tools)
        self.assertIn("fetch_url", tools)
        self.assertIn("save_memory", tools)
        self.assertIn("recall_memories", tools)

        # Verify permission override applied
        for name in tools:
            self.assertEqual(reg.get_tool(name).permission_level, PermissionLevel.ALLOW)

    def test_normalize_tool_registry_declarative(self):
        """normalize_tool_registry handles bundle strings, function objects, and registries."""
        # From list of bundle names and callables
        reg = normalize_tool_registry(["filesystem", sample_tool])
        self.assertTrue(reg.has_tool("read_file"))
        self.assertTrue(reg.has_tool("write_file"))
        self.assertTrue(reg.has_tool("sample_tool"))

        # From existing registry (passthrough)
        self.assertIs(normalize_tool_registry(reg), reg)

        # From None
        empty_reg = normalize_tool_registry(None)
        self.assertEqual(empty_reg.list_tools(), [])

    def test_agent_declarative_tools(self):
        """Agent constructor accepts list of bundle strings and tool callables."""
        agent = Agent(
            config=AgentConfig(provider="anthropic"),
            tools=["bash", sample_tool],
        )
        self.assertTrue(agent.tools.has_tool("run_bash"))
        self.assertTrue(agent.tools.has_tool("sample_tool"))

    def test_streaming_agent_declarative_tools(self):
        """Agent constructor accepts list of bundle strings and tool callables."""
        agent = Agent(
            config=AgentConfig(provider="anthropic"),
            tools=["filesystem", sample_tool],
        )
        self.assertTrue(agent.tools.has_tool("read_file"))
        self.assertTrue(agent.tools.has_tool("write_file"))
        self.assertTrue(agent.tools.has_tool("sample_tool"))

    async def test_filesystem_base_path_sandboxing(self):
        """Filesystem tools restrict access when base_path is provided."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = ToolRegistry()
            register_filesystem_tools(reg, base_path=tmp_dir, permission=PermissionLevel.ALLOW)

            # Write inside sandbox
            write_res = await reg.execute(
                ToolCall(id="1", name="write_file", input={"path": "hello.txt", "content": "world"})
            )
            self.assertFalse(write_res.is_error)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "hello.txt")))

            # Attempt directory traversal outside sandbox
            escape_res = await reg.execute(
                ToolCall(id="2", name="write_file", input={"path": "../escaped.txt", "content": "bad"})
            )
            self.assertTrue(escape_res.is_error)
            self.assertIn("PermissionError", escape_res.content)
            self.assertFalse(os.path.exists(os.path.join(tmp_dir, "../escaped.txt")))


if __name__ == "__main__":
    unittest.main()
