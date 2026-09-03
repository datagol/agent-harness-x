"""Multi-agent orchestration: an orchestrator delegates to specialist agents.

The orchestrator has delegation tools that spawn sub-agents with specific
system prompts and tool sets. Each sub-agent runs to completion and returns
its result to the orchestrator.

Run: python -m examples.multi_agent
"""

import asyncio

from datagol_agent_harness import Agent, AgentConfig, PermissionLevel
from examples._console import (
    get_user_input,
    print_banner,
    print_delegation,
    print_error,
    print_response,
    print_status,
)
from datagol_agent_harness.builtin.bash import register_bash_tools
from datagol_agent_harness.builtin.filesystem import register_filesystem_tools
from datagol_agent_harness.builtin.web import register_web_tools


async def run_code_review_agent(code: str, instructions: str) -> str:
    """Run a specialist code review agent."""
    reviewer = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are a code review specialist. Review the provided code "
                "and give detailed, constructive feedback on quality, bugs, "
                "security issues, and potential improvements. Be specific "
                "and reference line numbers when possible."
            ),
            max_iterations=5,
        )
    )
    return await reviewer.run(
        f"Review this code:\n```\n{code}\n```\n\nFocus on: {instructions}"
    )


async def run_research_agent(query: str) -> str:
    """Run a specialist research agent with web access."""
    researcher = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are a research specialist. Find accurate, up-to-date "
                "information to answer the query. Cite sources when possible."
            ),
            max_iterations=10,
        )
    )
    register_web_tools(researcher.tools)
    researcher.permissions.set_permission("fetch_url", PermissionLevel.ALLOW)
    return await researcher.run(query)


async def run_file_analysis_agent(path: str, question: str) -> str:
    """Run a specialist file analysis agent."""
    analyst = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are a file analysis specialist. Read and analyze files "
                "to answer questions about their contents, structure, and purpose."
            ),
            max_iterations=10,
        )
    )
    register_filesystem_tools(analyst.tools)
    register_bash_tools(analyst.tools)
    analyst.permissions.set_permission("read_file", PermissionLevel.ALLOW)
    analyst.permissions.set_permission("list_directory", PermissionLevel.ALLOW)
    analyst.permissions.set_permission("run_bash", PermissionLevel.ALLOW)
    return await analyst.run(f"Analyze '{path}': {question}")


async def main():
    # Create the orchestrator
    orchestrator = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are an orchestrator agent managing a team of specialists.\n\n"
                "Available specialists:\n"
                "- delegate_code_review: Reviews code for quality, bugs, security\n"
                "- delegate_research: Searches the web for information\n"
                "- delegate_file_analysis: Reads and analyzes files on disk\n\n"
                "You also have direct access to filesystem tools.\n\n"
                "For complex tasks, break them into subtasks and delegate to "
                "the appropriate specialist. Synthesize their results into a "
                "coherent response for the user."
            ),
            max_iterations=20,
        )
    )

    register_filesystem_tools(orchestrator.tools)
    orchestrator.permissions.set_permission("read_file", PermissionLevel.ALLOW)
    orchestrator.permissions.set_permission("list_directory", PermissionLevel.ALLOW)

    # Register delegation tools
    @orchestrator.tools.register(permission=PermissionLevel.ALLOW)
    async def delegate_code_review(code: str, instructions: str = "General review") -> str:
        """Delegate code review to a specialist agent.

        Args:
            code: The code to review.
            instructions: Specific aspects to focus on.
        """
        print_delegation("Orchestrator", "Code Review Agent")
        result = await run_code_review_agent(code, instructions)
        print_delegation("Code Review Agent", "Orchestrator")
        return result

    @orchestrator.tools.register(permission=PermissionLevel.ASK)
    async def delegate_research(query: str) -> str:
        """Delegate research to a specialist agent with web access.

        Args:
            query: The research question.
        """
        print_delegation("Orchestrator", "Research Agent")
        result = await run_research_agent(query)
        print_delegation("Research Agent", "Orchestrator")
        return result

    @orchestrator.tools.register(permission=PermissionLevel.ALLOW)
    async def delegate_file_analysis(path: str, question: str) -> str:
        """Delegate file analysis to a specialist agent.

        Args:
            path: Path to the file or directory to analyze.
            question: What to find out about the file(s).
        """
        print_delegation("Orchestrator", "File Analysis Agent")
        result = await run_file_analysis_agent(path, question)
        print_delegation("File Analysis Agent", "Orchestrator")
        return result

    print_banner(
        "DataGOL Agent Harness — Multi-Agent Orchestrator",
        subtitle="Specialists: code review, research, file analysis",
        commands={"quit": "Exit"},
    )

    while True:
        user_input = get_user_input()
        if user_input is None or user_input.lower() == "quit":
            break
        if not user_input:
            continue

        try:
            response = await orchestrator.run(user_input)
            print_response(response)
        except Exception as e:
            print_error(e)

    print_status({"Session stats": orchestrator.guardrails.usage_summary})


if __name__ == "__main__":
    asyncio.run(main())
