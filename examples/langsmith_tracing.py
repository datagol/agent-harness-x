"""LangSmith tracing example: full lifecycle observability for DataGOL agents.

Demonstrates:
  - Automatic LangSmith run tree creation for agent turns
  - Nested child spans for LLM calls with token usage
  - Child spans for tool executions with arguments and outputs
  - Advanced manual delegation for a child with its own tracing extension

For ordinary delegation, prefer Agent(subagents=[SubAgent(...)]); see multi_agent.py.

Prerequisites:
  pip install -e '.[langsmith]'
  export ANTHROPIC_API_KEY=sk-ant-...
  export LANGSMITH_API_KEY=lsv2_pt_...
  export LANGSMITH_PROJECT="harnessx-demo"

Run:
  python -m examples.langsmith_tracing
"""

import asyncio
import os

from harnessx import Agent, AgentConfig, LangSmithExtension, PermissionLevel
from examples._calculator import calculate
from examples._console import completed_output


async def run_specialist(topic: str) -> str:
    """Specialist sub-agent.

    Because LangSmithExtension uses contextvars for parent span tracking,
    this sub-agent automatically nests under the orchestrator's tool call span!
    """
    specialist = Agent(
        config=AgentConfig(
            model="claude-sonnet-4-6",
            system_prompt="You are a concise research specialist. Give 2-3 bullet points.",
            max_iterations=5,
        ),
        extensions=[
            LangSmithExtension(
                run_name="Research Specialist",
                tags=["specialist"],
            )
        ],
    )
    async with specialist:
        return completed_output(
            await specialist.run(f"Summarize key aspects of: {topic}")
        )


async def main() -> None:
    print("=" * 60)
    print("HarnessX — LangSmith Observability Demo")
    print("=" * 60)

    # 1. Initialize LangSmith extension
    project = os.getenv("LANGSMITH_PROJECT", "harnessx-demo")
    ls_ext = LangSmithExtension(
        project_name=project,
        run_name="Main Assistant",
        tags=["demo", "cli"],
        metadata={"environment": "development"},
    )

    # 2. Create the primary agent with LangSmith tracing enabled
    agent = Agent(
        config=AgentConfig(
            model="claude-sonnet-4-6",
            system_prompt=(
                "You are a helpful assistant. Use tools when helpful. "
                "Delegate in-depth research to the research_topic tool."
            ),
            max_iterations=10,
        ),
        extensions=[ls_ext],
    )

    async with agent:
        # 3. Register tools
        agent.tools.register_tool(
            calculate, permission=PermissionLevel.ALLOW, replay_policy="safe"
        )

        @agent.tools.register(permission=PermissionLevel.ALLOW)
        async def research_topic(topic: str) -> str:
            """Delegate research to a specialist sub-agent.

            Args:
                topic: The subject to research.
            """
            print(f"\n[Orchestrator] Delegating '{topic}' to Research Specialist...")
            return await run_specialist(topic)

        # 4. Run a turn that invokes both a tool and sub-agent
        prompt = "What is 144 / 12, and can you research the difference between OLTP and OLAP?"
        print(f"\nUser: {prompt}\n")
        print("Running agent (tracing to LangSmith)...")

        response = completed_output(await agent.run(prompt))

        print("\nAgent Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        print(f"\nConfigured LangSmith project: '{project}'")
        print("View your traces at: https://smith.langchain.com")

        # Flush traces cleanly on exit


if __name__ == "__main__":
    asyncio.run(main())
