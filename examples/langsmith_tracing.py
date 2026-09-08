"""LangSmith tracing example: full lifecycle observability for DataGOL agents.

Demonstrates:
  - Automatic LangSmith run tree creation for agent turns
  - Nested child spans for LLM calls with token usage
  - Child spans for tool executions with arguments and outputs
  - Seamless nested tracing for multi-agent delegation (sub-agents)

Prerequisites:
  export ANTHROPIC_API_KEY=sk-ant-...
  export LANGSMITH_API_KEY=lsv2_pt_...
  export LANGSMITH_PROJECT="datagol-agent-harness-demo"

Run:
  python -m examples.langsmith_tracing
"""

import asyncio
import os

from datagol_agent_harness import Agent, AgentConfig, LangSmithExtension, PermissionLevel


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
    return await specialist.run(f"Summarize key aspects of: {topic}")


async def main() -> None:
    print("=" * 60)
    print("DataGOL Agent Harness — LangSmith Observability Demo")
    print("=" * 60)

    # 1. Initialize LangSmith extension
    project = os.getenv("LANGSMITH_PROJECT", "datagol-agents-demo")
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

    # 3. Register tools
    @agent.tools.register(permission=PermissionLevel.ALLOW)
    def calculate(expression: str) -> str:
        """Safely evaluate simple arithmetic expressions.

        Args:
            expression: Arithmetic expression like '25 * 4 + 10'.
        """
        try:
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "Error: only basic arithmetic characters allowed"
            # eval safe for simple numbers
            return str(eval(expression, {"__builtins__": None}, {}))
        except Exception as e:
            return f"Error: {e}"

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

    response = await agent.run(prompt)

    print("\nAgent Response:")
    print("-" * 40)
    print(response)
    print("-" * 40)
    print(f"\nTrace sent to LangSmith project: '{project}'")
    print("View your traces at: https://smith.langchain.com")

    # Flush traces cleanly on exit
    await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
