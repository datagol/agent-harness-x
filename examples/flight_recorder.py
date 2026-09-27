"""Run a synthetic invoice analysis, then inspect its recording without a model.

Usage: python -m examples.flight_recorder --output /tmp/invoice-incident.hx
No API key or external service is used. The destination must not already exist.
"""

import argparse
import asyncio
import json
from pathlib import Path
import tempfile

from harnessx import (
    Agent,
    AgentRuntime,
    ExportPolicy,
    IncidentRecorder,
    Middleware,
    PermissionLevel,
    ProviderResponse,
    SQLiteBackend,
    ToolCall,
)
from harnessx.providers import LLMProvider


class FixtureProvider(LLMProvider):
    """A scripted fixture, not a production Datagol connector."""

    def __init__(self):
        self.calls = 0

    async def count_tokens(self, **kwargs):
        return 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("Injected model connection failure")
        if self.calls == 2:
            return ProviderResponse(
                tool_calls=[
                    ToolCall("overdue", "overdue_balance", {"account": "demo"})
                ],
                stop_reason="tool_use",
            )
        return ProviderResponse(text="Overdue balance: $1,500.")


class LabelFixtureOutput(Middleware):
    async def after_llm_call(self, response):
        if response.text:
            response.text += " (synthetic fixture)"
        return response


async def main(destination: Path):
    provider = FixtureProvider()
    tool_calls = []
    with tempfile.TemporaryDirectory(prefix="harnessx-recorder-") as directory:
        agent = Agent(provider=provider)
        agent.middleware.add(LabelFixtureOutput())

        @agent.tools.register(permission=PermissionLevel.ALLOW, replay_policy="safe")
        def overdue_balance(account: str) -> str:
            """Calculate the unpaid balance of synthetic overdue invoices."""
            tool_calls.append(account)
            invoices = [
                {"id": "inv-1", "amount": 1000, "overdue": True},
                {"id": "inv-2", "amount": 500, "overdue": True},
                {"id": "inv-3", "amount": 750, "overdue": False},
            ]
            return json.dumps(
                {
                    "account": account,
                    "overdue_balance": sum(
                        row["amount"] for row in invoices if row["overdue"]
                    ),
                    "invoices": invoices,
                }
            )

        backend = SQLiteBackend(str(Path(directory) / "runtime.db"))
        try:
            async with AgentRuntime(agent, backend=backend, recording=True) as runtime:
                result = await runtime.execute(
                    "Analyze overdue invoices for the demo account"
                )
                if result.status != "completed":
                    raise RuntimeError(f"Fixture failed: {result.error}")
                bundle = await runtime.export_incident(
                    result.run_id,
                    destination=destination,
                    policy=ExportPolicy(include_payloads=True),
                )
                print(result.output)
        finally:
            await backend.aclose()

    # Database and agent resources are now gone. Playback uses only the bundle.
    counts = (provider.calls, len(tool_calls))
    playback = await IncidentRecorder().playback(bundle)
    assert playback.report.valid and playback.report.complete
    assert (provider.calls, len(tool_calls)) == counts == (3, 1)
    for record in playback.records:
        if record["kind"].startswith(("model.", "tool.")):
            print(
                f"{record['seq']:3} step={record['step_id']} attempt={record['attempt_id']} {record['kind']}"
            )
    print(
        f"Verified {playback.report.record_count} records. Playback made zero model/tool calls."
    )
    print(f"Bundle: {bundle}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("invoice-incident.hx"))
    asyncio.run(main(parser.parse_args().output))
