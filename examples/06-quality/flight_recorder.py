"""Record a run, export it as an incident bundle, and play it back without a model or tools.

``AgentRuntime(..., recording=True)`` journals every model call, retry and tool call of a run.
``runtime.export_incident(run_id, destination=...)`` writes that journal to one portable ``.hx`` file,
and ``IncidentRecorder().playback(bundle)`` verifies and replays it later, after the agent, runtime
and database are gone, without calling the model or any tool again. The scripted model fails once
(an injected connection error) so the recording shows a retry, then calls a synthetic invoice tool.

Run:   python examples/06-quality/flight_recorder.py --output /tmp/invoice-incident.hx
Needs: Nothing: a scripted model, no network. The --output file must not already exist.
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from dotenv import load_dotenv

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

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables


class FixtureProvider(LLMProvider):
    """A scripted model that fails once, then calls the tool, then answers."""

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
                tool_calls=[ToolCall("overdue", "overdue_balance", {"account": "demo"})], stop_reason="tool_use",
            )
        return ProviderResponse(text="Overdue balance: $1,500.")


class LabelFixtureOutput(Middleware):
    """Mark the answer as synthetic, so nobody mistakes the fixture for real figures."""

    async def after_llm_call(self, response):
        if response.text:
            response.text += " (synthetic fixture)"
        return response


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("invoice-incident.hx"), help="Where to write the bundle")
    return parser.parse_args(argv)


async def main(destination: Path | None = None) -> None:
    if destination is None:
        destination = parse_args().output
    provider = FixtureProvider()
    tool_calls = []
    with tempfile.TemporaryDirectory(prefix="harnessx-recorder-") as directory:
        async with Agent(provider=provider) as agent:
            agent.middleware.add(LabelFixtureOutput())

            @agent.tools.register(permission=PermissionLevel.ALLOW, replay_policy="safe")
            def overdue_balance(account: str) -> str:
                """Calculate the unpaid balance of synthetic overdue invoices."""
                tool_calls.append(account)
                invoices: list[dict] = [
                    {"id": "inv-1", "amount": 1000, "overdue": True},
                    {"id": "inv-2", "amount": 500, "overdue": True},
                    {"id": "inv-3", "amount": 750, "overdue": False},
                ]
                balance = sum(row["amount"] for row in invoices if row["overdue"])
                return json.dumps({"account": account, "overdue_balance": balance, "invoices": invoices})

            async with SQLiteBackend(str(Path(directory) / "runtime.db")) as backend:
                async with AgentRuntime(agent, backend=backend, recording=True) as runtime:
                    result = await runtime.run("Analyze overdue invoices for the demo account")
                    if result.status != "completed":
                        print("Error:", result.error["message"] if result.error else result.status.value)
                        return
                    bundle = await runtime.export_incident(
                        result.run_id, destination=destination, policy=ExportPolicy(include_payloads=True),
                    )
                    print(result.output)

    # The database and the agent are gone now; playback reads only the bundle.
    counts = (provider.calls, len(tool_calls))
    playback = await IncidentRecorder().playback(bundle)
    if not (playback.report.valid and playback.report.complete):
        raise RuntimeError("Playback reported an invalid or incomplete bundle")
    if (provider.calls, len(tool_calls)) != counts or counts != (3, 1):
        raise RuntimeError("Playback must not invoke the provider or tools again")
    for record in playback.records:
        if record["kind"].startswith(("model.", "tool.")):
            print(f"{record['seq']:3} step={record['step_id']} attempt={record['attempt_id']} {record['kind']}")
    print(f"Verified {playback.report.record_count} records. Playback made zero model/tool calls.")
    print(f"Bundle: {bundle}")


if __name__ == "__main__":
    asyncio.run(main())
