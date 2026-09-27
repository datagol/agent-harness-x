"""I/O activities and payload offloading. Imported only with the Temporal extra."""

from __future__ import annotations
import asyncio
import copy
from temporalio import activity
from temporalio.converter import PayloadCodec
from temporalio.exceptions import ApplicationError
from temporalio.api.common.v1 import Payload

from ..artifacts import capture
from ..engine import (
    command,
    execute_tool,
    finish_tool,
    new_state,
    restore,
    snapshot,
    retryable,
)
from ..execution import RunEvent, RunEventType, wire, ToolApprovalRequired
from ..registry import AgentRef


class ArtifactCodec(PayloadCodec):
    def __init__(self, store, threshold=65536):
        self.store, self.threshold = store, threshold

    async def encode(self, payloads):
        result = []
        for payload in payloads:
            raw = payload.SerializeToString()
            if len(raw) > self.threshold:
                key = await self.store.put_artifact(raw)
                result.append(
                    Payload(
                        metadata={"encoding": b"binary/harness-artifact-v1"},
                        data=key.encode(),
                    )
                )
            else:
                result.append(payload)
        return result

    async def decode(self, payloads):
        result = []
        for payload in payloads:
            if payload.metadata.get("encoding") == b"binary/harness-artifact-v1":
                result.append(
                    Payload.FromString(
                        await self.store.get_artifact(payload.data.decode())
                    )
                )
            else:
                result.append(payload)
        return result


class AgentActivities:
    def __init__(self, registry, artifacts, events):
        self.registry, self.artifacts, self.events = registry, artifacts, events

    async def _agent(self, args):
        agent = self.registry.create(AgentRef(**args["agent"]))
        agent._artifact_store = self.artifacts
        agent._session_id = args["session_id"]
        try:
            if args.get("state"):
                state = args["state"]
                if state.get("config") and wire(agent.config) != state["config"]:
                    raise ValueError(
                        "Worker agent definition differs from persisted configuration"
                    )
                await restore(agent, state)
        except BaseException:
            await agent.aclose()
            raise
        return agent

    @activity.defn(name="harness_step_v1")
    async def step(self, args: dict) -> dict:
        agent = await self._agent(args)
        state = args.get("state")
        buffered = []

        async def emit(event):
            if event.type in (RunEventType.TEXT_DELTA, RunEventType.THINKING_DELTA):
                event.attempt_id = str(activity.info().attempt)
                event.event_id = f"{state['run_id']}:{state['iterations']}:{activity.info().activity_id}:{activity.info().attempt}:{len(buffered)}"
                buffered.append(None)
                try:
                    await self.events.publish(event)
                except Exception:
                    pass
            else:
                buffered.append(wire(event))

        try:
            if args["command"] == "new":
                new = new_state(
                    agent, args["message"], run_id=args["run_id"], durable=True
                )
                new.update(
                    **await snapshot(agent),
                    stream=args["stream"],
                    config=wire(agent.config),
                )
                return {"outcome": new, "events": []}
            if args["command"] == "finish_tool":
                result = await finish_tool(agent, args["entry"])
                outcome = {"result": result, **await snapshot(agent)}
            else:
                if args["command"] == "model" and activity.info().attempt > 1:
                    try:
                        await self.events.publish(
                            RunEvent(
                                RunEventType.ATTEMPT_RESET,
                                {"attempt": activity.info().attempt},
                                state["session_id"],
                                state["run_id"],
                            )
                        )
                    except Exception:
                        pass
                outcome = await command(agent, state, args["command"], emit)
            return {
                "outcome": await capture(agent, outcome),
                "events": [e for e in buffered if e is not None],
                "attempt": activity.info().attempt,
            }
        except Exception as exc:
            if not retryable(exc):
                raise ApplicationError(
                    str(exc), type=type(exc).__name__, non_retryable=True
                ) from None
            raise
        finally:
            await agent.aclose()

    @activity.defn(name="harness_tool_v1")
    async def tool(self, args: dict) -> dict:
        agent = await self._agent(args)
        entry = copy.deepcopy(args["entry"])
        entry["attempt"] = activity.info().attempt

        async def ignore(event):
            pass

        async def heartbeat():
            while True:
                activity.heartbeat(entry["execution_key"])
                await asyncio.sleep(5)

        heart = asyncio.create_task(heartbeat())
        try:
            # No middleware after execution: return the outcome to Temporal first.
            return await execute_tool(agent, args["state"], entry, ignore)
        except ToolApprovalRequired as exc:
            raise ApplicationError(
                str(exc), type="ToolApprovalRequired", non_retryable=True
            ) from None
        finally:
            heart.cancel()
            await asyncio.gather(heart, return_exceptions=True)
            await agent.aclose()

    @activity.defn(name="harness_publish_v1")
    async def publish(self, events: list[dict]) -> None:
        for event in events:
            await self.events.publish(RunEvent.from_dict(event))
