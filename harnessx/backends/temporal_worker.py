"""I/O activities and payload offloading. Imported only with the Temporal extra."""

from __future__ import annotations
import asyncio
import copy
from dataclasses import asdict
from datetime import timedelta
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
    retry_after,
    sync_accounting,
    provider_state,
    restore_provider_state,
)
from ..execution import RunEvent, RunEventType, wire, ToolApprovalRequired
from ..registry import AgentRef
from ..types import AgentConfig


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


def activity_accounting(agent, state):
    state = state or {}
    usage = asdict(agent.guardrails.total_usage)
    baseline = state.get("total_usage", {})
    return {"usage": {k: max(0, v - baseline.get(k, 0)) for k, v in usage.items()},
            "cost": max(0, agent.guardrails.estimated_cost - state.get("estimated_cost", agent.guardrails.cost_of(type(agent.guardrails.total_usage)(**baseline)))),
            "incomplete": agent.guardrails.usage_incomplete}


def activity_checkpoint(agent, state):
    return {"accounting": activity_accounting(agent, state), "provider_state": provider_state(agent)}


def restore_heartbeat(agent):
    for detail in activity.info().heartbeat_details:
        if isinstance(detail, dict) and "accounting" in detail:
            restore_provider_state(agent, detail.get("provider_state"))
            saved = detail["accounting"]
            from ..types import TokenUsage
            agent.guardrails.track_usage(TokenUsage(**saved.get("usage", {})), cost=saved.get("cost", 0))
            agent.guardrails.usage_incomplete = True
            break


async def heartbeat_loop(agent, state):
    while True:
        activity.heartbeat(activity_checkpoint(agent, state))
        await asyncio.sleep(5)


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
                if state.get("config") and wire(agent.config) != wire(AgentConfig.from_dict(state["config"])):
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
        modern = args.get("reliability_v2", False)
        if modern:
            restore_heartbeat(agent)
            agent.guardrails.usage_observer = lambda *_args, **_kwargs: activity.heartbeat(activity_checkpoint(agent, state))
        heart = asyncio.create_task(heartbeat_loop(agent, state)) if modern else None

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
                try:
                    outcome = await command(agent, state, args["command"], emit)
                except Exception as exc:
                    spent = not retryable(agent, exc) or activity.info().attempt >= agent.config.retry.attempts
                    if modern and args["command"] == "compact" and spent:
                        agent.guardrails.usage_incomplete = True
                        agent.memory.set_messages(copy.deepcopy(state.get("messages", [])))
                        fallback = {**state, "condense_without_model": True}
                        outcome = await command(agent, fallback, "compact", emit)
                    else:
                        raise
            if modern:
                accounting_state = {**(state or {}), **outcome}
                sync_accounting(agent, accounting_state)
                outcome.update({k: accounting_state[k] for k in ("usage", "total_usage", "estimated_cost", "usage_incomplete", "provider_state") if k in accounting_state})
            return {
                "outcome": await capture(agent, outcome),
                "events": [e for e in buffered if e is not None],
                "attempt": activity.info().attempt,
            }
        except BaseException as exc:
            if modern:
                accounting = activity_accounting(agent, state)
                accounting["incomplete"] |= args["command"] in ("model", "compact")
                checkpoint = {"accounting": accounting, "provider_state": provider_state(agent)}
                activity.heartbeat(checkpoint)
                retry = agent.config.retry
                delay = retry.wait_for(activity.info().attempt, retry_after(agent, exc))
                raise ApplicationError(
                    str(exc), checkpoint, type=type(exc).__name__,
                    non_retryable=isinstance(exc, asyncio.CancelledError) or not retryable(agent, exc),
                    next_retry_delay=timedelta(seconds=max(0.001, delay)),
                ) from None
            if not retryable(agent, exc):
                raise ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True) from None
            raise
        finally:
            if heart is not None:
                heart.cancel()
                await asyncio.gather(heart, return_exceptions=True)
            await agent.aclose()

    @activity.defn(name="harness_tool_v1")
    async def tool(self, args: dict) -> dict:
        agent = await self._agent(args)
        entry = copy.deepcopy(args["entry"])
        entry["attempt"] = activity.info().attempt

        async def ignore(event):
            pass

        modern = args.get("reliability_v2", False)
        if modern:
            restore_heartbeat(agent)
            agent.guardrails.usage_observer = lambda *_args, **_kwargs: activity.heartbeat(activity_checkpoint(agent, args["state"]))
        heart = asyncio.create_task(heartbeat_loop(agent, args["state"]))
        try:
            # Commit both the tool result and child charges before middleware.
            result = await execute_tool(agent, args["state"], entry, ignore)
            return {"result": result, "accounting": activity_accounting(agent, args["state"])} if modern else result
        except BaseException as exc:
            if modern:
                accounting = activity_accounting(agent, args["state"])
                accounting["incomplete"] = True
                checkpoint = {"accounting": accounting, "provider_state": provider_state(agent)}
                activity.heartbeat(checkpoint)
                raise ApplicationError(
                    str(exc), checkpoint, type=type(exc).__name__,
                    non_retryable=isinstance(exc, (ToolApprovalRequired, asyncio.CancelledError)),
                ) from None
            if isinstance(exc, ToolApprovalRequired):
                raise ApplicationError(str(exc), type="ToolApprovalRequired", non_retryable=True) from None
            raise
        finally:
            heart.cancel()
            await asyncio.gather(heart, return_exceptions=True)
            await agent.aclose()

    @activity.defn(name="harness_publish_v1")
    async def publish(self, events: list[dict]) -> None:
        for event in events:
            await self.events.publish(RunEvent.from_dict(event))
