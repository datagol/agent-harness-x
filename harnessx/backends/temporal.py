"""Optional Temporal client facade. Importing Harness never imports the SDK."""

from __future__ import annotations
import asyncio
import json
import uuid
from ..execution import RunEvent, RunEventType, RunStream, result_from_state, wire


class RedisEvents:
    def __init__(self, url=None, *, client=None, retention_seconds=86400):
        self._owns_client = client is None
        self.url, self.client, self.retention_seconds = url, client, retention_seconds

    async def initialize(self):
        if self.client is None:
            from redis.asyncio import Redis

            self.client = Redis.from_url(
                self.url, decode_responses=True, socket_timeout=2
            )

    async def publish(self, event):
        await self.initialize()
        key = "harness:events:" + event.run_id
        # Lua makes duplicate suppression and publication atomic.
        script = """
        if redis.call('SISMEMBER', KEYS[2], ARGV[1]) == 1 then return '' end
        local id = redis.call('XADD', KEYS[1], '*', 'event', ARGV[2])
        redis.call('SADD', KEYS[2], ARGV[1])
        redis.call('EXPIRE', KEYS[1], ARGV[3])
        redis.call('EXPIRE', KEYS[2], ARGV[3])
        return id
        """
        await self.client.eval(
            script,
            2,
            key,
            key + ":ids",
            event.event_id,
            json.dumps(wire(event)),
            self.retention_seconds,
        )

    async def read(self, run_id, after="0-0"):
        await self.initialize()
        key = "harness:events:" + run_id
        first = await self.client.xrange(key, count=1)
        expired = after != "0-0" and (
            not first
            or tuple(map(int, first[0][0].split("-")))
            > tuple(map(int, after.split("-")))
        )
        if expired:
            return [], True
        batches = await self.client.xread({key: after}, count=1000)
        events = []
        for _, rows in batches:
            for cursor, fields in rows:
                event = RunEvent.from_dict(json.loads(fields["event"]))
                event.cursor = cursor
                events.append(event)
        return events, False

    async def aclose(self):
        if self.client and self._owns_client:
            await self.client.aclose()


class TemporalRunHandle:
    def __init__(self, backend, session_id, run_id):
        self.backend, self.session_id, self.run_id = backend, session_id, run_id

    async def result(self):
        while True:
            state = await self.backend.handle(self.session_id).query(
                "get_run", self.run_id
            )
            if state["status"] != "running":
                return result_from_state(state)
            await asyncio.sleep(0.1)

    async def events(self, after="0-0"):
        gap_reported = False
        last_result_status = None
        while True:
            try:
                events, gap = await self.backend.events.read(self.run_id, after)
            except Exception:
                events, gap = [], True
            if gap and not gap_reported:
                yield RunEvent(
                    RunEventType.GAP,
                    {"reason": "event_history_unavailable"},
                    self.session_id,
                    self.run_id,
                )
                gap_reported = True
                after = "0-0"
            for event in events:
                after = event.cursor
                if event.type == RunEventType.RUN_RESULT:
                    last_result_status = event.data.status.value
                yield event
            if len(events) == 1000:
                continue
            state = await self.backend.handle(self.session_id).query(
                "get_run", self.run_id
            )
            if state["status"] != "running":
                if last_result_status != state["status"]:
                    yield RunEvent(
                        RunEventType.RUN_RESULT,
                        result_from_state(state),
                        self.session_id,
                        self.run_id,
                        event_id=f"{self.run_id}:result:{state['status']}",
                    )
                return
            await asyncio.sleep(0.1)

    def stream(self, after="0-0"):
        async def observe(emit):
            async for event in self.events(after):
                await emit(event)
            return await self.result()

        return RunStream(observe)


class TemporalBackend:
    remote = True

    def __init__(
        self,
        client,
        *,
        task_queue="harness-x-v1",  # Existing worker queue, not a Python import path.
        events: RedisEvents,
        artifact_store,
        registry=None,
    ):
        from ..registry import agents

        self.client, self.task_queue, self.events = client, task_queue, events
        self.artifact_store, self.registry = artifact_store, registry or agents

    @classmethod
    async def connect(
        cls, target, *, events, artifact_store, namespace="default", **kwargs
    ):
        from temporalio.client import Client
        from temporalio.converter import DataConverter
        from .temporal_worker import ArtifactCodec

        client = await Client.connect(
            target,
            namespace=namespace,
            data_converter=DataConverter(payload_codec=ArtifactCodec(artifact_store)),
        )
        return cls(client, events=events, artifact_store=artifact_store, **kwargs)

    async def initialize(self):
        # Redis may be down without blocking durable work.
        from .temporal_worker import ArtifactCodec

        converter = self.client.config()["data_converter"]
        if not isinstance(converter.payload_codec, ArtifactCodec):
            raise ValueError(
                "Use TemporalBackend.connect() or configure ArtifactCodec on the Temporal client"
            )

    def handle(self, session_id):
        return self.client.get_workflow_handle("harness-session:" + session_id)

    async def create_session(self, session_id, ref, *, runtime_config=None):
        from .temporal_workflow import AgentSessionWorkflow
        from temporalio.common import WorkflowIDReusePolicy

        await self.client.start_workflow(
            AgentSessionWorkflow.run,
            {
                "version": 1,
                "runtime_config": runtime_config or {},
                "session_id": session_id,
                "agent": ref,
                "runs": {},
                "requests": {},
                "active": None,
                "control": "",
            },
            id="harness-session:" + session_id,
            task_queue=self.task_queue,
            id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
        )

    async def get_session(self, session_id):
        return await self.handle(session_id).query("get_session")

    async def submit(self, session_id, message, request_id, stream):
        run_id = await self.handle(session_id).execute_update(
            "submit",
            {
                "message": message,
                "request_id": request_id,
                "stream": stream,
                "run_id": str(uuid.uuid4()),
            },
            id=request_id,
        )
        return TemporalRunHandle(self, session_id, run_id)

    async def resume(self, session_id):
        run_id = await self.handle(session_id).execute_update(
            "control", {"action": "resume"}
        )
        return TemporalRunHandle(self, session_id, run_id) if run_id else None

    async def resolve(self, session_id, key, **resolution):
        await self.handle(session_id).execute_update(
            "resolve", {"key": key, **resolution}
        )

    async def set_control(self, session_id, action):
        await self.handle(session_id).execute_update("control", {"action": action})
        # Pause/cancel return only after the workflow reaches a settled boundary.
        while True:
            data = await self.handle(session_id).query("get_session")
            if not data.get("busy"):
                return
            await asyncio.sleep(0.1)

    async def latest_run(self, session_id):
        session = await self.get_session(session_id)
        if session.get("active"):
            return await self.handle(session_id).query("get_run", session["active"])
        return None

    async def get_status(self, session_id):
        state = await self.latest_run(session_id)
        return {
            "session_id": session_id,
            "run": result_from_state(state) if state else None,
        }

    async def worker(self):
        from temporalio.worker import Worker
        from .temporal_worker import AgentActivities
        from .temporal_workflow import AgentSessionWorkflow

        activities = AgentActivities(self.registry, self.artifact_store, self.events)
        return Worker(
            self.client,
            task_queue=self.task_queue,
            workflows=[AgentSessionWorkflow],
            activities=[activities.step, activities.tool, activities.publish],
        )

    async def aclose(self):
        await self.events.aclose()
