"""Bounded local run ownership, streaming, terminal input, and process cleanup."""

import asyncio
import codecs
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import uuid

from .catalog import ROOT

FINAL = {"completed", "failed", "cancelled"}
MAX_EVENTS = 4000
MAX_EVENT_BYTES = 2_000_000


@dataclass
class Run:
    title: str
    example_id: str
    workdir: Path
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "starting"
    mode: str = "live"
    events: list = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task | None = None
    process: asyncio.subprocess.Process | None = None
    pending: dict | None = None
    answer: asyncio.Future | None = None
    url: str | None = None
    chat_id: str | None = None
    exit_code: int | None = None
    sequence: int = 0
    event_bytes: int = 0

    def public(self):
        return {
            "id": self.id,
            "title": self.title,
            "example_id": self.example_id,
            "created_at": self.created_at,
            "status": self.status,
            "mode": self.mode,
            "pending": self.pending,
            "url": self.url,
            "chat_id": self.chat_id,
            "exit_code": self.exit_code,
        }

    async def emit(self, event_type, **data):
        async with self.condition:
            self.sequence += 1
            event = {"seq": self.sequence, "type": event_type, **data}
            self.events.append(event)
            self.event_bytes += len(json.dumps(event))
            while len(self.events) > MAX_EVENTS or (
                self.event_bytes > MAX_EVENT_BYTES and len(self.events) > 1
            ):
                self.event_bytes -= len(json.dumps(self.events.pop(0)))
            self.condition.notify_all()

    async def state(self, status):
        self.status = status
        await self.emit("state", **self.public())

    async def question(self, prompt, kind="input", **extra):
        self.answer = asyncio.get_running_loop().create_future()
        self.pending = {"id": uuid.uuid4().hex, "prompt": prompt, "kind": kind, **extra}
        await self.emit("input_required", **self.pending)
        await self.state("waiting")
        try:
            return await self.answer
        finally:
            self.pending = None
            self.answer = None

    async def reply(self, prompt_id, value):
        if (
            not self.pending
            or self.pending["id"] != prompt_id
            or self.status != "waiting"
        ):
            raise ValueError("This prompt is no longer waiting for input")
        if self.pending["kind"] == "approval" and value not in ("y", "n"):
            raise ValueError("Choose Allow once or Deny")
        answer = self.answer
        if answer is None and not (
            self.process and self.process.stdin and self.process.returncode is None
        ):
            raise ValueError("This process is no longer accepting input")
        # Claim the prompt before any await. Two requests cannot approve it twice.
        self.pending = None
        await self.emit("input_sent", content=value)
        await self.state("running")
        if answer is not None:
            if answer.done():
                raise ValueError("This prompt is no longer accepting input")
            answer.set_result(value)
        else:
            assert self.process is not None and self.process.stdin is not None
            self.process.stdin.write(
                (json.dumps({"id": prompt_id, "value": value}) + "\n").encode()
            )
            await self.process.stdin.drain()

    async def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.status not in FINAL:
            self.pending = None
            await self.state("cancelled")

    async def subscribe(self, cursor=0):
        while True:
            gap = None
            async with self.condition:
                if self.events and cursor < self.events[0]["seq"] - 1:
                    gap = {
                        "type": "gap",
                        "seq": self.events[0]["seq"] - 1,
                        "content": "Earlier output was truncated; showing the retained tail.",
                    }
                    cursor = self.events[0]["seq"] - 1
                pending = [event for event in self.events if event["seq"] > cursor]
                finished = self.status in FINAL
                if not pending and not finished:
                    try:
                        await asyncio.wait_for(self.condition.wait(), timeout=15)
                    except asyncio.TimeoutError:
                        pass
            if gap:
                yield gap
            for event in pending:
                cursor = event["seq"]
                yield event
            if finished:
                return
            if not pending:
                yield {"type": "heartbeat", "seq": cursor}


class RunManager:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.runs = {}

    def create(self, title, example_id, *, workdir=None):
        if (
            sum(
                run.status not in FINAL or bool(run.task and not run.task.done())
                for run in self.runs.values()
            )
            >= 4
        ):
            raise ValueError("Stop an active run before starting another (limit: 4)")
        if len(self.runs) >= 100:
            oldest = next(
                (
                    key
                    for key, run in self.runs.items()
                    if run.status in FINAL and (not run.task or run.task.done())
                ),
                None,
            )
            if oldest is not None:
                del self.runs[oldest]
        run = Run(
            title=title,
            example_id=example_id,
            workdir=Path(workdir or self.directory / uuid.uuid4().hex),
        )
        run.workdir.mkdir(parents=True, exist_ok=True)
        self.runs[run.id] = run
        return run

    async def close(self):
        await asyncio.gather(*(run.stop() for run in list(self.runs.values())))


async def run_example(run, example, args, initial_prompt=""):
    control_read, control_write = os.pipe()
    transport = None
    readers = []
    try:
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
            "COLUMNS": "100",
            "HARNESS_WEB_CONTROL_FD": str(control_write),
            "AGENT_OUTPUT_DIR": str(run.workdir / "output"),
            "AGENT_SKILLS": os.environ.get(
                "AGENT_SKILLS", str(ROOT / "examples" / "skills")
            ),
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), str(ROOT / "harness-web"), os.environ.get("PYTHONPATH", "")]
            ),
        }
        run.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-m",
            "harness_web.worker",
            example.id,
            *args,
            cwd=run.workdir,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            pass_fds=(control_write,),
            start_new_session=True,
        )
        os.close(control_write)
        control_write = -1
        reader = asyncio.StreamReader(limit=131072)
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(control_read, "rb")
        )
        control_read = -1
        await run.state("running")

        async def output():
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            while chunk := await run.process.stdout.read(4096):
                await run.emit("output", content=decoder.decode(chunk))
            tail = decoder.decode(b"", final=True)
            if tail:
                await run.emit("output", content=tail)

        async def control():
            nonlocal initial_prompt
            while line := await reader.readline():
                event = json.loads(line)
                kind = event.pop("type")
                if kind == "input_required":
                    run.pending = event
                    await run.emit(kind, **event)
                    await run.state("waiting")
                    if event["kind"] == "message" and initial_prompt:
                        value, initial_prompt = initial_prompt, ""
                        await run.reply(event["id"], value)
                elif kind == "agent":
                    # A tool call or result from an agent inside the example.
                    await run.emit("agent", event=event["event"])
                elif kind == "web_ready":
                    run.url = event["url"]
                    await run.state("running")

        readers = [asyncio.create_task(output()), asyncio.create_task(control())]
        # Reader errors must fail the run, rather than leave an invisible prompt waiting.
        await asyncio.gather(run.process.wait(), *readers)
        run.exit_code = run.process.returncode
        run.pending = None
        await run.state("completed" if run.exit_code == 0 else "failed")
    except asyncio.CancelledError:
        run.pending = None
        await run.state("cancelled")
        raise
    except Exception as exc:
        run.pending = None
        await run.emit("error", content=str(exc))
        await run.state("failed")
    finally:
        if run.process:
            # Terminate the process group, including ordinary shell/MCP children.
            try:
                os.killpg(run.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(run.process.wait(), 2)
            except asyncio.TimeoutError:
                try:
                    os.killpg(run.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await run.process.wait()
            # The parent may exit before a child that ignores SIGTERM.
            try:
                os.killpg(run.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for task in readers:
            task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        if transport:
            transport.close()
        for fd in (control_read, control_write):
            if fd >= 0:
                os.close(fd)
