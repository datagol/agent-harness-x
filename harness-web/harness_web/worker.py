"""Run unmodified example scripts with browser-backed terminal input."""

import builtins
import json
import os
import runpy
import sys
import threading
import uuid

from .catalog import CATALOG

_input_lock = threading.Lock()
_control_lock = threading.Lock()


def send(event):
    data = (json.dumps(event) + "\n").encode()
    with _control_lock:
        fd = int(os.environ["HARNESS_WEB_CONTROL_FD"])
        while data:
            data = data[os.write(fd, data) :]


def ask(prompt="", *, kind="input"):
    with _input_lock:
        request_id = uuid.uuid4().hex
        text = str(prompt).strip()
        if text == "You:":
            # Examples ask for the next chat message with exactly this prompt.
            kind = "message"
        elif "allow" in text.lower() or "approve" in text.lower():
            kind = "approval"
        send(
            {
                "type": "input_required",
                "id": request_id,
                "prompt": str(prompt),
                "kind": kind,
            }
        )
        line = sys.stdin.readline()
        if not line:
            raise EOFError("Browser session closed")
        answer = json.loads(line)
        if answer.get("id") != request_id:
            raise ValueError("Input does not match the pending prompt")
        return answer["value"]


def observe_agents():
    """Report every tool call of every agent the example creates.

    The example stays unmodified: its own output still streams as text, and
    these hooks, added to each Agent as it is constructed, send the calls as
    structured events the browser shows as tool cards.
    """
    from harnessx import Agent
    from harnessx.hooks import HookEvent

    original = Agent.__init__
    agents = iter(range(1, 1_000_000))

    def init(agent, *args, **kwargs):
        original(agent, *args, **kwargs)
        number = next(agents)

        def event(kind, data):
            send({"type": "agent", "event": {"type": kind, "data": data}})

        def started(ctx):
            call = ctx.data["tool_call"]
            event("tool_call_start", {
                "id": f"{number}:{call.id}", "name": call.name, "agent": number,
                "input": json.loads(json.dumps(call.input, default=str)),
            })

        def ended(ctx):
            result = ctx.data["result"]
            content = result.content if isinstance(result.content, str) else json.dumps(result.content, default=str)
            event("tool_result", {
                "tool_call_id": f"{number}:{result.tool_call_id}",
                "content": content[:20000], "is_error": bool(result.is_error),
            })

        agent.hooks.on(HookEvent.TOOL_CALL_START, started)
        agent.hooks.on(HookEvent.TOOL_CALL_END, ended)

    Agent.__init__ = init


def main():
    example = CATALOG[sys.argv[1]]
    script = str(example.file)
    # Exactly what `python examples/<folder>/<file>.py` sets up: the script as
    # argv[0] and its own folder first on the import path.
    sys.argv = [script, *sys.argv[2:]]
    sys.path.insert(0, os.path.dirname(script))
    builtins.input = ask
    observe_agents()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
