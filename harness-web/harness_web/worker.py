"""Run unmodified example entry points with browser-backed terminal input."""

import builtins
from contextlib import asynccontextmanager
import json
import os
import runpy
import socket
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
        if "allow" in str(prompt).lower():
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


def serve_web_example():
    import uvicorn
    from examples.web_app.server import app

    original_lifespan = app.router.lifespan_context
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]

        @asynccontextmanager
        async def lifespan(application):
            async with original_lifespan(application):
                send({"type": "web_ready", "url": f"http://127.0.0.1:{port}"})
                yield

        app.router.lifespan_context = lifespan
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", log_level="info"))
        server.run(sockets=[listener])


def main():
    example = CATALOG[sys.argv[1]]
    sys.argv = [example.module, *sys.argv[2:]]
    builtins.input = ask
    from examples import _console

    _console.get_user_input = lambda prompt="You": ask(prompt, kind="message")
    if example.id == "web_app":
        serve_web_example()
    else:
        runpy.run_module(example.module, run_name="__main__")


if __name__ == "__main__":
    main()
