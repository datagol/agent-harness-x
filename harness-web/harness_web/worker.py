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


def main():
    example = CATALOG[sys.argv[1]]
    script = str(example.file)
    # Exactly what `python examples/<folder>/<file>.py` sets up: the script as
    # argv[0] and its own folder first on the import path.
    sys.argv = [script, *sys.argv[2:]]
    sys.path.insert(0, os.path.dirname(script))
    builtins.input = ask
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
