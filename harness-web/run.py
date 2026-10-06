"""Run the built local app: python3 harness-web/run.py."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main():
    parser = argparse.ArgumentParser(description="Start harness-web on localhost")
    parser.add_argument("--port", type=int, default=8765)
    # Localhost by default: this is a local developer tool, and binding every
    # interface by accident would expose an app that runs arbitrary examples.
    # A container has to opt in with --host 0.0.0.0 to be reachable at all.
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if not (HERE / "dist" / "index.html").is_file():
        parser.error(
            "Build the frontend first: npm --prefix harness-web ci && npm --prefix harness-web run build"
        )
    try:
        import uvicorn
    except ImportError:
        parser.error(
            "Install the server extra first: python3 -m pip install -e '.[server]'"
        )
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"harness-web → http://{shown}:{args.port}", flush=True)
    uvicorn.run("harness_web.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
