"""Run the built local app: python3 harness-web/run.py."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main():
    parser = argparse.ArgumentParser(description="Start harness-web on localhost")
    parser.add_argument("--port", type=int, default=8765)
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
    print(f"harness-web → http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run("harness_web.server:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
