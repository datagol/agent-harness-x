"""Runnable HarnessX examples; see examples/README.md for prerequisites.

Loads environment variables from the project root .env file
(e.g. ANTHROPIC_API_KEY). Run examples as modules from the project root.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
