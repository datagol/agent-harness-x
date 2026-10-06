"""Load an example script by its path, the way the examples are meant to be run.

Examples are standalone scripts (`python examples/<folder>/<file>.py`), not a
package, so tests load them from their files. Each load is a fresh module.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_loads = itertools.count()


def example_path(relative: str) -> Path:
    path = EXAMPLES / relative
    if not path.is_file():
        raise FileNotFoundError(f"No example at examples/{relative}")
    return path


def load_example(relative: str) -> ModuleType:
    """Import ``examples/<relative>`` without running its ``__main__`` block."""
    path = example_path(relative)
    name = f"example_{path.stem}_{next(_loads)}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and pickling look the module up by name
    spec.loader.exec_module(module)
    return module


def example_scripts() -> list[Path]:
    """Every runnable example: the .py files in the topic folders."""
    return sorted(path for path in EXAMPLES.glob("[0-9][0-9]-*/*.py") if not path.name.startswith("_"))
