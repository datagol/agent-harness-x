"""Helper to make tool-generated files downloadable via the UI."""

from __future__ import annotations

import os
import shutil
import uuid

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


def make_downloadable(file_path: str, display_name: str | None = None) -> str:
    """Copy a file into the output directory and return a __FILE__ marker string.

    The UI detects this marker and renders a download button instead of plain text.

    Args:
        file_path: Path to the file on disk (already created by the tool).
        display_name: Human-readable name for the download link. Defaults to the filename.
    """
    filename = os.path.basename(file_path)
    display = display_name or filename
    file_id = uuid.uuid4().hex[:12]
    dest_dir = os.path.join(OUTPUT_DIR, file_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    shutil.copy2(file_path, dest_path)

    url = f"/files/{file_id}/{filename}"
    return f"__FILE__:{url}:{display}"
