"""Helper to make tool-generated files downloadable via the UI."""

from __future__ import annotations

import os
import shutil
import uuid

OUTPUT_DIR = os.path.abspath(os.environ.get("AGENT_OUTPUT_DIR", os.path.join(os.getcwd(), "output")))


def make_content_downloadable(
    filename: str, content: bytes, display_name: str | None = None, *, output_dir: str | None = None,
) -> str:
    """Publish known bytes without reopening a generated workspace path.

    Preserve the UI marker contract. Storage is application-owned and defaults to
    AGENT_OUTPUT_DIR or ./output, matching the example web server.
    """
    filename = os.path.basename(filename)
    if filename in ("", ".", ".."):
        raise ValueError("A download filename is required")
    file_id = uuid.uuid4().hex
    dest_dir = os.path.join(output_dir or OUTPUT_DIR, file_id)
    os.makedirs(dest_dir, mode=0o700, exist_ok=False)
    with open(os.path.join(dest_dir, filename), "xb") as output:
        output.write(content)
    return f"__FILE__:/files/{file_id}/{filename}:{display_name or filename}"


def make_downloadable(file_path: str, display_name: str | None = None, *, output_dir: str | None = None) -> str:
    """Copy a file into the output directory and return a __FILE__ marker string.

    The UI detects this marker and renders a download button instead of plain text.

    Args:
        file_path: Path to the file on disk (already created by the tool).
        display_name: Human-readable name for the download link. Defaults to the filename.
    """
    filename = os.path.basename(file_path)
    display = display_name or filename
    file_id = uuid.uuid4().hex[:12]
    dest_dir = os.path.join(output_dir or OUTPUT_DIR, file_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    shutil.copy2(file_path, dest_path)

    url = f"/files/{file_id}/{filename}"
    return f"__FILE__:{url}:{display}"
