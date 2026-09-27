"""Download an already-opened, ordinary file without following path symlinks."""

import os
from pathlib import Path
import stat


def open_workspace_file(root: Path, name: str):
    parts = name.split("/")
    if any(not part or part.startswith(".") or "\0" in part for part in parts):
        raise FileNotFoundError(name)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = child
        fd = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise FileNotFoundError(name)
            return os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise
    finally:
        os.close(directory)
