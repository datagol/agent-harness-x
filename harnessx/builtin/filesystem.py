"""Bounded filesystem tools with descriptor-relative access on POSIX."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import errno
import os
import stat
from typing import TYPE_CHECKING
import uuid

from harnessx.types import PermissionLevel
from ._registration import select_tools

if TYPE_CHECKING:
    from harnessx.tools import ToolRegistry


def _positive_limit(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class _Filesystem:
    """Open each component without following links; never reopen a checked path.

    The configured root is canonicalized once. Access below that root rejects all
    symlinks, including links to in-root targets. This is not an OS sandbox: other
    tools and processes can still access the host filesystem.
    """

    def __init__(
        self, base_path=None, *, max_read_bytes=1_000_000,
        max_write_bytes=10_000_000, max_directory_entries=1_000, output_dir=None,
    ):
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise NotImplementedError("Filesystem tools require POSIX no-follow directory operations")
        self.base = os.path.abspath(base_path) if base_path is not None else None
        self.root = os.path.realpath(self.base) if self.base is not None else os.path.sep
        self.max_read_bytes = _positive_limit(max_read_bytes, "max_read_bytes")
        self.max_write_bytes = _positive_limit(max_write_bytes, "max_write_bytes")
        self.max_directory_entries = _positive_limit(max_directory_entries, "max_directory_entries")
        self.output_dir = os.path.abspath(output_dir) if output_dir is not None else None
        # Reject missing/invalid roots at registration; retain no descriptors.
        with self._directory([]):
            pass

    def _parts(self, path: str) -> list[str]:
        if not isinstance(path, str) or not path or "\0" in path:
            raise ValueError("path must be a nonempty string without NUL bytes")
        target = os.path.abspath(os.path.join(self.base or os.getcwd(), path))
        if self.base is None:
            # Unscoped access retains host path semantics, including system aliases
            # such as /tmp on macOS. Only a configured root provides confinement.
            target = os.path.realpath(target)
        # Accept configured spelling (/var/...) and canonical root (/private/var/
        # on macOS), but never resolve model-supplied links below a scoped root.
        root = self.base or self.root
        if os.path.commonpath([root, target]) != root:
            root = self.root
        if os.path.commonpath([root, target]) != root:
            raise PermissionError(f"Access denied: path {path!r} escapes base directory")
        relative = os.path.relpath(target, root)
        return [] if relative == "." else relative.split(os.path.sep)

    @staticmethod
    def _open_directory(name, parent_fd):
        try:
            return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise PermissionError(f"Directory component {name!r} is a symlink or not a directory") from exc
            raise

    @contextmanager
    def _directory(self, parts, *, create=False):
        fd = os.open(os.path.sep, os.O_RDONLY | os.O_DIRECTORY)
        try:
            # Traverse even the root's ancestors without following new symlinks.
            for component in self.root.split(os.path.sep):
                if component:
                    child = self._open_directory(component, fd)
                    os.close(fd)
                    fd = child
            for component in parts:
                if create:
                    try:
                        os.mkdir(component, dir_fd=fd)
                    except FileExistsError:
                        pass
                child = self._open_directory(component, fd)
                os.close(fd)
                fd = child
            yield fd
        finally:
            os.close(fd)

    def read(self, path, offset, limit, *, numbered=True):
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 0:
            raise ValueError("offset and limit must be nonnegative integers")
        parts = self._parts(path)
        if not parts:
            raise IsADirectoryError(path)
        with self._directory(parts[:-1]) as parent:
            try:
                fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise PermissionError("Reading symlinks is not allowed") from exc
                raise
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise PermissionError("Only regular files can be read")
                if limit == 0:
                    return ""
                remaining, line_number, output = self.max_read_bytes, 0, []
                while remaining > 0 and line_number < offset + limit:
                    line = stream.readline(remaining)
                    if not line:
                        break
                    remaining -= len(line)
                    if line_number >= offset:
                        text = line.decode('utf-8', errors='replace')
                        output.append(f"{line_number + 1}\t{text}" if numbered else text)
                    line_number += 1
                if stream.tell() < os.fstat(stream.fileno()).st_size:
                    output.append("\n[truncated: read byte or line limit reached]\n")
                return "".join(output)

    def write(self, path, content, overwrite):
        if not isinstance(content, str) or type(overwrite) is not bool:
            raise TypeError("content must be a string and overwrite must be a bool")
        if len(content) > self.max_write_bytes:
            raise ValueError("Content exceeds max_write_bytes")
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_write_bytes:
            raise ValueError("Content exceeds max_write_bytes")
        parts = self._parts(path)
        if not parts:
            raise IsADirectoryError(path)
        with self._directory(parts[:-1], create=True) as parent:
            name = parts[-1]
            try:
                existing = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if not stat.S_ISREG(existing.st_mode):
                    raise PermissionError("Only regular files can be replaced; symlinks are not allowed")
                if not overwrite:
                    raise FileExistsError("File exists; set overwrite=True to replace it")
            temporary = f".harness-{uuid.uuid4().hex}.tmp"
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                if overwrite:
                    # Rename replaces the entry itself: a racing destination
                    # symlink is replaced, never followed to its target.
                    os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
                else:
                    # Atomic create-if-absent; do not overwrite a racing destination.
                    os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
        return f"Written {len(encoded)} bytes to {path}"

    def list(self, path):
        lines, truncated = [], False
        with self._directory(self._parts(path)) as directory:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if len(lines) >= self.max_directory_entries:
                        truncated = True
                        break
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if stat.S_ISLNK(info.st_mode):
                        lines.append(f"  {entry.name} [symlink; access disabled]")
                    elif stat.S_ISDIR(info.st_mode):
                        lines.append(f"  {entry.name}/")
                    else:
                        lines.append(f"  {entry.name} ({info.st_size} bytes)")
        result = f"Directory: {path}\n" + "\n".join(sorted(lines)) if lines else f"Directory: {path} (empty)"
        return result + ("\n[truncated: directory entry limit reached]" if truncated else "")

    def generate(self, path, content, display_name, overwrite):
        from .file_output import make_content_downloadable

        self.write(path, content, overwrite)
        # Publish supplied bytes instead of reopening a possibly replaced path.
        return make_content_downloadable(os.path.basename(path), content.encode("utf-8"), display_name or path, output_dir=self.output_dir)


async def read_file(path: str, offset: int = 0, limit: int = 100_000) -> str:
    """Read a regular file with line numbers and a bounded byte budget.

    Args:
        path: Absolute or relative file path.
        offset: Zero-based starting line, nonnegative.
        limit: Maximum number of lines, nonnegative; the byte cap also applies.
    """
    return await asyncio.to_thread(_Filesystem().read, path, offset, limit)


async def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Atomically create a regular file and missing parent directories.

    Args:
        path: Absolute or relative file path.
        content: UTF-8 text to write.
        overwrite: Explicitly permit replacing an existing regular file.
    """
    return await asyncio.to_thread(_Filesystem().write, path, content, overwrite)


async def list_directory(path: str = ".") -> str:
    """List a bounded number of entries without following entry symlinks.

    Args:
        path: Absolute or relative directory path.
    """
    return await asyncio.to_thread(_Filesystem().list, path)


async def generate_file(path: str, content: str, display_name: str = "", overwrite: bool = False) -> str:
    """Write a file and publish a downloadable copy. Requires write permission.

    Args:
        path: Absolute or relative destination file path.
        content: UTF-8 text to write.
        display_name: Human-readable download name.
        overwrite: Explicitly permit replacing an existing regular file.
    """
    return await asyncio.to_thread(_Filesystem().generate, path, content, display_name, overwrite)


def register_filesystem_tools(
    registry: ToolRegistry, *, include: list[str] | None = None,
    exclude: list[str] | None = None, permission: PermissionLevel | None = None,
    base_path: str | None = None, max_read_bytes: int = 1_000_000,
    max_write_bytes: int = 10_000_000, max_directory_entries: int = 1_000,
    output_dir: str | None = None, replace: bool = False,
) -> list[str]:
    """Register filesystem tools; omitted permissions inherit the manager default.

    base_path restricts paths to an existing root, canonicalized at registration.
    Accessed components below that root reject symlinks. Without base_path the
    process's filesystem permissions apply. For inspection only, include read_file
    and list_directory. Limits and output_dir are application configuration, hidden
    from model input. Downloads use a separate application-owned output directory.
    """
    names = select_tools(registry, ["read_file", "write_file", "list_directory", "generate_file"], include, exclude, replace=replace)
    if not names:
        return []
    fs = _Filesystem(base_path, max_read_bytes=max_read_bytes, max_write_bytes=max_write_bytes,
                     max_directory_entries=max_directory_entries, output_dir=output_dir)

    async def scoped_read(path: str, offset: int = 0, limit: int = 100_000) -> str:
        return await asyncio.to_thread(fs.read, path, offset, limit)

    async def scoped_write(path: str, content: str, overwrite: bool = False) -> str:
        return await asyncio.to_thread(fs.write, path, content, overwrite)

    async def scoped_list(path: str = ".") -> str:
        return await asyncio.to_thread(fs.list, path)

    async def scoped_generate(path: str, content: str, display_name: str = "", overwrite: bool = False) -> str:
        return await asyncio.to_thread(fs.generate, path, content, display_name, overwrite)

    handlers = {
        "read_file": (scoped_read, read_file, True),
        "write_file": (scoped_write, write_file, False),
        "list_directory": (scoped_list, list_directory, True),
        "generate_file": (scoped_generate, generate_file, False),
    }
    for name in names:
        handler, public, concurrent = handlers[name]
        handler.__doc__ = public.__doc__
        registry.register_tool(handler, name=name, permission=permission,
                               concurrent=concurrent, replace=replace)
    return names
