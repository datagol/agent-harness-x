"""Bounded filesystem tools with descriptor-relative access on POSIX."""

from __future__ import annotations

import asyncio
import fnmatch
import re
from contextlib import contextmanager
import errno
import os
import stat
from typing import TYPE_CHECKING, Any
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

    def _read_all(self, parts):
        """Whole file, exactly as stored. Editing cannot work from a numbered or
        truncated view, so this is deliberately separate from read()."""
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
                size = os.fstat(stream.fileno()).st_size
                if size > self.max_read_bytes:
                    raise ValueError(f"File is {size} bytes, above max_read_bytes")
                return stream.read().decode("utf-8", errors="replace")

    def edit(self, path, old_string, new_string, replace_all):
        if not isinstance(old_string, str) or not isinstance(new_string, str):
            raise TypeError("old_string and new_string must be strings")
        if type(replace_all) is not bool:
            raise TypeError("replace_all must be a bool")
        if not old_string:
            raise ValueError("old_string must be nonempty; use write_file to create a file")
        if old_string == new_string:
            raise ValueError("old_string and new_string are identical")
        parts = self._parts(path)
        if not parts:
            raise IsADirectoryError(path)
        content = self._read_all(parts)
        occurrences = content.count(old_string)
        if occurrences == 0:
            raise ValueError("old_string was not found; it must match the file exactly")
        if occurrences > 1 and not replace_all:
            raise ValueError(
                f"old_string matches {occurrences} times; include more surrounding "
                "text to make it unique, or set replace_all=True"
            )
        updated = content.replace(old_string, new_string)
        self.write(path, updated, True)
        return f"Replaced {occurrences if replace_all else 1} occurrence(s) in {path}"

    def delete(self, path, recursive):
        if type(recursive) is not bool:
            raise TypeError("recursive must be a bool")
        parts = self._parts(path)
        if not parts:
            raise PermissionError("Refusing to delete the base directory itself")
        with self._directory(parts[:-1]) as parent:
            name = parts[-1]
            info = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                # Removing the link itself is safe and never touches its target.
                os.unlink(name, dir_fd=parent)
                return f"Deleted symlink {path}"
            if not stat.S_ISDIR(info.st_mode):
                os.unlink(name, dir_fd=parent)
                return f"Deleted {path}"
            if not recursive:
                os.rmdir(name, dir_fd=parent)
                return f"Deleted empty directory {path}"
        removed = self._remove_tree(parts)
        return f"Deleted {path} and {removed} entries below it"

    def _remove_tree(self, parts):
        """Remove the directory at ``parts`` and everything below it, descending
        only through real directories. Returns the count of entries inside it."""
        removed = 0
        with self._directory(parts) as directory:
            with os.scandir(directory) as entries:
                children = [(e.name, e.is_dir(follow_symlinks=False)) for e in entries]
            for name, is_directory in children:
                if is_directory:
                    removed += self._remove_tree(parts + [name]) + 1  # its contents, plus itself
                else:
                    os.unlink(name, dir_fd=directory)
                    removed += 1
        with self._directory(parts[:-1]) as parent:
            os.rmdir(parts[-1], dir_fd=parent)
        return removed

    def _walk(self, parts, depth=0):
        """Yield (relative parts, is_directory) below ``parts``, never via symlinks."""
        if depth > 24:
            return
        with self._directory(parts) as directory:
            with os.scandir(directory) as entries:
                children = sorted(
                    (e.name, e.is_dir(follow_symlinks=False), e.is_symlink())
                    for e in entries
                )
        for name, is_directory, is_link in children:
            if is_link:
                continue  # a link is listed by ls, but never traversed or matched
            child = parts + [name]
            yield child, is_directory
            if is_directory:
                yield from self._walk(child, depth + 1)

    def glob(self, pattern, path):
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern must be a nonempty string")
        base = self._parts(path)
        matches = []
        for child, is_directory in self._walk(base):
            relative = os.path.sep.join(child[len(base):])
            if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(os.path.basename(relative), pattern):
                matches.append(relative + (os.path.sep if is_directory else ""))
                if len(matches) >= self.max_directory_entries:
                    return self._listing(path, matches, f"pattern {pattern!r}", truncated=True)
        return self._listing(path, matches, f"pattern {pattern!r}")

    def grep(self, pattern, path, include):
        try:
            expression = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression: {exc}") from None
        base = self._parts(path)
        hits, budget = [], self.max_read_bytes
        for child, is_directory in self._walk(base):
            if is_directory:
                continue
            relative = os.path.sep.join(child[len(base):])
            if include and not fnmatch.fnmatch(os.path.basename(relative), include):
                continue
            try:
                content = self._read_all(child)
            except (OSError, ValueError, PermissionError):
                continue  # unreadable, oversized or not a regular file
            budget -= len(content)
            for number, line in enumerate(content.splitlines(), 1):
                if expression.search(line):
                    hits.append(f"  {relative}:{number}: {line.strip()[:200]}")
                    if len(hits) >= self.max_directory_entries:
                        return self._listing(path, hits, f"pattern {pattern!r}", truncated=True)
            if budget <= 0:
                return self._listing(path, hits, f"pattern {pattern!r}", truncated=True)
        return self._listing(path, hits, f"pattern {pattern!r}")

    @staticmethod
    def _listing(path, rows, what, *, truncated=False):
        if not rows:
            return f"No matches for {what} under {path}"
        body = "\n".join(rows if rows[0].startswith("  ") else (f"  {r}" for r in rows))
        header = f"Matches for {what} under {path}:"
        return f"{header}\n{body}" + ("\n[truncated: match limit reached]" if truncated else "")

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


async def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace an exact string in a file, leaving the rest untouched.

    Prefer this over rewriting a whole file: only the changed text passes
    through the reply, so a large file costs a few tokens to amend.

    Args:
        path: Absolute or relative file path.
        old_string: Text to replace. Must appear exactly once unless replace_all.
            Include surrounding lines to make it unique.
        new_string: Replacement text. Pass an empty string to delete the match.
        replace_all: Replace every occurrence instead of requiring a unique one.
    """
    return await asyncio.to_thread(_Filesystem().edit, path, old_string, new_string, replace_all)


async def delete(path: str, recursive: bool = False) -> str:
    """Remove a file, a symlink, or a directory.

    Args:
        path: Absolute or relative path to remove.
        recursive: Permit removing a directory that still has entries.
    """
    return await asyncio.to_thread(_Filesystem().delete, path, recursive)


async def glob(pattern: str, path: str = ".") -> str:
    """Find files whose path or name matches a shell pattern, such as '*.py'.

    Args:
        pattern: Shell-style pattern matched against each relative path and name.
        path: Directory to search below.
    """
    return await asyncio.to_thread(_Filesystem().glob, pattern, path)


async def grep(pattern: str, path: str = ".", include: str = "") -> str:
    """Search file contents for a regular expression, reporting path and line.

    Args:
        pattern: Regular expression to search for.
        path: Directory to search below.
        include: Optional shell pattern limiting which filenames are searched.
    """
    return await asyncio.to_thread(_Filesystem().grep, pattern, path, include)


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
    output_dir: str | None = None, replace: bool = False, sandbox: Any | None = None,
) -> list[str]:
    """Register filesystem tools; omitted permissions inherit the manager default.

    base_path restricts paths to an existing root, canonicalized at registration.
    Accessed components below that root reject symlinks. Without base_path the
    process's filesystem permissions apply. For inspection only, include read_file
    and list_directory. Limits and output_dir are application configuration, hidden
    from model input. Downloads use a separate application-owned output directory.

    A sandbox whose ``owns_filesystem`` is true (the registry's own when none is
    passed) takes the tools inside it, with the same output; the sandbox decides
    which paths exist, so base_path does not apply. The local ``Sandbox`` leaves
    them on the host.
    """
    names = select_tools(
        registry,
        ["read_file", "write_file", "edit_file", "delete", "glob", "grep",
         "list_directory", "generate_file"],
        include, exclude, replace=replace,
    )
    if not names:
        return []
    if sandbox is None:
        sandbox = getattr(registry, "sandbox", None)
    limits = dict(max_read_bytes=max_read_bytes, max_write_bytes=max_write_bytes,
                  max_directory_entries=max_directory_entries, output_dir=output_dir)
    if getattr(sandbox, "owns_filesystem", False):
        if base_path is not None:
            raise ValueError("base_path does not apply to a sandbox that owns its filesystem")
        from .sandbox_filesystem import SandboxFilesystem

        remote = SandboxFilesystem(sandbox, **limits)

        async def call(operation: str, *args: Any) -> str:
            return await getattr(remote, operation)(*args)
    else:
        fs = _Filesystem(base_path, **limits)

        async def call(operation: str, *args: Any) -> str:
            return await asyncio.to_thread(getattr(fs, operation), *args)

    async def scoped_read(path: str, offset: int = 0, limit: int = 100_000) -> str:
        return await call("read", path, offset, limit)

    async def scoped_write(path: str, content: str, overwrite: bool = False) -> str:
        return await call("write", path, content, overwrite)

    async def scoped_list(path: str = ".") -> str:
        return await call("list", path)

    async def scoped_generate(path: str, content: str, display_name: str = "", overwrite: bool = False) -> str:
        return await call("generate", path, content, display_name, overwrite)

    async def scoped_edit(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        return await call("edit", path, old_string, new_string, replace_all)

    async def scoped_delete(path: str, recursive: bool = False) -> str:
        return await call("delete", path, recursive)

    async def scoped_glob(pattern: str, path: str = ".") -> str:
        return await call("glob", pattern, path)

    async def scoped_grep(pattern: str, path: str = ".", include: str = "") -> str:
        return await call("grep", pattern, path, include)

    # (handler, public twin for the docstring, safe to run concurrently)
    handlers = {
        "read_file": (scoped_read, read_file, True),
        "write_file": (scoped_write, write_file, False),
        "edit_file": (scoped_edit, edit_file, False),
        "delete": (scoped_delete, delete, False),
        "glob": (scoped_glob, glob, True),
        "grep": (scoped_grep, grep, True),
        "list_directory": (scoped_list, list_directory, True),
        "generate_file": (scoped_generate, generate_file, False),
    }
    for name in names:
        handler, public, concurrent = handlers[name]
        handler.__doc__ = public.__doc__
        registry.register_tool(handler, name=name, permission=permission,
                               concurrent=concurrent, replace=replace)
    return names
