"""File tools that run inside an execution backend instead of on the host.

A backend that owns its filesystem (``ExecutionBackend.owns_filesystem``) gets
the same eight file tools as the host, with the same output: only where the
bytes live changes. Each operation is a short POSIX shell snippet sent through
``backend.execute_command``; file content travels as stdin on the way in and as
base64 on the way out. Paths go through ``backend.resolve_path`` first, which
decides what the model may reach.

The sandbox is the boundary here, not this module: it does not re-check every
path component for symlinks the way the host tools do. A symlink is still
never read, replaced, listed into, or followed by a walk.
"""

from __future__ import annotations

import base64
import builtins
import errno
import fnmatch
import os
import posixpath
import re
import shlex
import uuid
from typing import Any

from .filesystem import _Filesystem, _positive_limit

# Exit codes the snippets use to report a refusal; anything else is a failure.
_NOT_FOUND, _SYMLINK, _IS_DIRECTORY, _NOT_REGULAR, _EXISTS, _NOT_DIRECTORY, _NOT_EMPTY = range(40, 47)

# Paths with directory checks, shared by the snippets that work on a directory.
_DIRECTORY_CHECK = (
    '[ -L "$p" ] && exit 41\n'
    '[ -e "$p" ] || exit 40\n'
    '[ -d "$p" ] || exit 45\n'
)

# Matches _Filesystem._walk: it stops descending below 25 levels.
_MAX_DEPTH = 25
_MAX_WALK_ENTRIES = 200_000


def _not_a_directory(path: str) -> dict[int, BaseException]:
    refusal = PermissionError(f"Directory component {path!r} is a symlink or not a directory")
    return {_SYMLINK: refusal, _NOT_DIRECTORY: refusal}


class SandboxFilesystem:
    """The host ``_Filesystem`` operations, run inside ``backend``."""

    def __init__(
        self, backend: Any, *, max_read_bytes: int = 1_000_000,
        max_write_bytes: int = 10_000_000, max_directory_entries: int = 1_000,
        output_dir: str | None = None,
    ) -> None:
        self.backend = backend
        self.max_read_bytes = _positive_limit(max_read_bytes, "max_read_bytes")
        self.max_write_bytes = _positive_limit(max_write_bytes, "max_write_bytes")
        self.max_directory_entries = _positive_limit(max_directory_entries, "max_directory_entries")
        self.output_dir = os.path.abspath(output_dir) if output_dir is not None else None

    # ── plumbing ─────────────────────────────────────────────────────────

    def _resolve(self, path: str, *, write: bool = False) -> str:
        if not isinstance(path, str) or not path or "\0" in path:
            raise ValueError("path must be a nonempty string without NUL bytes")
        return self.backend.resolve_path(path, write=write)

    async def _run(self, script: str, path: str, *, stdin: bytes | None = None,
                   refusals: dict[int, BaseException] | None = None) -> str:
        """Run ``script``; map its refusal exit codes to the host tools' exceptions."""
        result = await self.backend.execute_command(script, stdin=stdin)
        if result.timed_out:
            raise TimeoutError(f"The sandbox did not finish the operation on {path!r} in time")
        if result.exit_code == 0:
            return result.stdout
        mapped = {
            _NOT_FOUND: FileNotFoundError(errno.ENOENT, "No such file or directory", path),
            _NOT_DIRECTORY: NotADirectoryError(errno.ENOTDIR, "Not a directory", path),
            **(refusals or {}),
        }
        if result.exit_code in mapped:
            raise mapped[result.exit_code]
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.exit_code}"
        raise OSError(f"Sandbox file operation on {path!r} failed: {detail}")

    async def _fetch(self, path: str, limit: int) -> tuple[int, bytes]:
        """(size, first ``limit`` bytes) of a regular file."""
        script = (
            f"p={shlex.quote(self._resolve(path))}\n"
            '[ -L "$p" ] && exit 41\n'
            '[ -e "$p" ] || exit 40\n'
            '[ -d "$p" ] && exit 42\n'
            '[ -f "$p" ] || exit 43\n'
            'wc -c < "$p"\n'
            f'head -c {limit} -- "$p" | base64\n'
        )
        output = await self._run(script, path, refusals={
            _SYMLINK: PermissionError("Reading symlinks is not allowed"),
            _IS_DIRECTORY: IsADirectoryError(path),
            _NOT_REGULAR: PermissionError("Only regular files can be read"),
        })
        size, _, encoded = output.partition("\n")
        return int(size.strip()), base64.b64decode(encoded)

    async def _walk(self, path: str) -> builtins.list[tuple[builtins.list[str], bool]]:
        """(relative parts, is_directory) below ``path``, in _Filesystem._walk order."""
        script = (
            f"p={shlex.quote(self._resolve(path))}\n" + _DIRECTORY_CHECK
            + 'cd -- "$p" || exit 1\n'
            f"find . -mindepth 1 -maxdepth {_MAX_DEPTH} "
            r"\( -type l -prune \) -o \( -type d -exec printf 'D\t%s\n' {} + \) -o -exec printf 'F\t%s\n' {} +"
            f" | head -n {_MAX_WALK_ENTRIES}\n"
        )
        output = await self._run(script, path, refusals=_not_a_directory(path))
        entries = []
        for line in output.splitlines():
            kind, _, name = line.partition("\t")
            if name.startswith("./"):
                entries.append((name[2:].split("/"), kind == "D"))
        # Depth-first with sorted children is lexicographic order on the parts.
        return sorted(entries, key=lambda entry: entry[0])

    # ── the eight operations ─────────────────────────────────────────────

    async def read(self, path, offset, limit, *, numbered=True):
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 0:
            raise ValueError("offset and limit must be nonnegative integers")
        size, data = await self._fetch(path, self.max_read_bytes)
        if limit == 0:
            return ""
        # Line by line with a byte budget, as _Filesystem.read does with readline().
        remaining, line_number, output, position = self.max_read_bytes, 0, [], 0
        while remaining > 0 and line_number < offset + limit:
            end = data.find(b"\n", position, position + remaining)
            line = data[position:end + 1] if end != -1 else data[position:position + remaining]
            if not line:
                break
            position += len(line)
            remaining -= len(line)
            if line_number >= offset:
                text = line.decode("utf-8", errors="replace")
                output.append(f"{line_number + 1}\t{text}" if numbered else text)
            line_number += 1
        if position < size:
            output.append("\n[truncated: read byte or line limit reached]\n")
        return "".join(output)

    async def _read_all(self, path):
        """Whole file, exactly as stored (see _Filesystem._read_all)."""
        size, data = await self._fetch(path, self.max_read_bytes + 1)
        if size > self.max_read_bytes:
            raise ValueError(f"File is {size} bytes, above max_read_bytes")
        return data.decode("utf-8", errors="replace")

    async def write(self, path, content, overwrite):
        if not isinstance(content, str) or type(overwrite) is not bool:
            raise TypeError("content must be a string and overwrite must be a bool")
        if len(content) > self.max_write_bytes:
            raise ValueError("Content exceeds max_write_bytes")
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_write_bytes:
            raise ValueError("Content exceeds max_write_bytes")
        target = self._resolve(path, write=True)
        directory, name = posixpath.split(target)
        if not name:
            raise IsADirectoryError(path)
        flag = "1" if overwrite else "0"
        script = (
            f"p={shlex.quote(target)}\n"
            f"d={shlex.quote(directory)}\n"
            f"t={shlex.quote(posixpath.join(directory, f'.harness-{uuid.uuid4().hex}.tmp'))}\n"
            '[ -L "$p" ] && exit 41\n'
            f'if [ -e "$p" ]; then [ -f "$p" ] || exit 43; [ {flag} = 1 ] || exit 44; fi\n'
            'mkdir -p -- "$d" 2>/dev/null || exit 45\n'
            'cat > "$t" || { rm -f -- "$t"; exit 1; }\n'
            # Rename replaces the entry itself; a hard link creates only if absent.
            f'if [ {flag} = 1 ]; then mv -f -- "$t" "$p"; exit $?; fi\n'
            'ln -- "$t" "$p" 2>/dev/null; s=$?\n'
            'rm -f -- "$t"\n'
            '[ $s -eq 0 ] || exit 44\n'
        )
        await self._run(script, path, stdin=encoded, refusals={
            _SYMLINK: PermissionError("Only regular files can be replaced; symlinks are not allowed"),
            _NOT_REGULAR: PermissionError("Only regular files can be replaced; symlinks are not allowed"),
            _EXISTS: FileExistsError("File exists; set overwrite=True to replace it"),
            _NOT_DIRECTORY: PermissionError(f"Directory component of {path!r} is a symlink or not a directory"),
        })
        return f"Written {len(encoded)} bytes to {path}"

    async def list(self, path):
        script = (
            f"p={shlex.quote(self._resolve(path))}\n" + _DIRECTORY_CHECK
            + 'cd -- "$p" || exit 1\n'
            'for f in * .[!.]* ..?*; do\n'
            '  if [ -L "$f" ]; then printf \'L\\t%s\\n\' "$f"\n'
            '  elif [ -d "$f" ]; then printf \'D\\t%s\\n\' "$f"\n'
            '  elif [ -f "$f" ]; then printf \'F\\t%s\\t%s\\n\' "$(wc -c < "$f" | tr -d \' \')" "$f"\n'
            '  elif [ -e "$f" ]; then printf \'F\\t0\\t%s\\n\' "$f"\n'
            '  fi\n'
            f'done | head -n {self.max_directory_entries + 1}\n'
        )
        output = await self._run(script, path, refusals=_not_a_directory(path))
        lines = []
        for row in output.splitlines():
            kind, _, rest = row.partition("\t")
            if kind == "L":
                lines.append(f"  {rest} [symlink; access disabled]")
            elif kind == "D":
                lines.append(f"  {rest}/")
            elif kind == "F":
                size, _, name = rest.partition("\t")
                lines.append(f"  {name} ({size} bytes)")
        truncated = len(lines) > self.max_directory_entries
        lines = lines[:self.max_directory_entries]
        result = f"Directory: {path}\n" + "\n".join(sorted(lines)) if lines else f"Directory: {path} (empty)"
        return result + ("\n[truncated: directory entry limit reached]" if truncated else "")

    async def edit(self, path, old_string, new_string, replace_all):
        if not isinstance(old_string, str) or not isinstance(new_string, str):
            raise TypeError("old_string and new_string must be strings")
        if type(replace_all) is not bool:
            raise TypeError("replace_all must be a bool")
        if not old_string:
            raise ValueError("old_string must be nonempty; use write_file to create a file")
        if old_string == new_string:
            raise ValueError("old_string and new_string are identical")
        self._resolve(path, write=True)  # refuse a read-only path before reading it
        content = await self._read_all(path)
        occurrences = content.count(old_string)
        if occurrences == 0:
            raise ValueError("old_string was not found; it must match the file exactly")
        if occurrences > 1 and not replace_all:
            raise ValueError(
                f"old_string matches {occurrences} times; include more surrounding "
                "text to make it unique, or set replace_all=True"
            )
        await self.write(path, content.replace(old_string, new_string), True)
        return f"Replaced {occurrences if replace_all else 1} occurrence(s) in {path}"

    async def delete(self, path, recursive):
        if type(recursive) is not bool:
            raise TypeError("recursive must be a bool")
        target = self._resolve(path, write=True)
        if target == self.backend.resolve_path("."):
            raise PermissionError("Refusing to delete the base directory itself")
        flag = "1" if recursive else "0"
        script = (
            f"p={shlex.quote(target)}\n"
            # Removing a link never touches its target.
            'if [ -L "$p" ]; then rm -f -- "$p" && echo L; exit $?; fi\n'
            '[ -e "$p" ] || exit 40\n'
            'if [ ! -d "$p" ]; then rm -f -- "$p" && echo F; exit $?; fi\n'
            f'if [ {flag} = 1 ]; then\n'
            '  n=$(find "$p" -mindepth 1 | wc -l | tr -d \' \')\n'
            '  rm -rf -- "$p" && echo "R $n"; exit $?\n'
            'fi\n'
            'rmdir -- "$p" 2>/dev/null && echo D || exit 46\n'
        )
        output = (await self._run(script, path, refusals={
            _NOT_EMPTY: OSError(errno.ENOTEMPTY, "Directory not empty", path),
        })).strip()
        if output == "L":
            return f"Deleted symlink {path}"
        if output == "F":
            return f"Deleted {path}"
        if output == "D":
            return f"Deleted empty directory {path}"
        return f"Deleted {path} and {output.split()[-1]} entries below it"

    async def glob(self, pattern, path):
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern must be a nonempty string")
        matches = []
        for parts, is_directory in await self._walk(path):
            relative = "/".join(parts)
            if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(posixpath.basename(relative), pattern):
                matches.append(relative + ("/" if is_directory else ""))
                if len(matches) >= self.max_directory_entries:
                    return _Filesystem._listing(path, matches, f"pattern {pattern!r}", truncated=True)
        return _Filesystem._listing(path, matches, f"pattern {pattern!r}")

    async def grep(self, pattern, path, include):
        try:
            expression = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression: {exc}") from None
        candidates = [
            "/".join(parts) for parts, is_directory in await self._walk(path)
            if not is_directory and (not include or fnmatch.fnmatch(parts[-1], include))
        ]
        hits, budget = [], self.max_read_bytes
        for relative, content in await self._contents(path, candidates):
            budget -= len(content)
            for number, line in enumerate(content.splitlines(), 1):
                if expression.search(line):
                    hits.append(f"  {relative}:{number}: {line.strip()[:200]}")
                    if len(hits) >= self.max_directory_entries:
                        return _Filesystem._listing(path, hits, f"pattern {pattern!r}", truncated=True)
            if budget <= 0:
                return _Filesystem._listing(path, hits, f"pattern {pattern!r}", truncated=True)
        return _Filesystem._listing(path, hits, f"pattern {pattern!r}")

    async def _contents(self, path: str, files: builtins.list[str]) -> builtins.list[tuple[str, str]]:
        """Regular files among ``files`` (relative to ``path``), in order, in one
        round trip, stopping once the read budget is spent. Oversized and
        unreadable files are skipped, as the host grep skips them."""
        if not files:
            return []
        script = (
            f"p={shlex.quote(self._resolve(path))}\n" + _DIRECTORY_CHECK
            + 'cd -- "$p" || exit 1\n'
            # The budget below is in characters, this one in bytes: allow for
            # 4-byte UTF-8 so the snippet never stops before the budget does.
            f"budget={4 * self.max_read_bytes}\n"
            "while IFS= read -r f; do\n"
            '  [ -f "$f" ] && [ ! -L "$f" ] && [ -r "$f" ] || continue\n'
            '  s=$(wc -c < "$f" | tr -d \' \')\n'
            f'  [ "$s" -le {self.max_read_bytes} ] || continue\n'
            "  printf '@@FILE\\t%s\\n' \"$f\"\n"
            '  base64 < "$f"\n'
            "  printf '@@END\\n'\n"
            "  budget=$((budget - s))\n"
            '  [ "$budget" -gt 0 ] || break\n'
            "done\n"
        )
        output = await self._run(script, path, stdin=("\n".join(files) + "\n").encode("utf-8"))
        found: builtins.list[tuple[str, str]] = []
        current: str | None = None
        chunks: builtins.list[str] = []
        for line in output.splitlines():
            if line.startswith("@@FILE\t"):
                current, chunks = line[len("@@FILE\t"):], []
            elif line == "@@END" and current is not None:
                found.append((current, base64.b64decode("".join(chunks)).decode("utf-8", errors="replace")))
                current = None
            elif current is not None:
                chunks.append(line)
        return found

    async def generate(self, path, content, display_name, overwrite):
        from .file_output import make_content_downloadable

        await self.write(path, content, overwrite)
        # The file stays in the sandbox; the download is published on the host.
        return make_content_downloadable(
            posixpath.basename(path), content.encode("utf-8"), display_name or path, output_dir=self.output_dir,
        )
