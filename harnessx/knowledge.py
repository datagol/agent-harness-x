"""Knowledge bundles in the Open Knowledge Format (OKF).

An OKF bundle is a directory of markdown files with YAML frontmatter
(spec: https://github.com/GoogleCloudPlatform/open-knowledge-format). There is
no manifest: every non-reserved ``.md`` file is a *concept* whose frontmatter
carries at least a ``type``; ``index.md`` files are directory listings for
progressive disclosure and ``log.md`` records change history. Links between
concepts are ordinary markdown links, so the knowledge graph is emergent.

The agent sees each bundle's root ``index.md`` in its system prompt and pulls
concepts into context lazily with three read-only tools:

- ``search_concepts`` — BM25F search over title, tags, description, body
- ``read_concept``    — one concept by bundle-relative path, with a metadata header
- ``get_neighbors``   — outbound links and backlinks of a concept

Usage:
    agent = Agent(
        config=AgentConfig(system_prompt="Answer from the knowledge bundle."),
        knowledge=[
            "./knowledge",                                        # local folder
            "https://github.com/org/repo/tree/main/bundles/ga4",  # git URL (cloned once)
        ],
    )

Git sources are shallow-cloned into ``.agent_knowledge/`` on first use and
reused afterwards; pass ``KnowledgeManager(..., refresh=True)`` to re-fetch.
Loading happens synchronously when the agent is constructed, like skills.
"""

from __future__ import annotations

import builtins
import fcntl
import hashlib
import math
import os
import posixpath
import re
import subprocess
import tempfile
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence
from urllib.parse import unquote

import yaml
from markdown_it import MarkdownIt

from .hooks import HookContext, HookEvent
from .types import PermissionLevel

if TYPE_CHECKING:
    from .core import Agent


RESERVED_FILENAMES = frozenset({"index.md", "log.md"})
KNOWLEDGE_BLOCK_OPEN = "<knowledge>"
KNOWLEDGE_BLOCK_CLOSE = "</knowledge>"
DEFAULT_CACHE_DIR = ".agent_knowledge"
STATUSES = ("draft", "stable", "deprecated")
TRUST_UNVERIFIED = "unverified"
TRUST_MACHINE = "machine-confirmed"
TRUST_HUMAN = "human-reviewed"

_MARKDOWN = MarkdownIt("commonmark").enable("table")
_HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_TOKEN_RE = re.compile(r"[^\W_]+")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_GIT_URL_RE = re.compile(r"^(https?://|ssh://|git://|git@)")
# https://github.com/org/repo/tree/<ref>/<subdir>  and GitLab's /-/tree/ form.
_TREE_URL_RE = re.compile(r"^(https?://[^/]+/[^/]+/[^/]+?)(?:\.git)?/(?:-/)?tree/([^/]+)(?:/(.*))?/?$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


# ── Value normalization ───────────────────────────────────────────────────────

def _to_datetime(value: Any) -> datetime | None:
    """Coerce a frontmatter value to an aware datetime; None when it cannot be."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z") or text.endswith("z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _iso(value: Any) -> Any:
    """ISO-format datetimes so tool results and hook payloads stay JSON-safe."""
    parsed = _to_datetime(value) if isinstance(value, (datetime, date)) else None
    return parsed.isoformat() if parsed is not None else value


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return _iso(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _actor_entry(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    entry = dict(value)
    if "at" in entry:
        entry["at"] = _iso(entry["at"])
    return entry


def _valid_actor(value: Any) -> bool:
    """Recognize human:<id>, process:<id>, and producer/version actors."""
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        return False
    if value.startswith(("human:", "process:")):
        return bool(value.partition(":")[2])
    producer, separator, version = value.partition("/")
    return bool(producer and separator and version) and ":" not in producer


def _string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tags(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(t).strip() for t in value if str(t).strip())
    if isinstance(value, str):
        return tuple(t.strip() for t in value.split(",") if t.strip())
    return ()


# ── Paths and links ───────────────────────────────────────────────────────────

def normalize_path(raw: Any) -> str | None:
    """Canonical bundle-relative path: leading slash, posix separators, no ``.``/``..``.

    Returns None when the path is empty, is the root itself, or would escape
    the bundle root. ``read_concept`` looks the result up in a dict and never
    touches the filesystem, so traversal is impossible by construction.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip().replace("\\", "/")
    if not text:
        return None
    parts: list[str] = []
    for segment in text.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(segment)
    if not parts:
        return None
    return "/" + "/".join(parts)


def resolve_link(target: str, from_path: str) -> str | None:
    """Resolve a markdown link target to a canonical bundle path.

    Returns None for external URLs, anchor-only links, and non-markdown targets.
    ``/x.md`` is bundle-root-relative; anything else is relative to ``from_path``.
    """
    text = target.strip()
    if not text or text.startswith(("#", "//")) or _SCHEME_RE.match(text):
        return None
    text = unquote(text.split("#", 1)[0].split("?", 1)[0]).replace("\\", "/")
    if not text:
        return None
    candidate = text if text.startswith("/") else posixpath.join(posixpath.dirname(from_path), text)
    if candidate.endswith("/"):
        candidate = posixpath.join(candidate, "index.md")
    normalized = normalize_path(candidate)
    if normalized is None or not normalized.lower().endswith(".md"):
        return None
    return normalized


def _links_in(body: str, from_path: str) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for block in _MARKDOWN.parse(body):
        # Reference links are resolved by the parser. Code, HTML, and image
        # contents are not link_open tokens in the inline stream.
        for token in block.children or []:
            target = token.attrGet("href") if token.type == "link_open" else None
            if isinstance(target, str):
                resolved = resolve_link(target, from_path)
                if resolved is not None and resolved != from_path:
                    seen.setdefault(resolved, None)
    return tuple(seen)


# ── Frontmatter and concepts ──────────────────────────────────────────────────

def split_frontmatter(text: str) -> tuple[str | None, str]:
    """Split a ``---`` frontmatter block from the body. Returns (yaml_text, body).

    ``yaml_text`` is None when the file has no frontmatter or the block is not
    closed (``---`` or ``...`` on its own line).
    """
    text = text.lstrip("\ufeff")
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, text
    for index in range(1, len(lines)):
        if lines[index].strip() in ("---", "..."):
            return "".join(lines[1:index]), "".join(lines[index + 1:]).lstrip("\n")
    return None, text


@dataclass(frozen=True)
class OKFConcept:
    """One OKF concept file, parsed and normalized.

    ``path`` is the canonical bundle-relative path (``/dir/file.md``). Dates
    are normalized to ISO strings inside ``generated``/``verified``/``sources``
    and to an aware datetime for ``stale_after``. ``frontmatter`` keeps the raw
    mapping so unknown keys are never lost.
    """

    path: str
    type: str
    title: str
    description: str
    tags: tuple[str, ...]
    status: str
    stale_after: datetime | None
    generated: dict[str, Any] | None
    verified: tuple[dict[str, Any], ...]
    sources: tuple[dict[str, Any], ...]
    resource: str | None
    frontmatter: dict[str, Any]
    body: str
    links: tuple[str, ...]
    broken_links: tuple[str, ...]
    source_path: Path

    @property
    def trust_tier(self) -> str:
        actors = [str(entry["by"]) for entry in self.verified if _valid_actor(entry.get("by"))]
        if not actors:
            return TRUST_UNVERIFIED
        if any(actor.startswith("human:") for actor in actors):
            return TRUST_HUMAN
        return TRUST_MACHINE

    def is_stale(self, now: datetime | None = None) -> bool:
        if self.stale_after is None:
            return False
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current >= self.stale_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "status": self.status,
            "stale_after": self.stale_after.isoformat() if self.stale_after else None,
            "generated": _jsonable(self.generated),
            "verified": _jsonable(list(self.verified)),
            "sources": _jsonable(list(self.sources)),
            "resource": self.resource,
            "trust_tier": self.trust_tier,
            "links": list(self.links),
            "broken_links": list(self.broken_links),
            "source_path": str(self.source_path),
        }


def parse_concept(rel_path: str, text: str, source_path: Path) -> tuple[OKFConcept | None, list[str]]:
    """Parse one concept file. Returns (concept, warnings); concept is None when
    the file is non-conformant (no frontmatter, invalid YAML, or empty ``type``).

    ``links`` on the returned concept holds every in-bundle candidate; the
    bundle later splits them into existing links and ``broken_links``.
    """
    warnings: list[str] = []
    yaml_text, body = split_frontmatter(text)
    if yaml_text is None:
        return None, [f"{rel_path}: missing frontmatter (non-conformant), skipped"]
    try:
        data = yaml.safe_load(yaml_text)
    except (yaml.YAMLError, ValueError, OverflowError) as exc:
        return None, [f"{rel_path}: invalid YAML frontmatter ({exc}), skipped"]
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return None, [f"{rel_path}: frontmatter is not a mapping, skipped"]

    concept_type = _string(data.get("type"))
    if not concept_type:
        return None, [f"{rel_path}: missing or empty `type` (non-conformant), skipped"]

    heading = _HEADING_RE.search(body)
    title = (
        _string(data.get("title"))
        or (heading.group(1).strip() if heading else None)
        or Path(rel_path).stem.replace("-", " ").replace("_", " ")
    )

    status = (_string(data.get("status")) or "stable").lower()
    if status not in STATUSES:
        warnings.append(f"{rel_path}: unknown status '{status}', treated as stable")
        status = "stable"

    stale_after = None
    if data.get("stale_after") is not None:
        stale_after = _to_datetime(data["stale_after"])
        if stale_after is None:
            warnings.append(f"{rel_path}: unparsable stale_after {data['stale_after']!r}, ignored")

    generated = _actor_entry(data.get("generated"))

    raw_verified = data.get("verified")
    if isinstance(raw_verified, dict):  # spec: a bare mapping is a one-element list
        raw_verified = [raw_verified]
    if not isinstance(raw_verified, list):
        if raw_verified is not None:
            warnings.append(f"{rel_path}: `verified` is not a mapping or list, ignored")
        raw_verified = []
    verified_entries: list[dict[str, Any]] = []
    for index, value in enumerate(raw_verified):
        entry = _actor_entry(value)
        if entry is None or not _valid_actor(entry.get("by")):
            warnings.append(f"{rel_path}: verified[{index}] has an invalid or missing actor, ignored")
            continue
        verified_entries.append(entry)
    verified = tuple(verified_entries)

    sources: list[dict[str, Any]] = []
    raw_sources = data.get("sources")
    if isinstance(raw_sources, list):
        for item in raw_sources:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            if "last_modified" in entry:
                entry["last_modified"] = _iso(entry["last_modified"])
            if not _string(entry.get("resource")):
                warnings.append(f"{rel_path}: source without `resource` ({entry.get('id') or '?'})")
            sources.append(entry)

    concept = OKFConcept(
        path=rel_path,
        type=concept_type,
        title=title,
        description=_string(data.get("description")) or "",
        tags=_tags(data.get("tags")),
        status=status,
        stale_after=stale_after,
        generated=generated,
        verified=verified,
        sources=tuple(sources),
        resource=_string(data.get("resource")),
        frontmatter=data,
        body=body.strip(),
        links=_links_in(body, rel_path),
        broken_links=(),
        source_path=source_path,
    )
    return concept, warnings


# ── Bundle ────────────────────────────────────────────────────────────────────

def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def _term_variants(term: str) -> list[str]:
    """Common English plural candidates; corpus aliases make matching symmetric."""
    variants = [term]
    if not term.isascii() or not term.isalpha() or len(term) < 3 or term in {"news", "series", "species"}:
        return variants
    if term.endswith("ies") and len(term) > 4:
        variants.append(term[:-3] + "y")
    elif term.endswith("es") and len(term) > 4 and term[-3] in ("s", "h", "x", "z"):
        variants.append(term[:-2])
    elif term.endswith("s") and not term.endswith(("ss", "us", "is")) and len(term) > 3:
        variants.append(term[:-1])
    if term.endswith("y") and term[-2] not in "aeiou":
        variants.append(term[:-1] + "ies")
    elif term.endswith(("ss", "us", "x", "z", "ch", "sh")):
        variants.append(term + "es")
    elif not term.endswith("s"):
        variants.append(term + "s")
    return variants


class _BM25Index:
    """BM25F with per-field length normalization and document-level IDF.

    Field frequencies are weighted and combined *before* saturation, as in
    Craswell et al., TREC 2005, section 2:
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/craswell_trec05.pdf
    Positive IDF keeps even single-document bundles searchable. Postings hold
    precomputed term scores, so queries visit only matching documents.
    """

    _WEIGHTS = (3.0, 2.5, 2.0, 0.5)  # title, tags, description, body
    _K1 = 1.2
    _B = 0.75

    def __init__(self, documents: dict[tuple[str, str], OKFConcept]) -> None:
        self._documents = documents
        self._postings: dict[str, dict[tuple[str, str], float]] = {}
        self._aliases: dict[str, set[str]] = {}
        if not documents:
            return
        fields = {
            key: tuple(Counter(_tokens(text)) for text in (
                concept.title, " ".join(concept.tags), concept.description, concept.body,
            ))
            for key, concept in documents.items()
        }
        averages = [
            sum(counts[index].total() for counts in fields.values()) / len(documents)
            for index in range(len(self._WEIGHTS))
        ]
        for key, counts in fields.items():
            for terms, weight, average in zip(counts, self._WEIGHTS, averages):
                if not terms:
                    continue
                norm = 1.0 - self._B + self._B * terms.total() / average
                for term, frequency in terms.items():
                    posting = self._postings.setdefault(term, {})
                    posting[key] = posting.get(key, 0.0) + weight * frequency / norm
        for posting in self._postings.values():
            frequency = len(posting)  # Count each document once across all fields.
            idf = math.log1p((len(documents) - frequency + 0.5) / (frequency + 0.5))
            for key, tf in posting.items():
                posting[key] = idf * (self._K1 + 1.0) * tf / (self._K1 + tf)
        for term in self._postings:
            for variant in _term_variants(term):
                if variant != term:
                    self._aliases.setdefault(variant, set()).add(term)

    def search(
        self,
        query: str,
        *,
        limit: int,
        type: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        scores: dict[tuple[str, str], float] = {}
        for query_term in dict.fromkeys(_tokens(query)):
            term_scores: dict[tuple[str, str], float] = {}
            variants = sorted(set(_term_variants(query_term)) | self._aliases.get(query_term, set()))
            for variant in variants:
                posting = self._postings.get(variant)
                if posting:
                    factor = 1.0 if variant == query_term else 0.95
                    for key, score in posting.items():
                        current = term_scores.get(key, 0.0)
                        if score * factor > current:
                            term_scores[key] = score * factor
            for key, score in term_scores.items():
                scores[key] = scores.get(key, 0.0) + score
        wanted_type = type.strip().casefold() if type else None
        wanted_tags = {tag.strip().casefold() for tag in tags or [] if tag.strip()}
        ranked = []
        for key, score in scores.items():
            concept = self._documents[key]
            if wanted_type and concept.type.casefold() != wanted_type:
                continue
            if wanted_tags and not wanted_tags <= {tag.casefold() for tag in concept.tags}:
                continue
            if concept.status == "deprecated":
                score *= 0.5
            elif concept.status == "draft":
                score *= 0.8
            ranked.append((score, key))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        hits = []
        for score, key in ranked[:max(1, min(int(limit), 50))]:
            concept = self._documents[key]
            hits.append({
                "bundle": key[0],
                "path": concept.path,
                "type": concept.type,
                "title": concept.title,
                "description": concept.description,
                "status": concept.status,
                "score": score,
            })
        return hits


class OKFBundle:
    """A loaded OKF bundle: concepts keyed by canonical path, index bodies,
    conformance warnings, and the emergent link graph."""

    def __init__(
        self,
        name: str,
        root: Path,
        concepts: dict[str, OKFConcept],
        indexes: dict[str, str],
        warnings: list[str],
        okf_version: str | None = None,
        index_links: dict[str, tuple[str, ...]] | None = None,
        logs: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.root = root
        self.okf_version = okf_version
        self.indexes = indexes
        self.logs = logs or {}
        self.warnings = warnings
        known = set(concepts) | set(indexes) | set(self.logs)
        self.concepts: dict[str, OKFConcept] = {}
        for path, concept in concepts.items():
            existing = tuple(link for link in concept.links if link in known)
            broken = tuple(link for link in concept.links if link not in known)
            self.concepts[path] = replace(concept, links=existing, broken_links=broken)
        self._index_links = {
            path: tuple(link for link in links if link in known)
            for path, links in (index_links or {}).items()
        }
        self._index_broken_links = {
            path: tuple(link for link in links if link not in known)
            for path, links in (index_links or {}).items()
        }
        self._backlinks: dict[str, list[str]] = {}
        for path, concept in self.concepts.items():
            for target in concept.links:
                self._backlinks.setdefault(target, []).append(path)
        for path, links in self._index_links.items():
            for target in links:
                self._backlinks.setdefault(target, []).append(path)
        for targets in self._backlinks.values():
            targets.sort()
        self._search = _BM25Index({(self.name, path): c for path, c in self.concepts.items()})

    # ── Loading ──

    @classmethod
    def load(cls, root: str | Path, *, name: str | None = None) -> OKFBundle:
        """Load a bundle from a directory. Never raises for non-conformant
        content: malformed concept files become entries in ``warnings``."""
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            raise FileNotFoundError(f"Knowledge bundle path is not a directory: {root_path}")

        concepts: dict[str, OKFConcept] = {}
        indexes: dict[str, str] = {}
        logs: dict[str, str] = {}
        index_links: dict[str, tuple[str, ...]] = {}
        warnings: list[str] = []
        okf_version: str | None = None

        for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for filename in sorted(filenames):
                if filename.startswith(".") or not filename.lower().endswith(".md"):
                    continue
                file_path = Path(dirpath) / filename
                rel = "/" + file_path.relative_to(root_path).as_posix()
                if file_path.is_symlink():
                    try:
                        resolved = file_path.resolve(strict=True)
                    except OSError:
                        warnings.append(f"{rel}: dangling symlink, skipped")
                        continue
                    if not resolved.is_relative_to(root_path):
                        warnings.append(f"{rel}: symlink outside the bundle, skipped")
                        continue
                try:
                    text = file_path.read_bytes().decode("utf-8")
                except UnicodeDecodeError:
                    warnings.append(f"{rel}: not UTF-8, skipped")
                    continue
                except OSError as exc:
                    warnings.append(f"{rel}: unreadable ({exc}), skipped")
                    continue

                if filename == "index.md":
                    yaml_text, body = split_frontmatter(text)
                    indexes[rel] = body.strip()
                    index_links[rel] = _links_in(body, rel)
                    if rel == "/index.md" and yaml_text:
                        try:
                            data = yaml.safe_load(yaml_text)
                        except (yaml.YAMLError, ValueError, OverflowError) as exc:
                            warnings.append(f"{rel}: invalid YAML frontmatter ({exc})")
                            data = None
                        if isinstance(data, dict) and data.get("okf_version") is not None:
                            okf_version = str(data["okf_version"])
                    continue
                if filename == "log.md":
                    logs[rel] = text.strip()
                    continue

                concept, concept_warnings = parse_concept(rel, text, file_path)
                warnings.extend(concept_warnings)
                if concept is not None:
                    concepts[rel] = concept

        return cls(
            name=name or root_path.name,
            root=root_path,
            concepts=concepts,
            indexes=indexes,
            warnings=warnings,
            okf_version=okf_version,
            index_links=index_links,
            logs=logs,
        )

    @classmethod
    def from_git(
        cls,
        url: str,
        *,
        ref: str | None = None,
        subdir: str | None = None,
        name: str | None = None,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        refresh: bool = False,
        timeout: float = 120,
    ) -> OKFBundle:
        """Clone (or reuse a cached clone of) a git repository and load a bundle
        from it. GitHub/GitLab ``.../tree/<ref>/<subdir>`` URLs are understood."""
        source = GitSource.parse(url) or GitSource(url)
        source = GitSource(
            url=source.url,
            ref=ref or source.ref,
            subdir=subdir or source.subdir,
            name=name or source.name,
        )
        # Hold the cache lock through parsing so refresh cannot change files
        # halfway through a load.
        with _git_source_root(source, cache_dir=cache_dir, refresh=refresh, timeout=timeout) as root:
            return cls.load(root, name=source.bundle_name)

    # ── Access ──

    @property
    def index(self) -> str | None:
        """Body of the root ``index.md``, if the bundle has one."""
        return self.indexes.get("/index.md")

    def get(self, path: str) -> OKFConcept | None:
        normalized = normalize_path(path)
        if normalized is None:
            return None
        concept = self.concepts.get(normalized)
        if concept is None and not normalized.endswith(".md"):
            concept = self.concepts.get(normalized + ".md")
        return concept

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        type: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25F search with title, tag, description, and body weights.

        Term rarity, frequency saturation, and per-field length normalization
        determine relevance. Deprecated and draft concepts are down-weighted.
        """
        return [
            {key: value for key, value in hit.items() if key != "bundle"}
            for hit in self._search.search(query, limit=limit, type=type, tags=tags)
        ]

    def _link_entry(self, path: str) -> dict[str, str]:
        concept = self.concepts.get(path)
        if concept is not None:
            return {"path": path, "title": concept.title, "type": concept.type}
        if path in self.logs:
            return {"path": path, "title": Path(path).parent.name or "root", "type": "log"}
        return {"path": path, "title": Path(path).parent.name or "root", "type": "index"}

    def neighbors(self, path: str) -> dict[str, list[dict[str, str]]] | None:
        """Outbound links, backlinks, and unresolved links of a concept, index, or log.
        None when ``path`` is not in the bundle."""
        normalized = normalize_path(path)
        if normalized is None:
            return None
        if normalized not in self.concepts and not normalized.endswith(".md") and (normalized + ".md") in self.concepts:
            normalized = normalized + ".md"
        elif normalized not in self.indexes and (normalized + "/index.md") in self.indexes:
            normalized = normalized + "/index.md"
        elif normalized not in self.logs and (normalized + "/log.md") in self.logs:
            normalized = normalized + "/log.md"

        if normalized in self.concepts:
            concept = self.concepts[normalized]
            outbound, unresolved = concept.links, concept.broken_links
        elif normalized in self.indexes:
            outbound = self._index_links.get(normalized, ())
            unresolved = self._index_broken_links.get(normalized, ())
        elif normalized in self.logs:
            outbound, unresolved = (), ()
        else:
            return None
        return {
            "outbound": [self._link_entry(p) for p in outbound],
            "backlinks": [self._link_entry(p) for p in self._backlinks.get(normalized, [])],
            "unresolved": [{"path": p} for p in unresolved],
        }

    def render_summary(self, *, char_budget: int = 4000) -> str:
        """The bundle's entry in the system prompt: heading plus the root index
        (or a generated listing when the bundle has none)."""
        version = f", okf_version {self.okf_version}" if self.okf_version else ""
        lines = [f"## Bundle: {self.name} ({len(self.concepts)} concepts{version})"]
        if self.index:
            body = self.index.rstrip()
            if len(body) > char_budget:
                body = body[:char_budget].rstrip() + "\n... [index truncated; use search_concepts]"
            lines.append(body)
        else:
            top_level = [c for p, c in sorted(self.concepts.items()) if p.count("/") == 1]
            shown = top_level[:30]
            for concept in shown:
                suffix = f": {concept.description}" if concept.description else ""
                lines.append(f"- {concept.path} — {concept.title}{suffix}")
            remaining = len(self.concepts) - len(shown)
            if remaining > 0:
                lines.append(f"- ... and {remaining} more concepts; use search_concepts")
            text = "\n".join(lines)
            if len(text) > char_budget:
                return text[:char_budget].rstrip() + "\n... [listing truncated; use search_concepts]"
        return "\n".join(lines)


# ── Git sources ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GitSource:
    """A bundle hosted in a git repository, optionally at a subdirectory/ref."""

    url: str
    ref: str | None = None
    subdir: str | None = None
    name: str | None = None

    @classmethod
    def parse(cls, raw: str) -> GitSource | None:
        """Interpret ``raw`` as a git URL; None when it looks like a local path.
        ``https://host/org/repo/tree/<ref>/<subdir>`` is split into its parts."""
        text = raw.strip()
        if not text:
            return None
        is_git = bool(_GIT_URL_RE.match(text)) or (text.endswith(".git") and "://" in text)
        if not is_git and text.startswith("file://"):
            is_git = True
        if not is_git:
            return None
        tree = _TREE_URL_RE.match(text)
        if tree:
            subdir = (tree.group(3) or "").strip("/") or None
            return cls(url=tree.group(1), ref=tree.group(2), subdir=subdir)
        return cls(url=text)

    @property
    def repo_slug(self) -> str:
        tail = self.url.rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
        if tail.endswith(".git"):
            tail = tail[:-4]
        return _SLUG_RE.sub("-", tail).strip("-") or "bundle"

    @property
    def bundle_name(self) -> str:
        if self.name:
            return self.name
        if self.subdir:
            return Path(self.subdir.strip("/")).name or self.repo_slug
        return self.repo_slug

    @property
    def cache_key(self) -> str:
        digest = hashlib.sha1(f"{self.url}|{self.ref or ''}".encode()).hexdigest()[:10]
        return f"{self.repo_slug}-{digest}"


def _run_git(args: list[str], *, timeout: float, url: str) -> None:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        completed = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=timeout, check=False, env=env,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "git is required to load knowledge from a URL; install git, or clone the "
            f"repository yourself and pass the local path instead of {url}"
        ) from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"git {args[0]} timed out after {timeout:g}s for {url}") from None
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"git {args[0]} failed for {url}: {detail or 'unknown error'}")


def fetch_git_source(
    source: GitSource,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    timeout: float = 120,
) -> Path:
    """Ensure a shallow clone of ``source`` exists under ``cache_dir`` and return
    the bundle root (the clone, or ``subdir`` inside it).

    A cached clone is reused as is unless ``refresh`` is true, in which case the
    ref is fetched again and the checkout reset to it. Cache updates are locked;
    use ``OKFBundle.from_git`` to also hold the lock while reading the files.
    """
    with _git_source_root(source, cache_dir=cache_dir, refresh=refresh, timeout=timeout) as root:
        return root


@contextmanager
def _cache_lock(dest: Path, timeout: float) -> Iterator[None]:
    """Serialize threads and processes using one cache entry. Keep the lock file
    in place so waiters always lock the same inode, even after a failed clone."""
    lock_path = dest.with_name(f".{dest.name}.lock")
    with lock_path.open("a+b") as lock:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Knowledge cache lock timed out after {timeout:g}s: {dest}") from None
                time.sleep(min(0.05, remaining))
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def _git_source_root(
    source: GitSource,
    *,
    cache_dir: str | Path,
    refresh: bool,
    timeout: float,
) -> Iterator[Path]:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Knowledge git timeout must be positive and finite")
    normalized = normalize_path(source.subdir) if source.subdir else None
    if source.subdir and normalized is None:
        raise FileNotFoundError(f"Invalid subdir {source.subdir!r} for {source.url}")
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / source.cache_key
    with _cache_lock(dest, timeout):
        _prepare_git_checkout(source, dest, refresh=refresh, timeout=timeout)
        clone_root = dest.resolve(strict=True)
        root = (dest / normalized.lstrip("/")) if normalized else dest
        try:
            root = root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FileNotFoundError(f"Subdirectory {source.subdir!r} not found in {source.url}") from exc
        if not root.is_relative_to(clone_root):
            raise ValueError(f"Subdirectory {source.subdir!r} resolves outside the repository: {source.url}")
        if not root.is_dir():
            raise FileNotFoundError(f"Subdirectory {source.subdir!r} not found in {source.url}")
        yield root


def _prepare_git_checkout(source: GitSource, dest: Path, *, refresh: bool, timeout: float) -> None:
    if dest.exists():
        if not (dest / ".git").exists():
            raise RuntimeError(
                f"{dest} exists but is not a git clone; remove it and load {source.url} again"
            )
        if refresh:
            fetch = ["-C", str(dest), "fetch", "--depth", "1", "origin"]
            if source.ref:
                fetch.extend(["--", source.ref])
            _run_git(fetch, timeout=timeout, url=source.url)
            _run_git(["-C", str(dest), "reset", "--hard", "FETCH_HEAD"], timeout=timeout, url=source.url)
    else:
        # Publish only a complete clone. TemporaryDirectory removes partial
        # repositories on errors/timeouts so a subsequent call can retry.
        with tempfile.TemporaryDirectory(prefix=f".{dest.name}-", dir=dest.parent) as staging:
            checkout = Path(staging) / "checkout"
            clone = ["clone", "--quiet", "--depth", "1"]
            if source.ref:
                clone.extend(["--branch", source.ref])
            clone.extend(["--", source.url, str(checkout)])
            _run_git(clone, timeout=timeout, url=source.url)
            checkout.rename(dest)


# ── Manager: many bundles, three tools, one prompt block ──────────────────────

class KnowledgeManager:
    """Loads OKF bundles and wires lazy retrieval into an Agent.

    The model sees each bundle's root index in the system prompt and calls
    ``search_concepts`` / ``read_concept`` / ``get_neighbors`` to pull
    knowledge into context only when needed. ``search`` is the seam to
    override for embedding-based retrieval.
    """

    def __init__(
        self,
        bundles: Sequence[OKFBundle] | None = None,
        *,
        index_char_budget: int = 4000,
        max_body_chars: int = 8000,
        default_limit: int = 10,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        refresh: bool = False,
    ) -> None:
        self.index_char_budget = index_char_budget
        self.max_body_chars = max_body_chars
        self.default_limit = default_limit
        self.cache_dir = cache_dir
        self.refresh = refresh
        self._bundles: dict[str, OKFBundle] = {}
        self._search_index: _BM25Index | None = None
        for bundle in bundles or []:
            self.add(bundle)

    @classmethod
    def from_paths(cls, sources: Sequence[str | Path | GitSource], **options: Any) -> KnowledgeManager:
        """Construct a manager by loading every source: a local directory, a
        git URL, or a ``GitSource``. Options go to the constructor."""
        manager = cls(**options)
        for source in sources:
            manager.add(manager.load_source(source))
        return manager

    def load_source(self, source: str | Path | GitSource) -> OKFBundle:
        if isinstance(source, GitSource):
            return OKFBundle.from_git(
                source.url, ref=source.ref, subdir=source.subdir, name=source.name,
                cache_dir=self.cache_dir, refresh=self.refresh,
            )
        if isinstance(source, str):
            git = GitSource.parse(source)
            if git is not None:
                return self.load_source(git)
        return OKFBundle.load(Path(source))

    def add(self, bundle: OKFBundle) -> None:
        if bundle.name in self._bundles:
            raise ValueError(
                f"Duplicate knowledge bundle name: {bundle.name!r}; pass OKFBundle.load(root, name=...)"
            )
        self._bundles[bundle.name] = bundle
        self._search_index = None

    def get(self, name: str) -> OKFBundle:
        if name not in self._bundles:
            raise KeyError(f"Knowledge bundle {name!r} not found. Available: {sorted(self._bundles)}")
        return self._bundles[name]

    def list(self) -> builtins.list[OKFBundle]:
        return builtins.list(self._bundles.values())

    def warnings(self) -> builtins.list[str]:
        return [f"{bundle.name}: {warning}" for bundle in self._bundles.values() for warning in bundle.warnings]

    def search(
        self,
        query: str,
        *,
        bundle: str | None = None,
        limit: int | None = None,
        type: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> builtins.list[dict[str, Any]]:
        """Search one bundle, or all bundles using shared BM25F statistics."""
        limit = limit or self.default_limit
        if bundle or len(self._bundles) == 1:
            target = self.get(bundle) if bundle else next(iter(self._bundles.values()))
            return [
                {**hit, "bundle": target.name}
                for hit in target.search(query, limit=limit, type=type, tags=tags)
            ]
        if self._search_index is None:
            self._search_index = _BM25Index({
                (target.name, path): concept
                for target in self._bundles.values()
                for path, concept in target.concepts.items()
            })
        return self._search_index.search(query, limit=limit, type=type, tags=tags)

    # ── Prompt ──

    def render_catalog(self) -> str:
        """Render the ``<knowledge>`` block appended to the system prompt."""
        if not self._bundles:
            return ""
        lines = [
            KNOWLEDGE_BLOCK_OPEN,
            "You have access to knowledge bundles in the Open Knowledge Format: concept documents "
            "with metadata, linked into a graph. Workflow: consult the bundle index below, call "
            "`search_concepts` to find candidates, `read_concept` to read one (paths are "
            "bundle-relative, e.g. /metrics/revenue.md), and `get_neighbors` to follow links. Cite "
            "the concept path and its listed sources when you answer from a concept. Prefer stable, "
            "human-reviewed, non-stale concepts and say so when a concept is draft, deprecated, or stale.",
        ]
        if len(self._bundles) > 1:
            lines.append("Several bundles are loaded; pass `bundle` to the tools to choose one.")
        for bundle in self._bundles.values():
            lines.append("")
            lines.append(bundle.render_summary(char_budget=self.index_char_budget))
        lines.append(KNOWLEDGE_BLOCK_CLOSE)
        return "\n".join(lines)

    # ── Rendering for tool results ──

    def render_concept(self, bundle: OKFBundle, path: str, *, max_chars: int | None = None) -> str | None:
        """Metadata header plus body for ``read_concept``; None when not found."""
        normalized = normalize_path(path)
        if normalized is None:
            return None
        concept = bundle.concepts.get(normalized)
        if concept is None and not normalized.endswith(".md"):
            concept = bundle.concepts.get(normalized + ".md")
            if concept is not None:
                normalized = normalized + ".md"
        budget = max(200, int(max_chars or self.max_body_chars))
        if concept is None:
            idx_path = normalized if normalized in bundle.indexes else (normalized + "/index.md" if (normalized + "/index.md") in bundle.indexes else None)
            if idx_path is not None:
                header = [f"# {Path(idx_path).parent.name or 'root'} index", f"path: {idx_path}   bundle: {bundle.name}", "type: index"]
                return "\n".join(header) + "\n\n" + _truncate(bundle.indexes[idx_path], budget)
            log_path = normalized if normalized in bundle.logs else (normalized + "/log.md" if (normalized + "/log.md") in bundle.logs else None)
            if log_path is not None:
                header = [f"# {Path(log_path).parent.name or 'root'} log", f"path: {log_path}   bundle: {bundle.name}", "type: log"]
                return "\n".join(header) + "\n\n" + _truncate(bundle.logs[log_path], budget)
            return None
        stale = concept.is_stale()
        header = [
            f"# {concept.title}",
            f"path: {concept.path}   bundle: {bundle.name}",
            f"type: {concept.type}   status: {concept.status}   trust: {concept.trust_tier}   "
            f"stale: {'yes' if stale else 'no'} ({concept.stale_after.isoformat() if concept.stale_after else 'n/a'})",
        ]
        if concept.tags:
            header.append("tags: " + ", ".join(concept.tags))
        if concept.resource:
            header.append(f"resource: {concept.resource}")
        if concept.generated:
            header.append(f"generated: {concept.generated.get('by', '?')} at {concept.generated.get('at', '?')}")
        if concept.verified:
            header.append("verified: " + "; ".join(f"{v.get('by', '?')} at {v.get('at', '?')}" for v in concept.verified))
        if concept.sources:
            header.append("sources:")
            for source in concept.sources:
                label = source.get("title") or source.get("resource") or "?"
                extra: list[str] = []
                if source.get("author"):
                    extra.append(f"author: {source['author']}")
                if source.get("usage_count") is not None:
                    extra.append(f"usage: {source['usage_count']}")
                if source.get("last_modified"):
                    extra.append(f"modified: {source['last_modified']}")
                suffix = f" ({', '.join(extra)})" if extra else ""
                header.append(f"  [{source.get('id') or '-'}] {label} — {source.get('resource', '?')}{suffix}")
        if concept.type.casefold() == "attested computation" or "runtime" in concept.frontmatter:
            runtime = concept.frontmatter.get("runtime")
            if runtime:
                header.append(f"runtime: {runtime}")
            comp = concept.frontmatter.get("computation")
            if comp:
                header.append(f"computation: {comp}")
            params = concept.frontmatter.get("parameters")
            if isinstance(params, list) and params:
                param_strs: list[str] = []
                for p in params:
                    if isinstance(p, dict):
                        p_name = p.get("name", "?")
                        p_type = p.get("type")
                        p_req = "required" if p.get("required") else "optional"
                        param_strs.append(f"{p_name}: {p_type} ({p_req})" if p_type else f"{p_name} ({p_req})")
                    else:
                        param_strs.append(str(p))
                header.append("parameters: " + ", ".join(param_strs))
            executor = concept.frontmatter.get("executor")
            if isinstance(executor, dict):
                res = executor.get("resource", "?")
                receipt = executor.get("receipt")
                receipt_str = f" [receipt: {', '.join(str(r) for r in receipt)}]" if isinstance(receipt, list) else ""
                header.append(f"executor: {res}{receipt_str}")
            attester = concept.frontmatter.get("attester")
            if isinstance(attester, dict):
                res = attester.get("resource", "?")
                header.append(f"attester: {res}")
        if concept.links:
            header.append("links: " + ", ".join(concept.links))
        return "\n".join(header) + "\n\n" + _truncate(concept.body, budget)

    # ── Install ──

    def _resolve(self, bundle: str | None) -> OKFBundle | None:
        if len(self._bundles) == 1:
            return next(iter(self._bundles.values()))
        return self._bundles.get(bundle or "")

    def _unknown_bundle(self, bundle: str | None) -> str:
        return f"Unknown bundle {bundle!r}. Available: {sorted(self._bundles)}"

    def install(self, agent: Agent) -> None:
        """Register the three knowledge tools on the agent and append the
        ``<knowledge>`` block to its system prompt. Fires
        ``HookEvent.KNOWLEDGE_ACCESSED`` on every tool call."""

        async def emit(data: dict[str, Any]) -> None:
            await agent.hooks.emit(
                HookEvent.KNOWLEDGE_ACCESSED,
                HookContext(event=HookEvent.KNOWLEDGE_ACCESSED, agent=agent, data=data),
            )

        async def search_concepts(
            query: str,
            limit: int | None = None,
            type: str | None = None,
            tags: Sequence[str] | None = None,
            bundle: str | None = None,
        ) -> str:
            target = self._resolve(bundle)
            if target is None:
                await emit({"tool": "search_concepts", "bundle": bundle or "", "query": query, "hits": 0, "found": False})
                return self._unknown_bundle(bundle)
            hits = self.search(query, bundle=target.name, limit=limit, type=type, tags=tags)
            await emit({"tool": "search_concepts", "bundle": target.name, "query": query, "hits": len(hits), "found": True})
            if not hits:
                return f"No concepts matched {query!r}."
            lines = []
            if len(self._bundles) > 1:
                lines.append(f"bundle={target.name}")
            for index, hit in enumerate(hits, start=1):
                lines.append(
                    f"{index}. {hit['path']} — {hit['title']} [{hit['type']}, {hit['status']}] (score {hit['score']:.1f})"
                )
                if hit["description"]:
                    lines.append(f"   {hit['description']}")
            return "\n".join(lines)

        async def read_concept(path: str, max_chars: int | None = None, bundle: str | None = None) -> str:
            target = self._resolve(bundle)
            if target is None:
                await emit({"tool": "read_concept", "bundle": bundle or "", "path": path, "found": False})
                return self._unknown_bundle(bundle)
            rendered = self.render_concept(target, path, max_chars=max_chars)
            normalized = normalize_path(path) or path
            await emit({
                "tool": "read_concept", "bundle": target.name, "path": normalized,
                "found": rendered is not None, "body_chars": len(rendered) if rendered else 0,
            })
            return rendered if rendered is not None else f"Concept not found: {path}"

        async def get_neighbors(path: str, bundle: str | None = None) -> str:
            target = self._resolve(bundle)
            if target is None:
                await emit({"tool": "get_neighbors", "bundle": bundle or "", "path": path, "found": False})
                return self._unknown_bundle(bundle)
            result = target.neighbors(path)
            normalized = normalize_path(path) or path
            hits = len(result["outbound"]) + len(result["backlinks"]) if result else 0
            await emit({
                "tool": "get_neighbors", "bundle": target.name, "path": normalized,
                "found": result is not None, "hits": hits,
            })
            if result is None:
                return f"Concept not found: {path}"
            lines = []
            for label, key in (("Outbound", "outbound"), ("Backlinks", "backlinks"), ("Unresolved links", "unresolved")):
                lines.append(f"{label}:")
                entries = result[key]
                if not entries:
                    lines.append("  (none)")
                for entry in entries:
                    title = entry.get("title")
                    lines.append(f"  - {entry['path']}" + (f" — {title}" if title else ""))
            return "\n".join(lines)

        bundle_property: dict[str, Any] = {}
        if len(self._bundles) > 1:
            bundle_property = {
                "bundle": {
                    "type": "string",
                    "description": f"Bundle to query; one of {sorted(self._bundles)}.",
                },
            }

        agent.tools.register_with_schema(
            name="search_concepts",
            description=(
                "Search the knowledge bundle for concepts matching a query. Returns ranked hits with "
                "path, type, title, description, and score. Call this before read_concept."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords or a short phrase."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Max hits (default 10)."},
                    "type": {"type": "string", "description": "Only concepts of this OKF type."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Require all of these tags."},
                    **bundle_property,
                },
                "required": ["query"],
            },
            handler=search_concepts,
            permission=PermissionLevel.ALLOW,
        )
        agent.tools.register_with_schema(
            name="read_concept",
            description=(
                "Read one concept by its bundle-relative path (e.g. /metrics/revenue.md, as returned by "
                "search_concepts or listed in the index). Returns a metadata header (type, status, trust "
                "tier, staleness, tags, sources) followed by the markdown body."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle-relative path of the concept."},
                    "max_chars": {"type": "integer", "minimum": 200, "description": "Body character budget (default 8000)."},
                    **bundle_property,
                },
                "required": ["path"],
            },
            handler=read_concept,
            permission=PermissionLevel.ALLOW,
        )
        agent.tools.register_with_schema(
            name="get_neighbors",
            description=(
                "List the concepts a concept links to (outbound) and the concepts that link to it "
                "(backlinks), with titles. Use it to explore related knowledge from a concept you have read."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle-relative path of the concept."},
                    **bundle_property,
                },
                "required": ["path"],
            },
            handler=get_neighbors,
            permission=PermissionLevel.ALLOW,
        )

        catalog = self.render_catalog()
        if catalog:
            existing = agent.config.system_prompt or ""
            if catalog not in existing:
                agent.config.system_prompt = f"{existing}\n\n{catalog}" if existing else catalog


def _truncate(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return (
        text[:budget].rstrip()
        + f"\n\n[truncated: showing {budget} of {len(text)} characters; call read_concept with a larger max_chars]"
    )


__all__ = [
    "DEFAULT_CACHE_DIR",
    "GitSource",
    "KnowledgeManager",
    "OKFBundle",
    "OKFConcept",
    "fetch_git_source",
    "normalize_path",
    "parse_concept",
    "resolve_link",
    "split_frontmatter",
]
