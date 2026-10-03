"""Open Knowledge Format bundles: parsing, graph, search, git sources, and agent tools."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from harnessx import (
    Agent,
    AgentConfig,
    GitSource,
    HookEvent,
    KnowledgeManager,
    OKFBundle,
    PermissionLevel,
    ProviderResponse,
    ToolCall,
)
from harnessx.hooks import KnowledgeAccessedData
from harnessx.knowledge import (
    KNOWLEDGE_BLOCK_OPEN,
    fetch_git_source,
    normalize_path,
    resolve_link,
    split_frontmatter,
)
from harnessx.providers import LLMProvider
import harnessx.knowledge as knowledge_module


class ScriptedProvider(LLMProvider):
    def __init__(self, responses):
        self._responses = iter(responses)

    async def create(self, **kwargs):
        return next(self._responses)

    async def count_tokens(self, **kwargs):
        return 0


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_bundle(root: Path) -> Path:
    write(root / "index.md", '---\nokf_version: "0.2"\n---\n\n# Catalog\n\n- [Alpha](a.md) - the first concept\n- [Tables](/sub/index.md)\n')
    write(root / "log.md", "# Log\n\n## 2026-09-27\n- created\n")
    write(
        root / "a.md",
        """---
type: Concept
title: Alpha Revenue Metric
description: Total recognized revenue per month.
tags: [finance, revenue]
status: stable
stale_after: 2099-01-01T00:00:00Z
generated:
  by: reference_agent/gemini-2.5-pro
  at: 2026-09-01T00:00:00Z
verified:
  - by: human:analyst
    at: 2026-09-02T00:00:00Z
sources:
  - id: dbt-model
    resource: /sub/b.md
    title: Events table
    last_modified: 2026-08-30T00:00:00Z
resource: bq://proj.finance.revenue
extra_key: preserved
---

# Alpha Revenue Metric

Revenue is summed from the [events table](./sub/b.md) every night.[^dbt-model]
See also the [missing page](/missing.md) and [the spec](https://example.com/spec).

[^dbt-model]: Events table.
""",
    )
    write(
        root / "sub" / "b.md",
        """---
type: BigQuery Table
description: Raw revenue events.
tags: "one, two"
verified: {by: process:nightly, at: 2026-09-03T00:00:00Z}
---

Rows land hourly. Back to [alpha](/a.md). Revenue revenue revenue revenue.
""",
    )
    write(root / "sub" / "index.md", "# Sub\n\n- [B](b.md)\n")
    write(
        root / "sub" / "c.md",
        """---
type: Playbook
title: Alpha Revenue Metric
description: Deprecated revenue playbook.
status: deprecated
stale_after: not-a-date
---

Old revenue guidance.
""",
    )
    write(root / "bad.md", "# No frontmatter here\n")
    write(root / "empty_type.md", '---\ntype: ""\ntitle: Empty\n---\n\nBody.\n')
    write(root / ".hidden" / "h.md", "---\ntype: Concept\n---\nhidden\n")
    (root / "bin.md").write_bytes("---\ntype: Concept\n---\ncaf\xe9".encode("latin-1"))
    return root


@pytest.fixture
def bundle_root(tmp_path: Path) -> Path:
    return build_bundle(tmp_path / "kb")


@pytest.fixture
def bundle(bundle_root: Path) -> OKFBundle:
    return OKFBundle.load(bundle_root)


# ── Loading and parsing ───────────────────────────────────────────────────────

def test_load_collects_concepts_indexes_and_warnings(bundle: OKFBundle) -> None:
    assert sorted(bundle.concepts) == ["/a.md", "/sub/b.md", "/sub/c.md"]
    assert sorted(bundle.indexes) == ["/index.md", "/sub/index.md"]
    assert bundle.okf_version == "0.2"
    assert bundle.name == "kb"
    joined = "\n".join(bundle.warnings)
    assert "/bad.md: missing frontmatter" in joined
    assert "/empty_type.md" in joined
    assert "/bin.md: not UTF-8" in joined
    assert "/sub/c.md: unparsable stale_after" in joined
    assert "hidden" not in joined and "/.hidden/h.md" not in bundle.concepts


def test_concept_fields_are_normalized(bundle: OKFBundle) -> None:
    a = bundle.get("/a.md")
    assert a is not None
    assert a.type == "Concept" and a.title == "Alpha Revenue Metric"
    assert a.tags == ("finance", "revenue") and a.status == "stable"
    assert a.resource == "bq://proj.finance.revenue"
    assert a.generated == {"by": "reference_agent/gemini-2.5-pro", "at": "2026-09-01T00:00:00+00:00"}
    assert a.verified[0]["by"] == "human:analyst" and a.trust_tier == "human-reviewed"
    assert a.sources[0]["id"] == "dbt-model" and a.sources[0]["last_modified"] == "2026-08-30T00:00:00+00:00"
    assert a.frontmatter["extra_key"] == "preserved"
    assert a.stale_after == datetime(2099, 1, 1, tzinfo=timezone.utc)
    assert a.is_stale() is False
    assert a.is_stale(now=datetime(2100, 1, 1, tzinfo=timezone.utc)) is True
    assert a.to_dict()["stale_after"] == "2099-01-01T00:00:00+00:00"

    b = bundle.get("sub/b.md")  # no leading slash also resolves
    assert b is not None
    assert b.title == "b" and b.status == "stable"
    assert b.tags == ("one", "two")
    assert len(b.verified) == 1 and b.trust_tier == "machine-confirmed"

    c = bundle.get("/sub/c.md")
    assert c is not None
    assert c.stale_after is None and c.status == "deprecated" and c.trust_tier == "unverified"


def test_title_falls_back_to_heading_then_filename(tmp_path: Path) -> None:
    write(tmp_path / "kb" / "with-heading.md", "---\ntype: Concept\n---\n\n# Heading Title\n\nbody\n")
    write(tmp_path / "kb" / "plain_name.md", "---\ntype: Concept\n---\nbody\n")
    loaded = OKFBundle.load(tmp_path / "kb")
    assert loaded.concepts["/with-heading.md"].title == "Heading Title"
    assert loaded.concepts["/plain_name.md"].title == "plain name"


def test_unknown_types_and_status_are_tolerated(tmp_path: Path) -> None:
    write(tmp_path / "kb" / "x.md", "---\ntype: Totally New Thing\nstatus: weird\n---\nbody\n")
    loaded = OKFBundle.load(tmp_path / "kb")
    concept = loaded.concepts["/x.md"]
    assert concept.type == "Totally New Thing" and concept.status == "stable"
    assert any("unknown status" in w for w in loaded.warnings)


def test_split_frontmatter_requires_closing_fence() -> None:
    assert split_frontmatter("---\ntype: A\n") == (None, "---\ntype: A\n")
    assert split_frontmatter("\ufeff---\ntype: A\n...\nbody") == ("type: A\n", "body")
    assert split_frontmatter("no fence") == (None, "no fence")


@pytest.mark.parametrize("value", ["2026-99-99", "2026-02-30", "2026-09-27T25:00:00Z"])
@pytest.mark.parametrize("filename", ["bad.md", "index.md"])
def test_invalid_yaml_dates_warn_without_aborting_bundle(tmp_path: Path, filename: str, value: str) -> None:
    write(tmp_path / "good.md", "---\ntype: Concept\n---\nGood concept")
    write(tmp_path / filename, f'---\ntype: Concept\nokf_version: "0.2"\nstale_after: {value}\n---\nCatalog or body')
    loaded = OKFBundle.load(tmp_path)
    assert list(loaded.concepts) == ["/good.md"]
    assert any(f"/{filename}: invalid YAML frontmatter" in warning for warning in loaded.warnings)
    if filename == "index.md":
        assert loaded.index == "Catalog or body"


@pytest.mark.parametrize("entry", [{}, {"by": "human:"}, {"by": "process:"}, {"by": " /v1"},
                                  {"by": "agent/"}, {"by": "agent"}, {"by": 123}, {"by": ["human:reviewer"]}])
def test_invalid_verifiers_do_not_raise_trust(tmp_path: Path, entry: dict) -> None:
    metadata = yaml.safe_dump({"type": "Concept", "verified": entry})
    write(tmp_path / "concept.md", f"---\n{metadata}---\nBody")
    loaded = OKFBundle.load(tmp_path)
    concept = loaded.concepts["/concept.md"]
    assert concept.trust_tier == "unverified"
    assert concept.verified == ()
    assert concept.frontmatter["verified"] == entry
    assert any("invalid or missing actor" in warning for warning in loaded.warnings)


@pytest.mark.parametrize("actor,tier", [("human:reviewer", "human-reviewed"),
                                       ("process:nightly", "machine-confirmed"),
                                       ("agent/v1", "machine-confirmed")])
def test_valid_verifiers_survive_invalid_entries(tmp_path: Path, actor: str, tier: str) -> None:
    metadata = yaml.safe_dump({"type": "Concept", "verified": [{}, {"by": actor}, {"by": "human:"}]})
    write(tmp_path / "concept.md", f"---\n{metadata}---\nBody")
    loaded = OKFBundle.load(tmp_path)
    concept = loaded.concepts["/concept.md"]
    assert concept.trust_tier == tier
    assert concept.verified == ({"by": actor},)
    assert len(loaded.warnings) == 2


def test_load_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OKFBundle.load(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        KnowledgeManager.from_paths([str(tmp_path / "nope")])


# ── Paths and links ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("/a.md", "/a.md"),
        ("a.md", "/a.md"),
        ("./sub/b.md", "/sub/b.md"),
        ("\\sub\\b.md", "/sub/b.md"),
        ("/sub/../a.md", "/a.md"),
        ("/../etc/passwd", None),
        ("../a.md", None),
        ("", None),
        ("/", None),
        (None, None),
    ],
)
def test_normalize_path(raw, expected) -> None:
    assert normalize_path(raw) == expected


def test_resolve_link_forms() -> None:
    assert resolve_link("/x.md", "/dir/from.md") == "/x.md"
    assert resolve_link("./y.md#section", "/dir/from.md") == "/dir/y.md"
    assert resolve_link("../z.md", "/dir/from.md") == "/z.md"
    assert resolve_link("https://example.com/p.md", "/from.md") is None
    assert resolve_link("#anchor", "/from.md") is None
    assert resolve_link("image.png", "/from.md") is None


def test_links_backlinks_and_traversal_guard(bundle: OKFBundle) -> None:
    a = bundle.concepts["/a.md"]
    assert a.links == ("/sub/b.md",)
    assert a.broken_links == ("/missing.md",)
    neighbors = bundle.neighbors("sub/b.md")
    assert neighbors is not None
    assert [n["path"] for n in neighbors["backlinks"]] == ["/a.md", "/sub/index.md"]
    assert [n["path"] for n in neighbors["outbound"]] == ["/a.md"]
    root_index = bundle.neighbors("/index.md")
    assert root_index is not None
    assert {n["path"] for n in root_index["outbound"]} == {"/a.md", "/sub/index.md"}
    assert bundle.get("/../a.md") is None
    assert bundle.neighbors("/nope.md") is None


def test_graph_parses_reference_links_and_ignores_examples(tmp_path: Path) -> None:
    write(tmp_path / "from.md", """---
type: Concept
---
[Full reference][target], [collapsed][], [shortcut], and [nested](notes/a(b).md).
[Space](<notes/with space.md>) and [encoded](notes/with%20space.md).
[Web](//example.com/external.md) and ![Image](/image.md).

`[Inline code](/inline.md)`

```markdown
[Fenced example](/fenced.md)
```

    [Indented example](/indented.md)

<!-- [Comment](/comment.md) -->

[target]: /target.md
[collapsed]: /target.md
[shortcut]: /target.md
""")
    for name in ("target.md", "notes/a(b).md", "notes/with space.md"):
        write(tmp_path / name, "---\ntype: Concept\n---\nTarget")
    loaded = OKFBundle.load(tmp_path)
    concept = loaded.concepts["/from.md"]
    assert concept.links == ("/target.md", "/notes/a(b).md", "/notes/with space.md")
    assert concept.broken_links == ()
    assert [entry["path"] for entry in loaded.neighbors("/target.md")["backlinks"]] == ["/from.md"]


def test_index_keeps_unresolved_links_and_directory_edges(tmp_path: Path) -> None:
    write(tmp_path / "index.md", "[Subdirectory](sub/)\n[Missing][later]\n\n[later]: /later.md")
    write(tmp_path / "sub" / "index.md", "# Subdirectory")
    loaded = OKFBundle.load(tmp_path)
    neighbors = loaded.neighbors("/index.md")
    assert [entry["path"] for entry in neighbors["outbound"]] == ["/sub/index.md"]
    assert neighbors["unresolved"] == [{"path": "/later.md"}]


# ── Search and prompt rendering ───────────────────────────────────────────────

def test_search_ranks_relevant_concepts_and_filters(bundle: OKFBundle) -> None:
    hits = bundle.search("alpha revenue")
    paths = [h["path"] for h in hits]
    assert paths[0] == "/a.md"
    assert set(paths) == {"/a.md", "/sub/b.md", "/sub/c.md"}
    assert bundle.search("revenue", type="bigquery table") and all(
        h["type"] == "BigQuery Table" for h in bundle.search("revenue", type="bigquery table")
    )
    assert [h["path"] for h in bundle.search("revenue", tags=["finance"])] == ["/a.md"]
    assert bundle.search("zzzz") == []
    assert bundle.search("") == []
    assert len(bundle.search("revenue", limit=1)) == 1


def test_bm25_rare_term_outweighs_repeated_common_term(tmp_path: Path) -> None:
    write(tmp_path / "rare.md", "---\ntype: Concept\ntitle: Entry\n---\nquasar background")
    for index in range(10):
        write(tmp_path / f"common-{index}.md", "---\ntype: Concept\ntitle: Entry\n---\n" + "common " * 20)
    loaded = OKFBundle.load(tmp_path)
    hits = loaded.search("common quasar", limit=50)
    assert len(hits) == 11
    assert hits[0]["path"] == "/rare.md"
    assert hits[0]["score"] > hits[1]["score"] > 0


def test_bm25_prefers_focused_body_over_long_unrelated_body(tmp_path: Path) -> None:
    for name, body in (("short", "revenue guide"), ("long", "revenue " + "unrelated " * 100)):
        write(tmp_path / f"{name}.md", f"---\ntype: Concept\ntitle: Entry\n---\n{body}")
    hits = OKFBundle.load(tmp_path).search("revenue")
    assert [hit["path"] for hit in hits] == ["/short.md", "/long.md"]
    assert hits[0]["score"] > hits[1]["score"]


def test_bm25_term_frequency_saturates_without_a_hard_cap(tmp_path: Path) -> None:
    for frequency in (1, 2, 4, 20):
        body = "revenue " * frequency + "background " * (100 - frequency)
        write(tmp_path / f"{frequency}.md", f"---\ntype: Concept\ntitle: Entry\n---\n{body}")
    scores = {hit["path"]: hit["score"] for hit in OKFBundle.load(tmp_path).search("revenue")}
    one, two, four, twenty = (scores[f"/{frequency}.md"] for frequency in (1, 2, 4, 20))
    assert 0 < one < two < four < twenty
    assert two - one > (four - two) / 2 > (twenty - four) / 16


def test_bm25_field_weights_prefer_title_then_tags_then_description(tmp_path: Path) -> None:
    for field in ("title", "tags", "description", "body"):
        title = "nebula" if field == "title" else "Entry"
        tag = "nebula" if field == "tags" else "general"
        description = "nebula" if field == "description" else "overview"
        body = "nebula" if field == "body" else "background"
        write(tmp_path / f"{field}.md", (
            f"---\ntype: Concept\ntitle: {title}\ntags: [{tag}]\ndescription: {description}\n---\n{body}"
        ))
    hits = OKFBundle.load(tmp_path).search("nebula")
    assert [hit["path"] for hit in hits] == ["/title.md", "/tags.md", "/description.md", "/body.md"]
    assert all(first["score"] > second["score"] for first, second in zip(hits, hits[1:]))


def test_bm25_penalizes_draft_and_deprecated_concepts(tmp_path: Path) -> None:
    for status in ("deprecated", "draft", "stable"):
        write(tmp_path / f"{status}.md", f"---\ntype: Concept\ntitle: Revenue\nstatus: {status}\n---\nRecognition.")
    hits = OKFBundle.load(tmp_path).search("revenue")
    assert [hit["status"] for hit in hits] == ["stable", "draft", "deprecated"]


def test_bm25_handles_unicode_empty_fields_and_single_document(tmp_path: Path) -> None:
    assert OKFBundle.load(tmp_path).search("revenue") == []
    write(tmp_path / "unicode.md", "---\ntype: Concept\ntitle: Café Straße\n---\n")
    loaded = OKFBundle.load(tmp_path)
    hits = loaded.search("CAFE\u0301 STRASSE")
    assert [hit["path"] for hit in hits] == ["/unicode.md"]
    assert hits[0]["score"] > 0
    assert loaded.search("café café straße") == hits
    assert loaded.search("café absent straße") == hits
    assert loaded.search("!!!") == []
    assert loaded.search("absent") == []


def test_manager_bm25_uses_shared_statistics_and_rebuilds_after_add(tmp_path: Path) -> None:
    text = "---\ntype: Metric\ntitle: Entry\ntags: [finance]\n---\nrevenue"
    write(tmp_path / "one" / "match.md", text)
    write(tmp_path / "two" / "match.md", text)
    for index in range(5):
        write(tmp_path / "two" / f"other-{index}.md", "---\ntype: Concept\ntitle: Entry\n---\nbackground")
    manager = KnowledgeManager.from_paths([tmp_path / "one", tmp_path / "two"])
    hits = manager.search("revenue", type="Metric", tags=["finance"])
    assert [hit["bundle"] for hit in hits] == ["one", "two"]
    assert hits[0]["score"] == pytest.approx(hits[1]["score"])
    assert manager.search("revenue", limit=1) == hits[:1]
    assert manager.search("revenue", type="Concept") == []
    assert manager.search("revenue", tags=["unrelated"]) == []
    scoped = manager.search("revenue", bundle="one")
    assert scoped == [{**hit, "bundle": "one"} for hit in manager.get("one").search("revenue")]

    write(tmp_path / "three" / "match.md", text)
    manager.add(OKFBundle.load(tmp_path / "three"))
    assert [hit["bundle"] for hit in manager.search("revenue")] == ["one", "three", "two"]


def test_render_summary_uses_index_and_truncates(bundle: OKFBundle) -> None:
    summary = bundle.render_summary()
    assert summary.startswith("## Bundle: kb (3 concepts, okf_version 0.2)")
    assert "[Alpha](a.md)" in summary
    short = bundle.render_summary(char_budget=10)
    assert "[index truncated; use search_concepts]" in short


def test_render_summary_falls_back_to_listing(tmp_path: Path) -> None:
    write(tmp_path / "kb" / "top.md", "---\ntype: Concept\ntitle: Top\ndescription: A top-level concept.\n---\nbody\n")
    write(tmp_path / "kb" / "deep" / "inner.md", "---\ntype: Concept\n---\nbody\n")
    summary = OKFBundle.load(tmp_path / "kb").render_summary()
    assert "- /top.md — Top: A top-level concept." in summary
    assert "and 1 more concepts; use search_concepts" in summary


# ── Agent integration ─────────────────────────────────────────────────────────

def make_agent(bundle_root: Path, *responses: ProviderResponse, **kwargs) -> Agent:
    provider = ScriptedProvider(list(responses) or [ProviderResponse(text="done")])
    return Agent(
        config=AgentConfig(system_prompt="Base prompt."),
        provider=provider,
        knowledge=[str(bundle_root)],
        **kwargs,
    )


def test_agent_without_knowledge_has_none() -> None:
    agent = Agent(provider=ScriptedProvider([]))
    assert agent.knowledge is None
    assert not agent.tools.has_tool("search_concepts")


def test_install_registers_tools_and_prompt_once(bundle_root: Path) -> None:
    agent = make_agent(bundle_root)
    assert agent.knowledge is not None
    for name in ("search_concepts", "read_concept", "get_neighbors"):
        assert agent.tools.get_tool(name).permission_level is PermissionLevel.ALLOW
    prompt = agent.config.system_prompt
    assert prompt.startswith("Base prompt.")
    assert prompt.count(KNOWLEDGE_BLOCK_OPEN) == 1
    assert "## Bundle: kb" in prompt
    # single bundle: no `bundle` argument exposed to the model
    assert "bundle" not in agent.tools.get_tool("search_concepts").input_schema["properties"]
    # a restored session already carries the block (load_session passes the
    # saved prompt back in); installing again must not duplicate it
    restored = Agent(
        config=AgentConfig(system_prompt=prompt),
        provider=ScriptedProvider([]),
        knowledge=agent.knowledge,
    )
    assert restored.config.system_prompt.count(KNOWLEDGE_BLOCK_OPEN) == 1
    assert restored.config.system_prompt == prompt


@pytest.mark.asyncio
async def test_tools_run_through_the_engine_and_fire_hooks(bundle_root: Path) -> None:
    agent = make_agent(
        bundle_root,
        ProviderResponse(
            tool_calls=[
                ToolCall("s", "search_concepts", {"query": "revenue"}),
                ToolCall("r", "read_concept", {"path": "a.md", "max_chars": 200}),
                ToolCall("n", "get_neighbors", {"path": "/a.md"}),
                ToolCall("x", "read_concept", {"path": "/../etc/passwd"}),
            ],
            stop_reason="tool_use",
        ),
        ProviderResponse(text="Answer from the bundle."),
    )
    events = []
    agent.hooks.on(HookEvent.KNOWLEDGE_ACCESSED, lambda ctx: events.append(dict(ctx.data)))

    result = await agent.run("What is revenue?")
    assert result.ok
    allowed = set(KnowledgeAccessedData.__annotations__)
    assert all(set(event) <= allowed for event in events), events
    tools = [e["tool"] for e in events]
    assert tools == ["search_concepts", "read_concept", "get_neighbors", "read_concept"]
    assert events[0]["hits"] >= 2 and events[0]["bundle"] == "kb"
    assert events[1]["found"] is True and events[1]["path"] == "/a.md"
    assert events[3]["found"] is False

    results = [
        block["content"]
        for message in agent.memory.get_messages()
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    search_out, read_out, neighbors_out, bad_out = [str(r) for r in results]
    assert search_out.startswith("1. /a.md — Alpha Revenue Metric [Concept, stable]")
    assert "trust: human-reviewed" in read_out and "stale: no" in read_out
    assert "[dbt-model] Events table — /sub/b.md" in read_out
    assert "[truncated: showing 200 of" in read_out
    assert "Outbound:\n  - /sub/b.md — b" in neighbors_out
    assert "Backlinks:\n  - /index.md" in neighbors_out
    assert "Unresolved links:\n  - /missing.md" in neighbors_out
    assert bad_out == "Concept not found: /../etc/passwd"


@pytest.mark.asyncio
async def test_read_concept_serves_nested_index(bundle_root: Path) -> None:
    agent = make_agent(bundle_root)
    text = await agent.tools.get_tool("read_concept").handler(path="/sub/index.md")
    assert "type: index" in text and "- [B](b.md)" in text


@pytest.mark.asyncio
async def test_search_tool_uses_manager_override_and_preserves_hook(bundle: OKFBundle) -> None:
    calls = []

    class CustomSearch(KnowledgeManager):
        def search(self, query, **kwargs):
            calls.append((query, kwargs))
            return super().search("revenue", **kwargs)

    agent = Agent(provider=ScriptedProvider([]), knowledge=CustomSearch([bundle]))
    events = []
    agent.hooks.on(HookEvent.KNOWLEDGE_ACCESSED, lambda ctx: events.append(dict(ctx.data)))
    result = await agent.tools.get_tool("search_concepts").handler(
        query="income", limit=1, type="Concept", tags=["finance"],
    )
    assert result.startswith("1. /a.md — Alpha Revenue Metric")
    assert calls == [("income", {"bundle": "kb", "limit": 1, "type": "Concept", "tags": ["finance"]})]
    assert events == [{"tool": "search_concepts", "bundle": "kb", "query": "income", "hits": 1, "found": True}]


@pytest.mark.asyncio
async def test_multiple_bundles_require_a_bundle_argument(tmp_path: Path) -> None:
    first = build_bundle(tmp_path / "one")
    second = build_bundle(tmp_path / "two")
    manager = KnowledgeManager.from_paths([first, second])
    agent = Agent(provider=ScriptedProvider([]), knowledge=manager)
    schema = agent.tools.get_tool("search_concepts").input_schema
    assert "bundle" in schema["properties"]
    assert "Several bundles are loaded" in agent.config.system_prompt

    events = []
    agent.hooks.on(HookEvent.KNOWLEDGE_ACCESSED, lambda ctx: events.append(dict(ctx.data)))
    handler = agent.tools.get_tool("search_concepts").handler
    assert (await handler(query="revenue", bundle="nope")).startswith("Unknown bundle 'nope'")
    assert events[-1]["found"] is False
    assert (await handler(query="revenue", bundle="two")).startswith("bundle=two\n1. /a.md")

    hits = manager.search("revenue", limit=2)
    assert [h["bundle"] for h in hits] == ["one", "two"]
    assert manager.warnings() and manager.warnings()[0].startswith("one: ")


def test_duplicate_bundle_names_are_rejected(bundle_root: Path) -> None:
    manager = KnowledgeManager([OKFBundle.load(bundle_root)])
    with pytest.raises(ValueError, match="Duplicate knowledge bundle"):
        manager.add(OKFBundle.load(bundle_root))
    manager.add(OKFBundle.load(bundle_root, name="kb-copy"))
    assert [b.name for b in manager.list()] == ["kb", "kb-copy"]
    with pytest.raises(KeyError):
        manager.get("missing")


# ── Git sources ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://github.com/org/repo/tree/main/bundles/ga4", ("https://github.com/org/repo", "main", "bundles/ga4")),
        ("https://gitlab.com/org/repo/-/tree/v1/kb/", ("https://gitlab.com/org/repo", "v1", "kb")),
        ("https://github.com/org/repo.git", ("https://github.com/org/repo.git", None, None)),
        ("git@github.com:org/repo.git", ("git@github.com:org/repo.git", None, None)),
        ("./local/folder", None),
        ("/abs/folder", None),
        ("", None),
    ],
)
def test_git_source_parse(raw, expected) -> None:
    parsed = GitSource.parse(raw)
    if expected is None:
        assert parsed is None
    else:
        assert (parsed.url, parsed.ref, parsed.subdir) == expected


def test_git_source_names_and_cache_keys() -> None:
    source = GitSource("https://github.com/org/repo.git", ref="main", subdir="bundles/ga4")
    assert source.repo_slug == "repo" and source.bundle_name == "ga4"
    assert source.cache_key.startswith("repo-") and len(source.cache_key) == len("repo-") + 10
    assert GitSource("git@github.com:org/other.git").bundle_name == "other"
    assert GitSource("https://x/y/z", name="custom").bundle_name == "custom"


def test_git_errors_are_reported(tmp_path: Path, monkeypatch) -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="git is required"):
        fetch_git_source(GitSource("https://example.com/org/repo.git"), cache_dir=tmp_path)

    def failing(*args, **kwargs):
        return subprocess.CompletedProcess(args, 128, stdout="", stderr="fatal: repository not found")

    monkeypatch.setattr(subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="repository not found"):
        fetch_git_source(GitSource("https://example.com/org/repo.git"), cache_dir=tmp_path)


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git binary not available")


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
        env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
             "GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ["PATH"], "HOME": str(repo)},
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    build_bundle(repo / "bundles" / "sample")
    write(repo / "README.md", "not a bundle root\n")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "init")
    return repo


@needs_git
def test_from_git_clones_caches_and_refreshes(tmp_path: Path, git_repo: Path, monkeypatch) -> None:
    url = git_repo.as_uri()
    cache = tmp_path / "cache"
    loaded = OKFBundle.from_git(url, subdir="bundles/sample", cache_dir=cache)
    assert loaded.name == "sample" and sorted(loaded.concepts) == ["/a.md", "/sub/b.md", "/sub/c.md"]
    clones = [p for p in cache.iterdir() if (p / ".git").exists()]
    assert len(clones) == 1

    calls = []
    real_run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a[0]) or real_run(*a, **k))
    OKFBundle.from_git(url, subdir="bundles/sample", cache_dir=cache)
    assert calls == []  # cached clone reused without touching git

    write(git_repo / "bundles" / "sample" / "new.md", "---\ntype: Concept\ntitle: New\n---\nfresh\n")
    git(git_repo, "add", ".")
    git(git_repo, "commit", "-q", "-m", "add")
    stale = OKFBundle.from_git(url, subdir="bundles/sample", cache_dir=cache)
    assert "/new.md" not in stale.concepts
    fresh = OKFBundle.from_git(url, subdir="bundles/sample", cache_dir=cache, refresh=True)
    assert "/new.md" in fresh.concepts
    assert any(a[0] == "git" and "fetch" in a for a in calls)

    with pytest.raises(FileNotFoundError, match="Subdirectory"):
        OKFBundle.from_git(url, subdir="bundles/none", cache_dir=cache)


@needs_git
def test_manager_accepts_git_urls_and_sources(tmp_path: Path, git_repo: Path) -> None:
    url = git_repo.as_uri()
    manager = KnowledgeManager.from_paths(
        [GitSource(url, subdir="bundles/sample", name="from-source"), url],
        cache_dir=tmp_path / "cache",
    )
    names = [b.name for b in manager.list()]
    assert names == ["from-source", "repo"]
    whole_repo = manager.get("repo")
    assert "/bundles/sample/a.md" in whole_repo.concepts
    assert any("/README.md: missing frontmatter" in w for w in whole_repo.warnings)


@needs_git
@pytest.mark.parametrize("subdir", ["escaped", "escaped/nested"])
def test_git_subdirectory_cannot_escape_via_symlink(tmp_path: Path, git_repo: Path, subdir: str) -> None:
    outside = tmp_path / "outside"
    write(outside / "index.md", "External fixture")
    write(outside / "nested" / "index.md", "Nested external fixture")
    (git_repo / "escaped").symlink_to(outside, target_is_directory=True)
    git(git_repo, "add", ".")
    git(git_repo, "commit", "-q", "-m", "symlink fixture")
    with pytest.raises(ValueError, match="outside the repository"):
        OKFBundle.from_git(git_repo.as_uri(), subdir=subdir, cache_dir=tmp_path / "cache")


@needs_git
def test_git_subdirectory_allows_internal_symlink(tmp_path: Path, git_repo: Path) -> None:
    (git_repo / "linked").symlink_to("bundles/sample", target_is_directory=True)
    git(git_repo, "add", ".")
    git(git_repo, "commit", "-q", "-m", "internal link")
    loaded = OKFBundle.from_git(git_repo.as_uri(), subdir="linked", cache_dir=tmp_path / "cache")
    assert "/a.md" in loaded.concepts


@needs_git
def test_concurrent_git_loads_publish_one_complete_clone(tmp_path: Path, git_repo: Path, monkeypatch) -> None:
    cache = tmp_path / "cache"
    source = GitSource(git_repo.as_uri(), subdir="bundles/sample")
    dest = cache / source.cache_key
    started, release = threading.Event(), threading.Event()
    calls = []
    original = knowledge_module._run_git

    def delayed_clone(args, **kwargs):
        calls.append(args)
        original(args, **kwargs)
        if args[0] == "clone":
            started.set()
            assert release.wait(5)

    monkeypatch.setattr(knowledge_module, "_run_git", delayed_clone)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(OKFBundle.from_git, source.url, subdir=source.subdir, cache_dir=cache)
        try:
            assert started.wait(5)
            assert not dest.exists(), "A clone must be complete before it becomes the shared cache entry"
            second = pool.submit(OKFBundle.from_git, source.url, subdir=source.subdir, cache_dir=cache)
            with pytest.raises(FutureTimeoutError):
                second.result(timeout=0.15)
        finally:
            release.set()
        assert sorted(first.result(timeout=5).concepts) == sorted(second.result(timeout=5).concepts)
    assert sum(args[0] == "clone" for args in calls) == 1


@needs_git
def test_failed_git_clone_is_cleaned_up_and_can_retry(tmp_path: Path, git_repo: Path, monkeypatch) -> None:
    cache = tmp_path / "cache"

    def fail_clone(args, **kwargs):
        (Path(args[-1]) / ".git").mkdir(parents=True)
        raise RuntimeError("interrupted clone")

    with monkeypatch.context() as patcher:
        patcher.setattr(knowledge_module, "_run_git", fail_clone)
        with pytest.raises(RuntimeError, match="interrupted clone"):
            OKFBundle.from_git(git_repo.as_uri(), cache_dir=cache)
    assert not any(path.is_dir() for path in cache.iterdir())
    loaded = OKFBundle.from_git(git_repo.as_uri(), subdir="bundles/sample", cache_dir=cache)
    assert "/a.md" in loaded.concepts


@needs_git
def test_git_refresh_waits_for_another_process_to_finish_loading(tmp_path: Path, git_repo: Path, monkeypatch) -> None:
    cache, ready, release = tmp_path / "cache", tmp_path / "ready", tmp_path / "release"
    script = """
import sys, time
from pathlib import Path
from harnessx import OKFBundle
url, cache, ready, release = sys.argv[1:]
original = OKFBundle.load.__func__
def slow_load(cls, *args, **kwargs):
    Path(ready).touch()
    deadline = time.monotonic() + 10
    while not Path(release).exists():
        if time.monotonic() > deadline:
            raise TimeoutError('test reader was not released')
        time.sleep(0.01)
    return original(cls, *args, **kwargs)
OKFBundle.load = classmethod(slow_load)
OKFBundle.from_git(url, cache_dir=cache)
"""
    reader = subprocess.Popen([sys.executable, "-c", script, git_repo.as_uri(), str(cache), str(ready), str(release)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    refresh_started = threading.Event()
    original = knowledge_module._run_git

    def observe_refresh(args, **kwargs):
        refresh_started.set()
        return original(args, **kwargs)

    monkeypatch.setattr(knowledge_module, "_run_git", observe_refresh)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and reader.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "The other process did not reach bundle loading"
        with pytest.raises(TimeoutError, match="cache lock timed out"):
            fetch_git_source(GitSource(git_repo.as_uri()), cache_dir=cache, timeout=0.1)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(OKFBundle.from_git, git_repo.as_uri(), cache_dir=cache, refresh=True)
            try:
                assert not refresh_started.wait(0.15), "Refresh started while another process was reading"
            finally:
                release.touch()
            assert "/bundles/sample/a.md" in pending.result(timeout=5).concepts
        assert refresh_started.is_set()
        stdout, stderr = reader.communicate(timeout=5)
        assert reader.returncode == 0, stdout + stderr
    finally:
        release.touch()
        if reader.poll() is None:
            reader.terminate()
        reader.communicate(timeout=5)


# ── Optimizations & Enhanced v0.2 Features ─────────────────────────────────────

def test_search_matches_inflections_and_plurals(tmp_path: Path) -> None:
    write(tmp_path / "kb" / "metric.md", "---\ntype: Metric\ntitle: Active User Metric\n---\nCounts unique users.")
    bundle = OKFBundle.load(tmp_path / "kb")
    # Singular in doc, plural in query:
    hits_plural = bundle.search("metrics")
    assert len(hits_plural) == 1
    assert hits_plural[0]["path"] == "/metric.md"
    # Plural in doc, singular in query:
    write(tmp_path / "kb" / "tables.md", "---\ntype: Table\ntitle: Event Tables\n---\nLogs landing hourly.")
    bundle2 = OKFBundle.load(tmp_path / "kb")
    hits_singular = bundle2.search("table")
    assert any(h["path"] == "/tables.md" for h in hits_singular)


@pytest.mark.parametrize("singular,plural", [("class", "classes"), ("box", "boxes"), ("monkey", "monkeys"),
                                           ("policy", "policies"), ("status", "statuses"), ("bus", "buses"),
                                           ("house", "houses"), ("table", "tables")])
def test_plural_matches_work_in_both_directions(tmp_path: Path, singular: str, plural: str) -> None:
    for document, query in ((singular, plural), (plural, singular)):
        write(tmp_path / "topic.md", f"---\ntype: Concept\ntitle: Entry\n---\n{document}")
        hits = OKFBundle.load(tmp_path).search(query)
        assert [hit["path"] for hit in hits] == ["/topic.md"]


def test_resolve_directory_links_and_read_log(bundle_root: Path) -> None:
    bundle = OKFBundle.load(bundle_root)
    # Directory link ending with slash resolves to index.md
    assert resolve_link("sub/", "/index.md") == "/sub/index.md"
    assert resolve_link("/sub/", "/a.md") == "/sub/index.md"
    # log.md is captured in bundle.logs
    assert "/log.md" in bundle.logs
    assert "created" in bundle.logs["/log.md"]
    # read_concept can read /log.md
    manager = KnowledgeManager([bundle])
    rendered_log = manager.render_concept(bundle, "/log.md")
    assert rendered_log is not None
    assert "type: log" in rendered_log
    assert "created" in rendered_log


def test_read_concept_attested_computation_and_credibility(tmp_path: Path) -> None:
    write(
        tmp_path / "kb" / "revenue.md",
        """---
type: Attested Computation
title: Fiscal Revenue
runtime: bigquery
computation: references/sql/rev.sql
parameters:
  - { name: year, type: integer, required: true }
  - { name: region, required: false }
executor:
  resource: references/skills/run-bq.md
  receipt: [job_id, executed_sql]
attester:
  resource: references/attesters/verify.py
sources:
  - id: policy
    resource: https://wiki.example/rev
    title: Policy
    author: team:finance
    usage_count: 1200
    last_modified: 2026-06-01T00:00:00Z
---
# Computation
SELECT sum(val) FROM t WHERE y = @year
""",
    )
    bundle = OKFBundle.load(tmp_path / "kb")
    manager = KnowledgeManager([bundle])
    # Read with explicit path:
    rendered = manager.render_concept(bundle, "/revenue.md")
    assert rendered is not None
    assert "runtime: bigquery" in rendered
    assert "computation: references/sql/rev.sql" in rendered
    assert "parameters: year: integer (required), region (optional)" in rendered
    assert "executor: references/skills/run-bq.md [receipt: job_id, executed_sql]" in rendered
    assert "attester: references/attesters/verify.py" in rendered
    assert "author: team:finance" in rendered
    assert "usage: 1200" in rendered
    assert "modified: 2026-06-01T00:00:00+00:00" in rendered

    # Read without .md extension:
    rendered_no_ext = manager.render_concept(bundle, "/revenue")
    assert rendered_no_ext == rendered
