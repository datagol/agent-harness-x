# Knowledge bundles (OKF)

HarnessX consumes [Open Knowledge Format v0.2](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md)
bundles: directories of Markdown concepts with YAML metadata, provenance, and
links. An agent searches with BM25F, reads a concept, and follows its neighbors
using three retrieval tools.

Files are read and indexed synchronously when the bundle loads. Inclusion in
model context is lazy: the root index enters the system prompt, and the agent
requests individual documents as needed.

## Load a bundle

```python
from harnessx import Agent, AgentConfig

agent = Agent(
    config=AgentConfig(system_prompt="Answer from the knowledge bundle and cite sources."),
    knowledge=["./knowledge"],
)
```

Each non-reserved `.md` file is a concept with a non-empty `type` in its
frontmatter. An `index.md` provides a directory listing; a `log.md` records
change history. Both are readable through `read_concept`, but are excluded from
concept search. Hidden files and directories are excluded from loading.

```markdown
---
type: Metric
title: Active Users
description: Users with at least one qualifying product event.
tags: [engagement, metric]
status: stable
stale_after: 2027-09-27T00:00:00Z
verified:
  - by: human:analytics-lead
    at: 2026-09-27T00:00:00Z
sources:
  - id: events
    resource: /tables/events.md
---

Count distinct users from the [events table](/tables/events.md).
```

Files with missing frontmatter, an empty `type`, malformed YAML, or an invalid
YAML timestamp are skipped with a warning. Invalid root-index metadata produces
a warning while its body remains readable. Unknown concept types and additional
frontmatter keys are preserved; missing optional fields are accepted.

```python
from harnessx import OKFBundle

bundle = OKFBundle.load("./knowledge", name="analytics")
print(bundle.warnings)
print(bundle.get("/metrics/active-users.md"))
```

## Search with BM25F

BM25F combines matches across fields using these weights:

| Field | Weight |
| --- | ---: |
| Title | 3 |
| Tags | 2.5 |
| Description | 2 |
| Body | 0.5 |

The score accounts for term rarity, diminishing returns from repeated terms,
and the length of each field (`k1=1.2`, `b=0.75`). Draft scores are multiplied by
0.8 and deprecated scores by 0.5. Trust and staleness are shown when a concept is
read; they do not change the search score.

Tokens use Unicode normalization and case folding. Common English singular and
plural forms match in both directions, including `class/classes`, `box/boxes`,
and `monkey/monkeys`. An expanded match has a 0.95 multiplier; equivalent forms
of one query term contribute their best score per document. This is a limited
plural heuristic, without general stemming, synonym expansion, or embeddings.

```python
hits = bundle.search("active users", limit=5, type="Metric", tags=["engagement"])
for hit in hits:
    print(hit["path"], hit["title"], hit["score"])
```

`type` and `tags` filters are case-insensitive; all requested tags must match.
The result limit is clamped to 1–50. Empty or unmatched queries return an empty
list. Scores express relevance within the searched corpus; comparing scores
from independent bundle searches is not meaningful.

## Use multiple bundles or custom retrieval

```python
from harnessx import KnowledgeManager

manager = KnowledgeManager.from_paths(["./sales-knowledge", "./product-knowledge"])
hits = manager.search("active users", limit=5)  # Shared corpus statistics.
sales_hits = manager.search("active users", bundle="sales-knowledge")
```

Searching all bundles uses a combined index with shared term and field-length
statistics. Adding a bundle invalidates that combined index. A search scoped to
one bundle uses that bundle's statistics. Bundle names must be unique; load an
`OKFBundle` with an explicit `name` when folder names collide.

Override the synchronous `KnowledgeManager.search` method to implement
embedding or hybrid retrieval. The agent's `search_concepts` tool calls that
method, passing its query, bundle, limit, type, and tags. Return hits with the
same `path`, `type`, `title`, `description`, `status`, `score`, and `bundle` keys.

## Read and follow links

| Tool | Inputs | Output |
| --- | --- | --- |
| `search_concepts` | `query`, optional `limit`, `type`, `tags` | Ranked concept paths, titles, metadata, and scores |
| `read_concept` | `path`, optional `max_chars` | Metadata header and Markdown body; also reads indexes and logs |
| `get_neighbors` | `path` | Outbound links, backlinks, and unresolved targets |

The tools are registered with `ALLOW` permission. With multiple bundles, pass a
`bundle` argument to select one. The agent tool searches one bundle per call;
the Python `manager.search()` API also supports searching all bundles.

`/tables/events.md` is relative to the bundle root; `../tables/events.md` is
relative to the current document. Directory links such as `tables/` resolve to
`tables/index.md`. Inline and reference-style Markdown links become graph edges.
Code examples, images, HTML comments, and external URLs do not become edges.
Missing targets remain visible in `get_neighbors`, including links from indexes.

```python
neighbors = bundle.neighbors("/metrics/active-users.md")
print(neighbors["outbound"], neighbors["backlinks"], neighbors["unresolved"])
```

Concept reads accept paths with or without `.md`; a directory path can read its
index. Reads look up already loaded documents in memory. File symlinks outside
the bundle are skipped, and paths that traverse above the bundle root are
rejected.

The default body budget is 8,000 characters; pass `max_chars` to request a larger
body. Metadata is additional to that budget. The root index has a default
4,000-character budget per bundle in the system prompt. Configure these with
`KnowledgeManager(max_body_chars=..., index_char_budget=...)`.

## Trust, freshness, and computation contracts

Trust is derived from valid `verified` actors:

| Verification | Trust tier |
| --- | --- |
| No valid actor | `unverified` |
| `process:<id>` or `<producer>/<version>` actors | `machine-confirmed` |
| At least one `human:<id>` actor | `human-reviewed` |

Identifiers must contain a non-empty ID or producer/version and no whitespace.
Malformed entries such as `{}` or `{by: "human:"}` produce warnings and are
excluded from normalized verification metadata. Their original values remain in
`concept.frontmatter`. These are author-supplied claims; HarnessX does not
authenticate actor identities.

A concept is stale on or after `stale_after`. Missing status defaults to
`stable`; `draft` and `deprecated` are also supported. The prompt asks the agent
to prefer stable, reviewed, current concepts and disclose lifecycle concerns.

`read_concept` includes source credibility fields and, for Attested Computation
concepts, `runtime`, `parameters`, `computation`, `executor`, its receipt fields,
and `attester`. The application supplies execution and attestation; the
knowledge tools retrieve the contract and its references.

## Load from Git

```python
from harnessx import GitSource, KnowledgeManager

manager = KnowledgeManager.from_paths(
    [GitSource("https://github.com/acme/knowledge.git", ref="main", subdir="bundles/sales")],
    cache_dir=".agent_knowledge",
    refresh=True,
)
```

GitHub/GitLab tree URLs can select a branch or tag and subdirectory. The `git`
binary is required. Archives and other remote formats must be extracted to a
local directory before loading.

The first shallow clone is created in a temporary directory and published to
the cache only when complete. Failed clones are cleaned up so a later load can
retry. A lock per repository/ref serializes threads and processes. Refresh and
`OKFBundle.from_git` parsing hold the same lock, preventing readers from seeing
files change midway through a load. The lock file remains alongside the cache.

Subdirectories are resolved before reading and must remain inside the cloned
repository, including when symlinks are involved. Escaping symlinks raise
`ValueError`; missing or invalid subdirectories raise `FileNotFoundError`.
Cached content is reused until `refresh=True` is requested.

`OKFBundle.from_git(..., timeout=30)` sets the lock-wait and individual Git
command timeouts. Lock contention that exceeds the timeout raises
`TimeoutError`; Git failures raise `RuntimeError`. The lower-level
`fetch_git_source` returns a path after unlocking; use `OKFBundle.from_git` for
loading protected against concurrent refreshes.

## Observe retrieval

```python
from harnessx import HookEvent

def on_knowledge(ctx):
    print(ctx.data["tool"], ctx.data.get("path") or ctx.data.get("query"), ctx.data["found"])

agent.hooks.on(HookEvent.KNOWLEDGE_ACCESSED, on_knowledge)
```

Successful searches report `hits`; reads report returned character counts;
neighbor calls report the number of resolved outbound links and backlinks.
Missing bundles or documents emit an event with `found=False`.

Run `python -m examples.knowledge_agent` for the interactive example. Its sample
bundle lives in `examples/knowledge/`; `AGENT_KNOWLEDGE` selects another folder
or Git URL. The same example is available in the harness-web launcher.
