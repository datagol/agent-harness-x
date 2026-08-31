---
name: commit-message
description: Write a clear, conventional commit message from a diff or change description
---

# Commit Message

When the user shows you a diff or describes changes, produce a single commit message.

**Format:**

```
<type>(<scope>): <short summary in imperative, ≤50 chars>

<body: why this change exists — not what changed. Wrap at 72 cols.>
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`.

**Rules:**

- Imperative mood ("add", not "added").
- The summary should make sense to a reviewer who hasn't seen the diff.
- Skip the body for trivial changes (typo fixes, dep bumps).
- Never include AI-attribution lines, emoji, or "Co-Authored-By: Claude".

Output the commit message only — no surrounding explanation.
