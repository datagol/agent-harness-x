---
name: code-review
description: Review a unified diff for bugs, style issues, and security concerns
---

# Code Review

When the user asks for a code review, follow this process:

1. **Read the diff carefully.** Note added vs. removed lines.
2. **Flag bugs first.** Off-by-one errors, null derefs, race conditions.
3. **Then style.** Naming, formatting, missing docstrings.
4. **Then security.** Injection points, secrets in code, weak crypto.

Output format: one bulleted list grouped by category. Cite line numbers.
