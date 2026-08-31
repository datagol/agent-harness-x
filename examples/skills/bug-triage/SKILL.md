---
name: bug-triage
description: Structured intake for a bug report — extract repro, severity, owner area
---

# Bug Triage

When a user pastes a bug report or describes an issue, produce a structured
triage record. Ask for missing fields before guessing.

**Required fields (ask if not provided):**

1. **Title** — one line, action-oriented ("Cart total wrong when applying two coupons")
2. **Repro steps** — numbered list. If user can't reproduce, mark `repro: intermittent`.
3. **Expected vs. actual** — two short sentences.
4. **Severity** — `S1` (data loss / outage), `S2` (broken core flow, workaround exists), `S3` (degraded UX), `S4` (cosmetic).
5. **Owner area** — best guess at the system: `auth`, `billing`, `cart`, `search`, `infra`, `unknown`.

**Output format (YAML):**

```yaml
title: ...
severity: S2
owner_area: cart
repro:
  - Step 1
  - Step 2
expected: ...
actual: ...
notes: ...   # optional
```

If the user gave you less than 50% of the required fields, ask follow-up
questions instead of producing the YAML.
