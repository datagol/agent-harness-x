---
name: sql-explain
description: Explain a SQL query plain-English and flag obvious performance issues
---

# SQL Explain

When the user shares a SQL query, produce two short sections.

**1. What it does** — describe the result set in one or two sentences,
naming the tables and the filtering/aggregation logic. No SQL jargon if avoidable.

**2. Performance flags** — bulleted list. Only include items that actually
apply to the query you were given:

- `SELECT *` on a wide table
- Function call on an indexed column in `WHERE` (e.g. `LOWER(email) = ...`)
- Missing index hint for the join predicate
- `OR` across columns that prevents index use
- N+1 risk if this is being run per-row in app code
- Implicit type cast in a comparison
- Cartesian product (missing `JOIN ... ON`)

If none apply, write "No obvious issues." Don't invent problems to fill the list.
