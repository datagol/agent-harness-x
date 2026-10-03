---
type: Metric
title: Active Users
description: Daily and monthly active users, counted from qualifying product events.
tags: [metric, engagement, dau, mau]
status: stable
stale_after: 2027-09-27T00:00:00Z
generated:
  by: reference_agent/claude-fable-5-1
  at: 2026-09-27T00:00:00Z
verified:
  - by: human:analytics-lead
    at: 2026-09-27T00:00:00Z
sources:
  - id: dau-model
    resource: /tables/events.md
    title: Events table
    author: human:data-team
    last_modified: 2026-09-20T00:00:00Z
  - id: metric-policy
    resource: https://example.com/wiki/metric-policy
    title: Metric definitions policy
---

# Active Users

A user is **active on a day** when they emit at least one qualifying event that day.[^dau-model]
Qualifying events are `session_start`, `feature_used`, and `purchase`; background
syncs and error events do not count.[^metric-policy]

- **DAU**: distinct active users per UTC calendar day.
- **MAU**: distinct users active on at least one of the trailing 30 days.

Both are computed from the [events table](/tables/events.md). Internal test
accounts (`is_internal = true`) are excluded.

[^dau-model]: Nightly aggregation over the events table.
[^metric-policy]: Company metric definitions policy.
