---
type: BigQuery Table
title: Events table
description: One row per product event, partitioned by event date.
resource: bq://acme-analytics.product.events
tags: [table, events, raw]
status: stable
stale_after: 2027-03-31T00:00:00Z
generated:
  by: process:schema-export
  at: 2026-09-20T00:00:00Z
verified:
  - by: process:schema-export
    at: 2026-09-20T00:00:00Z
---

# Events table

Raw product telemetry. Columns:

| column | type | notes |
|---|---|---|
| event_ts | TIMESTAMP | UTC |
| user_id | STRING | stable per account |
| event_name | STRING | e.g. session_start, feature_used, purchase |
| is_internal | BOOL | true for staff and test accounts |
| properties | JSON | event-specific payload |

Partitioned by `DATE(event_ts)`; roughly 40M rows per day. Feeds
[active users](/metrics/active-users.md) and every other engagement metric.
