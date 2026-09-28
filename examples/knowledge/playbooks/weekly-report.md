---
type: Playbook
title: Weekly usage report
description: Steps to produce the Monday usage report from the standard metrics.
tags: [playbook, reporting]
status: draft
generated:
  by: reference_agent/claude-fable-5-1
  at: 2026-09-27T00:00:00Z
---

# Weekly usage report

1. Pull DAU and MAU for the last 7 days from [active users](/metrics/active-users.md).
2. Compare with the prior week; flag any day-over-day change above 15%.
3. Note upstream incidents that affected the [events table](/tables/events.md).
4. Post the summary in the analytics channel before 10:00 UTC on Monday.

This playbook is still a draft; confirm thresholds with the analytics lead.
