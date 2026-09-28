---
type: Concept
title: Analytics Overview
description: How product events flow from the app into metrics and reports.
tags: [analytics, architecture]
status: stable
generated:
  by: human:data-team
  at: 2026-09-27T00:00:00Z
verified:
  - by: human:data-team
    at: 2026-09-27T00:00:00Z
---

# Analytics Overview

The product emits one row per user action into the [events table](/tables/events.md).
Nightly jobs aggregate those rows into metrics such as [active users](/metrics/active-users.md).
Reports follow the [weekly report playbook](/playbooks/weekly-report.md).

Event timestamps are UTC; all daily metrics use the UTC calendar day.
