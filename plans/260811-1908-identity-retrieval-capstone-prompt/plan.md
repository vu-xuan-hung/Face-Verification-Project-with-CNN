---
title: "Identity Retrieval Metrics and Capstone Prompt"
description: "Add auditable identity-level Precision@5/Recall@5 and a source-grounded GPT Plus prompt for the V-Shield capstone document."
status: pending
priority: P1
effort: 4h
branch: fixgit
tags: [feature, ai-ml, evaluation, docs]
created: 2026-08-11
---

# Identity Retrieval Metrics and Capstone Prompt

## Overview

Add top-5 identity retrieval without changing authentication decisions. Evaluate external probe embeddings against the enrollment gallery, then document the current repository and provide a copy-ready GPT Plus capstone prompt. Do not publish fabricated metrics while the recognition protocol and enrollment data are absent.

## Decisions

- Database means `data/faces/<username>/*` enrollment embeddings; SQLite remains login audit only.
- Rank unique identities by minimum normalized L2 distance across their templates.
- `Precision@5 = hits / 5`; `Recall@5 = hits / relevant identities`.
- Current closed-set probe has one relevant identity, so Recall@5 equals Hit Rate@5.
- Ranking ignores authentication threshold and margin; existing `search()` behavior stays compatible.
- Empty gallery is an error. Empty probe set returns zero metrics with `query_count=0`.
- No CLI or API endpoint until a versioned gallery/probe protocol exists.
- Real repository metric remains `N/A`; synthetic unit tests prove only implementation behavior.

## Phases

| # | Phase | Status | Effort | Link |
|---|---|---|---:|---|
| 1 | Ranked identity retrieval contract | Pending | 1.5h | [phase-01](./phase-01-ranked-identity-retrieval.md) |
| 2 | Metrics and regression tests | Pending | 1.5h | [phase-02](./phase-02-recognition-metrics-and-tests.md) |
| 3 | Repository summary, capstone prompt, verification | Pending | 1h | [phase-03](./phase-03-capstone-docs-and-verification.md) |

## Dependency Flow

```text
Ranked Identity API -> Retrieval Metrics + Tests -> Capstone Docs + Full Verification
```

## Scope Boundaries

- No changes to `/predict`, authentication status, SQLite schema, PAD metrics, or frontend.
- No use of enrollment images as probes.
- Open-set probes require FPIR/FNIR and are outside Precision@5/Recall@5 scope.
- No claims based on dataset v1 or blocked dataset v2.

## Definition of Done

- Top-5 returns deterministic unique identities on NumPy and FAISS backends.
- Metrics use fixed `k=5`, macro-average by probe, and expose Recall@5/Hit Rate@5 equivalence.
- Tests cover success, misses, multi-template identities, invalid inputs, empty states, and compatibility.
- Docs summarize current code and data state with source paths.
- GPT Plus prompt explicitly prohibits invented experiments and labels unavailable results `N/A`.
- Targeted tests, full pytest, and Ruff checks pass.

## Unresolved Questions

- None. A future versioned recognition protocol must define probe IDs and leakage exclusion before real reporting.
