---
title: Three-role user management and embedding identity
description: Migrate existing SQLite and bearer sessions to hierarchical RBAC and managed face enrollment.
status: completed
priority: P1
effort: 10h
branch: add
tags: [auth, backend, frontend, database]
created: 2026-09-09
---

# Three-role user management

Software implementation completed and verified 2026-09-10. [Final implementation and verification report](../reports/implementation-260910-0818-three-role-user-management.md): 256 Python tests, 13 JavaScript tests, production build, lint, compilation, dependency check and independent review passed. Operational follow-ups below are not software completion claims.

User explicitly requested analysis then implementation. Existing FastAPI, React/Vite, SQLite, FaceNet512 and Chroma retained. Known unavailable planner model: plan inline. Preserve unrelated dataset work.

Before: PAD -> FaceNet -> username vector match -> SQLite two roles -> opaque session. CLI enrollment only.
After: PAD -> FaceNet -> stable numeric identity -> existing SQLite status and three roles -> same sessions and scoped management APIs/UI. No model retraining.

## Decisions

- Canonical SUPER_ADMIN, ADMIN, USER; migrate lowercase roles without automatic privilege elevation.
- Add profile, status and audit fields to existing users table; nullable password_hash since password login is not requested. Embeddings remain private manifest and Chroma, excluded from user API responses.
- ADMIN manages only USER accounts; SUPER_ADMIN manages USER and ADMIN. Only SUPER_ADMIN changes USER/ADMIN role. No HTTP elevation to SUPER_ADMIN; explicit guarded first-owner CLI bootstrap.
- Preserve existing username gallery directories; new managed index identifies immutable user IDs.
- Validate role again inside database transaction. Reject duplicate username/email/face, unknown fields and untrusted role input.
- Soft delete sets DELETED, revokes sessions and excludes vectors; retains files/audit under documented retention. No physical deletion.
- Revision-driven live index refresh for creation/disable/delete; no stale fallback on invalid gallery. Role-only updates require no vector change.
- Fail-closed PAD remains; synthetic tests are code evidence, not recognition accuracy.

## Phases

1. [Database/RBAC](phase-01-database-rbac.md): DB worker migrations/store; main session/routes.
2. [Enrollment](phase-02-enrollment-identity.md): main photo preparation, publish, stable ID lookup, live sync, CLI.
3. [UI/verification](phase-03-ui-validation.md): frontend worker; then tests/review/docs.

## Checklist

- [x] Inspect actual tree, frameworks, schemas, sessions, face pipeline and API gaps.
- [x] Migrate existing database and implement backend hierarchy.
- [x] Implement managed enrollment and live identity lookup without training.
- [x] Implement three-role management UI and multicapture.
- [x] Required nine permission cases and regressions.
- [x] Full tests, lint/build, review, docs and plan sync.

## Operational follow-ups (not completed)

- [ ] Supply compatible validated PAD model and FaceNet weights; do not bypass PAD.
- [ ] Supply consented owner photos and explicitly enroll/bootstrap the first SUPER_ADMIN; no real account was created.
- [ ] Connect browser/camera and perform live enrollment/login E2E; browser runtime was unavailable and no biometric accuracy is claimed.
- [ ] Configure production backup, ACL/TLS and biometric retention/purge, including old revision collections.

Earlier dataset acquisition/release work belongs to its separate plan and is unchanged; no new dataset samples or real identity associations are claimed here.

## Risks

SQLite, files and Chroma are not a single transaction: publish validated drafts, recover failed publication, retain DB status as final authority. Serialise concurrent management and recheck permissions. Never auto-assign photos/admin privileges. Real camera checks need model and consented users; report those prerequisites separately.
