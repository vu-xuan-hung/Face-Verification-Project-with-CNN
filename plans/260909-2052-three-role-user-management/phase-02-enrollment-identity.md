# Enrollment and identity

Status: completed (2026-09-10). Main owns enrollment/authentication services, gallery/Chroma/live-index components and identity CLI. Evidence: [final report](../reports/implementation-260910-0818-three-role-user-management.md).

- [x] Reusable bounded photo preparation and single-face pretrained embedding.
- [x] Consent, duplicate image/face and multiple-view consistency checks.
- [x] Transactional account publication, draft recovery and stable ID association.
- [x] Revision-driven live refresh; never query stale gallery after failure.
- [x] Guarded first SUPER_ADMIN bootstrap, no default promotion.
- [x] Tests no training call, identity ID, duplicate/disabled/deleted behavior.

Role changes only affect SQLite. Soft-delete retained files documented; revoked users excluded from active search.

Model artifacts and consented owner enrollment remain operational follow-ups in plan.md. No actual owner was bootstrapped or real recognition accuracy established.
