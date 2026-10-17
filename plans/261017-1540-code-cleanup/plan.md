# Safe code cleanup

Scope: remove code verified unused across source, scripts, tests, notebooks and
entry points; organize API responsibilities without changing HTTP behavior.
Preserve existing workspace changes, data, weights and CLI workflows.

- [x] Audit backend and frontend references, dependencies and entry points.
- [x] Remove orphan response schemas and unused legacy helpers/constants.
- [x] Remove unused collector resource initialization and misleading comments.
- [x] Split API lifecycle, image decoding, authentication and log routes.
- [x] Remove duplicate frontend query helper and unused role constant.
- [x] Run backend tests, frontend tests/build and scoped lint/compile checks.
- [x] Review changes and document final structure and validation.

Known boundaries: eye alignment still uses Haar; legacy training serialization,
collector CLI, dataset tools and evaluation commands remain supported.

Status: complete. See `../reports/cleanup-261017-1540-code-cleanup.md`.
