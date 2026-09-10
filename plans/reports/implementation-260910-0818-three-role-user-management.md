# Three-role RBAC implementation handoff

Verified 2026-09-10, Asia/Bangkok. Software implementation complete; real model/camera checks remain operational prerequisites. No commit/push, no production account creation, no arbitrary face association.

## Architecture

Before: FastAPI, React/Vite, SQLite with lowercase admin/user, opaque hashed bearer sessions, FaceNet512 unit embeddings, username-keyed Chroma and trusted local enrollment only.

After: same stack and session mechanism. Detection/alignment -> fail-closed PAD -> pretrained FaceNet embedding -> revision-specific Chroma identity search -> numeric immutable user_id -> SQLite current role/status -> session and backend management guards. Embeddings never encode permission. Adding users or changing roles does not train models.

## Database

Additive idempotent migration on existing database initialization. Preserves user IDs and session username foreign-key structure. Canonicalizes roles to USER/ADMIN/SUPER_ADMIN without promoting old ADMIN to SUPER_ADMIN; old sessions revoked only once at migration.

Existing users gains name, email, password_hash(nullable), status, created_at, updated_at, created_by. Existing username/id/active/enrollment_id remain. SQLite triggers enforce role/status and active/status consistency; case-insensitive email index prevents duplicate account email, including soft-deleted rows.

New schema_migrations, identity_revision and user_management_audit tables. Password authentication not added; password_hash reserved and never exposed. Biometric data remains private image/manifest plus vector store, not a public users.face_embedding response field.

## APIs

| Endpoint | Permission |
|---|---|
| POST /predict | Public request, but session only after PAD, identity and active account validation |
| GET /auth/me; POST /auth/logout | Authenticated active account |
| GET /users | ADMIN sees USER records; SUPER_ADMIN sees all, SUPER_ADMIN records read-only |
| POST /users | ADMIN or SUPER_ADMIN; server assigns USER |
| POST /admins | SUPER_ADMIN only; server assigns ADMIN |
| PATCH /users/{id} | ADMIN manages USER; SUPER_ADMIN manages USER/ADMIN; name/email only |
| PATCH /users/{id}/role | SUPER_ADMIN only; USER or ADMIN only |
| PATCH /users/{id}/status | Same managed-account scope; ACTIVE/DISABLED |
| DELETE /users/{id} | Same managed-account scope; soft delete and session revocation |
| GET /logs; GET /logs/export | ADMIN or SUPER_ADMIN |

No HTTP operation can create, demote, disable or delete SUPER_ADMIN. Explicit local bootstrap promotes one operator-selected, already enrolled active account only when no SUPER_ADMIN exists. Do not use a browser role or a model output as authority.

## Enrollment and identity

HTTP creation accepts username/name/email, 2-10 image data URIs and explicit consent; no role, file path, embedding, created_by or password_hash accepted from clients. Decode/image/body bounds precede use. FaceNet generates normalized templates; reject identical photos, inconsistent identities and faces already enrolled to active or disabled accounts.

Actor permission rechecked in BEGIN IMMEDIATE after inference; duplicate gate rechecks serialized state. Validated private draft published with numeric association in same account-creation workflow, files moved back on database failure. Created-by and creation audit recorded. Committed account with failed vector refresh returns vector_sync_status=pending, not false creation failure. Future recognition retries refresh and fails closed on invalid gallery.

Role updates touch SQLite and revoke sessions, not vectors/models. Enrollment/status/deletion increment revision; each worker refreshes its index before recognition. Revision-specific collections isolate old workers from current membership. Generic legacy vector tests/API retained. CLI also checks duplicates inside SQLite publication transaction and checks consistent identities.

## Security addressed

- Backend actor and managed-account role scope, with transaction-time revalidation.
- No role mass assignment or ADMIN -> ADMIN/SUPER_ADMIN escalation.
- Preserved hashed one-hour bearer sessions, current database role/status, revocation.
- Disabled/deleted recognized face cannot obtain authorization.
- Immutable numeric identity; no username/profile overwrite from managed enrollment.
- Duplicate face/user/email, bounded images, single face, consent and canonical photos.
- Safe private paths, fail-closed gallery validation, recovery of failed publication.
- Revision isolation for multi-worker Chroma, including same-count replacement regression.
- Sensitive validation inputs omitted from errors, no-store API responses.
- Frontend scope-aware controls, authenticated exports, no localStorage role authority; normal CRUD403 does not destroy valid session.

## Files changed for this request

- Backend: api/roles.py, migrations.py, database.py, user_store.py, bootstrap.py, sessions.py, user_schemas.py, user_routes.py, request_limits.py, app.py, schemas.py.
- Recognition/enrollment: services/enrollment.py, enrollment_images.py, user_management.py, authentication.py; core/authorization_gallery.py, managed_identity_index.py, chroma_identity_index.py.
- CLI: scripts/manage-identities.py.
- Frontend: role-permissions.js, face-enrollment.jsx, user-enrollment-form.jsx, user-management.jsx; auth-api.js, auth-context.jsx, App.jsx, pages and index.css; auth/role helper tests.
- Python tests: test_rbac_api.py, test_rbac_database.py, test_rbac_identity.py, plus existing authorization/gallery/core/authentication tests updated for the intentionally changed role/principal contracts.
- Docs: README.md, docs/face-authorization.md, docs/codebase-summary.md; plan and this report. Test scratch directories ignored.

Paths under api/core/services above are relative to src/vshield; frontend paths relative to frontend/src. Earlier external dataset implementation remains preserved and outside this request.

## Verification evidence

- Final full Python suite: **256 passed**, 3 upstream dependency warnings, 114.47s. Project interpreter; pytest flags `-q --basetemp tmp/rbac-final-260910 --tb=short -o log_cli=false`.
- Frontend Node tests: **13 passed**, zero failures. Command `node --test tests/auth-api.test.js tests/role-permissions.test.js` in frontend.
- Production build: `npm.cmd run build` passed, 1507 modules. Existing Vite/plugin deprecation notices remain.
- Ruff changed implementation/RBAC tests, compileall src/scripts, git diff whitespace check: passed.
- `uv pip check`: 162 installed packages compatible. Sandbox cache denial resolved with approved escalation.
- CLI help includes explicit bootstrap. No real bootstrap/account enrollment executed.
- Independent edge scout and code reviewer closed shared-Chroma and mixed-identity enrollment findings; no remaining critical/high RBAC blocker identified in scoped review.
- Browser runtime queried on 2026-09-10: no browser available, list empty. No screenshots, actual camera interaction or live biometric accuracy claim.

All nine requested allow/deny scenarios covered: USER cannot create USER/ADMIN; ADMIN can create USER but not ADMIN; SUPER_ADMIN can create both; ADMIN cannot promote USER; SUPER_ADMIN can; disabled recognized account fails authorization. Tests also cover extra-field injection, privileged-account mutation scope, soft-delete/re-enable, duplicate rollback, numeric identity, live refresh, migration, bootstrap and CLI consistency.

## Run and operational prerequisites

See docs/face-authorization.md for exact PowerShell migration, owner enrollment, guarded bootstrap, server/frontend and test commands. Startup runs migration; back up live SQLite and private biometric data before deployment. This turn tested temporary databases, not a populated deployment migration.

Unresolved operational inputs: compatible validated PAD artifact and FaceNet weights, consented owner photos, connected browser/camera for live E2E. Current model folder had no model files at inspection. Existing 54 unlabelled photos untouched.

Soft deletion does not erase photos, manifests, old Chroma revisions, drafts or backups; retention/purge and encryption/ACL/TLS remain operator responsibilities. No replacement-face API, password login or production biometric calibration added. Previous dataset acquisition still contains zero new real training samples; no change to that evidence.
