# Database and RBAC

Status: completed (2026-09-10). Preserved same SQLite users/sessions/audit. Evidence: [final report](../reports/implementation-260910-0818-three-role-user-management.md).

- [x] Enum and idempotent migration, preserve IDs, no automatic super admin.
- [x] Profile/status/audit columns and gallery revision counter.
- [x] Transaction-time actor and managed-account scope checks, duplicate username/email protection.
- [x] CRUD, role, disable and soft-delete operations and guards.
- [x] Existing sessions return numeric principal and live current role/status.
- [x] Migration and escalation/revocation tests.

DB worker owns database.py, roles.py, migrations.py, user_store.py and DB tests. Main owns sessions.py, app.py, user_routes.py, user_schemas.py. No overlap. Password login out of scope; hash nullable. Preserve records and identifiers on soft delete.
