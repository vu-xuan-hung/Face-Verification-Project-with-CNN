# Code cleanup

Completed 2026-10-17. Preserved existing uncommitted work and runtime contracts.

## Removed

- Orphan `api/schemas.py`; remaining response classes had no imports.
- Unused `roles.Status`, `embedder.img_to_encoding_frame`, `IdentityIndex._decide`,
  `verifier.build_faiss_index`, legacy `predict_is_real` and its exception/import.
- Collector's unused EAR helper, eye constants, FaceMesh/MOG2 initialization and
  misleading blink/motion claims. Active cvzone/MediaPipe detection remains.
- Frontend unused `ROLES` export and obsolete `logsPath` helper. Existing query
  encoding regression now exercises the active `accessLogsPath` helper.
- Redundant test-local sys.path mutation; shared conftest already configures it.

## Structure

- `api/app.py`: lifecycle, middleware and router assembly; 490 to 97 lines.
- `api/image_decoding.py`: bounded image decoding shared by login/enrollment.
- `api/prediction_routes.py`: multipart authentication and audit dispatch.
- `api/auth_routes.py`: readiness and bearer-session endpoints.
- `api/log_routes.py`: log queries, CSV exports and dashboard statistics.
- Removed imports from route/middleware modules back into app assembly.
- Updated README, codebase summary and decoder references in evaluation audit.

Kept Haar eye alignment, offline Keras serialization support, legacy verifier
helpers exercised by existing tests, training/evaluation/collection CLIs,
notebooks, data, model weights and prior reports.

## Validation

- Full backend before and after: 363 passed, 1 skipped. Skip requires unavailable
  local consented `data/DataCollect` fixture. Same dependency warnings as baseline.
- Frontend: 14 passed; Vite production build passed.
- Python compileall and full Ruff rules on changed Python files passed.
- Unused-name/import checks across src/scripts and git whitespace check passed.
- Exact OpenAPI comparison and normalized AST comparison of route extraction
  passed before formatting; all route behavior tests passed afterward.
- Independent reviewer found no actionable issues; 75 focused tests passed.

Commit requested after validation. Unresolved questions: none.
