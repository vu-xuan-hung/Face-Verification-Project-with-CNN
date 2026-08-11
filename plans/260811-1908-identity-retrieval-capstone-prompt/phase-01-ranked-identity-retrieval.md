# Phase 01 — Ranked Identity Retrieval

## Context

- [Plan](./plan.md)
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\src\vshield\core\identity_index.py`
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\src\vshield\core\verifier.py`
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\tests\test_authentication_architecture.py`

## Overview

**Priority:** P1  
**Status:** Pending  
**Effort:** 1.5h

Expose deterministic top-k identity ranking while preserving the existing threshold-and-margin authentication path.

## Requirements

- Add immutable ranked result containing `username` and L2 `distance`.
- Add public `ranked_identities(embedding, k=5)` on `IdentityIndex`.
- Normalize and validate query through the existing embedding utility.
- Search enough templates to produce up to `k` unique identities.
- Collapse templates by username using minimum distance.
- Sort by `(distance, username)` for reproducible ties.
- Do not apply `distance_threshold` or `min_margin` to ranking.
- Preserve `search() -> MatchResult | None` and its callers.

## Related Files

- Modify `C:\Users\LOQ\Desktop\ml-ai-dl\project\src\vshield\core\identity_index.py`.
- Optionally modify `C:\Users\LOQ\Desktop\ml-ai-dl\project\src\vshield\core\verifier.py` only to re-export the result type.
- Modify `C:\Users\LOQ\Desktop\ml-ai-dl\project\tests\test_authentication_architecture.py`.

## Implementation Steps

1. Define a frozen ranked identity dataclass beside `MatchResult`.
2. Implement ranking with the existing FAISS/NumPy candidate search and normalized L2 semantics.
3. Use all gallery templates when needed; do not confuse constructor `search_k` with top-k identities.
4. Validate `k >= 1`; raise the existing unavailable error for an empty gallery.
5. Keep runtime authentication logic and return types unchanged.
6. Add focused tests before moving to metric aggregation.

## Todo

- [ ] Add ranked result contract.
- [ ] Implement unique identity top-k ranking.
- [ ] Preserve FAISS squared-L2 square root behavior.
- [ ] Add deterministic tie-breaking.
- [ ] Test multi-template collapse, invalid `k`, empty gallery, and search compatibility.
- [ ] Test FAISS/NumPy ranking parity when FAISS is installed.

## Success Criteria

- A user with many templates occupies one rank only.
- Ranking contains at most `min(k, identity_count)` rows.
- NumPy and FAISS produce equivalent names and distances.
- Existing authentication tests remain unchanged and pass.

## Risks

- Fetching only five templates can return fewer than five identities. Mitigation: rank from enough gallery templates.
- Changing `_decide()` ordering could alter production decisions. Mitigation: isolate ranking and preserve `search()` contract.
- Equal-distance ties can be backend-order dependent. Mitigation: explicit username tie-break.

## Security Considerations

- Do not expose ranked usernames or distances through the authentication API.
- Ranking is an offline evaluation interface only.

## Next

Phase 02 computes auditable query-level and macro retrieval metrics.
