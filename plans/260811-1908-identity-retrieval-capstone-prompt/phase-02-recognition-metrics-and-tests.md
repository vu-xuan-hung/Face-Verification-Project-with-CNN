# Phase 02 — Recognition Metrics and Tests

## Context

- [Plan](./plan.md)
- [Phase 01](./phase-01-ranked-identity-retrieval.md)
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\src\vshield\evaluation\pad_metrics.py`
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\tests\evaluation\test_shortcut_and_pad_metrics.py`

## Overview

**Priority:** P1  
**Status:** Pending  
**Effort:** 1.5h

Implement pure, testable identity retrieval evaluation over external labeled probe embeddings.

## Metric Contract

For probe `q`, relevant identity set `G(q)`, and first five unique retrieved identities `R5(q)`:

```text
Precision@5(q) = |R5(q) ∩ G(q)| / 5
Recall@5(q)    = |R5(q) ∩ G(q)| / |G(q)|
```

Aggregate by macro mean across probes. Current closed-set contract has one ground-truth identity per probe; therefore `Recall@5 == Hit Rate@5`.

## Requirements

- Represent a probe with expected identity and normalized-compatible embedding.
- Return `k`, query count, macro Precision@k, Recall@k, and Hit Rate@k.
- Default and documented capstone metric is `k=5`.
- Empty probe list returns count zero and zero metric values.
- Empty gallery propagates a clear evaluation error.
- Missing expected identity in gallery is a miss, not an exception.
- Reject empty expected identity and invalid `k`.
- Keep PAD evaluation independent.

## Related Files

- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\src\vshield\evaluation\recognition_metrics.py`.
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\tests\evaluation\test_recognition_metrics.py`.
- Modify `C:\Users\LOQ\Desktop\ml-ai-dl\project\src\vshield\evaluation\__init__.py` only if public exports add value.

## Implementation Steps

1. Add small frozen dataclasses or typed structures for probe and aggregate result.
2. Evaluate each external probe through `IdentityIndex.ranked_identities()`.
3. Calculate per-probe hit, fixed-denominator precision, and recall.
4. Macro-average without weighting identities by template count.
5. Include Hit Rate@k explicitly to prevent misreading singleton Recall@k.
6. Add deterministic synthetic tests; do not label these as model performance.

## Todo

- [ ] Implement probe and aggregate metric contracts.
- [ ] Implement macro Precision@k, Recall@k, and Hit Rate@k.
- [ ] Cover all-hit, partial-hit, all-miss, and missing-gallery-identity cases.
- [ ] Cover empty probes, empty gallery, invalid embedding, invalid label, and invalid `k`.
- [ ] Verify template count cannot weight macro metrics.
- [ ] Run targeted core and evaluation tests.

## Success Criteria

- Two probes with one top-5 hit produce Recall@5 and Hit Rate@5 of `0.5` and Precision@5 of `0.1`.
- Metrics are independent of templates per identity.
- Empty and invalid states follow the documented contract.
- No PAD metric or promotion gate is silently changed.

## Risks

- Precision@5 appears low because its singleton maximum is `0.2`. Mitigation: document formula and Hit Rate@5.
- Gallery/probe leakage inflates results. Mitigation: evaluator accepts external probes only; real reporting remains blocked until a protocol includes sample/group identifiers.
- Synthetic tests may be mistaken for experiment results. Mitigation: label them implementation verification only.

## Security Considerations

- Metric results should aggregate identities; avoid publishing per-user embeddings or ranked names in capstone artifacts.

## Next

Phase 03 creates factual documentation and verifies the repository.
