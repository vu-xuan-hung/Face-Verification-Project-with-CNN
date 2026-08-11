# Phase 03 — Capstone Docs and Verification

## Context

- [Plan](./plan.md)
- [Phase 02](./phase-02-recognition-metrics-and-tests.md)
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\README.md`
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\plans\vshield-ai-ml-audit-2026-07-30.md`
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\artifacts\evaluation\v1\INVALID.md`
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\artifacts\evaluation\v2\data-card.md`
- `C:\Users\LOQ\Desktop\ml-ai-dl\project\artifacts\evaluation\v2\data-release-audit.json`

## Overview

**Priority:** P1  
**Status:** Pending  
**Effort:** 1h

Create a concise repository source-of-truth summary and a copy-ready Vietnamese prompt for GPT Plus to draft an academically honest capstone document.

## Requirements

- Summarize goal, client-server architecture, ML pipeline, tech stack, API/UI, storage, training, evaluation, limitations, and current release state.
- Distinguish PAD classification from identity retrieval.
- State verified v1 integrity failures and v2 release block with source paths.
- Explain Precision@5/Recall@5 formulas and mark real results `N/A` until a valid gallery/probe protocol exists.
- Prompt GPT Plus to cite uploaded source files, use academic Vietnamese, and never invent results.
- Request useful chapters, tables, formulas, diagrams, experiment protocol, threats to validity, and future work.
- Keep generated docs under the configured `docs/` directory.

## Related Files

- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\docs\codebase-summary.md`.
- Create `C:\Users\LOQ\Desktop\ml-ai-dl\project\docs\gpt-plus-capstone-prompt.md`.

## Implementation Steps

1. Reconcile current code with README and audit artifacts.
2. Write a compact factual codebase summary with local source references.
3. Write one self-contained prompt ready for copy/paste after the user uploads source files to GPT Plus.
4. Add explicit anti-hallucination and unavailable-evidence rules.
5. Include recognition metric definitions and the difference from precision/recall for binary PAD.
6. Run targeted tests, full pytest, Ruff, and inspect the final diff for unrelated changes.

## Todo

- [ ] Create current codebase summary.
- [ ] Create complete Vietnamese GPT Plus capstone prompt.
- [ ] Include verified data-integrity facts and source list.
- [ ] Include Precision@5/Recall@5 definitions and `N/A` reporting rule.
- [ ] Run targeted tests and full pytest.
- [ ] Run Ruff on changed Python files.
- [ ] Review edge cases and compatibility.
- [ ] Sync completed checkboxes and plan status.

## Success Criteria

- Prompt can be copied directly into GPT Plus with an explicit upload checklist.
- Every numeric claim maps to a repository artifact.
- No model accuracy, FAR/FRR, Precision@5, or Recall@5 value is fabricated.
- Documentation matches actual endpoints, modules, and data status.
- Fresh verification reports zero test failures and no lint errors in changed Python files.

## Risks

- README contains historical claims that conflict with the audit. Mitigation: audit artifacts win and discrepancies are stated.
- GPT Plus cannot access local paths. Mitigation: prompt begins with a required file-upload checklist.
- A polished report can hide invalid evaluation. Mitigation: mandatory limitations and threats-to-validity sections.

## Security Considerations

- Do not include real usernames, face images, embeddings, login records, or database contents in the prompt.
- Do not upload secrets, model weights, or private biometric data to GPT Plus.

## Unresolved Questions

- None for prompt generation. University formatting rules can be supplied later as an extra source.

## Next

Ask the user whether they want the resulting capstone content generated as Markdown or DOCX after the prompt is reviewed.
