# Dataset and evaluation v1 are invalid

Do not use this split for model selection, threshold calibration, early stopping,
model promotion, or production accuracy claims.

Measured contamination:

- 264 exact duplicate groups cross train/val/test, containing 746 image references.
- 1,567 cross-split dHash candidate pairs at Hamming distance <= 4.
- A classifier using only image dimensions, byte size, brightness, contrast, and
  blur achieved 99.59% test accuracy and ROC-AUC 1.0.

These results show that the old test protocol measures acquisition shortcuts and
content leakage. They do not establish anti-spoof capability.

Canonical evidence: `plans/vshield-ai-ml-audit-2026-07-30.md`.
