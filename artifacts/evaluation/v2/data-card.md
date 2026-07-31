# V-Shield anti-spoof dataset v2 data card

**Status:** blocked — collection/provenance required.

## Required collection design

- Every subject contributes both bona fide and attack samples.
- Subject, session, clip, device, attack instrument, lighting, and background are recorded.
- Real and fake samples use the same detector, crop, codec, resolution, and storage pipeline.
- Fake samples target the same identities as bona fide samples.
- Train, validation, and test are assigned by group before augmentation.

## Current release-gate result

- Released samples: 0.
- Quarantined historical samples: 2,415.
- Blocking cause: missing subject/session/clip/device provenance.
- Historical metadata-only baseline: accuracy 99.59%, ROC-AUC 1.0.
- Low-resolution, color-histogram, background-only, and center-only probes each
  reached ROC-AUC 1.0 on the historical split.
- New conservative OpenCV dHash scan produced 5,998 cross-split candidates.
  These are pending candidates, not 5,998 confirmed duplicates.
- Current data must not be presented as an anti-spoof benchmark.

See `artifacts/evaluation/v2/data-release-audit.json` for machine-readable results.
