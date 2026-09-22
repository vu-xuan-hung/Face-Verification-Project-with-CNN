# VShield production biometric evaluation audit

Audit date: 2026-09-15 (Asia/Bangkok)
Repository commit: `1b5d2da9de5735a28f079779fa4ae59701ac7443`
Scope: the production request, biometric, gallery, account, and access-decision paths.

This document is source-first. Values below are implementation facts, not measured accuracy or recommended operating points.

## Production decision flow

`POST /predict` decodes a bounded base64 image to OpenCV BGR `uint8`, then calls `AuthenticationService.authenticate` (`src/vshield/api/app.py:40-71,145-166`). Authentication requires exactly one detected face, runs PAD on the original frame and detected bounding box, and runs FaceNet plus identity search only after PAD returns `REAL` (`src/vshield/services/authentication.py:55-99`). A biometric match is not sufficient for access: `/predict` calls `sessions.create_session`, which reloads the matched account from SQLite and requires `active=1`, `status='ACTIVE'`, and a recognized role (`src/vshield/api/app.py:250-307`; `src/vshield/api/sessions.py:23-60`).

## Face detection, crop, and alignment

- Detector: OpenCV `haarcascade_frontalface_default.xml`; eye detector: `haarcascade_eye.xml` (`src/vshield/core/face_preprocessor.py:39-51`).
- Detection input: three-channel BGR. Detection uses grayscale plus histogram equalization and `detectMultiScale(scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))` (`face_preprocessor.py:53-70`).
- Exactly one face is required. Zero or multiple detections raise `InvalidFaceCountError` (`face_preprocessor.py:71-73`).
- The recognition crop expands the face by 10% left/right, 30% above, and 5% below (`face_preprocessor.py:84-93`). Eye alignment rotates the crop when exactly two usable eye detections exist; otherwise the expanded crop is returned unchanged (`face_preprocessor.py:95-132`).
- `FaceCrops.anti_spoof` is a legacy-looking 128x128 expanded crop, but the current production MiniFASNet path does not consume it. PAD receives the original frame and raw Haar bounding box (`face_preprocessor.py:75-82`; `authentication.py:69`).

## PAD implementation

| Property | Verified production value | Evidence |
|---|---|---|
| Model | MiniFASNetV2 | `configs/ai-models.yaml:2` |
| Version | `minifasnet-v2-b6d5f04-a5eb02e1-opset17` | `configs/ai-models.yaml:3` |
| Artifact | `artifacts/models/minifasnet-v2.onnx` | `configs/ai-models.yaml:5` |
| Artifact SHA-256 | `04e8890346498741e655adfcb67b48ee2620fc2d251321a8f0ad3262bb842067` | `configs/ai-models.yaml:6`; verified file hash |
| Runtime | ONNX Runtime, `CPUExecutionProvider` only | `src/vshield/core/minifasnet_onnx.py:13-28` |
| Input | one `[1,3,80,80]` float32 tensor | `minifasnet_onnx.py:20-25` |
| Input color/layout | BGR, NCHW | `configs/ai-models.yaml:8-10`; `src/vshield/core/pad_preprocessing.py:39-42` |
| Context crop | scale 2.7 from original frame and detector bbox | `pad_preprocessing.py:13-40` |
| Resize | 80x80, OpenCV bilinear | `pad_preprocessing.py:39-40` |
| Normalization | none; values remain 0-255 | `pad_preprocessing.py:41-42` |
| Output | three logits for `[spoof_0, real, spoof_2]` | `configs/ai-models.yaml:13-15`; `minifasnet_onnx.py:54-64` |
| Continuous score | softmax probability for real class index 1 | `minifasnet_onnx.py:60-64` |
| Threshold | 0.8, explicitly provisional | `configs/ai-models.yaml:17-18` |

The exact decision rule is: non-real argmax is `FAKE`; real argmax with score below 0.8 is `UNCERTAIN`; real argmax with score greater than or equal to 0.8 is `REAL` (`minifasnet_onnx.py:63-71`). Both `FAKE` and `UNCERTAIN` deny access and skip recognition (`authentication.py:70-77`). Raw logits and the full probability vector are not exposed by `PadResult`; it exposes status, real score, predicted class, model version, threshold, and reason (`src/vshield/core/anti_spoof.py:16-30`).

The model loader verifies an exact contract, artifact checksum, upstream source commit, and parity report before creating the ONNX session. Failures return an unavailable, fail-closed PAD service; production never falls back to the legacy Keras PAD model (`src/vshield/core/model_registry.py:39-88`; `src/vshield/core/anti_spoof.py:50-58,89-93`).

## Recognition implementation

| Property | Verified production value | Evidence |
|---|---|---|
| Model API | lazy `keras_facenet.FaceNet()`; installed default `20180402-114759` InceptionResNetV1 | `src/vshield/core/embedder.py:41-59`; resolved `keras-facenet` 0.3.2 package metadata |
| Model artifact/version | production code does not pin it; resolved cached weights hash is `8b71e7045497e841c00ee568f031d1a4d30908fceadf6884aef2dec4d545202b` | `embedder.py:49-59`; installed `keras_facenet/metadata.py`; verified cache hash |
| Input | eye-aligned expanded BGR face crop | `authentication.py:78-80`; `face_preprocessor.py:75-132` |
| Model preprocessing | BGR to RGB, then package resize to 160x160 and fixed standardization `(x - 127.5) / 127.5` | `embedder.py:66-74`; installed `keras_facenet/__init__.py` `embeddings()` implementation and metadata |
| Embedding | exactly 512 finite float32 values | `embedder.py:12,20-33` |
| Normalization | explicit L2 unit normalization | `embedder.py:34-38,80` |
| Metric | Euclidean L2 between normalized embeddings, lower is better | `src/vshield/core/identity_index.py:159-189`; `RecognitionDecision.to_dict()` at lines 51-53 |
| Gallery aggregation | minimum template distance per identity; no centroid/average | `identity_index.py:194-218` |
| Unknown threshold | 0.9 | `identity_index.py:19,59-80,200-203` |
| Ambiguity margin | 0.05 | `identity_index.py:20,59-80,201-205` |
| Threshold equality | distance exactly 0.9 accepts | `identity_index.py:202-203` |
| Margin equality | margin exactly 0.05 accepts | `identity_index.py:204-205` |

FAISS uses `IndexFlatL2` and production square-roots its squared distances; the fallback computes NumPy L2 directly (`identity_index.py:85-96,159-189`). Chroma uses HNSW L2, requests every local template so a competing identity cannot be omitted, and square-roots returned squared L2 distances (`src/vshield/core/chroma_identity_index.py:58-62,152-190`). Chroma failures fall back to the validated in-memory FAISS/NumPy index (`chroma_identity_index.py:196-228`).

The recognition threshold and margin are uncalibrated code defaults, not configuration values. They are also duplicated as literals in Chroma and enrollment duplicate/consistency checks (`chroma_identity_index.py:26-43`; `src/vshield/services/enrollment.py:83-85`; `src/vshield/services/enrollment_images.py:38-39,59-67`).

## Enrollment, gallery, database, and RBAC

The authoritative gallery is built only from active, non-deleted SQLite accounts with consented enrollment manifests under `data/authorization/faces/<username>/enrollment.json`. The loader verifies username, enrollment ID, contract `facenet-512-l2-bgr-v1`, consent, 1-20 templates, image sizes and hashes, optional user ID association, and every stored embedding (`src/vshield/core/authorization_gallery.py:11,36-80`). Authentication uses stable numeric account IDs as vector identities and revision-specific Chroma collections under `data/authorization/chroma` (`src/vshield/core/managed_identity_index.py:20-63`).

Managed HTTP enrollment requires 2-10 images and verifies every sample with the same face preprocessor and production PAD before extracting embeddings (`src/vshield/services/enrollment_images.py:12-50`; `src/vshield/services/user_management.py:23-94`). The trusted local enrollment flow permits 1-20 images and uses the same services (`src/vshield/services/enrollment.py:31-103`).

After a biometric `MATCH`, SQLite remains authoritative: `sessions.create_session` reloads by user ID and rejects missing, disabled, deleted, or invalid-role accounts. RBAC is checked again on protected requests; ADMIN/SUPER_ADMIN and SUPER_ADMIN-only dependencies are in `src/vshield/api/sessions.py:63-101`.

## Evaluation-to-production reuse map

| Evaluation component | Production implementation reused |
|---|---|
| Image decoding | OpenCV BGR decode with the production size constraints mirrored by the dataset validator; API data-URI decoding remains `vshield.api.app.decode_image` |
| Face detector/crops | `vshield.core.face_preprocessor.FacePreprocessor.extract` |
| PAD preprocessing/model/decision | `vshield.core.anti_spoof.build_pad_service` and `check_pad`, which call `MiniFASNetONNX` and `prepare_pad_input` |
| FaceNet preprocessing/model/normalization | `vshield.core.embedder.FaceEmbedder.encode` and `normalize_embedding` |
| Gallery flattening | `vshield.core.identity_index_support.flatten_database_embeddings` through `IdentityIndex` |
| Matching metric/threshold/ambiguity | `vshield.core.identity_index.IdentityIndex.search_decision` |
| Ranking and second identity | `IdentityIndex.search_decision`, whose decision object now exposes the production-ranked best and runner-up identities; pair analysis uses `ranked_identities` |
| Full biometric orchestration | `vshield.services.authentication.AuthenticationService.authenticate` |
| Production gallery mode | `vshield.core.managed_identity_index.ManagedIdentityIndex` and `load_authorization_gallery` |
| Account/access authorization | documented from `/predict` and `sessions.create_session`; manifest-based tests report biometric ALLOW/DENY without creating real sessions or mutating the production database |

## Current evidence and data status

No valid released PAD or recognition evaluation dataset exists in the repository. `artifacts/evaluation/v2/data-release-audit.json` reports zero released samples and empty train/validation/test protocols. Historical PAD v1 is explicitly quarantined for exact cross-split leakage, missing provenance, and shortcut confounding (`artifacts/evaluation/v1/INVALID.md`; `artifacts/evaluation/v2/data-card.md`). The private authorization gallery is enrollment data and must not be reused as test probes.

## Production inconsistencies relevant to evaluation

1. `scripts/evaluate_pad_directory.py` passes a project directory where `build_pad_service` expects a config file, substitutes a whole-frame bbox after face-detection failure, silently drops unreadable/error samples, and uses `score > threshold`. All differ from production (`scripts/evaluate_pad_directory.py:73-105,121-122`; `anti_spoof.py:50-55`; `minifasnet_onnx.py:67-70`).
2. The pre-existing PAD metric code used strict `score > threshold`, while production accepts equality. This integration corrects the shared evaluator to the production `>=` boundary (`src/vshield/evaluation/pad_metrics.py:error_rates,evaluate_scores`).
3. `RecognitionDecision` was extended with best and runner-up identity names so evaluation can record production ranking without a second decision path. This is an observability-only change; thresholds and acceptance behavior are unchanged (`identity_index.py:32-55,194-218`). Genuine/impostor score generation separately uses `ranked_identities()` to preserve the same production distance implementation.
4. Recognition model weights/version/hash are not pinned by VShield. Cache fingerprints can include package versions and production source/config hashes, but cannot claim an exact FaceNet artifact hash until production exposes it.
5. `load_authorization_gallery` classifies missing liveness provenance as legacy-unverified but does not exclude such enrollments from authentication (`authorization_gallery.py:14-21,36-80`).
6. `load_pad_contract` permits a threshold equal to 1.0, but `check_pad` rejects result thresholds that are not strictly below 1.0. A configured value of 1.0 would therefore load and then fail every prediction as `MODEL_ERROR` (`model_registry.py:57-59`; `anti_spoof.py:69-74`). The evaluation CLI restricts overrides to the actually runnable interval `(0.5, 1)`.

## Metric definitions and scientific boundaries

- APCER: attack presentations classified bona fide divided by scored attack presentations.
- BPCER: bona-fide presentations classified attack divided by scored bona-fide presentations.
- ACER: `(APCER + BPCER) / 2`, only when both classes exist.
- Verification-style FAR/FMR and FRR/FNMR are computed only when valid continuous genuine and impostor scores exist. Each score is the production minimum gallery-template distance for an identity, not an independent all-template pair trial; reports label this protocol explicitly.
- Open-set identification metrics use known/unknown probes and include rejected known probes in their denominators.
- Threshold recommendations use calibration rows only and never update production configuration automatically.
- Computing APCER/BPCER does not establish ISO certification or compliance.
