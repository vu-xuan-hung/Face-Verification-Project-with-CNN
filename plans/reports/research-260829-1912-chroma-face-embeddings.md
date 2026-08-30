# Research Report: Local Chroma Face Embeddings

Date: 2026-08-29

## Scope

Evaluate ChromaDB as a local persistent vector store for normalized 512-D FaceNet embeddings after the V-Shield anti-spoof gate. This is not an LLM, RAG, MCP, or text-embedding feature.

## Findings

- `PersistentClient(path=...)` stores and reloads local data automatically. It is intended for local development/small deployments; server-backed Chroma is recommended for larger production deployments.
- Collections accept precomputed embeddings through `add`/`upsert` and accept `query_embeddings` for nearest-neighbor search. V-Shield must continue using FaceNet to produce embeddings.
- Chroma HNSW supports `l2`, `cosine`, and inner-product spaces. Its documented L2 distance is squared L2, while V-Shield thresholds currently use Euclidean L2. The adapter must apply `sqrt` before threshold and margin checks.
- Persistence is automatic; the obsolete `.persist()` call must not be used.
- Telemetry is disabled in the local client settings. The store contains embeddings and minimal identity metadata, not source face images.

## Recommendation

Use `chromadb.PersistentClient` at `data/chroma/` and a collection configured for L2. Keep the existing identity decision logic for normalized embeddings, threshold, and runner-up margin. Prefer Chroma at startup and retain FAISS/NumPy as fallback. Seed from `data/faces/<username>/` only when the collection is empty.

## Security and Operations

- Face embeddings are biometric data and must remain ignored by Git and outside reports/logs.
- Local persistence has no application-level encryption or authorization added by this change. OS filesystem permissions remain required.
- There is no enrollment API or automatic gallery reconciliation. A controlled rebuild mechanism is future work.
- Recognition metrics remain N/A until a versioned gallery/probe protocol is evaluated.

## Sources

- [Chroma Python client and PersistentClient](https://docs.trychroma.com/reference/python/client)
- [Chroma collection add, upsert, and query API](https://docs.trychroma.com/reference/python/collection)
- [Chroma collection distance configuration](https://docs.trychroma.com/docs/collections/configure)
- [ChromaDB 1.5.9 package metadata](https://pypi.org/project/chromadb/)

## Unresolved Questions

- Define an explicit enrollment/rebuild workflow before identities are updated after the first seed.
- Calibrate the recognition threshold on a real versioned gallery/probe protocol; do not assume the existing threshold is validated for deployment.
