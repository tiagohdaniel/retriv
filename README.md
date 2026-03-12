# Semantic Docs API

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![Tests](https://img.shields.io/badge/tests-13%20passed-brightgreen)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

A REST API for semantic search over documentation.
Index any text content, ask natural language questions, and get LLM-generated answers grounded on retrieved context — with source references and relevance scores.

Built with **FastAPI**, **ChromaDB**, **ONNX Runtime** embeddings, and **Claude** as the reasoning layer.

---

## Quick overview

- **Index** any documentation via `POST /index` — text is chunked, embedded, and stored in a vector database
- **Ask** natural language questions via `POST /ask` — relevant chunks are retrieved, filtered by semantic relevance, and sent to an LLM for a grounded answer with source citations
- **Manage** indexed sources via `GET /sources` and `DELETE /sources/{source_id}`
- No GPU required — embeddings run via ONNX Runtime; the full test suite runs without API keys

---

## The problem

Keyword search breaks on documentation. A user searches for *"how to handle payment failures"* but the relevant paragraph says *"retry logic for declined transactions"* — zero keyword overlap, relevant content missed.

Semantic search solves this by encoding content and queries into the same vector space and finding chunks by *meaning*, not by word match. The LLM then synthesizes a grounded answer from the retrieved chunks.

---

## RAG pipeline

```
POST /index                          POST /ask
     │                                    │
     ▼                                    ▼
TextChunker                    query → EmbeddingService
(500 chars, 50 overlap)                   │
     │                                    ▼
     ▼                          ChromaDB cosine search
EmbeddingService                (top-k most similar chunks)
(ONNX all-MiniLM-L6-v2)                  │
     │                                    ▼
     ▼                          Filter by max_distance (default 0.8)
ChromaDB upsert               (discard semantically unrelated chunks)
                                          │
                                          ▼
                                Guard: no chunks? → skip LLM
                                         │
                                         ▼
                                  Build prompt with context
                                         │
                                         ▼
                                  LLM Client (Claude)
                                         │
                                         ▼
                                   AskResponse
                               (answer + sources + scores)
```

---

## Design decisions & tradeoffs

### Embedding: ONNX Runtime over sentence-transformers

The `all-MiniLM-L6-v2` model runs via ONNX Runtime instead of the `sentence-transformers` library.

**Why:** `sentence-transformers` pulls in PyTorch as a dependency. PyTorch CPU-only is ~500MB. ONNX Runtime is ~10MB and runs the same exported model directly. At container build time this matters significantly — a Docker image that installs PyTorch just for inference is solving the wrong problem.

**Tradeoff:** ONNX models can't be fine-tuned at runtime. If you need to adapt the embedding model to a specific domain (e.g., medical or legal terminology), you'd fine-tune with PyTorch and export to ONNX as a build step. For general-purpose documentation search, the base model is sufficient.

**Fallback chain:** `ONNXEmbedding → SentenceTransformerEmbedding → HashEmbedding`. The factory tries each level in order. In tests, `HashEmbedding` runs with zero dependencies — no model download, no GPU, deterministic output.

---

### Vector store: ChromaDB over Pinecone / pgvector

**Why ChromaDB:** Runs in-process, persists to disk, zero external dependencies. The alternative patterns each have a cost:

| Option | Good for | Problem |
|--------|----------|---------|
| Pinecone | Managed, scalable | External service, cost, vendor lock-in |
| pgvector | Already have Postgres | Query complexity, separate index tuning |
| Weaviate | Rich filtering | Heavy to operate |
| ChromaDB | Local / single-instance | Not horizontally scalable |

ChromaDB is the right call when the deployment is a single container or a single-tenant service. The moment you need multiple API workers sharing the same vector store, you'd switch to a server-mode ChromaDB or Pinecone.

**Similarity metric: cosine, not L2.**

Cosine similarity measures the *angle* between vectors — it ignores magnitude. Two documents that say the same thing with different verbosity have the same direction but different magnitude. L2 (Euclidean distance) would score them differently. For semantic similarity, direction is meaning; magnitude is noise.

---

### Chunking: size 500, overlap 50

Text is split into 500-character chunks with a 50-character overlap between consecutive chunks.

**Why overlap:** A sentence that sits on a chunk boundary gets split. Without overlap, searching for content from that sentence might return neither chunk with good relevance. Overlap ensures boundary content appears in both adjacent chunks.

**The size tradeoff:**

- **Too small (< 100 chars):** Individual sentences lose context. "It does not support this." — support *what*? The LLM gets fragments without enough information.
- **Too large (> 1000 chars):** One chunk covers multiple topics. A query about topic A retrieves a chunk that's 80% about topic B, reducing precision. Also, each chunk becomes a larger slice of the LLM's context window.
- **500 chars:** Roughly 2-4 sentences. Enough context, specific enough to score well on a single-topic query.

---

### Relevance threshold: max_distance filter

ChromaDB always returns the top-k closest chunks — even when the query is completely unrelated to the indexed content. Without a threshold, asking *"what is the capital of France?"* against a Python documentation index would still retrieve chunks and send them to the LLM, which would then produce a hallucinated or confused answer.

The `max_distance` parameter (default `0.8`) discards any chunk whose cosine distance exceeds the threshold before the guard clause runs:

```python
# vector_store.py — applied after ChromaDB query
if distance > max_distance:
    continue  # chunk is not semantically related, discard
```

**The tradeoff:** `0.8` was chosen empirically for general documentation. A tighter threshold (`0.5`) reduces noise but may discard valid chunks for niche queries. A looser threshold (`1.0`) accepts more context but risks sending irrelevant chunks to the LLM. The value is exposed as a per-request parameter so callers can tune it for their domain.

**Why this exposed a test limitation:** `HashEmbedding` (used in tests to avoid model downloads) produces vectors based on SHA-256 — no semantic meaning, all distances are effectively random. Tests that assert LLM behavior must explicitly pass `max_distance=2.0` to bypass the filter. This is documented in the test code and is a known, intentional tradeoff of hash-based mock embeddings.

---

### Guard clause before LLM call

```python
if not docs:
    return AskResponse(answer="No relevant documentation found...")
```

If the vector search returns nothing after the distance filter, the LLM is never called.

**Why this matters:**

1. **Cost:** An LLM call with an empty context still charges tokens for the prompt template.
2. **Hallucination:** A well-prompted LLM *might* still answer from its training data when given no context. That's the wrong behavior for a documentation API — the answer should come from your docs, not from the model's general knowledge.
3. **Signal:** "No relevant documentation found" tells the user exactly what happened. An LLM-generated "I don't have information about this" is vaguer and costs 10x more to produce.

---

### Prompt design

The prompt passed to Claude has two sections, separated by Markdown headings:

```
## Documentation

[1] Source: FastAPI Introduction
FastAPI is a modern web framework...

[2] Source: FastAPI Introduction
It uses Pydantic for data validation...

## Question

What does FastAPI use for data validation?
```

The system prompt instructs the model to answer **only** from the provided documentation:

```python
SYSTEM_PROMPT = """You are a technical documentation assistant.
Answer questions based ONLY on the provided documentation excerpts.
If the documentation does not contain enough information, say so clearly.
Be concise, accurate, and cite the source document when relevant."""
```

**Why `temperature=0`:** RAG answers should be deterministic and grounded, not creative. With `temperature=0`, the model always picks the highest-probability token — given the same context, it produces the same answer. This makes behaviour predictable and easier to evaluate.

**Why numbered source blocks:** Numbering chunks (`[1]`, `[2]`) gives the model a natural way to reference sources in its answer ("According to [1]..."). It also makes the context structure visible when debugging retrieval quality.

**Why Markdown headings (`##`) to separate sections:** Claude responds better to structured prompts. A clear visual boundary between *context* and *question* reduces the chance the model treats part of the documentation as an instruction.

---

### Idempotent indexing: delete-before-upsert

Re-indexing the same `source_id` deletes all existing chunks first, then inserts the new ones.

**Why:** If a document is updated and re-indexed without deleting the old chunks, old and new chunks coexist. A query might retrieve an outdated chunk alongside a new one, producing a contradictory answer.

**The tradeoff:** There's a brief window between delete and upsert where the source has zero chunks. During that window, a concurrent `/ask` query would return "no relevant documentation found" for that source. For a single-writer, low-concurrency use case, this is acceptable. For high-concurrency systems, an atomic swap strategy (write to a temp collection, then rename) would be safer.

---

### Dependency injection with `Depends()` and `@lru_cache`

Every service dependency (embedding, vector store, LLM client, chunker) is wired through FastAPI's `Depends()`.

**Why `@lru_cache` on factory functions:** Each factory function is called once per process and the result is cached. This gives singleton behavior without a global variable — the same pattern as a DI container, without a framework.

**Why this enables clean testing:** FastAPI's `app.dependency_overrides` only works when routes declare their dependencies via `Depends()`. Direct function calls in route handlers bypass the override mechanism entirely. With `Depends()`, every integration can be swapped in tests:

```python
# In tests: no API key, no model download, no disk I/O
app.dependency_overrides = {
    get_vector_store: lambda: DocsVectorStore(EphemeralClient(), collection_name=uuid),
    get_embedding_service: lambda: HashEmbedding(),
    get_llm_client: lambda: mock_llm,
}
```

**The `@lru_cache` tradeoff:** Singletons are process-scoped. With `uvicorn --workers 4`, each worker process has its own ChromaDB client pointing to the same directory. For a single persistent ChromaDB directory, multiple writers can cause corruption. The solution for multi-worker deployments is ChromaDB server mode (HTTP client) or an external vector store.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/index` | Index a document (chunked + embedded) |
| `POST` | `/ask` | Ask a question, get a grounded answer |
| `GET` | `/sources` | List all indexed sources with chunk counts |
| `DELETE` | `/sources/{source_id}` | Remove all chunks for a source |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI (auto-generated) |

### Index a document

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "fastapi-intro",
    "title": "FastAPI Introduction",
    "content": "FastAPI is a modern, fast web framework for building APIs with Python..."
  }'
```

```json
{
  "status": "ok",
  "source_id": "fastapi-intro",
  "chunks_indexed": 3
}
```

### Ask a question

```bash
# Default: max_distance=0.8 (discards unrelated chunks)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does FastAPI handle async requests?"}'

# Filter to specific sources + custom threshold
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does FastAPI handle async requests?", "source_ids": ["fastapi-intro"], "max_distance": 0.6}'
```

```json
{
  "answer": "FastAPI handles async requests using Python's asyncio...",
  "sources": [
    {
      "source_id": "fastapi-intro",
      "title": "FastAPI Introduction",
      "excerpt": "FastAPI supports async/await natively...",
      "relevance_score": 0.1234
    }
  ],
  "tokens_used": 312,
  "model": "claude-sonnet-4-20250514"
}
```

---

## Running locally

### With Docker (recommended)

```bash
git clone https://github.com/tiagohdaniel/semantic-docs-api
cd semantic-docs-api

cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

docker compose up
```

API at `http://localhost:8000` — Swagger UI at `http://localhost:8000/docs`

### Run the end-to-end demo

With the API running, execute the demo script to see the full pipeline in action:

```bash
bash examples/demo.sh
```

It indexes two documents, asks questions (including a filtered query and an out-of-scope question that triggers the fallback), re-indexes to verify idempotency, and deletes a source — all with pass/fail output.

### Without Docker

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

uvicorn app.main:app --reload --port 8000
```

---

## Running tests

No API key needed. Tests use in-memory ChromaDB (isolated per test via UUID collection names), hash-based embeddings, and a mocked LLM client.

```bash
pytest tests/ -v
```

```
tests/test_ask.py::test_ask_with_indexed_content PASSED
tests/test_ask.py::test_ask_without_indexed_content_returns_fallback PASSED
tests/test_ask.py::test_ask_with_source_id_filter PASSED
tests/test_ask.py::test_ask_requires_question PASSED
tests/test_health.py::test_health PASSED
tests/test_index.py::test_index_returns_chunk_count PASSED
tests/test_index.py::test_index_is_idempotent PASSED
tests/test_index.py::test_index_requires_source_id PASSED
tests/test_index.py::test_index_requires_content PASSED
tests/test_sources.py::test_list_sources_empty PASSED
tests/test_sources.py::test_list_sources_after_index PASSED
tests/test_sources.py::test_delete_source PASSED
tests/test_sources.py::test_delete_nonexistent_source_returns_404 PASSED

13 passed in 0.34s
```

---

## Project structure

```
app/
├── core/
│   ├── chunker.py       # Text splitting with configurable size and overlap
│   ├── embeddings.py    # ONNX embedding with SentenceTransformer/hash fallback
│   ├── vector_store.py  # ChromaDB wrapper (upsert, cosine search, delete, list)
│   └── llm_client.py    # Anthropic Claude async client
├── services/
│   ├── index_service.py # Orchestrates: chunk → embed → upsert (idempotent)
│   └── ask_service.py   # Orchestrates: embed → search → guard → prompt → generate
├── api/
│   ├── routes_index.py  # POST /index — dependencies injected via Depends()
│   ├── routes_ask.py    # POST /ask
│   └── routes_sources.py
├── schemas/
│   └── models.py        # Pydantic v2 request/response models
├── dependencies.py      # DI wiring: @lru_cache factories + Depends()
├── settings.py          # Config from .env (pydantic-settings)
└── main.py
tests/
├── conftest.py          # Test fixtures: isolated store, hash embedding, mock LLM
├── test_health.py
├── test_index.py
├── test_ask.py
└── test_sources.py
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `MODEL_NAME` | `claude-sonnet-4-20250514` | Claude model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Where ChromaDB stores data |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |

---

## Stack

- **Python 3.11**
- **FastAPI** + Uvicorn
- **ChromaDB** — local persistent vector store
- **ONNX Runtime** — `all-MiniLM-L6-v2` (384-dim embeddings, no GPU required)
- **Anthropic Claude** — LLM for answer synthesis
- **Pydantic v2** — request/response validation
- **pytest** — test suite (no external dependencies required)
- **Docker** + docker-compose

---

## Future improvements

The items below are conscious next steps, not missing fundamentals.

- **Hybrid search** — combine semantic search with BM25 keyword search; semantic search excels at meaning but misses exact-match queries (product codes, error codes, version numbers)
- **Re-ranking** — add a cross-encoder after initial retrieval to re-score the top-k chunks with higher precision; the tradeoff is latency (cross-encoders are 10-50x slower than bi-encoders)
- **Streaming responses** — stream LLM output token-by-token using `client.messages.stream()` to reduce perceived latency on longer answers
- **Multi-tenant isolation** — API key per tenant, with `tenant_id` injected into every ChromaDB metadata filter so each tenant's data is fully isolated
- **Semantic chunking** — split by paragraph and sentence boundaries instead of character count, preserving semantic coherence within each chunk
- **RAG evaluation** — integrate [RAGAS](https://github.com/explodinggradients/ragas) to measure context precision, context recall, and answer faithfulness against a golden dataset

---

## Known limitations & production considerations

This project is intentionally scoped as a single-tenant, single-instance service. The following are the gaps between this implementation and a production-grade deployment, and the reasoning behind each.

**No authentication**
Every request has full read/write access to the index. In production, you'd add API key validation as FastAPI middleware — each key maps to a tenant, and every ChromaDB query includes a `tenant_id` metadata filter to isolate data.

**ChromaDB runs in-process with a single writer**
`chromadb.PersistentClient` writes directly to disk. With `uvicorn --workers N`, each worker process has its own client instance pointing to the same directory — concurrent writes can corrupt the index. The fix is ChromaDB server mode (one HTTP server, N clients) or replacing ChromaDB with a managed vector store (Pinecone, Weaviate cloud) behind an interface.

**Character-based chunking**
Splitting at 500 characters can break mid-sentence or mid-code block. A better approach is semantic chunking: split by paragraph, then merge small paragraphs, then split oversized ones. This requires a tokenizer-aware splitter (e.g., LangChain's `RecursiveCharacterTextSplitter` with token counting).

**No re-ranking**
The retrieval step uses a bi-encoder (fast, approximate). For higher precision, a cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) can re-score the top-k results before sending them to the LLM. The tradeoff is latency: cross-encoders are 10-50x slower than bi-encoders.

**No LLM response streaming**
`AnthropicClient.generate()` waits for the full response before returning. For a user-facing product, streaming the response token-by-token (using `client.messages.stream()`) would significantly improve perceived latency.

**No RAG evaluation**
There is no measurement of retrieval quality or answer faithfulness. In production, you'd integrate a framework like [RAGAS](https://github.com/explodinggradients/ragas) to track metrics like context precision, context recall, and answer relevance over a golden dataset.

**`max_distance=0.8` is empirical**
The threshold was chosen by testing against general documentation. Different domains (legal, medical, code) have different embedding distributions — the right value should be calibrated per domain using a labeled evaluation set.

---

## License

MIT
