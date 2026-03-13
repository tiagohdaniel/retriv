# retriv

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![Tests](https://img.shields.io/badge/tests-35%20passed-brightgreen)
![CI](https://github.com/tiagohdaniel/retriv/actions/workflows/ci.yml/badge.svg)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

A domain-agnostic RAG API. Index any text content, ask natural language questions, and get LLM-generated answers grounded on retrieved context — with source references and relevance scores.

Built with **FastAPI**, **ChromaDB**, **fastembed** (ONNX Runtime, no PyTorch), and **Claude** as the reasoning layer.

---

## Quick overview

- **Index** any documentation via `POST /index` — text is chunked, embedded, and stored in a vector database
- **Ask** natural language questions via `POST /ask` — relevant chunks are retrieved and sent to an LLM for a grounded answer with source citations
- **Stream** responses token-by-token via `POST /ask/stream` — SSE for real-time output
- **Manage** indexed sources via `GET /sources` (paginated) and `DELETE /sources/{source_id}`
- **Observe** via Prometheus metrics at `GET /metrics` — aggregated across all Gunicorn workers
- **Evaluate** RAG quality automatically via LLM-as-judge scoring sent to Langfuse (optional)
- **Secure** via `X-API-Key` header — disabled by default for local dev, enabled in production
- No GPU required — embeddings run via ONNX Runtime; tests run without API keys

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
(500 chars, 50 overlap)         (nomic-embed-text-v1.5)
     │                                    │
     ▼                                    ▼
EmbeddingService               VectorStore cosine search
(nomic-embed-text-v1.5)        (top-k most similar chunks)
     │                                    │
     ▼                                    ▼
VectorStore upsert             Filter by max_distance (default 0.8)
                               (discard semantically unrelated chunks)
                                          │
                                          ▼
                                Guard: no chunks? → skip LLM
                                          │
                                          ▼
                                 Build prompt with context
                                          │
                                          ▼
                                  LLMClient (Claude)
                                          │
                                          ▼
                                    AskResponse
                                (answer + sources + scores)
                                          │
                               [async background task]
                                          ▼
                               LLM-as-judge evaluation
                               (faithfulness + relevancy)
                                          ▼
                                  Langfuse trace + scores
```

---

## Architecture

retriv follows Hexagonal Architecture (Ports & Adapters). The core never knows about infrastructure.

```
┌────────────────────────────────────────────────────────────┐
│                      CONTRACTS (ports)                     │
│   VectorStoreBase · LLMClientBase · ObservabilityBase      │
└────────┬───────────────────────┬───────────────────────────┘
         │                       │
┌────────▼────────┐   ┌──────────▼──────────────────────────┐
│    BACKENDS     │   │             SERVICES                 │
│  chroma.py      │   │  AskService   — RAG pipeline         │
│  anthropic.py   │   │  IndexService — chunk/embed/store    │
│  langfuse_      │   └─────────────────────────────────────┘
│  observability  │
│  null_          │
│  observability  │
└─────────────────┘
```

Adding a new vector DB, LLM provider, or observability backend = new file in `app/core/backends/`. Core services never change.

---

## Design decisions & tradeoffs

### Embedding: fastembed + nomic-embed-text-v1.5

Text is embedded using `nomic-embed-text-v1.5` (768 dimensions) via the `fastembed` library, which runs models in ONNX format — no PyTorch, no GPU required.

**Why nomic over MiniLM:** `all-MiniLM-L6-v2` (384 dims, 2021) was the default for quick RAG setups but is dated. `nomic-embed-text-v1.5` significantly outperforms it on retrieval benchmarks with a richer 768-dim vector space.

**Why fastembed over sentence-transformers:** `sentence-transformers` pulls in PyTorch (~500MB). fastembed downloads ONNX models from HuggingFace and runs them directly via ONNX Runtime (~10MB overhead).

**Fallback chain:** `FastEmbedEmbedding → ONNXEmbedding (MiniLM only) → HashEmbedding`. In tests, `HashEmbedding` runs deterministically with zero model downloads.

> ⚠️ Changing `EMBEDDING_MODEL` after indexing requires reindexing all sources — vector spaces are incompatible across models.

---

### Vector store: ChromaDB

Runs embedded (local dev) or in server mode (production via docker-compose). The `VectorStoreBase` interface decouples all code from ChromaDB — swapping to Pinecone or pgvector requires only a new backend file.

**Similarity metric: cosine, not L2.** Cosine measures the angle between vectors — it ignores magnitude. For semantic similarity, direction is meaning; magnitude is noise.

---

### Chunking: size 500, overlap 50

- **Too small (< 100 chars):** Individual sentences lose context.
- **Too large (> 1000 chars):** One chunk covers multiple topics, reducing precision.
- **500 chars:** ~2-4 sentences. Enough context, specific enough to score well on a single-topic query.
- **50 char overlap:** Ensures content at chunk boundaries appears in both adjacent chunks.

---

### Relevance threshold: max_distance filter

ChromaDB always returns top-k chunks — even when the query is unrelated to indexed content. The `max_distance` filter (default `0.8`) discards chunks beyond the threshold before the guard clause runs. Exposed as a per-request parameter for domain-specific tuning.

---

### Guard clause before LLM call

```python
if not docs:
    return AskResponse(answer="No relevant documentation found...")
```

If the vector search returns nothing after the distance filter, the LLM is never called — saves tokens and prevents hallucination from an empty context.

---

### Idempotent indexing: delete-before-upsert

Re-indexing the same `source_id` deletes all existing chunks first. Old and new chunks never coexist — a re-indexed source always reflects the latest version.

---

### RAG quality evaluation

Every `/ask` query fires an async background evaluation after the response is sent — zero added latency to the client.

Evaluation uses Anthropic Claude (Haiku by default — cheap and fast) as an LLM judge to score:
- **faithfulness** — is the answer supported by the retrieved context, or did the model hallucinate?
- **answer_relevancy** — does the answer actually address the question?

Scores and full traces are sent to [Langfuse](https://langfuse.com) for historical analysis and dashboarding. Disabled by default — enable with `EVAL_ENABLED=true` and Langfuse credentials.

**Why this matters:** reranking and hybrid search are not implemented yet — they will only be added if evaluation data shows the current retrieval quality is insufficient. Building features without metrics is guesswork.

---

## Endpoints

| Method | Path | Auth required | Description |
|--------|------|---------------|-------------|
| `POST` | `/index` | Yes | Index a document |
| `POST` | `/ask` | Yes | Ask a question, get a grounded answer |
| `POST` | `/ask/stream` | Yes | Ask with streaming (SSE) |
| `GET` | `/sources` | Yes | List indexed sources (paginated) |
| `DELETE` | `/sources/{source_id}` | Yes | Remove all chunks for a source |
| `GET` | `/health` | No | Health check (verifies ChromaDB connectivity) |
| `GET` | `/metrics` | No | Prometheus metrics (aggregated across workers) |
| `GET` | `/docs` | No | Swagger UI |

---

## Running locally

```bash
git clone https://github.com/tiagohdaniel/retriv
cd retriv

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

python run.py
```

API at `http://localhost:8001` — Swagger UI at `http://localhost:8001/docs`

---

## Running in production

```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY, API_AUTH_ENABLED=true, API_KEY=<strong-random-key>

docker compose up -d
```

This starts two containers:
- `chromadb` — vector store in server mode, internal only
- `api` — exposed on port `8001`, starts after ChromaDB is healthy, runs as non-root user

Gunicorn manages multiple Uvicorn workers (`WEB_CONCURRENCY`, default `2`). Prometheus metrics are aggregated across all workers via shared mmap files.

---

## Authentication

Authentication is disabled by default for local dev. To enable:

```env
API_AUTH_ENABLED=true
API_KEY=your-strong-random-key
```

All endpoints except `/health` and `/metrics` require the header:

```
X-API-Key: your-strong-random-key
```

---

## RAG evaluation setup

To enable quality monitoring:

1. Create a free account at [langfuse.com](https://langfuse.com) (or self-host)
2. Copy your project keys
3. Add to `.env`:

```env
EVAL_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Every `/ask` query will be evaluated asynchronously and appear in your Langfuse dashboard with faithfulness and answer relevancy scores.

---

## Running tests

No API key needed. Tests use in-memory ChromaDB, hash-based embeddings, and a mocked LLM client.

```bash
pytest tests/ -q
```

```
35 passed in 0.72s
```

Tests are also run automatically on every push and pull request via GitHub Actions.

---

## Project structure

```
app/
├── core/
│   ├── vector_store.py          # VectorStoreBase port
│   ├── llm_client.py            # LLMClientBase port
│   ├── observability.py         # ObservabilityBase port
│   ├── chunker.py               # Text splitting
│   ├── embeddings.py            # fastembed (ONNX) with fallback chain
│   ├── auth.py                  # API key verification
│   ├── rate_limit.py            # SlowAPI per-endpoint limits
│   ├── logging_config.py        # structlog JSON/console config
│   ├── metrics.py               # Custom Prometheus counters
│   ├── agent/
│   │   ├── base.py              # ToolBase interface
│   │   ├── orchestrator.py      # Generic agentic loop
│   │   └── tools/rag_tool.py    # RAG as a pluggable tool
│   └── backends/
│       ├── chroma.py            # ChromaVectorStore
│       ├── anthropic.py         # AnthropicClient
│       ├── langfuse_            # LangfuseObservability (eval + tracing)
│       │   observability.py
│       └── null_observability.py # No-op (eval disabled)
├── middleware/
│   └── logging_middleware.py    # Request ID correlation
├── services/
│   ├── ask_service.py           # RAG pipeline orchestration
│   └── index_service.py         # Indexing pipeline orchestration
├── api/
│   ├── routes_index.py
│   ├── routes_ask.py
│   └── routes_sources.py
├── schemas/models.py
├── dependencies.py              # DI wiring
├── settings.py                  # Config from .env
└── main.py
tests/
├── conftest.py
├── test_health.py
├── test_index.py
├── test_ask.py
├── test_sources.py
├── test_request_id.py
├── test_input_validation.py
├── test_metrics.py
├── test_exception_handler.py
├── test_llm_backend.py
└── test_multiprocess_metrics.py
.github/workflows/ci.yml         # GitHub Actions CI
gunicorn.conf.py                  # Worker lifecycle hooks
Dockerfile
docker-compose.yml
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `LLM_BACKEND` | `anthropic` | LLM provider (`anthropic` — others future) |
| `MODEL_NAME` | `claude-sonnet-4-20250514` | Claude model for answer generation |
| `LLM_TIMEOUT` | `30.0` | LLM request timeout in seconds |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model (⚠️ reindex required on change) |
| `API_AUTH_ENABLED` | `false` | Enable API key authentication |
| `API_KEY` | *(empty)* | API key for authentication |
| `CHROMA_MODE` | `embedded` | `embedded` (local) or `server` (production) |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Path for embedded mode |
| `CHROMA_HOST` | `localhost` | ChromaDB server host |
| `CHROMA_PORT` | `8000` | ChromaDB server port |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `RATE_LIMIT_ENABLED` | `false` | Enable per-endpoint rate limiting |
| `RATE_LIMIT_INDEX` | `10/minute` | Rate limit for `POST /index` |
| `RATE_LIMIT_ASK` | `30/minute` | Rate limit for `POST /ask` |
| `RATE_LIMIT_SOURCES` | `60/minute` | Rate limit for `GET /sources` |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | `json` (production) or `console` (local dev) |
| `CORS_ORIGINS` | `*` | Allowed origins (comma-separated or `*`) |
| `WEB_CONCURRENCY` | `2` | Number of Gunicorn worker processes |
| `METRICS_ENABLED` | `true` | Expose `GET /metrics` in Prometheus format |
| `EVAL_ENABLED` | `false` | Enable async RAG quality evaluation |
| `EVAL_MODEL` | `claude-haiku-4-5-20251001` | Model used as LLM judge for evaluation |
| `LANGFUSE_PUBLIC_KEY` | *(empty)* | Langfuse project public key |
| `LANGFUSE_SECRET_KEY` | *(empty)* | Langfuse project secret key |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse server URL |

---

## Stack

- **Python 3.11**
- **FastAPI** + Uvicorn + Gunicorn (multi-worker)
- **ChromaDB** — vector store (embedded or server mode)
- **fastembed** — `nomic-embed-text-v1.5` via ONNX Runtime, no GPU required
- **Anthropic Claude** — LLM for answer synthesis and evaluation
- **SlowAPI** — per-endpoint rate limiting
- **structlog** — structured JSON logging with request ID correlation
- **Prometheus** — metrics via `prometheus-fastapi-instrumentator`, multiprocess-safe
- **Langfuse** — RAG evaluation tracing and dashboarding (optional)
- **Pydantic v2** — request/response validation
- **pytest** — 35 tests, no external dependencies required
- **GitHub Actions** — CI on every push and pull request
- **Docker** + docker-compose

---

## Roadmap

The following features are intentionally not implemented yet. They will only be added once evaluation data from Langfuse shows they are needed:

- **Reranking** — cross-encoder reranking of retrieved chunks before LLM call (improves ordering)
- **Hybrid search** — BM25 + vector search combined via Reciprocal Rank Fusion (improves recall for exact-term queries)

Building these without metrics would be guesswork. The evaluation pipeline exists precisely to make this decision data-driven.

---

## License

MIT
