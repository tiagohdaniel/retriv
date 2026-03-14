# retriv

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![Tests](https://img.shields.io/badge/tests-43%20passed-brightgreen)
![CI](https://github.com/tiagohdaniel/retriv/actions/workflows/ci.yml/badge.svg)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

A domain-agnostic RAG API. Index any text content, ask natural language questions, and get LLM-generated answers grounded on retrieved context — with source references and relevance scores.

Built with **FastAPI**, **ChromaDB**, **fastembed** (ONNX Runtime, no PyTorch), and **Claude** as the reasoning layer.

---

## Quick overview

- **Index** any documentation via `POST /index` — text is chunked semantically, embedded, and stored in a vector database
- **Ask** natural language questions via `POST /ask` — relevant chunks are retrieved via hybrid search (semantic + BM25) and sent to an LLM for a grounded answer with source citations
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

## RAG pipeline (v1.1)

```
POST /index                          POST /ask
     │                                    │
     ▼                                    ▼
SemanticChunker                  query → EmbeddingService
(paragraphs → merge up to 800ch)  (nomic-embed-text-v1.5)
     │                                    │
     ▼                              ┌─────┴──────────────────┐
EmbeddingService                   │                        │
(nomic-embed-text-v1.5)       Semantic search           BM25 search
     │                        (vector cosine)        (rank-bm25 + RRF)
     ▼                              │                        │
VectorStore upsert                 └─────────┬──────────────┘
                                             ▼
                                      RRF merge + top-k
                                             │
                                             ▼
                                    Filter max_distance
                                             │
                                             ▼
                                Guard: no chunks? → skip LLM
                                             │
                                       [optional]
                                             ▼
                                    CrossEncoder reranker
                                    (BAAI/bge-reranker-base)
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

### Chunking: semantic, 800 chars, 100 overlap

The default chunker splits text by paragraph boundaries (double newlines), merges small paragraphs greedily up to 800 characters, and carries a tail overlap of ~100 characters into the next chunk to avoid cutting context at boundaries. Oversized paragraphs are split at sentence boundaries.

- **Semantic vs fixed:** Fixed chunking cuts every N characters regardless of content. Semantic chunking respects paragraph and sentence boundaries, keeping related sentences together and improving retrieval precision.
- **800 chars:** ~3-6 sentences per chunk. Large enough to carry context, specific enough to score well on single-topic queries.
- **100 char overlap:** Ensures content at paragraph boundaries appears in both adjacent chunks.

Set `CHUNKING_STRATEGY=fixed` to revert to character-based splitting.

---

### Hybrid search: BM25 + semantic → RRF

When `HYBRID_ENABLED=true`, retrieval runs two independent searches and merges them:

1. **Semantic search** — cosine similarity in the embedding vector space (finds by *meaning*)
2. **BM25 search** — keyword frequency/IDF scoring via `rank-bm25` (finds by *exact terms*)

Results are merged with **Reciprocal Rank Fusion** (RRF):

```
score(d) = Σ 1 / (k + rank_i(d))
```

Documents appearing in both lists get boosted. BM25 complements semantic search for exact-term queries (product codes, acronyms, proper nouns) where semantic similarity alone can miss.

**Tokenization:** BM25 tokenises with a language-agnostic plural normaliser (`_normalize_token`) — strips common plural endings that work for both English and Portuguese without language-specific assumptions. Minimum stem length: 3 characters to avoid over-stripping short words.

---

### Cross-encoder reranker (optional)

When `RERANKER_ENABLED=true`, after retrieving `RERANKER_TOP_K_FETCH` candidates (default 15), each `(query, document)` pair is scored independently by a cross-encoder (`BAAI/bge-reranker-base` — multilingual). The top `RERANKER_TOP_N` are kept and sent to the LLM.

**Why cross-encoder over bi-encoder for reranking:** Bi-encoders (embedding models) compress query and document independently — they cannot model their interaction directly. Cross-encoders see both texts together, producing more accurate relevance scores at the cost of higher latency. This is why reranking is a second pass, not the primary retrieval step.

Model download happens once on first use. Set `RERANKER_ENABLED=false` to skip entirely (zero overhead).

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

---

### Known limitations

**Semantic search bridges synonyms, but struggles with technical acronyms.**

The embedding model understands natural language synonyms well (e.g. `falecer` → `morte`, `prazo de espera` → `carência`). However, it can fail when the document uses technical acronyms without expanding them in context (e.g. `IFPD` — *Invalidez Funcional Permanente e Total por Doença*), because the embedding space has limited signal to link an unexpanded acronym to its full description.

Similarly, BM25 keyword search cannot match a query term to a synonym — it only scores exact or normalised-plural matches. Hybrid search mitigates this but does not eliminate it.

**Mitigations to consider:**
- Expand acronyms in the source document before indexing
- Include a glossary section in indexed content that maps acronyms to their full forms
- Increase `top_k` to surface more candidate chunks, giving the LLM a wider context

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
43 passed in 0.72s
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
│   ├── chunker.py               # SemanticChunker (default) + TextChunker (fixed)
│   ├── embeddings.py            # fastembed (ONNX) with fallback chain
│   ├── hybrid_search.py         # BM25Searcher + RRF merge
│   ├── reranker.py              # FastEmbedReranker + NullReranker
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
├── test_chunker.py
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
| `API_KEY` | *(empty)* | API key for single-tenant auth |
| `API_KEYS` | *(empty)* | Multi-tenant keys: `key1:tenant1,key2:tenant2` |
| `CHROMA_MODE` | `embedded` | `embedded` (local) or `server` (production) |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Path for embedded mode |
| `CHROMA_HOST` | `localhost` | ChromaDB server host |
| `CHROMA_PORT` | `8000` | ChromaDB server port |
| `CHUNKING_STRATEGY` | `semantic` | `semantic` (paragraph-aware) or `fixed` (character-based) |
| `CHUNK_SIZE` | `800` | Target characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `HYBRID_ENABLED` | `false` | Enable BM25 + semantic hybrid search with RRF |
| `HYBRID_BM25_CORPUS_LIMIT` | `200` | Max chunks loaded into BM25 corpus (≤ 200 for Chroma Cloud) |
| `RERANKER_ENABLED` | `false` | Enable cross-encoder reranking after retrieval |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder model (multilingual) |
| `RERANKER_TOP_K_FETCH` | `15` | Candidates retrieved before reranking |
| `RERANKER_TOP_N` | `5` | Candidates kept after reranking (sent to LLM) |
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
- **rank-bm25** — BM25Okapi for keyword search in hybrid mode
- **Anthropic Claude** — LLM for answer synthesis and evaluation
- **SlowAPI** — per-endpoint rate limiting
- **structlog** — structured JSON logging with request ID correlation
- **Prometheus** — metrics via `prometheus-fastapi-instrumentator`, multiprocess-safe
- **Langfuse** — RAG evaluation tracing and dashboarding (optional)
- **Pydantic v2** — request/response validation
- **pytest** — 43 tests, no external dependencies required
- **GitHub Actions** — CI on every push and pull request
- **Docker** + docker-compose

---

## Roadmap

### v1.1 — Retrieval quality ✅ complete

- **Semantic chunking** — splits at paragraph/sentence boundaries instead of fixed character counts, preserving context
- **Cross-encoder reranking** — optional second-pass scoring of `(query, doc)` pairs with `BAAI/bge-reranker-base`
- **BM25 hybrid search** — keyword search merged with semantic via Reciprocal Rank Fusion (RRF)

### v1.2 — Analytics pipeline (next)

The goal is to expose aggregated usage and quality data without requiring Langfuse.

Planned additions:
- `POST /analyze` — run a batch of test questions and return precision/recall metrics
- Per-source retrieval stats (which sources are cited most, which return low-relevance chunks)
- Answer quality trends over time (faithfulness, relevancy scores stored locally)
- Dashboard integration via the existing `/metrics` endpoint or a new `/analytics` endpoint

---

## License

MIT
