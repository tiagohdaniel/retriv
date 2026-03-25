# retriv

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![Tests](https://img.shields.io/badge/tests-80%20passed-brightgreen)
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
- **Analyze** structured data via `POST /analyze` — send a CSV/XLSX/JSON file and a natural language question; pandas computes deterministically, the LLM only formats the result
- **Manage** indexed sources via `GET /sources` (paginated) and `DELETE /sources/{source_id}`
- **Connect agents** via `GET /mcp` — MCP server (Streamable HTTP, MCP spec 2025-03-26); any MCP-compatible agent (AGNO, Claude Desktop, LangGraph) connects without custom HTTP wrappers
- **Observe** via Prometheus metrics at `GET /metrics` — aggregated across all Gunicorn workers
- **Evaluate** RAG quality automatically via LLM-as-judge scoring sent to Langfuse (optional)
- **Secure** via `X-API-Key` or `Authorization: Bearer` — disabled by default for local dev, enabled in production
- No GPU required — embeddings run via ONNX Runtime; tests run without API keys

---

## The problem

Keyword search breaks on documentation. A user searches for *"how to handle payment failures"* but the relevant paragraph says *"retry logic for declined transactions"* — zero keyword overlap, relevant content missed.

Semantic search solves this by encoding content and queries into the same vector space and finding chunks by *meaning*, not by word match. The LLM then synthesizes a grounded answer from the retrieved chunks.

---

## RAG pipeline (`POST /ask`)

```
POST /index                          POST /ask
     │                                    │
     ▼                                    ▼
SemanticChunker                  query → EmbeddingService
(paragraphs → sentence units)    (nomic-embed-text-v1.5)
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
                                       [optional]
                                             ▼
                                  Source diversity cap
                                  (max N chunks per source_id)
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

## Analytics pipeline (`POST /analyze`)

```
POST /analyze
     │
     ▼
base64 decode (CSV / XLSX / JSON)
     │
     ▼
PandasAnalyzer
├── load DataFrame
├── infer intent from question (heuristics, no LLM)
│   └── sum · mean · max · min · count · list · describe
├── detect target column (name found in question)
├── detect group column ("por X" / "by X")
└── execute deterministically → AnalysisResult
     │
     ▼
Build prompt: dataset info + computed result
     │
     ▼
LLMClient (Claude) — formats result into natural language
     │
     ▼
AnalyzeResponse
├── answer (natural language)
└── computation (raw result for client use)
```

**Why this matters:** LLMs cannot aggregate data reliably. Asking Claude to sum 10,000 rows from a spreadsheet will produce wrong answers. This pipeline computes first, then formats — correctness is guaranteed by pandas, not by the model.

**Supported file types:** CSV, XLSX, JSON (base64-encoded in the request body)

**Operations inferred from the question (PT and EN):**

| Keywords | Operation |
|---|---|
| `total`, `soma`, `sum` | Sum |
| `média`, `average`, `mean` | Mean |
| `maior`, `máximo`, `max`, `highest` | Max |
| `menor`, `mínimo`, `min`, `lowest` | Min |
| `quantos`, `count`, `how many` | Count |
| `listar`, `list`, `top N` | List first N rows |
| *(no keyword matched)* | Descriptive stats |

**Grouping:** Include "por X" or "by X" in the question to group results (e.g. *"total de vendas por região"* → `groupby("região").sum()`).

---

## Architecture

retriv follows Hexagonal Architecture (Ports & Adapters). The core never knows about infrastructure.

```
┌──────────────────────────────────────────────────────────────────┐
│                         CONTRACTS (ports)                        │
│  VectorStoreBase · LLMClientBase · ObservabilityBase             │
│  DataAnalyzerBase                                                │
└────────┬─────────────────────────┬───────────────────────────────┘
         │                         │
┌────────▼────────┐   ┌────────────▼────────────────────────────┐
│    BACKENDS     │   │              SERVICES                    │
│  chroma.py      │   │  AskService      — RAG pipeline          │
│  anthropic.py   │   │  IndexService    — chunk/embed/store     │
│  pandas_        │   │  AnalyticsService — analyze → LLM format │
│  analyzer.py    │   └──────────────────────────────────────────┘
│  langfuse_      │
│  observability  │
│  null_          │
│  observability  │
└─────────────────┘
```

Adding a new vector DB, LLM provider, observability backend, or data analyzer = new file in `app/core/backends/`. Core services never change.

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

### Chunking: semantic, 400 chars, strategy-C overlap

The default chunker splits text at paragraph boundaries (double newlines), breaks oversized paragraphs at sentence boundaries, and merges short units greedily up to `CHUNK_SIZE` characters.

**Overlap (Strategy C):** The last complete semantic unit (sentence or paragraph) is always carried into the next chunk as overlap — never truncated to a raw character suffix. If the budget (`CHUNK_OVERLAP`) allows it, additional units are prepended working backwards. This ensures the overlap is always a grammatically complete sentence, which embeds and reads correctly. Content produced by `_hard_split` (tables, legal enumerations, PDFs without punctuation) receives a char-suffix fallback instead.

**Section header detection:** Numbered headings (`1.5. CONCEITOS`, `2.3.1. DEFINIÇÃO`) are recognised as hard boundaries and always start a new chunk, regardless of available space. This prevents section titles from being silently merged with the preceding section's content — a common failure mode in legal and government PDFs.

- **Semantic vs fixed:** Fixed chunking cuts every N characters regardless of content. Semantic chunking respects paragraph and sentence boundaries, keeping related sentences together and improving retrieval precision.
- **400 chars:** ~2-4 sentences per chunk. Smaller chunks score better on focused single-fact queries (e.g. a legal limit or a specific figure). Increase to 600-800 for documents with long paragraphs.
- **Strategy-C overlap:** Carries the full last sentence (even if > `CHUNK_OVERLAP` budget) so that boundary content is semantically complete in both the ending chunk and the beginning of the next.

Set `CHUNKING_STRATEGY=fixed` to revert to character-based sliding-window splitting.

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

When `RERANKER_ENABLED=true`, after retrieving `RERANKER_TOP_K_FETCH` candidates (default 30), each `(query, document)` pair is scored independently by a cross-encoder (`BAAI/bge-reranker-base` — multilingual). The top `RERANKER_TOP_N` are kept and sent to the LLM.

**Why cross-encoder over bi-encoder for reranking:** Bi-encoders (embedding models) compress query and document independently — they cannot model their interaction directly. Cross-encoders see both texts together, producing more accurate relevance scores at the cost of higher latency. This is why reranking is a second pass, not the primary retrieval step.

Model download happens once on first use. Set `RERANKER_ENABLED=false` to skip entirely (zero overhead).

---

### Source diversity cap (optional)

When `SOURCE_DIVERSITY_ENABLED=true`, after retrieval (and before reranking), the candidate pool is filtered to at most `SOURCE_DIVERSITY_MAX_PER_SOURCE` chunks per `source_id`. This prevents a single large document from dominating the top-k results and crowding out context from other sources.

**When this matters:** If a tenant has one very large document (e.g. a 300-page reference guide with 800 chunks) and several smaller ones, a query likely to match the large document will fill all top-k slots with chunks from it alone, potentially missing a more precise answer from a smaller source.

The filter preserves ranking order — the first N chunks per source_id by rank are kept, the rest are dropped. The reranker then selects the final top-k from this diverse pool.

---

### Relevance threshold: max_distance filter

ChromaDB always returns top-k chunks — even when the query is unrelated to indexed content. The `max_distance` filter (default `0.6`) discards chunks beyond the threshold before the guard clause runs. Exposed as a per-request parameter for domain-specific tuning.

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

**Glossary/acronym chunks can be missed by short queries.**

A query like *"o que é DAS"* (3 tokens) may not surface the glossary chunk that defines the acronym, even when it exists in the document. The vector similarity between a 3-word query and a dense glossary block tends to be lower than its similarity to content chunks that discuss the concept in context.

Mitigation: index the glossary as a standalone source with repeated acronym → full-form mappings, or increase `top_k`.

---

## Endpoints

| Method | Path | Auth required | Description |
|--------|------|---------------|-------------|
| `POST` | `/index` | Yes | Index a document |
| `POST` | `/ask` | Yes | Ask a question, get a grounded answer |
| `POST` | `/ask/stream` | Yes | Ask with streaming (SSE) |
| `POST` | `/analyze` | Yes | Analyze structured data (CSV/XLSX/JSON) with a natural language question |
| `GET` | `/sources` | Yes | List indexed sources (paginated) |
| `DELETE` | `/sources/{source_id}` | Yes | Remove all chunks for a source |
| `*` | `/mcp/mcp` | Yes | MCP server — 5 tools for external agents (Streamable HTTP transport). Note: FastMCP's `http_app()` adds an internal `/mcp` segment, so when mounted at `/mcp` the full path becomes `/mcp/mcp`. |
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
71 passed in ~0.8s
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
│   ├── data_analyzer.py         # DataAnalyzerBase port
│   ├── chunker.py               # SemanticChunker (default) + TextChunker (fixed)
│   ├── embeddings.py            # fastembed (ONNX) with fallback chain
│   ├── hybrid_search.py         # BM25Searcher + RRF merge
│   ├── reranker.py              # FastEmbedReranker + NullReranker
│   ├── auth.py                  # API key verification
│   ├── rate_limit.py            # SlowAPI per-endpoint limits
│   ├── logging_config.py        # structlog JSON/console config
│   ├── metrics.py               # Custom Prometheus counters
│   └── backends/
│       ├── chroma.py            # ChromaVectorStore
│       ├── anthropic.py         # AnthropicClient
│       ├── pandas_analyzer.py   # PandasAnalyzer (intent inference + computation)
│       ├── langfuse_            # LangfuseObservability (eval + tracing)
│       │   observability.py
│       └── null_observability.py # No-op (eval disabled)
├── middleware/
│   └── logging_middleware.py    # Request ID correlation
├── services/
│   ├── ask_service.py           # RAG pipeline orchestration
│   ├── index_service.py         # Indexing pipeline orchestration
│   └── analytics_service.py     # Analytics pipeline orchestration
├── api/
│   ├── routes_index.py
│   ├── routes_ask.py
│   ├── routes_sources.py
│   └── routes_analyze.py
├── mcp/
│   └── server.py                # MCP server — 5 tools for external agents
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
├── test_analyze.py
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
| `MODEL_NAME` | `claude-sonnet-4-6` | Claude model for answer generation |
| `LLM_TIMEOUT` | `30.0` | LLM request timeout in seconds |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model (⚠️ reindex required on change) |
| `API_AUTH_ENABLED` | `false` | Enable API key authentication |
| `API_KEY` | *(empty)* | API key for single-tenant auth |
| `API_KEYS` | *(empty)* | Multi-tenant keys: `key1:tenant1,key2:tenant2` |
| `CHROMA_MODE` | `embedded` | `embedded` (local), `server` (docker-compose), or `cloud` (Chroma Cloud) |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Path for embedded mode |
| `CHROMA_HOST` | `localhost` | ChromaDB server host |
| `CHROMA_PORT` | `8000` | ChromaDB server port |
| `CHROMA_CLOUD_API_KEY` | *(empty)* | Chroma Cloud API key (required when `CHROMA_MODE=cloud`) |
| `CHROMA_CLOUD_TENANT` | *(empty)* | Chroma Cloud tenant (required when `CHROMA_MODE=cloud`) |
| `CHROMA_CLOUD_DATABASE` | *(empty)* | Chroma Cloud database (required when `CHROMA_MODE=cloud`) |
| `CHUNKING_STRATEGY` | `semantic` | `semantic` (paragraph/sentence-aware) or `fixed` (character-based sliding window) |
| `CHUNK_SIZE` | `800` | Target characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap budget in characters — Strategy C carries complete sentences up to this budget |
| `HYBRID_ENABLED` | `false` | Enable BM25 + semantic hybrid search with RRF merge |
| `HYBRID_BM25_CORPUS_LIMIT` | `200` | Max chunks loaded into BM25 corpus (≤ 200 for Chroma Cloud) |
| `RERANKER_ENABLED` | `false` | Enable cross-encoder reranking after retrieval |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder model (multilingual; alt: `Xenova/ms-marco-MiniLM-L-6-v2`) |
| `RERANKER_TOP_K_FETCH` | `15` | Candidates retrieved before reranking (increase to 30 for denser corpora) |
| `RERANKER_TOP_N` | `5` | Candidates kept after reranking (sent to LLM) |
| `SOURCE_DIVERSITY_ENABLED` | `false` | Cap chunks per source in retrieval pool |
| `SOURCE_DIVERSITY_MAX_PER_SOURCE` | `3` | Max chunks from the same `source_id` in the retrieval pool |
| `ANALYTICS_MAX_ROWS` | `100000` | Max rows allowed in uploaded datasets |
| `ANALYTICS_MAX_FILE_SIZE_MB` | `10` | Max file size for `POST /analyze` (MB) |
| `RATE_LIMIT_ENABLED` | `false` | Enable per-endpoint rate limiting |
| `RATE_LIMIT_INDEX` | `10/minute` | Rate limit for `POST /index` |
| `RATE_LIMIT_ASK` | `30/minute` | Rate limit for `POST /ask` |
| `RATE_LIMIT_SOURCES` | `60/minute` | Rate limit for `GET /sources` |
| `RATE_LIMIT_ANALYZE` | `20/minute` | Rate limit for `POST /analyze` |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | `json` (production) or `console` (local dev) |
| `CORS_ORIGINS` | `*` | Allowed origins (comma-separated or `*`) |
| `WEB_CONCURRENCY` | `2` | Number of Gunicorn worker processes |
| `METRICS_ENABLED` | `true` | Expose `GET /metrics` in Prometheus format |
| `METRICS_USERNAME` | *(empty)* | Basic auth username for `/metrics` (leave empty to disable auth) |
| `METRICS_PASSWORD` | *(empty)* | Basic auth password for `/metrics` |
| `EVAL_ENABLED` | `false` | Enable async RAG quality evaluation |
| `EVAL_MODEL` | `claude-haiku-4-5-20251001` | Model used as LLM judge for evaluation |
| `LANGFUSE_PUBLIC_KEY` | *(empty)* | Langfuse project public key |
| `LANGFUSE_SECRET_KEY` | *(empty)* | Langfuse project secret key |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse server URL |

---

## Stack

- **Python 3.11**
- **FastAPI** + Uvicorn + Gunicorn (multi-worker)
- **fastmcp** — MCP server (Streamable HTTP transport, MCP spec 2025-03-26)
- **ChromaDB** — vector store (embedded or server mode)
- **fastembed** — `nomic-embed-text-v1.5` via ONNX Runtime, no GPU required
- **rank-bm25** — BM25Okapi for keyword search in hybrid mode
- **pandas** + **openpyxl** — deterministic data analysis for the analytics pipeline
- **Anthropic Claude** — LLM for answer synthesis and evaluation
- **SlowAPI** — per-endpoint rate limiting
- **structlog** — structured JSON logging with request ID correlation
- **Prometheus** — metrics via `prometheus-fastapi-instrumentator`, multiprocess-safe
- **Langfuse** — RAG evaluation tracing and dashboarding (optional)
- **Pydantic v2** — request/response validation
- **pytest** — 71 tests, no external dependencies required
- **GitHub Actions** — CI on every push and pull request
- **Docker** + docker-compose

---

## Roadmap

### v1.1 — Retrieval quality ✅ complete

- **Semantic chunking** — splits at paragraph/sentence boundaries instead of fixed character counts, preserving context
- **Cross-encoder reranking** — optional second-pass scoring of `(query, doc)` pairs with `BAAI/bge-reranker-base`
- **BM25 hybrid search** — keyword search merged with semantic via Reciprocal Rank Fusion (RRF)
- **Section header detection** — numbered headings (`1.5. CONCEITOS`) always start a new chunk, preventing section merges in legal/government PDFs
- **Strategy-C overlap** — overlap carries complete semantic units (never raw character fragments), preserving sentence integrity at chunk boundaries
- **Source diversity cap** — optional `SOURCE_DIVERSITY_ENABLED` limits chunks per `source_id` in the retrieval pool, preventing large documents from crowding out smaller sources
- **Richer Langfuse traces** — `source_distribution` and `hybrid` flags per query for retrieval diagnostics

### v1.2 — Analytics pipeline ✅ complete

- **`POST /analyze`** — analyze structured data (CSV, XLSX, JSON) with a natural language question
- Heuristic intent inference (PT + EN): sum, mean, max, min, count, list, describe
- Automatic grouping: *"total de vendas por região"* → `groupby("região").sum()`
- Deterministic computation via pandas — the LLM only formats the result, never aggregates raw data
- File size guard (default 10 MB), row limit (default 100k), Prometheus metrics

### v1.2.1 — Retrieval pipeline fixes ✅ complete

Diagnosed and resolved a set of retrieval failures that were limiting benchmark accuracy to 28%.

- **TOC chunk detection fix** — `_is_toc_chunk` now runs on raw chunks *before* `_clean_pdf_text` strips the dot leaders that identify table-of-contents lines. Previously, TOC chunks were passing the filter and dominating top-k results, surfacing section names instead of actual content
- **Single-pass indexing** — replaced the error-prone double-chunking approach (raw for detection, clean for embedding) with a single pass: chunk raw → detect TOC on raw chunk → clean individual chunk → filter. Eliminates index-misalignment risk
- **PDF noise cleaning** — `_clean_pdf_text` removes navigation footers (*"Voltar ao Sumário N"*), broken hyphenation across line breaks, dot leader lines (`........38`), and excess whitespace — common artifacts in scanned government PDFs
- **Section header overlap fix** — section headers now always start a fresh chunk with no overlap from the preceding section. Previously, overlap could carry context from the previous section into the header, splitting the header from its body in the index
- **Duplicate chunk elimination** — fixed an overlap guard bug where a single large unit that filled an entire chunk was being emitted again as the overlap prefix, creating exact duplicates in the index. Fix: trim oldest overlap units until the remainder fits instead of emitting a standalone duplicate
- **ALL-CAPS section header detection** — `_is_section_header` now recognises unnumbered all-caps multi-word headings (e.g. `CONCEITOS PRELIMINARES`) in addition to numbered headings (`1.5. CONCEITOS`). These are treated as hard boundaries that never merge with preceding content
- **Min-content filter at retrieval time** — chunks shorter than 80 chars are discarded after retrieval (headers, footers, and navigation fragments that survived indexing produce high-similarity false positives)
- **Short-chunk distance penalty** — chunks under 200 chars receive a `+0.1` distance penalty in the retrieval ranking, pushing stub chunks below content-rich chunks of similar score
- **`top_k` default 5 → 10** — the single highest-impact change; doubles recall by giving the LLM twice as many candidate chunks to work from
- **`max_distance` default 0.45 → 0.6** — the previous threshold was too tight, cutting relevant content chunks while TOC chunks (cosine distance ~0.40) still passed
- **Domain query normalizer** — `RuleBasedQueryNormalizer` maps informal vocabulary to document vocabulary before embedding. Domain-specific rule sets (e.g. `accounting`) correct vocabulary mismatches that neither semantic nor BM25 search can bridge — e.g. *"posso ganhar"* → *"limite de receita bruta"*. New domains are added as a list of `{pattern, replacement, name}` rules; no code change required beyond the registry

Benchmark result (guia PGDAS-D, 25 informal questions):

| Configuration | Correct | Score |
|---|---|---|
| Before (top_k=5, max_distance=0.45, no PDF cleaning) | 7/25 | 28% |
| After pipeline fixes (v1.2.1) | 19/25 | **76%** (fair eval) |
| After + chunker dedup fix + query normalizer | 20/25 | **80%** |
| After, within-scope questions only | 20/21 | **95%** |

The 5 remaining misses are out-of-scope questions (MEI-specific content not present in a PGDAS-D guide). The within-scope score of 95% reflects the retrieval ceiling for a single-document index — the next quality lever is indexing additional relevant documents.

### v1.3 — MCP Server ✅ complete

- **`/mcp/mcp`** — MCP Streamable HTTP server (MCP spec 2025-03-26)
- 5 tools: `ask`, `search`, `index_document`, `list_sources`, `delete_source`
- Any MCP-compatible agent (AGNO, Claude Desktop, LangGraph, etc.) connects without custom HTTP wrappers
- Auth reuses `API_KEYS` — multi-tenant isolation preserved across all tools
- Removed `AgentOrchestrator` / `ToolBase` stubs — agent orchestration belongs in domain agent repos
- **Integration note:** FastMCP requires its `lifespan` to be wired into the FastAPI instance (`lifespan=_mcp_http_app.lifespan`) — without this the `StreamableHTTPSessionManager` is never initialized and all MCP requests return 500. See `app/main.py`.

### v1.4 — Resiliência

- **Redis job store** — replace `/tmp` with Redis to survive restarts and multiple workers
- **Automatic retry** — retry failed indexing jobs with exponential backoff
- **Usage analytics** — track questions, latencies, and failures per tenant

---

## License

MIT
