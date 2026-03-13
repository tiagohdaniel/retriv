# retriv

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![Tests](https://img.shields.io/badge/tests-13%20passed-brightgreen)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

A domain-agnostic RAG API. Index any text content, ask natural language questions, and get LLM-generated answers grounded on retrieved context — with source references and relevance scores.

Built with **FastAPI**, **ChromaDB**, **ONNX Runtime** embeddings, and **Claude** as the reasoning layer.

---

## Quick overview

- **Index** any documentation via `POST /index` — text is chunked, embedded, and stored in a vector database
- **Ask** natural language questions via `POST /ask` — relevant chunks are retrieved and sent to an LLM for a grounded answer with source citations
- **Stream** responses token-by-token via `POST /ask/stream` — SSE for real-time output
- **Manage** indexed sources via `GET /sources` and `DELETE /sources/{source_id}`
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
(500 chars, 50 overlap)                   │
     │                                    ▼
     ▼                          VectorStore cosine search
EmbeddingService                (top-k most similar chunks)
(ONNX all-MiniLM-L6-v2)                  │
     │                                    ▼
     ▼                          Filter by max_distance (default 0.8)
VectorStore upsert            (discard semantically unrelated chunks)
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
```

---

## Architecture

retriv follows Hexagonal Architecture (Ports & Adapters). The core never knows about infrastructure.

```
┌──────────────────────────────────────────────────────────┐
│                     CONTRACTS (ports)                    │
│   VectorStoreBase · LLMClientBase · ToolBase             │
└────────┬──────────────────────┬──────────────────────────┘
         │                      │
┌────────▼────────┐   ┌─────────▼──────────────────────────┐
│    BACKENDS     │   │            SERVICES                 │
│  chroma.py      │   │  AskService   — RAG pipeline        │
│  anthropic.py   │   │  IndexService — chunk/embed/store   │
│  openai.py ...  │   │  AgentService — tool orchestration  │
└─────────────────┘   └────────────────────────────────────┘
```

Adding a new vector DB or LLM provider = new file in `app/core/backends/`. Core services never change.

---

## Design decisions & tradeoffs

### Embedding: ONNX Runtime over sentence-transformers

The `all-MiniLM-L6-v2` model runs via ONNX Runtime instead of the `sentence-transformers` library.

**Why:** `sentence-transformers` pulls in PyTorch as a dependency. PyTorch CPU-only is ~500MB. ONNX Runtime is ~10MB and runs the same exported model directly.

**Fallback chain:** `ONNXEmbedding → SentenceTransformerEmbedding → HashEmbedding`. In tests, `HashEmbedding` runs with zero dependencies — no model download, deterministic output.

---

### Vector store: ChromaDB

Runs embedded (local dev) or in server mode (production). The `VectorStoreBase` interface decouples the rest of the code from ChromaDB — swapping to Pinecone or pgvector requires only a new backend file.

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

## Endpoints

| Method | Path | Auth required | Description |
|--------|------|---------------|-------------|
| `POST` | `/index` | Yes | Index a document |
| `POST` | `/ask` | Yes | Ask a question, get a grounded answer |
| `POST` | `/ask/stream` | Yes | Ask with streaming (SSE) |
| `GET` | `/sources` | Yes | List all indexed sources |
| `DELETE` | `/sources/{source_id}` | Yes | Remove all chunks for a source |
| `GET` | `/health` | No | Health check |
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
- `api` — exposed on port `8001`, starts after ChromaDB is healthy

---

## Authentication

Authentication is disabled by default for local dev. To enable:

```env
API_AUTH_ENABLED=true
API_KEY=your-strong-random-key
```

All endpoints except `/health` require the header:

```
X-API-Key: your-strong-random-key
```

---

## Running tests

No API key needed. Tests use in-memory ChromaDB, hash-based embeddings, and a mocked LLM client.

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
│   ├── vector_store.py      # VectorStoreBase interface
│   ├── llm_client.py        # LLMClientBase interface
│   ├── chunker.py           # Text splitting
│   ├── embeddings.py        # ONNX embedding with fallback chain
│   ├── auth.py              # API key verification dependency
│   ├── agent/
│   │   ├── base.py          # ToolBase interface
│   │   ├── orchestrator.py  # Generic agentic loop
│   │   └── tools/
│   │       └── rag_tool.py  # RAG as a pluggable tool
│   └── backends/            # Swappable implementations
│       ├── chroma.py        # ChromaVectorStore
│       └── anthropic.py     # AnthropicClient
├── agents/                  # Platform-specific agents (future)
├── services/
│   ├── ask_service.py       # RAG pipeline orchestration
│   ├── index_service.py     # Indexing pipeline orchestration
│   └── agent_service.py     # Agent orchestration
├── api/
│   ├── routes_index.py
│   ├── routes_ask.py
│   └── routes_sources.py
├── schemas/
│   └── models.py
├── dependencies.py          # DI wiring
├── settings.py              # Config from .env
└── main.py
tests/
├── conftest.py
├── test_health.py
├── test_index.py
├── test_ask.py
└── test_sources.py
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `LLM_BACKEND` | `anthropic` | LLM provider |
| `MODEL_NAME` | `claude-sonnet-4-20250514` | Claude model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `API_AUTH_ENABLED` | `false` | Enable API key auth |
| `API_KEY` | *(empty)* | API key for authentication |
| `CHROMA_MODE` | `embedded` | `embedded` or `server` |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Path for embedded mode |
| `CHROMA_HOST` | `localhost` | ChromaDB server host |
| `CHROMA_PORT` | `8000` | ChromaDB server port |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |

---

## Stack

- **Python 3.11**
- **FastAPI** + Uvicorn
- **ChromaDB** — vector store (embedded or server mode)
- **ONNX Runtime** — `all-MiniLM-L6-v2` embeddings, no GPU required
- **Anthropic Claude** — LLM for answer synthesis
- **Pydantic v2** — request/response validation
- **pytest** — test suite, no external dependencies required
- **Docker** + docker-compose

---

## License

MIT
