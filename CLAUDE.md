# retriv — Project Context for Claude Code

## What This Is

retriv is a RAG-based AI assistant that answers questions grounded in indexed documents.
It is built as a product — not a script, not a POC — designed to be installed in any
client environment and operated without developer intervention for day-to-day use.

Current integration: Magento Admin (chat interface + document management module).
Architecture: decoupled Python API (FastAPI) + Magento PHP module communicating via HTTP.

First target client: FedEx (via McFadyen Digital).

---

## Architecture Philosophy

The single most important rule: **the core never knows about infrastructure**.

This project follows Hexagonal Architecture (Ports & Adapters):

```
┌─────────────────────────────────────────────────────┐
│                     CORE (domain)                   │
│  AskService · IndexService                          │
│  Pure Python · No external imports · Never changes  │
└────────────┬───────────────────────┬────────────────┘
             │ depends on            │ depends on
        (interfaces/ports)     (interfaces/ports)
             │                       │
┌────────────▼───┐         ┌─────────▼──────────────┐
│   PORTS        │         │   PORTS                 │
│ VectorStoreBase│         │ LLMClientBase (future)  │
│ EmbeddingBase  │         │ EmbeddingBase (future)  │
└────────────────┘         └─────────────────────────┘
             │                       │
┌────────────▼───────────────────────▼────────────────┐
│                   ADAPTERS                          │
│  backends/chroma.py · AnthropicClient               │
│  SentenceTransformerEmbedding                       │
│  Future: backends/pinecone.py · OpenAIClient        │
└─────────────────────────────────────────────────────┘
```

**Consequence**: adding a new vector DB, LLM provider, or embedding model = add a new
adapter file. Core services are never touched.

---

## Incremental Evolution Rules

Every future capability must follow this pattern: **add, don't rewrite**.

### Adding a new vector DB backend
1. Create `app/core/backends/<name>.py`
2. Implement `VectorStoreBase`
3. Add `elif settings.vector_backend == "<name>"` in `dependencies.py`
4. Core, services, routes: untouched

### Adding tenant isolation
1. Add `tenant_id: str = "default"` to `IndexRequest` and `AskRequest` schemas
2. Pass `tenant_id` through to `upsert_chunks` and `search` as metadata filter
3. Each adapter filters by `tenant_id` in its own way (ChromaDB: `where` clause)
4. No core logic changes — only schemas + adapters

### Adding a new LLM provider
1. Create `app/core/llm_client.py` as `LLMClientBase` (abstract, same as VectorStoreBase)
2. Move `AnthropicClient` to `app/core/backends/anthropic.py`
3. New providers: `app/core/backends/openai.py`, etc.
4. `dependencies.py` resolves which one based on `LLM_BACKEND` env var

### Adding a new domain (e.g. HR, Legal, Support)
1. Domain = a `collection_name` or a `tenant_id` prefix
2. No schema changes, no new tables — just a naming convention
3. Multi-domain in same deployment = multiple collections in same vector DB

### Swapping the embedding model
1. `EMBEDDING_MODEL` is already an env var
2. `create_embedding_service()` in `embeddings.py` resolves the model by name
3. Change the env var → different model, same interface

---

## What Never Changes (The Core Contract)

These files should rarely if ever be modified:

| File | Why it's stable |
|------|----------------|
| `app/services/ask_service.py` | Pure RAG orchestration — embed, search, prompt, generate |
| `app/services/index_service.py` | Pure indexing — chunk, embed, store |
| `app/schemas/models.py` | Public API contract — only additive changes allowed |
| `app/core/vector_store.py` | The VectorStoreBase interface |

If you feel the urge to modify `ask_service.py` or `index_service.py` to accommodate
a new infrastructure choice, stop — the right answer is a new adapter, not a rewrite.

---

## Gitflow

```
main        → production only — receives merges from develop or hotfix/*
develop     → integration — all features land here first
feature/*   → one feature per branch, branched from develop
hotfix/*    → urgent production fix, branched from main, merged to main + develop
```

Rules:
- Never commit directly to `main` or `develop`
- Every feature goes through a PR
- Tags on `main` follow semver: `v1.0.0`, `v1.1.0`, `v2.0.0`
- Hotfixes get a patch bump: `v1.0.1`

---

## Commit Style

```
type: short description in imperative mood

Optional body explaining WHY, not what.
```

Types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`

Examples:
```
feat: add tenant_id filtering to vector store interface
refactor: extract LLMClientBase to enable provider swapping
fix: handle empty collection on ChromaDB count()
chore: add CHROMA_HOST to .env.example
```

**No co-authorship lines. No "Generated with Claude". Commits are authored by the developer only.**

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | — | Required |
| `MODEL_NAME` | `claude-sonnet-4-20250514` | LLM model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `CHROMA_MODE` | `embedded` | `embedded` (local) or `server` (production) |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Path for embedded mode |
| `CHROMA_HOST` | `localhost` | ChromaDB server host |
| `CHROMA_PORT` | `8000` | ChromaDB server port |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |

Local dev: copy `.env.example` to `.env`, fill `ANTHROPIC_API_KEY`, run `python run.py`.
Production: `docker-compose up` — API + ChromaDB server start together.

---

## Production Readiness Checklist

Features implemented:
- [x] RAG pipeline (embed → search → prompt → generate)
- [x] Streaming responses (SSE)
- [x] Vector store abstraction (swappable backends)
- [x] Magento module with chat + document management
- [x] ACL (role-based access in Magento Admin)
- [x] Configurable API URL per environment

Features pending (in order of priority):
- [ ] API authentication (API key header) — `feature/api-authentication`
- [ ] Docker Compose production setup (ChromaDB server mode) — `feature/production-docker`
- [ ] Structured logging — `feature/observability`
- [ ] LLMClientBase abstraction (same pattern as VectorStoreBase) — `feature/llm-abstraction`
- [ ] Tenant isolation — `feature/multi-tenant`

---

## What This Is Not (Yet)

- Not an agent — does not take actions (cancel orders, send emails)
- Not connected to live store data (orders, inventory, customers)
- Not multi-tenant — single collection, single client per deployment
- Not multi-LLM — Anthropic only

These are Phase 2+ items. Do not design for them now beyond the abstraction layer.
The abstraction layer exists precisely so these can be added without touching the core.
