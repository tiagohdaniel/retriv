# retriv — Project Context for Claude Code

## What This Is

retriv is a domain-agnostic AI assistant that answers questions grounded in indexed documents.
It is built as a product — not a script, not a POC — designed to be installed in any
client environment and operated without developer intervention for day-to-day use.

The RAG pipeline is the foundation. It is platform-agnostic and domain-agnostic —
the same core serves any industry (retail, legal, healthcare, finance) and any platform
(Magento, BigCommerce, Oracle Commerce) without modification.

Current integration: Magento Admin (chat interface + document management module).
Architecture: decoupled Python API (FastAPI) + Magento PHP module communicating via HTTP.

First target client: FedEx (via McFadyen Digital).

---

## Architecture Philosophy

The single most important rule: **the core never knows about infrastructure or platform.**

This project follows Hexagonal Architecture (Ports & Adapters):

```
┌──────────────────────────────────────────────────────────────┐
│                        CONTRACTS (ports)                     │
│   VectorStoreBase · LLMClientBase · ToolBase · AgentBase    │
│   — define what exists, never change —                       │
└────────┬───────────────────┬──────────────────┬─────────────┘
         │                   │                  │
┌────────▼────────┐ ┌────────▼────────┐ ┌───────▼──────────────┐
│    BACKENDS     │ │     TOOLS       │ │      AGENTS          │
│  chroma.py      │ │  rag_tool.py    │ │  magento/ (future)   │
│  pinecone.py    │ │  order_tool.py  │ │  bigcommerce/        │
│  anthropic.py   │ │  catalog_tool.py│ │  oracle/             │
│  openai.py      │ │  ...            │ │  ...                 │
└─────────────────┘ └─────────────────┘ └──────────────────────┘
         │                   │                  │
┌────────▼───────────────────▼──────────────────▼─────────────┐
│                        SERVICES                              │
│  AskService    — RAG pure pipeline, never changes            │
│  IndexService  — chunk/embed/store, never changes            │
│  AgentService  — generic orchestrator, uses tools + LLM     │
└──────────────────────────────────────────────────────────────┘
```

**Adding a new platform agent (Magento, BigCommerce):** new folder in `app/agents/`.
**Adding a new vector DB:** new file in `app/core/backends/`.
**Adding a new LLM provider:** new file in `app/core/backends/`.
**Adding a new domain (legal, healthcare):** index different documents. Zero code.

Core services are never touched.

---

## Target Folder Structure

```
app/
├── api/                        # routes — HTTP layer only
│   ├── routes_ask.py
│   ├── routes_index.py
│   └── routes_sources.py
├── core/
│   ├── vector_store.py         # VectorStoreBase (contract)
│   ├── llm_client.py           # LLMClientBase (contract) — feature/llm-abstraction
│   ├── agent/
│   │   ├── base.py             # AgentBase, ToolBase (contracts) — feature/agent-foundation
│   │   └── orchestrator.py     # generic agentic loop — feature/agent-foundation
│   └── backends/               # all swappable implementations
│       ├── chroma.py           # done
│       ├── anthropic.py        # feature/llm-abstraction
│       └── openai.py           # future
├── agents/                     # platform-specific agent implementations
│   ├── magento/                # future — feature/magento-agent
│   │   └── tools.py
│   └── bigcommerce/            # future
│       └── tools.py
├── services/
│   ├── ask_service.py          # RAG pure — stable, never changes
│   ├── index_service.py        # indexing — stable, never changes
│   └── agent_service.py        # agentic orchestration — feature/agent-foundation
├── schemas/
│   └── models.py               # public API contract — additive only
├── dependencies.py             # wires everything together
└── settings.py                 # all config via env vars
```

---

## Incremental Evolution Rules

Every future capability must follow this pattern: **add, don't rewrite**.

### Adding a new vector DB backend
1. Create `app/core/backends/<name>.py`
2. Implement `VectorStoreBase`
3. Add `elif settings.vector_backend == "<name>"` in `dependencies.py`
4. Core, services, routes: untouched

### Adding a new LLM provider
1. Create `app/core/backends/<provider>.py`
2. Implement `LLMClientBase`
3. Add `elif settings.llm_backend == "<provider>"` in `dependencies.py`
4. Core, services, routes: untouched

### Adding a platform-specific agent
1. Create `app/agents/<platform>/tools.py`
2. Implement each tool extending `ToolBase`
3. Register tools in `dependencies.py`
4. `AgentService` orchestrator: untouched
5. `AskService`, `IndexService`: untouched

### Adding a new domain (legal, healthcare, HR)
1. Index documents from that domain — zero code changes
2. The RAG pipeline handles any domain natively
3. If the domain needs custom tools (e.g. query a specific API): new tool in `app/agents/`

### Adding tenant isolation
1. Add `tenant_id: str = "default"` to `IndexRequest` and `AskRequest` schemas
2. Pass `tenant_id` to `upsert_chunks` and `search` as metadata filter
3. Each adapter filters by `tenant_id` in its own way
4. No core logic changes — only schemas + adapters

### Swapping the embedding model
1. `EMBEDDING_MODEL` is already an env var
2. Change the env var → different model, same interface, zero code

---

## What Never Changes (The Core Contract)

These files are closed for modification:

| File | Why it's stable |
|------|----------------|
| `app/services/ask_service.py` | Pure RAG pipeline |
| `app/services/index_service.py` | Pure indexing pipeline |
| `app/schemas/models.py` | Public API contract — additive only |
| `app/core/vector_store.py` | VectorStoreBase interface |
| `app/core/llm_client.py` | LLMClientBase interface (once created) |
| `app/core/agent/base.py` | AgentBase, ToolBase interfaces (once created) |
| `app/core/agent/orchestrator.py` | Generic agentic loop (once created) |

If you feel the urge to modify a service or a contract file to accommodate a new
infrastructure choice, stop — the right answer is a new adapter or a new agent folder.

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
| `LLM_BACKEND` | `anthropic` | LLM provider — feature/llm-abstraction |
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
- [x] Vector store abstraction (VectorStoreBase + ChromaVectorStore)
- [x] Magento module with chat + document management
- [x] ACL (role-based access in Magento Admin)
- [x] Configurable API URL per environment

Features pending (in order of priority):
- [ ] API authentication (API key header) — `feature/api-authentication`
- [ ] LLMClientBase abstraction — `feature/llm-abstraction`
- [ ] Agent foundation (AgentBase, ToolBase, orchestrator) — `feature/agent-foundation`
- [ ] Docker Compose production setup (ChromaDB server mode) — `feature/production-docker`
- [ ] Structured logging — `feature/observability`
- [ ] Tenant isolation — `feature/multi-tenant`
- [ ] Magento agent (Tool Use for orders, catalog, cart) — `feature/magento-agent`

---

## What This Is Not (Yet)

- Not an agent — does not take actions (cancel orders, send emails)
- Not connected to live store data (orders, inventory, customers)
- Not multi-tenant — single collection, single client per deployment
- Not multi-LLM — Anthropic only
- `feature/agent-foundation` creates the structure, not the agent itself

These are Phase 2+ items. Do not design for them prematurely.
The abstraction layer exists precisely so these can be added without touching the core.
