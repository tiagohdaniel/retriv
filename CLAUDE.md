# retriv — Project Context

## O que é

retriv é uma API RAG (Retrieval-Augmented Generation) domain-agnóstica.
Recebe documentos via `POST /index`, armazena como vetores no ChromaDB, e responde perguntas em linguagem natural via `POST /ask` (síncrono) ou `POST /ask/stream` (SSE).

É um produto — não um script. Projetado para ser instalado em qualquer ambiente de cliente sem modificação no core.

---

## Arquitetura

```
Dashboards / Browsers          Agents externos (AGNO, Claude, LangGraph...)
        │                                      │
        │  X-API-Key → tenant_id               │  Bearer <api-key> → tenant_id
        ▼                                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI (main.py)                        │
│                                                             │
│  REST API                          MCP Server               │
│  /index  /ask  /ask/stream         /mcp  (Streamable HTTP)  │
│  /sources  /analyze                5 tools expostos         │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │   Services      │
        │  IndexService   │  chunk → embed → store
        │  AskService     │  embed → search → prompt → LLM → trace
        └────────┬────────┘
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
ChromaDB    Anthropic    Langfuse
(vetores)   (LLM+eval)  (observability)
```

**Regra principal: o core nunca conhece infraestrutura.**
Toda dependência de infra é injetada via `dependencies.py`. Nunca modifique `ask_service.py` ou `index_service.py` para acomodar infra — crie um novo adapter em `app/core/backends/`.

---

## MCP Server (`/mcp`)

O retriv expõe um MCP server em `/mcp` usando o transporte Streamable HTTP (padrão MCP spec 2025-03-26). Qualquer agent MCP-compatível (AGNO, Claude Desktop, LangGraph, etc.) se conecta sem precisar escrever um wrapper HTTP customizado.

**Auth:** mesmo mecanismo da REST API — `Authorization: Bearer <api-key>` ou `X-API-Key: <chave>`. Multi-tenant preservado: cada key resolve um `tenant_id` diferente.

**Tools expostas (`app/mcp/server.py`):**

| Tool | O que faz |
|---|---|
| `ask` | Pipeline RAG completo: retrieve → LLM → resposta com fontes |
| `search` | Retrieval puro sem LLM — para agents que fazem seu próprio prompting |
| `index_document` | Indexa conteúdo texto na knowledge base |
| `list_sources` | Lista fontes indexadas do tenant |
| `delete_source` | Remove uma fonte do índice |

**Padrão de integração nos `*-agent`:**
```python
# accounting-agent, ecommerce-agent, insurance-agent...
from agno.tools.mcp import MCPTools

retriv_tools = MCPTools(
    url="https://retriv-production.up.railway.app/mcp",
    headers={"Authorization": "Bearer chave-do-tenant"},
)
agent = Agent(model=Claude(...), tools=[retriv_tools, DomainSpecificTool()])
```
Zero código de integração manual. Sem `RetrivTool` customizada em cada agent.

---

## Multi-tenancy

Um único deployment do retriv serve múltiplos clientes com isolamento total.

**Como funciona:**
- Cada cliente tem uma API key única
- A key é mapeada para um `tenant_id` via env var `API_KEYS`
- Todos os chunks indexados carregam `tenant_id` nos metadados do ChromaDB
- Todas as buscas filtram por `tenant_id` automaticamente

**Configuração:**
```env
API_AUTH_ENABLED=true
API_KEYS=abc123:guerreiro,xyz789:porto,def456:outro-cliente
```

**Fluxo:**
```
X-API-Key: abc123
    → verify_api_key() → tenant_id = "guerreiro"
    → IndexService.index(request, tenant_id="guerreiro")
    → chunk IDs: "guerreiro__apolice-vida__chunk_0"
    → metadata: { tenant_id: "guerreiro", source_id: "apolice-vida" }

X-API-Key: xyz789
    → tenant_id = "porto"
    → busca filtrada: WHERE tenant_id = "porto"
    → nunca vê dados do tenant "guerreiro"
```

**Modo single-tenant (compatibilidade):**
- `API_AUTH_ENABLED=false` → `tenant_id=None` → sem filtragem
- `API_KEY=chave` único → `tenant_id="default"`

---

## Observabilidade (Langfuse)

Avalia qualidade do RAG automaticamente após cada query (streaming e síncrono).

**Métricas geradas:**
- `faithfulness` (0-1): a resposta se baseia apenas no contexto?
- `answer_relevancy` (0-1): a resposta endereça a pergunta?

**Como funciona:**
- Modelo leve (`EVAL_MODEL`, default: `claude-haiku`) julga a resposta
- Trace enviado ao Langfuse via `langfuse.flush()` antes do coroutine terminar
- Desabilitado por padrão (`EVAL_ENABLED=false`)

---

## Variáveis de Ambiente

| Variável | Default | Descrição |
|----------|---------|-----------|
| `ANTHROPIC_API_KEY` | — | Obrigatório |
| `MODEL_NAME` | `claude-sonnet-4-20250514` | Modelo LLM principal |
| `LLM_BACKEND` | `anthropic` | Provider LLM |
| `LLM_TIMEOUT` | `30.0` | Timeout requests LLM (segundos) |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | ⚠️ Trocar exige reindexação total |
| `CHROMA_MODE` | `embedded` | `embedded` / `server` / `cloud` |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Path para modo embedded |
| `CHROMA_HOST` | `localhost` | Host ChromaDB (modo server) |
| `CHROMA_PORT` | `8000` | Porta ChromaDB (modo server) |
| `CHROMA_CLOUD_API_KEY` | — | Chroma Cloud |
| `CHROMA_CLOUD_TENANT` | — | Chroma Cloud |
| `CHROMA_CLOUD_DATABASE` | — | Chroma Cloud |
| `CHROMA_COLLECTION` | `documents` | Nome da coleção |
| `CHUNKING_STRATEGY` | `semantic` | `semantic` (por parágrafo/frase) / `fixed` (N chars fixos) |
| `HYBRID_ENABLED` | `false` | Habilita busca híbrida BM25 + semântica (RRF) |
| `HYBRID_BM25_CORPUS_LIMIT` | `500` | Máx. chunks carregados no corpus BM25 |
| `RERANKER_ENABLED` | `false` | Habilita cross-encoder reranker pós-retrieval |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Modelo de reranking (multilingual) |
| `RERANKER_TOP_K_FETCH` | `15` | Candidatos buscados no vector store antes do reranking |
| `RERANKER_TOP_N` | `5` | Chunks mantidos após reranking (enviados ao LLM) |
| `CHUNK_SIZE` | `800` | Limite de chars por chunk (semântico usa como soft limit) |
| `CHUNK_OVERLAP` | `100` | Sobreposição entre chunks |
| `API_AUTH_ENABLED` | `false` | Habilita autenticação por API key |
| `API_KEY` | — | Chave única (single-tenant) |
| `API_KEYS` | — | Mapa `key:tenant_id,key:tenant_id` (multi-tenant) |
| `RATE_LIMIT_ENABLED` | `false` | Habilita rate limiting |
| `RATE_LIMIT_INDEX` | `10/minute` | Limite POST /index |
| `RATE_LIMIT_ASK` | `30/minute` | Limite POST /ask e /ask/stream |
| `RATE_LIMIT_SOURCES` | `60/minute` | Limite GET/DELETE /sources |
| `LOG_LEVEL` | `INFO` | Nível de log |
| `LOG_FORMAT` | `json` | `json` (prod) / `console` (dev) |
| `CORS_ORIGINS` | `*` | Origins permitidas |
| `WEB_CONCURRENCY` | `2` | Workers Gunicorn |
| `METRICS_ENABLED` | `true` | Expõe GET /metrics (Prometheus) |
| `METRICS_USERNAME` | — | Basic auth para /metrics |
| `METRICS_PASSWORD` | — | Basic auth para /metrics |
| `EVAL_ENABLED` | `false` | Habilita avaliação RAG via Langfuse |
| `EVAL_MODEL` | `claude-haiku-4-5-20251001` | Modelo leve para LLM-as-judge |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | — | Langfuse secret key |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | US: `https://us.cloud.langfuse.com` |

---

## Estrutura de Arquivos

```
app/
├── api/
│   ├── routes_ask.py           # POST /ask, POST /ask/stream
│   ├── routes_index.py         # POST /index
│   ├── routes_sources.py       # GET /sources, DELETE /sources/{id}
│   └── routes_analyze.py       # POST /analyze
├── mcp/
│   └── server.py               # MCP server — 5 tools para agents externos
├── core/
│   ├── auth.py                 # verify_api_key → retorna tenant_id | None
│   ├── vector_store.py         # VectorStoreBase (contrato — não modificar)
│   ├── llm_client.py           # LLMClientBase (contrato — não modificar)
│   ├── observability.py        # ObservabilityBase (contrato — não modificar)
│   ├── data_analyzer.py        # DataAnalyzerBase (contrato — não modificar)
│   ├── chunker.py              # SemanticChunker + TextChunker
│   ├── embeddings.py           # create_embedding_service (fastembed/ONNX)
│   ├── hybrid_search.py        # BM25Searcher + RRF merge
│   ├── reranker.py             # FastEmbedReranker + NullReranker
│   ├── rate_limit.py           # SlowAPI
│   ├── logging_config.py       # structlog JSON
│   ├── metrics.py              # Prometheus counters
│   └── backends/
│       ├── chroma.py                    # ChromaVectorStore — ativo
│       ├── anthropic.py                 # AnthropicClient — ativo
│       ├── pandas_analyzer.py           # PandasAnalyzer — analytics pipeline
│       ├── langfuse_observability.py    # LangfuseObservability — ativo se EVAL_ENABLED=true
│       └── null_observability.py        # NullObservability — fallback (no-op)
├── services/
│   ├── ask_service.py          # RAG: embed → search → prompt → LLM → trace
│   ├── index_service.py        # chunk → embed → upsert no vector store
│   └── analytics_service.py    # analytics: decode → pandas → LLM format
├── schemas/
│   └── models.py               # Contratos públicos da API (Pydantic) — apenas additive
├── dependencies.py             # Wiring: instancia e injeta todos os serviços via FastAPI Depends
├── settings.py                 # Todas as configs via env vars (pydantic-settings)
└── main.py                     # App FastAPI + middlewares + routers + /mcp mount
```

---

## Roadmap do core

### v1.1 — Qualidade RAG
- Chunking semântico (por parágrafo/seção, não N caracteres fixos)
- Reranker (cross-encoder reordena chunks após retrieval)
- Híbrido BM25 + semântico (rejeita perguntas sem relação com os docs)

### v1.2 — Analytics pipeline — ENTREGUE (v1.2.0)
- `POST /analyze` — CSV/XLSX/JSON + pergunta em linguagem natural → resposta formatada
- Computação determinística com pandas, LLM só formata o resultado
- Inferência de intenção por heurística: sum, mean, max, min, count, list, describe (PT + EN)
- Agrupamento automático via "por X" / "by X"
- `DataAnalyzerBase` port + `PandasAnalyzer` adapter — hexagonal intocado
- Os **conectores** com fontes específicas do cliente ficam no `*-agent`, não aqui

### v1.3 — MCP Server — ENTREGUE
- `GET /mcp` — endpoint MCP Streamable HTTP (padrão MCP spec 2025-03-26)
- 5 tools: `ask`, `search`, `index_document`, `list_sources`, `delete_source`
- Qualquer agent MCP-compatível (AGNO, Claude Desktop, LangGraph) conecta sem wrapper customizado
- Auth reutiliza `API_KEYS` — multi-tenant preservado
- Removido: `AgentOrchestrator`, `ToolBase`, `AgentService` — orchestração pertence aos `*-agent`

---

## O que NÃO pertence ao retriv

**Tools de domínio específico nunca entram aqui.**

O retriv é o core genérico — knowledge engine com MCP. Quando um cliente precisar de agents com ações específicas (abrir sinistro, rastrear pedido, protocolar documento), isso vai em um repositório separado que usa AGNO como framework e se conecta ao retriv via MCP.

**Convenção de nomes:** o repositório se chama pelo **domínio** em inglês, nunca pelo nome da empresa.

```
# ERRADO — nunca fazer isso:
app/agents/guerreiro/tools/abrir_sinistro.py  ← lógica de negócio dentro do retriv

# ERRADO — nome da empresa, não do domínio:
repositório: guerreiro-agent/

# CERTO — nome do domínio + AGNO como framework:
repositório: insurance-agent/
    requirements.txt → agno, anthropic, httpx
    tools/abrir_sinistro.py   ← domain logic
    agent.py                  ← AGNO Agent com MCPTools apontando para /mcp

repositório: accounting-agent/
    requirements.txt → agno, anthropic, httpx
    tools/simulador_fiscal.py
    agent.py
```

Agents verticais planejados (servem múltiplos clientes do mesmo setor):
- `insurance-agent` — seguradoras (1º cliente: Guerreiro)
- `accounting-agent` — escritórios contábeis (1º cliente: Maranata)
- `ecommerce-agent` — e-commerce (consome também a REST API do retriv-magento2)
- `legal-agent` — escritórios jurídicos

**Nada de `AgentOrchestrator`, `ToolBase` ou `AgentService` dentro do retriv** — esses arquivos foram removidos. A orchestração de agents acontece nos repositórios `*-agent`, não no core. O MCP server em `app/mcp/server.py` é o único ponto de extensão para agents.

---

## Como Adicionar

### Novo tenant/cliente
1. Gerar uma API key
2. Adicionar `nova_key:tenant_id` em `API_KEYS` no `.env`
3. Zero código

### Novo vector store
1. `app/core/backends/<nome>.py` implementando `VectorStoreBase`
2. `elif` em `dependencies.get_vector_store()`
3. Core e services: intocados

### Novo LLM provider
1. `app/core/backends/<provider>.py` implementando `LLMClientBase`
2. `elif settings.llm_backend == "<provider>"` em `dependencies.get_llm_client()`

### Novo domínio (jurídico, saúde, RH)
1. Indexar documentos do domínio via `POST /index`
2. Zero código — o pipeline é domain-agnostic

---

## Gitflow

```
main      → produção — só recebe merge de develop ou hotfix/*
develop   → integração — todas as features chegam aqui primeiro
feat/*    → nova feature, saindo de develop
fix/*     → correção de bug, saindo de develop
hotfix/*  → correção urgente em prod, sai de main, mergeia em main + develop
```

**Regras:**
- Nunca commitar direto em `main` ou `develop`
- Todo código passa por PR
- Tags em `main` seguem semver: `v1.0.0`, `v1.1.0`

---

## Padrão de Commits

```
tipo: descrição curta no imperativo
```

Tipos: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`

**Sem linhas de co-autoria. Sem referências a Claude/AI.**

---

## Deploy

**Produção atual:**
- retriv API → Railway (Docker), região US East, porta 8001
- ChromaDB → Chroma Cloud (CHROMA_MODE=cloud)

**Local/staging:**
```bash
docker compose up --build -d
```
Sobe: `retriv-api` (porta 8001) + `chromadb` (porta 8000).

**Healthcheck ChromaDB:** usa `grep -q ':1F40 ' /proc/net/tcp` — a imagem `chromadb/chroma:1.0.0` não tem `curl` nem `httpx`.

---

## System Prompt

O LLM é restrito ao conteúdo indexado:
1. Responde apenas com informação dos documentos
2. Sem contexto suficiente: _"Não encontrei informações sobre isso na base de conhecimento."_
3. Fora do escopo: _"Só consigo responder perguntas relacionadas aos documentos indexados."_
4. Responde no mesmo idioma da pergunta

`max_distance` padrão: `0.45` — chunks com distância coseno acima desse valor são descartados antes de chegar no LLM.
