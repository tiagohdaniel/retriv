# retriv dashboard — plano de implementação

Interface web para o retriv. Usuários fazem login, admins indexam documentos, consumers usam o chat.

---

## Arquitetura

```
┌─────────────────────┐        ┌──────────────────────┐
│   retriv-dashboard  │──────▶│      retriv API       │
│   Next.js (Railway) │  HTTP  │   (Railway — atual)   │
└────────┬────────────┘        └──────────────────────┘
         │
         ▼
┌─────────────────────┐
│      Supabase       │
│  Auth + PostgreSQL  │
│  (usuários + roles) │
└─────────────────────┘
```

**Por que projetos separados:**
- retriv continua sendo uma API pura — não sabe nada de usuário ou frontend
- o dashboard é um cliente como qualquer outro (consome a API via API Key guardada no servidor)
- podem evoluir, escalar e ser deployados independentemente

**Segurança:** a API Key do retriv fica apenas nas variáveis de ambiente do Next.js (server-side). O browser nunca a vê. O Supabase cuida de quem pode acessar o quê.

---

## Roles

| Role | Acesso |
|---|---|
| `admin` | Chat + indexar documentos + listar/deletar fontes |
| `consumer` | Somente chat |

---

## Stack

| Camada | Tecnologia | Por quê |
|---|---|---|
| Frontend | Next.js 14 (App Router) | Padrão atual, SSR, API routes server-side |
| Auth + usuários | Supabase | Auth pronto, PostgreSQL, JWT, free tier |
| Deploy | Railway (novo serviço) | Já estamos lá, fácil de conectar |

---

## Fases

### Fase 1 — Setup do projeto
- Criar projeto Next.js (`npx create-next-app retriv-dashboard`)
- Criar projeto no Supabase
- Configurar Supabase Auth (email/senha)
- Criar tabela `profiles` no Supabase com campo `role` (admin | consumer)
- Trigger automático: ao criar usuário no Auth, insere registro em `profiles`

### Fase 2 — Autenticação
- Página de login (`/login`) — email + senha via Supabase
- Middleware Next.js: rotas protegidas redirecionam para `/login` se não autenticado
- Após login: admin vai para `/admin`, consumer vai para `/chat`
- Página de logout

### Fase 3 — Chat (consumer)
- Interface de chat em `/chat`
- Campo de texto + botão enviar
- Chama `POST /ask` do retriv via Next.js API route (a chave fica no servidor)
- Exibe resposta em markdown + fontes com relevance score
- Histórico de conversa na sessão (estado local)

### Fase 4 — Painel admin
- `/admin` — visão geral: total de fontes indexadas
- `/admin/index` — formulário para indexar documento (source_id, título, conteúdo)
- `/admin/sources` — lista de fontes com opção de deletar
- Admin também tem acesso ao `/chat`

### Fase 5 — Deploy no Railway
- Criar novo serviço no Railway no mesmo projeto (`alluring-benevolence`)
- Configurar variáveis de ambiente:
  - `RETRIV_API_URL` — URL do Railway do retriv
  - `RETRIV_API_KEY` — API Key do retriv (quando ativarmos auth)
  - `NEXT_PUBLIC_SUPABASE_URL` — URL do projeto Supabase
  - `NEXT_PUBLIC_SUPABASE_ANON_KEY` — chave pública do Supabase
  - `SUPABASE_SERVICE_ROLE_KEY` — chave privada (só server-side)
- Gerar domínio público no Railway

---

## O que o retriv precisa antes do dashboard ir a ar

- [ ] `API_AUTH_ENABLED=true` no Railway (hoje está desabilitado — qualquer um acessa)
- [ ] Gerar `API_KEY` forte e configurar no Railway
- [ ] Mergear `fix/metrics-auth` (proteção do `/metrics`)

---

## Repositório

Projeto separado: `retriv-dashboard` (novo repositório GitHub)

O retriv não muda — continua exatamente como está.
