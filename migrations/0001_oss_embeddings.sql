-- Migration 0001: open-weight embedding store (+ optional lexical fts column)
-- pgvector 0.8.0; paper_id is INTEGER in SWRD.

-- Long-format store for open-model document embeddings (one row per paper x model).
create table if not exists public.oss_paper_embeddings (
  paper_id   int    not null references public.papers(id) on delete cascade,
  model      text   not null,
  dim        int    not null,
  input_kind text   not null default 'title_abstract',
  embedding  vector not null,
  created_at timestamptz not null default now(),
  primary key (paper_id, model)
);
create index if not exists oss_emb_model_idx on public.oss_paper_embeddings (model);
alter table public.oss_paper_embeddings enable row level security;
-- No policy: service-role only. The pipeline writes with a service key.

-- Optional lexical baseline: a generated full-text-search column on papers.
-- NOTE: a stored generated column rewrites all rows and can exceed short statement
-- timeouts on large tables; apply via a direct psql connection rather than a pooled
-- MCP/HTTP session. The bundled pipeline uses bm25s for the lexical baseline instead,
-- so this block is optional.
-- alter table public.papers
--   add column if not exists fts tsvector
--   generated always as (to_tsvector('english', title || ' ' || coalesce(abstract,''))) stored;
-- create index if not exists papers_fts_idx on public.papers using gin (fts);
