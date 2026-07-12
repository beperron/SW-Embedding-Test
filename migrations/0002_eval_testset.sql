-- Migration 0002: evaluation test collection and results tables.

create table if not exists public.eval_queries (
  query_id   text primary key,
  subset     text not null,            -- synthetic | keyword | known_item
  query_text text not null,
  seed_paper_id int references public.papers(id),
  source     text,                     -- generator model
  created_at timestamptz not null default now()
);

create table if not exists public.eval_qrels (
  query_id   text not null references public.eval_queries(query_id) on delete cascade,
  paper_id   int  not null references public.papers(id) on delete cascade,
  relevance  int  not null,            -- 0..3 graded; known_item uses 3
  judge      text not null,            -- 'llm' | 'human' | 'known_item' | 'claude-swarm' | 'seed'
  judge_model text,
  created_at timestamptz not null default now(),
  primary key (query_id, paper_id, judge)
);

create table if not exists public.eval_metrics (
  run_id   text not null,
  system   text not null,              -- retriever[/reranker@depth]
  metric   text not null,
  value    double precision not null,
  ci_low   double precision,
  ci_high  double precision,
  judge_source text not null default 'llm',
  created_at timestamptz not null default now(),
  primary key (run_id, metric, judge_source)
);

alter table public.eval_queries enable row level security;
alter table public.eval_qrels   enable row level security;
alter table public.eval_metrics enable row level security;
