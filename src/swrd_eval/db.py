"""Supabase access via PostgREST (anon, read-only) for the corpus.

Writes (final metrics) are small and pushed out-of-band via the MCP tool / direct SQL;
this module is read-only by design (local-first storage per project decision).
"""
from __future__ import annotations
import time
import httpx
from . import config


def _headers() -> dict:
    key = config.SUPABASE_ANON_KEY
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def _rest(path: str, params: dict, rng: tuple[int, int] | None = None, count: bool = False):
    headers = dict(_headers())
    if rng is not None:
        headers["Range"] = f"{rng[0]}-{rng[1]}"
    if count:
        headers["Prefer"] = "count=exact"
    url = f"{config.SUPABASE_URL}/rest/v1/{path}"
    for attempt in range(5):
        try:
            r = httpx.get(url, headers=headers, params=params, timeout=120)
            if r.status_code in (200, 206):
                cr = r.headers.get("content-range")
                total = int(cr.split("/")[-1]) if cr and "/" in cr and cr.split("/")[-1] != "*" else None
                return r.json(), total
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
        except httpx.RequestError:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"PostgREST request failed after retries: {path} {params}")


def count_papers(abstract_only: bool = True) -> int:
    params = {"select": "id"}
    if abstract_only:
        params["abstract"] = "not.is.null"
    _, total = _rest("papers", params, rng=(0, 0), count=True)
    return total or 0


def iter_papers(abstract_only: bool = True, page: int = 1000):
    """Yield pages of paper rows (id, title, abstract, publication_year, document_type, data_source, journal_id)."""
    params = {
        "select": "id,title,abstract,publication_year,document_type,data_source,journal_id",
        "order": "id.asc",
    }
    if abstract_only:
        params["abstract"] = "not.is.null"
    offset = 0
    while True:
        rows, total = _rest("papers", params, rng=(offset, offset + page - 1))
        if not rows:
            break
        yield rows
        offset += len(rows)
        if total is not None and offset >= total:
            break
        if len(rows) < page:
            break
