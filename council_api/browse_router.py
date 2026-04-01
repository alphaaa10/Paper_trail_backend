from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from council_api.extraction import _next_groq_api_key as _next_groq_key

router = APIRouter(prefix="/browse", tags=["browse"])

ROOT_DIR = Path(__file__).resolve().parents[1]
FINAL_REPORT_PATH = ROOT_DIR / "data" / "reports" / "final_report.json"
EXTRACTED_DIR = ROOT_DIR / "data" / "extracted"
PDF_DIR = ROOT_DIR / "data" / "pdf"
METADATA_DIR = ROOT_DIR / "data" / "metadata"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
MAX_SNAPSHOT_CHARS = 220_000
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}

SESSIONS: dict[str, dict] = {}


class BrowseStartRequest(BaseModel):
    session_id: str | None = None


@router.post("/start")
async def browse_start(payload: BrowseStartRequest | None = None) -> dict:
    session_id = payload.session_id if payload and payload.session_id else str(uuid.uuid4())
    report = _load_final_report()
    papers = _load_extracted_papers()
    gaps = _extract_gaps(report)
    top_methods = _top_methods(papers)
    contradictions = _contradiction_titles(report, papers)
    fallback_reason = ""
    try:
        queries = await _generate_queries_with_groq(gaps, top_methods, contradictions)
        if not queries:
            fallback_reason = "Groq returned empty query list."
            queries = _fallback_queries(gaps, top_methods, contradictions)
    except Exception as exc:  # noqa: BLE001
        fallback_reason = f"Groq unavailable: {exc}"
        queries = _fallback_queries(gaps, top_methods, contradictions)

    SESSIONS[session_id] = _new_session_state(queries)
    if fallback_reason:
        SESSIONS[session_id]["log"].append(f"Using deterministic fallback queries. Reason: {fallback_reason}")
    return {"session_id": session_id, "queries": queries}


@router.post("/run/{session_id}")
async def browse_run(session_id: str) -> dict:
    session = _session_or_404(session_id)
    if session["status"] == "running":
        return {"session_id": session_id, "status": "running"}
    if session.get("worker_started"):
        return {"session_id": session_id, "status": session["status"]}
    session["status"] = "running"
    session["log"].append("Run started.")
    session["worker_started"] = True
    thread = threading.Thread(target=_run_browsing_sync, args=(session_id,), daemon=True)
    thread.start()
    return {"session_id": session_id, "status": "running"}


@router.get("/status/{session_id}")
def browse_status(session_id: str) -> dict:
    session = _session_or_404(session_id)
    return {
        "session_id": session_id,
        "status": session["status"],
        "current_url": session["current_url"],
        "raw_xml": session["raw_xml"],
        "rendered_html": session["rendered_html"],
        "screenshot_base64": session["screenshot_base64"],
        "rendered_html_base64": session["rendered_html_base64"],
        "papers_found": session["papers_found"],
        "log": session["log"],
        "queries_done": session["current_query_index"],
        "queries_total": len(session["queries"]),
    }


@router.get("/sessions")
def browse_sessions() -> dict:
    sessions = [
        {
            "session_id": session_id,
            "status": state["status"],
            "created_at": state["created_at"],
            "papers_found": len(state.get("papers_found", [])),
        }
        for session_id, state in SESSIONS.items()
    ]
    return {"sessions": sessions, "total": len(sessions)}


def _new_session_state(queries: list[str]) -> dict:
    waiting_html = _waiting_html()
    return {
        "queries": queries,
        "current_query_index": 0,
        "papers_found": [],
        "raw_xml": "",
        "rendered_html": waiting_html,
        "screenshot_base64": "",
        "rendered_html_base64": _to_base64(waiting_html),
        "current_url": "",
        "status": "idle",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "worker_started": False,
        "log": ["Session created."],
    }


def _session_or_404(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


def _load_final_report() -> dict:
    if not FINAL_REPORT_PATH.exists():
        return {}
    return json.loads(FINAL_REPORT_PATH.read_text(encoding="utf-8"))


def _load_extracted_papers() -> list[dict]:
    if not EXTRACTED_DIR.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(EXTRACTED_DIR.glob("*.json"))]


def _extract_gaps(report: dict) -> list[str]:
    gaps = report.get("gaps", [])
    if not gaps:
        gaps = report.get("which_question_has_been_circling_but_unanswered", [])
    return _clean_strings(gaps, limit=10, max_chars=240)


def _top_methods(papers: list[dict]) -> list[str]:
    counter: Counter[str] = Counter()
    for paper in papers:
        for method in paper.get("methods", []):
            text = " ".join(str(method).split())
            if text:
                counter[text] += 1
    return [item for item, _ in counter.most_common(8)]


def _contradiction_titles(report: dict, papers: list[dict]) -> list[str]:
    by_id = {str(item.get("paper_id", "")).strip(): str(item.get("title", "")).strip() for item in papers}
    items = report.get("contradicting_papers", [])
    if not items:
        items = report.get("contradicting_papers_defying_targeted_research", [])
    titles: list[str] = []
    for item in items:
        if isinstance(item, str):
            titles.append(item)
            continue
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id", "")).strip()
        title = str(item.get("title", "")).strip() or by_id.get(paper_id, "")
        if title:
            titles.append(title)
    return _clean_strings(titles, limit=12, max_chars=200)


def _clean_strings(values: list, limit: int, max_chars: int) -> list[str]:
    out: list[str] = []
    for value in values:
        text = " ".join(str(value).split())
        if not text or text in out:
            continue
        out.append(text[:max_chars])
        if len(out) >= limit:
            break
    return out


async def _generate_queries_with_groq(gaps: list[str], methods: list[str], contradictions: list[str]) -> list[str]:
    keys = _groq_keys_from_env()
    if not keys:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    prompt = _browse_prompt(gaps, methods, contradictions)
    body = {
        "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient(timeout=35.0) as client:
        response = await client.post(GROQ_URL, headers=_groq_headers(keys), json=body)
    if response.status_code >= 400:
        raise RuntimeError(f"Groq query generation failed (HTTP {response.status_code}).")
    queries = _parse_queries(response.json())
    if not queries:
        raise RuntimeError("Groq returned no queries.")
    return queries


def _groq_keys_from_env() -> list[str]:
    pooled = os.getenv("GROQ_API_KEY", "")
    keys = [item.strip() for item in pooled.split(",") if item.strip()]
    if not keys:
        return []
    unique: list[str] = []
    for key in keys:
        if key not in unique:
            unique.append(key)
    return unique


def _groq_headers(keys: list[str]) -> dict:
    return {
        "authorization": f"Bearer {_next_groq_key(keys)}",
        "content-type": "application/json",
    }


def _browse_prompt(gaps: list[str], methods: list[str], contradictions: list[str]) -> str:
    return (
        "You are a PhD researcher. Given these research gaps and contradictions, generate 4-6 "
        "precise academic search queries to find papers that resolve these gaps. "
        "Return JSON only: {queries: [string]}\n\n"
        f"gaps: {json.dumps(gaps, ensure_ascii=True)}\n"
        f"top_methods: {json.dumps(methods, ensure_ascii=True)}\n"
        f"contradicting_titles: {json.dumps(contradictions, ensure_ascii=True)}"
    )


def _parse_queries(payload: dict) -> list[str]:
    choices = payload.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return []
    message = choices[0].get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    parsed = json.loads(content) if isinstance(content, str) and content else {}
    values = parsed.get("queries", []) if isinstance(parsed, dict) else []
    return _clean_strings(values if isinstance(values, list) else [], limit=6, max_chars=220)


def _fallback_queries(gaps: list[str], methods: list[str], contradictions: list[str]) -> list[str]:
    candidates = [*gaps, *methods, *contradictions]
    queries: list[str] = []
    for item in candidates:
        tokens = _keyword_tokens(item)
        if not tokens:
            continue
        query = "all:" + " AND ".join(tokens[:5])
        if query not in queries:
            queries.append(query)
        if len(queries) >= 6:
            break
    if queries:
        return queries
    return [
        "all:research AND gap AND methodology",
        "all:systematic AND review AND contradiction",
        "all:benchmark AND reproducibility AND evaluation",
    ]


def _keyword_tokens(text: str) -> list[str]:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in str(text))
    parts = [part for part in cleaned.split() if len(part) > 2]
    out: list[str] = []
    for part in parts:
        if part not in out:
            out.append(part)
    return out


def _run_browsing_sync(session_id: str) -> None:
    try:
        asyncio.run(_run_browsing(session_id))
    except Exception as exc:  # noqa: BLE001
        session = SESSIONS.get(session_id)
        if session:
            session["status"] = "error"
            session["log"].append(f"Run crashed: {exc}")


async def _run_browsing(session_id: str) -> None:
    session = _session_or_404(session_id)
    timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=30.0)
    max_retries = 3
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for index, query in enumerate(session["queries"]):
                session["current_query_index"] = index
                session["log"].append(f"Calling arXiv API for query: {query}")
                session["current_url"] = _arxiv_request_url(query)

                # Retry loop with exponential backoff for 429 rate-limiting
                response = None
                for attempt in range(max_retries):
                    response = await client.get(
                        ARXIV_API_URL,
                        params={"search_query": query, "start": 0, "max_results": 10},
                    )
                    if response.status_code == 429:
                        wait = 3 * (2 ** attempt)  # 3s, 6s, 12s
                        session["log"].append(
                            f"Rate-limited (429), retrying in {wait}s (attempt {attempt + 1}/{max_retries})…"
                        )
                        await asyncio.sleep(wait)
                        continue
                    break  # got a non-429 response

                if response is None or response.status_code >= 400:
                    code = response.status_code if response else "N/A"
                    session["log"].append(f"arXiv API failure (HTTP {code}) after {max_retries} attempts")
                    session["current_query_index"] = index + 1
                    await asyncio.sleep(3)
                    continue

                raw_text = _trim_snapshot(response.text)
                parsed_xml = _sanitize_xml(raw_text)
                rendered_html = _render_arxiv_html(parsed_xml)

                session["raw_xml"] = raw_text
                session["rendered_html"] = rendered_html
                session["screenshot_base64"] = _to_base64(raw_text)
                session["rendered_html_base64"] = _to_base64(rendered_html)

                new_papers = _merge_papers(session["papers_found"], parsed_xml)
                added = len(new_papers) - len(session["papers_found"])
                session["papers_found"] = new_papers
                session["log"].append(f"Fetched XML and parsed {added} new paper(s) from query {index + 1}.")
                session["current_query_index"] = index + 1
                # Respectful delay between arXiv queries to avoid rate-limiting
                await asyncio.sleep(3)
        session["status"] = "done"
        session["log"].append(f"Run completed. Total papers: {len(session['papers_found'])}.")
    except Exception as exc:  # noqa: BLE001
        session["status"] = "error"
        session["log"].append(f"Run failed: {exc}")


def _arxiv_request_url(query: str) -> str:
    return f"{ARXIV_API_URL}?search_query={quote_plus(query)}&start=0&max_results=10"


_XML_DECL_RE = re.compile(r"<\?xml[^?]*\?>")


def _sanitize_xml(text: str) -> str:
    """Strip BOM and XML processing instruction that can break ET.fromstring."""
    cleaned = text.lstrip("\ufeff").strip()
    cleaned = _XML_DECL_RE.sub("", cleaned, count=1).lstrip()
    return cleaned


def _trim_snapshot(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) <= MAX_SNAPSHOT_CHARS:
        return cleaned
    return f"{cleaned[:MAX_SNAPSHOT_CHARS]}\n\n<!-- trimmed -->"


def _to_base64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8", errors="ignore")).decode("ascii")


def _waiting_html() -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Waiting</title>"
        "<style>body{font-family:Arial,sans-serif;background:#f7fafc;color:#1f2937;padding:24px;}"
        ".box{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:18px;max-width:780px;}"
        "</style></head><body><div class='box'><h2>Waiting for arXiv response...</h2>"
        "<p>Start or resume a session to load rendered result cards.</p></div></body></html>"
    )


def _render_arxiv_html(xml_text: str) -> str:
    parsed = _parse_arxiv_feed(xml_text)
    if not parsed:
        return _render_raw_fallback_html(xml_text, "Could not parse arXiv feed XML.")

    entries_html = "".join(_entry_card_html(item) for item in parsed["entries"])
    meta = parsed["meta"]
    query_text = _friendly_query_text(meta["title"])
    header_title = "arXiv Results"
    if not query_text:
        header_title = meta["title"]

    query_block = ""
    if query_text:
        query_block = (
            "<div class='query'><span class='query-label'>Search query</span>"
            f"{escape(query_text)}</div>"
        )

    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>arXiv Feed Preview</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;background:#f3f6fb;color:#0f172a;margin:0;padding:20px;}"
        ".wrap{max-width:1040px;margin:0 auto;}"
        ".header{background:#ffffff;border:1px solid #d7dfeb;border-radius:14px;padding:16px 18px;margin-bottom:14px;box-shadow:0 6px 20px rgba(15,23,42,0.04);}"
        ".header h1{margin:0 0 10px 0;font-size:22px;line-height:1.3;}"
        ".query{font-size:14px;color:#1f2937;background:#f8fafc;border:1px solid #dbe3ef;border-radius:10px;padding:10px 12px;margin-bottom:12px;}"
        ".query-label{display:inline-block;font-size:11px;letter-spacing:.04em;text-transform:uppercase;background:#dbeafe;color:#1d4ed8;border-radius:999px;padding:2px 8px;margin-right:8px;}"
        ".meta{font-size:13px;color:#475569;display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;}"
        ".results{display:grid;gap:12px;}"
        ".card{background:#ffffff;border:1px solid #d7dfeb;border-radius:12px;padding:14px;box-shadow:0 4px 16px rgba(15,23,42,0.03);}"
        ".title{margin:0 0 8px 0;font-size:28px;line-height:1.35;color:#0f172a;}"
        ".meta-line{font-size:13px;color:#556274;margin-bottom:8px;}"
        ".summary{font-size:13px;line-height:1.55;white-space:pre-wrap;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px;max-height:170px;overflow:auto;}"
        ".links{font-size:13px;margin-top:10px;}"
        "a{color:#0b63ce;text-decoration:none;font-weight:600;}a:hover{text-decoration:underline;}</style>"
        "</head><body><main class='wrap'>"
        f"<section class='header'><h1>{escape(header_title)}</h1>{query_block}"
        f"<div class='meta'><div><strong>Updated:</strong> {escape(meta['updated'])}</div>"
        f"<div><strong>Total:</strong> {meta['total_results']}</div>"
        f"<div><strong>Start:</strong> {meta['start_index']}</div>"
        f"<div><strong>Items/page:</strong> {meta['items_per_page']}</div></div></section>"
        f"<section class='results'>{entries_html or '<div class=\'card\'>No entries for this query.</div>'}</section>"
        "</main></body></html>"
    )


def _parse_arxiv_feed(xml_text: str) -> dict | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    def text_or_empty(node: ET.Element | None) -> str:
        if node is None or node.text is None:
            return ""
        return " ".join(node.text.split())

    entries: list[dict] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        authors = [text_or_empty(author.find("atom:name", ATOM_NS)) for author in entry.findall("atom:author", ATOM_NS)]
        links = entry.findall("atom:link", ATOM_NS)
        abs_link = ""
        pdf_link = ""
        for link in links:
            href = (link.attrib.get("href") or "").strip()
            rel = (link.attrib.get("rel") or "").strip()
            title = (link.attrib.get("title") or "").strip().lower()
            if not href:
                continue
            if rel == "alternate" and not abs_link:
                abs_link = href
            if title == "pdf" or "/pdf/" in href:
                pdf_link = href
        entries.append(
            {
                "id": text_or_empty(entry.find("atom:id", ATOM_NS)),
                "title": text_or_empty(entry.find("atom:title", ATOM_NS)),
                "published": text_or_empty(entry.find("atom:published", ATOM_NS)),
                "updated": text_or_empty(entry.find("atom:updated", ATOM_NS)),
                "authors": [author for author in authors if author],
                "summary": text_or_empty(entry.find("atom:summary", ATOM_NS)),
                "abs_link": abs_link,
                "pdf_link": pdf_link,
            }
        )

    meta = {
        "title": text_or_empty(root.find("atom:title", ATOM_NS)) or "arXiv Search Feed",
        "updated": text_or_empty(root.find("atom:updated", ATOM_NS)),
        "total_results": _safe_int(text_or_empty(root.find("opensearch:totalResults", ATOM_NS))),
        "start_index": _safe_int(text_or_empty(root.find("opensearch:startIndex", ATOM_NS))),
        "items_per_page": _safe_int(text_or_empty(root.find("opensearch:itemsPerPage", ATOM_NS))),
    }
    return {"meta": meta, "entries": entries}


def _entry_card_html(item: dict) -> str:
    links: list[str] = []
    if item.get("abs_link"):
        links.append(f"<a href='{escape(item['abs_link'])}' target='_blank' rel='noreferrer'>Abstract</a>")
    if item.get("pdf_link"):
        links.append(f"<a href='{escape(item['pdf_link'])}' target='_blank' rel='noreferrer'>PDF</a>")
    links_html = " | ".join(links) if links else "No links"
    return (
        "<article class='card'>"
        f"<h3 class='title'>{escape(item.get('title', 'Untitled'))}</h3>"
        f"<div class='meta-line'><strong>Published:</strong> {escape(item.get('published', ''))} "
        f"<strong>Updated:</strong> {escape(item.get('updated', ''))}</div>"
        f"<div class='meta-line'><strong>Authors:</strong> {escape(', '.join(item.get('authors', [])) or 'Unknown')}</div>"
        f"<div class='summary'>{escape(item.get('summary', ''))}</div>"
        f"<div class='links'>{links_html}</div>"
        "</article>"
    )


def _friendly_query_text(feed_title: str) -> str:
    match = re.search(r"search_query=([^&]+)", feed_title)
    if not match:
        return ""
    query = " ".join(match.group(1).replace("+", " ").split())
    query = re.sub(r"\ball:", "", query, flags=re.IGNORECASE)
    return " ".join(query.split())


def _render_raw_fallback_html(raw_text: str, message: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Raw Feed</title>"
        "<style>body{font-family:Arial,sans-serif;background:#f8fafc;color:#111827;padding:20px;}"
        "pre{background:#fff;border:1px solid #d1d5db;border-radius:10px;padding:12px;white-space:pre-wrap;}"
        "p{color:#92400e;background:#fef3c7;padding:8px;border-radius:8px;border:1px solid #f59e0b;}</style>"
        "</head><body>"
        f"<p>{escape(message)}</p><pre>{escape(raw_text)}</pre></body></html>"
    )


def _safe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _merge_papers(existing: list[dict], xml_text: str) -> list[dict]:
    parsed = _parse_arxiv_feed(xml_text)
    if not parsed:
        return existing
    known_ids = {item.get("paper_id", "") for item in existing}
    merged = list(existing)
    for entry in parsed["entries"]:
        paper_id = _paper_id_from_arxiv_id(entry.get("id", ""))
        if not paper_id or paper_id in known_ids:
            continue
        merged.append(
            {
                "paper_id": paper_id,
                "title": entry.get("title", "Untitled"),
                "url": entry.get("abs_link", ""),
            }
        )
        known_ids.add(paper_id)
    return merged


def _paper_id_from_arxiv_id(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if "/abs/" in text:
        return text.rsplit("/abs/", maxsplit=1)[-1]
    return text.rsplit("/", maxsplit=1)[-1]
