from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from council_api.extraction import _next_groq_api_key as _next_groq_key
from research_crawler.models import PaperRecord
from research_crawler.utils import build_paper_id

router = APIRouter(prefix="/browse", tags=["browse"])

ROOT_DIR = Path(__file__).resolve().parents[1]
FINAL_REPORT_PATH = ROOT_DIR / "data" / "reports" / "final_report.json"
EXTRACTED_DIR = ROOT_DIR / "data" / "extracted"
PDF_DIR = ROOT_DIR / "data" / "pdf"
METADATA_DIR = ROOT_DIR / "data" / "metadata"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

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
    queries = await _generate_queries_with_groq(gaps, top_methods, contradictions)
    SESSIONS[session_id] = _new_session_state(queries)
    return {"session_id": session_id, "queries": queries}


@router.post("/run/{session_id}")
async def browse_run(session_id: str) -> dict:
    session = _session_or_404(session_id)
    if session["status"] == "running":
        return {"session_id": session_id, "status": "running"}
    session["status"] = "running"
    session["log"].append("Run started.")
    asyncio.create_task(_run_browsing(session_id))
    return {"session_id": session_id, "status": "running"}


@router.get("/status/{session_id}")
def browse_status(session_id: str) -> dict:
    session = _session_or_404(session_id)
    return {
        "status": session["status"],
        "current_url": session["current_url"],
        "screenshot_base64": session["screenshot_base64"],
        "papers_found": session["papers_found"],
        "log": session["log"],
        "queries_done": session["current_query_index"],
        "queries_total": len(session["queries"]),
    }


@router.get("/sessions")
def browse_sessions() -> list[dict]:
    return [
        {"session_id": session_id, "status": state["status"]}
        for session_id, state in SESSIONS.items()
    ]


def _new_session_state(queries: list[str]) -> dict:
    return {
        "queries": queries,
        "current_query_index": 0,
        "papers_found": [],
        "screenshot_base64": "",
        "current_url": "",
        "status": "idle",
        "log": ["Session created."],
    }


def _session_or_404(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


def _load_final_report() -> dict:
    if not FINAL_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="final_report.json not found.")
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
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")

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
        raise HTTPException(status_code=502, detail="Groq query generation failed.")
    queries = _parse_queries(response.json())
    if not queries:
        raise HTTPException(status_code=502, detail="Groq returned no queries.")
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


async def _run_browsing(session_id: str) -> None:
    session = _session_or_404(session_id)
    try:
        timeout = httpx.Timeout(connect=10.0, read=25.0, write=25.0, pool=25.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            await _run_query_loop(client, session)
        session["status"] = "done"
        session["log"].append("Run completed.")
    except Exception as exc:  # noqa: BLE001
        session["status"] = "error"
        session["log"].append(f"Run crashed: {exc}")
    finally:
        session["screenshot_base64"] = ""


async def _run_query_loop(client: httpx.AsyncClient, session: dict) -> None:
    max_papers = max(1, len(session.get("queries", [])) * 3)
    for query in session["queries"]:
        await _discover_papers(client, session, query)
        session["current_query_index"] += 1
    await _process_papers(client, session, max_papers=max_papers)


async def _discover_papers(client: httpx.AsyncClient, session: dict, query: str) -> None:
    search_url = f"https://arxiv.org/search/?searchtype=all&query={quote_plus(query)}"
    session["log"].append(f"Searching arxiv: {query}")
    session["current_url"] = search_url
    session["screenshot_base64"] = ""
    response = await client.get(search_url)
    if response.status_code >= 400:
        session["log"].append(f"Search failed: HTTP {response.status_code}")
        return
    for abs_url in _collect_abs_links(response.text):
        if _has_abs_url(session["papers_found"], abs_url):
            continue
        session["papers_found"].append({"url": abs_url, "status": "discovered"})


def _collect_abs_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for node in soup.select("a[href*='/abs/']"):
        href = " ".join(str(node.get("href", "")).split())
        if not href:
            continue
        url = href if href.startswith("http") else f"https://arxiv.org{href}"
        if url not in links:
            links.append(url)
    return links


def _has_abs_url(found: list[dict], abs_url: str) -> bool:
    for item in found:
        if " ".join(str(item.get("url", "")).split()) == abs_url:
            return True
    return False


async def _process_papers(client: httpx.AsyncClient, session: dict, max_papers: int) -> None:
    candidates = [item for item in session["papers_found"] if item.get("status") == "discovered"]
    for row in candidates[:max_papers]:
        abs_url = " ".join(str(row.get("url", "")).split())
        if not abs_url:
            continue
        session["current_url"] = abs_url
        await _process_single_paper(client, session, row, abs_url)
        await asyncio.sleep(1.0)


async def _process_single_paper(client: httpx.AsyncClient, session: dict, row: dict, abs_url: str) -> None:
    page_response = await client.get(abs_url)
    if page_response.status_code >= 400:
        row["status"] = "failed"
        session["log"].append(f"Processing failed: HTTP {page_response.status_code} {abs_url}")
        return
    title, abstract, pdf_url = _parse_abs_page(page_response.text, abs_url)
    session["log"].append(f"Processing: {title}")
    record = PaperRecord(title=title, source="arxiv", paper_url=abs_url, paper_id=build_paper_id("", title))
    row.update(asdict(record))
    row["abstract"] = abstract
    if not pdf_url:
        row["status"] = "failed"
        return
    await _download_and_store_pdf(client, record, row, pdf_url)


def _parse_abs_page(html: str, abs_url: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = " ".join(soup.select_one("h1.title").get_text(" ", strip=True).replace("Title:", "").split()) if soup.select_one("h1.title") else abs_url
    abstract = " ".join(soup.select_one("#abs").get_text(" ", strip=True).split()) if soup.select_one("#abs") else ""
    if not abstract and soup.select_one("blockquote.abstract"):
        abstract = " ".join(soup.select_one("blockquote.abstract").get_text(" ", strip=True).split())
    pdf_url = _resolve_pdf_url(soup, abs_url)
    return title, abstract, pdf_url


def _resolve_pdf_url(soup: BeautifulSoup, abs_url: str) -> str:
    for node in soup.select("a[href]"):
        href = " ".join(str(node.get("href", "")).split())
        if not href:
            continue
        if href.endswith(".pdf") or "/pdf/" in href:
            return href if href.startswith("http") else f"https://arxiv.org{href}"
    if "/abs/" in abs_url:
        suffix = abs_url.split("/abs/", maxsplit=1)[1]
        return f"https://arxiv.org/pdf/{suffix}.pdf"
    return ""


async def _download_and_store_pdf(
    client: httpx.AsyncClient,
    record: PaperRecord,
    row: dict,
    pdf_url: str,
) -> None:
    pdf_response = await client.get(pdf_url)
    data = pdf_response.content if pdf_response.status_code < 400 else b""
    if not data.startswith(b"%PDF-"):
        row["status"] = "failed"
        return
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PDF_DIR / f"{record.paper_id}.pdf"
    metadata_path = METADATA_DIR / f"{record.paper_id}.json"
    pdf_path.write_bytes(data)
    metadata = {
        "title": record.title,
        "authors": [],
        "doi": "",
        "year": "",
        "source": record.source,
        "paper_url": record.paper_url,
        "pdf_url": pdf_url,
        "pdf_path": str(pdf_path.as_posix()),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    row["pdf_url"] = pdf_url
    row["status"] = "saved"
