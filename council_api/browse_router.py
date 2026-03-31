from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote_plus

import httpx
from fastapi import APIRouter, HTTPException
from playwright.async_api import Browser, Page, async_playwright
from pydantic import BaseModel

from council_api.extraction import _next_groq_api_key as _next_groq_key
from research_crawler.models import PaperRecord
from research_crawler.utils import build_paper_id

router = APIRouter(prefix="/browse", tags=["browse"])

ROOT_DIR = Path(__file__).resolve().parents[1]
FINAL_REPORT_PATH = ROOT_DIR / "data" / "reports" / "final_report.json"
EXTRACTED_DIR = ROOT_DIR / "data" / "extracted"
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
    browser: Browser | None = None
    playwright = await async_playwright().start()
    try:
        browser = await playwright.chromium.launch(headless=False)
        page = await browser.new_page()
        await _run_query_loop(page, session)
        session["status"] = "done"
        session["log"].append("Run completed.")
    finally:
        if browser:
            await browser.close()
        await playwright.stop()


async def _run_query_loop(page: Page, session: dict) -> None:
    for query in session["queries"]:
        search_url = f"https://arxiv.org/search/?searchtype=all&query={quote_plus(query)}"
        await _safe_navigate(page, search_url, session)
        await _capture_session_shot(page, session)
        links = await _collect_abs_links(page)
        for link in links[:3]:
            await _safe_navigate(page, link, session)
            await _capture_session_shot(page, session)
            await _capture_paper(page, link, session)
            await asyncio.sleep(1.5)
        session["current_query_index"] += 1


async def _safe_navigate(page: Page, url: str, session: dict) -> None:
    try:
        await page.goto(url, wait_until="domcontentloaded")
    except Exception as exc:  # noqa: BLE001
        session["log"].append(f"Navigation failed for {url}: {exc}")
        return
    session["current_url"] = page.url
    session["log"].append(f"Visited {page.url}")


async def _capture_session_shot(page: Page, session: dict) -> None:
    raw = await page.screenshot(type="png")
    encoded = base64.b64encode(raw).decode("utf-8")
    session["screenshot_base64"] = encoded


async def _collect_abs_links(page: Page) -> list[str]:
    links = await page.eval_on_selector_all(
        "a[href*='/abs/']",
        "nodes => nodes.map(n => n.href)",
    )
    unique: list[str] = []
    for link in links:
        text = " ".join(str(link).split())
        if text and text not in unique:
            unique.append(text)
    return unique


async def _capture_paper(page: Page, link: str, session: dict) -> None:
    title = await _read_text(page, "h1.title")
    abstract = await _read_text(page, "#abs")
    if not abstract:
        abstract = await _read_text(page, "blockquote.abstract")
    clean_title = title.replace("Title:", "").strip() or link
    paper = PaperRecord(
        title=clean_title,
        source="arXiv",
        paper_url=link,
        paper_id=build_paper_id("", clean_title),
    )
    row = asdict(paper)
    row["abstract"] = abstract
    session["papers_found"].append(row)
    session["log"].append(f"Captured paper: {clean_title}")


async def _read_text(page: Page, selector: str) -> str:
    locator = page.locator(selector).first
    count = await locator.count()
    if count == 0:
        return ""
    text = await locator.inner_text()
    return " ".join(text.split())
