"""
Visual crawling router — discover, download, extract papers via Playwright with real-time progress.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, quote_plus

import httpx
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel

from research_crawler.pipeline import PDFResolver
from council_api.extraction import extract_from_pdf

router = APIRouter(prefix="/crawl-visual", tags=["crawl-visual"])

# Session storage: { session_id: { query, max_papers, limit_per_source, status, papers_discovered, current_url, screenshot_base64, log, stats } }
_sessions: dict = {}
_sessions_lock = threading.Lock()

ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT_DIR / "data" / "pdf"
METADATA_DIR = ROOT_DIR / "data" / "metadata"
EXTRACTED_DIR = ROOT_DIR / "data" / "extracted"


class CrawlVisualStartRequest(BaseModel):
    session_id: str | None = None
    query: str
    max_papers: int = 10
    limit_per_source: int = 5


@router.post("/start")
async def crawl_visual_start(payload: CrawlVisualStartRequest):
    """Initialize a visual crawl session without starting the crawl."""
    session_id = payload.session_id or str(uuid.uuid4())

    with _sessions_lock:
        _sessions[session_id] = {
            "query": payload.query,
            "max_papers": payload.max_papers,
            "limit_per_source": payload.limit_per_source,
            "status": "idle",
            "papers_discovered": [],
            "current_url": "",
            "screenshot_base64": "",
            "screenshot_data_url": "",
            "log": [],
            "stats": {
                "discovered": 0,
                "downloading": 0,
                "downloaded": 0,
                "failed_download": 0,
                "extracting": 0,
                "extracted": 0,
                "failed_extract": 0,
            },
        }
    
    return {"session_id": session_id}


@router.post("/run/{session_id}")
async def crawl_visual_run(session_id: str):
    """Start the visual crawl background task."""
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        _sessions[session_id]["status"] = "crawling"
        _sessions[session_id]["log"].append("Run queued.")

    asyncio.create_task(_run_crawl_visual(session_id))
    return {"session_id": session_id, "status": "crawling"}


@router.get("/status/{session_id}")
async def crawl_visual_status(session_id: str):
    """Get the current state of a crawl session."""
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return _sessions[session_id]


@router.get("/sessions")
async def crawl_visual_sessions():
    """List all active crawl sessions."""
    with _sessions_lock:
        return [
            {"session_id": sid, "status": sess["status"], "query": sess["query"], "stats": sess["stats"]}
            for sid, sess in _sessions.items()
        ]


async def _run_crawl_visual(session_id: str):
    """Runs the visual crawl in a worker thread to avoid Windows selector-loop limitations."""
    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id]["log"].append("Worker dispatched.")
    await asyncio.to_thread(_run_crawl_visual_worker, session_id)


def _run_crawl_visual_worker(session_id: str) -> None:
    try:
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_run_crawl_visual_async(session_id))
            finally:
                loop.close()
            return

        asyncio.run(_run_crawl_visual_async(session_id))
    except Exception as e:  # noqa: BLE001
        with _sessions_lock:
            if session_id in _sessions:
                _sessions[session_id]["status"] = "error"
                _sessions[session_id]["log"].append(f"Worker error: {str(e)}")


async def _run_crawl_visual_async(session_id: str):
    """Main background task: discovery → download → extraction."""
    browser = None
    try:
        with _sessions_lock:
            if session_id in _sessions:
                _sessions[session_id]["log"].append("Worker started.")

        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=False)
                with _sessions_lock:
                    _sessions[session_id]["log"].append("Browser launched (headed).")
            except Exception:
                browser = await p.chromium.launch(headless=True)
                with _sessions_lock:
                    _sessions[session_id]["log"].append("Browser launched (headless fallback).")

            # Publish an initial screenshot so frontend can render immediately.
            bootstrap_page = await browser.new_page()
            await bootstrap_page.goto("about:blank")
            await _update_screenshot(session_id, bootstrap_page, "about:blank")
            await bootstrap_page.close()

            # Phase 1: Discovery
            await _phase_discovery(session_id, browser)

            # Phase 2: Download
            await _phase_download(session_id, browser)

            # Phase 3: Extraction
            await _phase_extraction(session_id, browser)

            with _sessions_lock:
                _sessions[session_id]["status"] = "done"
    except Exception as e:
        with _sessions_lock:
            _sessions[session_id]["status"] = "error"
            _sessions[session_id]["log"].append(f"ERROR: {str(e)}")
    finally:
        if browser:
            await browser.close()


async def _phase_discovery(session_id: str, browser) -> None:
    """PHASE 1: Search arXiv and Semantic Scholar."""
    page = await browser.new_page()
    try:
        page.set_default_timeout(15000)
        page.set_default_navigation_timeout(15000)

        with _sessions_lock:
            _sessions[session_id]["status"] = "crawling"

        with _sessions_lock:
            query = _sessions[session_id]["query"]
            limit = _sessions[session_id]["limit_per_source"]
            _sessions[session_id]["log"].append(f"Discovery started for query: {query}")
        
        # arXiv search
        arxiv_url = f"https://arxiv.org/search/?searchtype=all&query={quote(query)}"
        try:
            await page.goto(arxiv_url, wait_until="commit", timeout=15000)
            await _update_screenshot(session_id, page, arxiv_url)
            with _sessions_lock:
                _sessions[session_id]["log"].append(f"Searching arxiv for: {query}")

            # Scrape arXiv results
            arxiv_links = await page.locator("a[href*='/abs/']").all()
            for link in arxiv_links[: limit * 2]:
                href = await link.get_attribute("href")
                if href:
                    paper_id = href.split("/abs/")[-1]
                    with _sessions_lock:
                        _sessions[session_id]["papers_discovered"].append({
                            "source": "arxiv",
                            "paper_id": paper_id,
                            "url": f"https://arxiv.org{href}",
                            "title": "",
                            "status": "discovered"
                        })
                        _sessions[session_id]["stats"]["discovered"] += 1
                        _sessions[session_id]["log"].append(f"Found: arxiv:{paper_id}")
        except Exception as e:
            with _sessions_lock:
                _sessions[session_id]["log"].append(f"Arxiv page scrape failed, using API fallback: {str(e)}")
            fallback_papers = await _discover_arxiv_via_api(query=query, limit=max(1, limit * 2))
            for paper in fallback_papers:
                with _sessions_lock:
                    _sessions[session_id]["papers_discovered"].append(paper)
                    _sessions[session_id]["stats"]["discovered"] += 1
                    _sessions[session_id]["log"].append(f"Found(api): arxiv:{paper['paper_id']}")

        ss_papers = await _discover_semantic_scholar(query=query, limit=limit)
        for paper in ss_papers:
            with _sessions_lock:
                _sessions[session_id]["papers_discovered"].append(paper)
                _sessions[session_id]["stats"]["discovered"] += 1
                _sessions[session_id]["log"].append(f"Found: ss:{paper['paper_id']}")

        with _sessions_lock:
            _sessions[session_id]["log"].append(
                f"Discovery completed with {_sessions[session_id]['stats']['discovered']} items."
            )
    
    finally:
        await page.close()


async def _phase_download(session_id: str, browser) -> None:
    """PHASE 2: Download PDFs for discovered papers."""
    with _sessions_lock:
        _sessions[session_id]["status"] = "downloading"
        papers = _sessions[session_id]["papers_discovered"][:_sessions[session_id]["max_papers"]]

    page = await browser.new_page()
    page.set_default_timeout(12000)
    page.set_default_navigation_timeout(12000)
    resolver = PDFResolver()

    try:
        for paper in papers:
            with _sessions_lock:
                _sessions[session_id]["log"].append(f"Downloading: {paper['paper_id']}")
                paper["status"] = "downloading"

            try:
                # Try to resolve and download PDF
                url = paper["url"] if "url" in paper else f"https://arxiv.org/abs/{paper['paper_id']}"
                try:
                    await page.goto(url, wait_until="commit", timeout=12000)
                    await _update_screenshot(session_id, page, url)
                except Exception:
                    with _sessions_lock:
                        _sessions[session_id]["current_url"] = url
                pdf_url = resolver.resolve_pdf_url(url)

                if not pdf_url:
                    with _sessions_lock:
                        paper["status"] = "failed"
                        _sessions[session_id]["stats"]["failed_download"] += 1
                    continue

                # Download PDF bytes
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    resp = await client.get(pdf_url, timeout=30)
                    if resp.status_code != 200 or not resp.content.startswith(b"%PDF-"):
                        raise RuntimeError(f"Invalid PDF: {pdf_url}")

                # Save PDF and metadata
                paper_id = paper["paper_id"]
                pdf_path = PDF_DIR / f"{paper_id}.pdf"
                pdf_path.write_bytes(resp.content)

                metadata = {
                    "paper_id": paper_id,
                    "source": paper["source"],
                    "title": paper.get("title", ""),
                    "url": url,
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }
                (METADATA_DIR / f"{paper_id}.json").write_text(json.dumps(metadata, indent=2))

                with _sessions_lock:
                    paper["status"] = "saved"
                    _sessions[session_id]["stats"]["downloaded"] += 1

                await asyncio.sleep(1.0)

            except Exception as e:
                with _sessions_lock:
                    paper["status"] = "failed"
                    _sessions[session_id]["stats"]["failed_download"] += 1
                    _sessions[session_id]["log"].append(f"Failed download: {paper['paper_id']} - {str(e)}")

    finally:
        await page.close()


async def _phase_extraction(session_id: str, browser) -> None:
    """PHASE 3: Extract text from downloaded papers."""
    with _sessions_lock:
        _sessions[session_id]["status"] = "extracting"
        papers = [p for p in _sessions[session_id]["papers_discovered"] if p["status"] == "saved"]

    page = await browser.new_page()

    try:
        for paper in papers:
            paper_id = paper["paper_id"]
            pdf_path = PDF_DIR / f"{paper_id}.pdf"

            if not pdf_path.exists():
                continue

            with _sessions_lock:
                paper["status"] = "extracting"
                _sessions[session_id]["log"].append(f"Extracting: {paper_id}")

            try:
                await page.set_content(
                    "<html><body style='font-family:sans-serif;padding:16px'>"
                    f"<h3>Extracting {paper_id}</h3>"
                    "<p>Parsing downloaded PDF and building extracted JSON...</p>"
                    "</body></html>"
                )
                await _update_screenshot(session_id, page, f"extract://{paper_id}")
            except Exception:
                pass

            try:
                metadata_path = METADATA_DIR / f"{paper_id}.json"
                metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

                # Call extract_from_pdf (synchronous, so run in thread pool)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    extract_from_pdf,
                    paper_id,
                    str(pdf_path),
                    str(EXTRACTED_DIR),
                    metadata
                )

                with _sessions_lock:
                    paper["status"] = "extracted"
                    _sessions[session_id]["stats"]["extracted"] += 1

            except Exception as e:
                with _sessions_lock:
                    paper["status"] = "extract_failed"
                    _sessions[session_id]["stats"]["failed_extract"] += 1
                    _sessions[session_id]["log"].append(f"Extract failed: {paper_id} - {str(e)}")

    finally:
        await page.close()


async def _update_screenshot(session_id: str, page, url: str) -> None:
    """Capture screenshot and store as base64."""
    try:
        screenshot_bytes = await page.screenshot(type="png")
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        with _sessions_lock:
            _sessions[session_id]["current_url"] = url
            _sessions[session_id]["screenshot_base64"] = screenshot_b64
            _sessions[session_id]["screenshot_data_url"] = f"data:image/png;base64,{screenshot_b64}"
    except Exception as e:
        with _sessions_lock:
            _sessions[session_id]["log"].append(f"Screenshot error: {str(e)}")


def _extract_arxiv_id_from_text(value: str) -> str:
    text = " ".join(str(value).split())
    if not text:
        return ""
    # arXiv ids like 1706.03762 or 2401.01234v2
    import re

    match = re.search(r"\b\d{4}\.\d{4,5}(v\d+)?\b", text)
    return match.group(0) if match else ""


async def _discover_arxiv_via_api(query: str, limit: int) -> list[dict]:
    api_url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=all:{quote_plus(query)}&start=0&max_results={max(1, min(limit, 30))}"
    )
    papers: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(api_url)
        if resp.status_code >= 400:
            return papers

        content = resp.text
        import re

        ids = re.findall(r"<id>https?://arxiv\.org/abs/([^<]+)</id>", content)
        titles = re.findall(r"<title>([^<]+)</title>", content)
        for idx, arxiv_id in enumerate(ids):
            if not arxiv_id:
                continue
            title = ""
            if idx + 1 < len(titles):
                title = " ".join(titles[idx + 1].split())
            papers.append(
                {
                    "source": "arxiv",
                    "paper_id": arxiv_id,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "title": title,
                    "status": "discovered",
                }
            )
    except Exception:
        return []

    return papers


async def _discover_semantic_scholar(query: str, limit: int) -> list[dict]:
    papers: list[dict] = []
    ss_url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={quote(query)}&limit={max(1, min(limit, 25))}"
        "&fields=title,externalIds"
    )
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(ss_url)
        if resp.status_code >= 400:
            return papers
        data = resp.json()
        for paper in data.get("data", []):
            title = " ".join(str(paper.get("title", "")).split())
            arxiv_id = " ".join(str(paper.get("externalIds", {}).get("ArXiv", "")).split())
            if not arxiv_id:
                arxiv_id = _extract_arxiv_id_from_text(title)
            if not arxiv_id:
                continue
            papers.append(
                {
                    "source": "semanticscholar",
                    "paper_id": arxiv_id,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "title": title,
                    "status": "discovered",
                }
            )
    except Exception:
        return []

    return papers
