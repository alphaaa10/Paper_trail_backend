from __future__ import annotations

import asyncio
import json
import os
import sys
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from research_crawler.config import load_environment
from research_crawler.pipeline import ResearchOrchestrator

from council_api.extraction import build_final_report, build_report, extract_from_pdf
from council_api.feature_accuracy import router as accuracy_router
from council_api.feature_citation import router as citation_router
from council_api.feature_citation_chat import router as citation_chat_router
from council_api.feature_debate import router as debate_router
from council_api.feature_heatmap import router as heatmap_router
from council_api.feature_qa import router as qa_router
from council_api.browse_router import router as browse_router
from council_api.timeline_router import router as timeline_router

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PDF_DIR = DATA_DIR / "pdf"
METADATA_DIR = DATA_DIR / "metadata"
EXTRACTED_DIR = DATA_DIR / "extracted"
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = DATA_DIR / "logs"

_log_lock = threading.Lock()
_active_log_path: Path | None = None

load_environment()

app = FastAPI(title="Council API", version="0.1.0")
orchestrator = ResearchOrchestrator(pdf_dir=PDF_DIR, metadata_dir=METADATA_DIR)
app.include_router(accuracy_router)
app.include_router(citation_router)
app.include_router(citation_chat_router)
app.include_router(debate_router)
app.include_router(heatmap_router)
app.include_router(qa_router)
app.include_router(browse_router)
app.include_router(timeline_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ensure_data_dirs = [PDF_DIR, METADATA_DIR, EXTRACTED_DIR, REPORTS_DIR, LOGS_DIR]
for _dir in _ensure_data_dirs:
    _dir.mkdir(parents=True, exist_ok=True)

# Serve downloaded PDFs over HTTP so frontend can open them directly in a browser tab.
app.mount("/pdf", StaticFiles(directory=str(PDF_DIR)), name="pdf_files")


class CrawlRequest(BaseModel):
    query: str = Field(default="", description="Topic, DOI, or title")
    question: str = Field(default="", description="Natural-language research question")
    topic_count: int = Field(default=4, ge=1, le=12)
    limit_per_source: int = Field(default=10, ge=1, le=50)
    max_papers: int = Field(default=20, ge=1, le=200)
    concurrency: int = Field(default=6, ge=1, le=20)


class AnalyzeRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)


class FinalReportRequest(BaseModel):
    target_research_finding: str = Field(default="", description="Main finding to validate")
    top_k: int = Field(default=10, ge=1, le=50)
    paper_ids: list[str] = Field(default_factory=list)


class CrawlAndReportRequest(CrawlRequest):
    target_research_finding: str = Field(default="", description="Main finding to validate")
    top_k: int = Field(default=10, ge=1, le=50)


class ExtractTaskStatus(BaseModel):
    status: str = "idle"
    started_at: str = ""
    finished_at: str = ""
    processed_count: int = 0
    skipped_count: int = 0
    processed: list[dict] = Field(default_factory=list)
    skipped: list[dict] = Field(default_factory=list)
    error: str = ""


extract_all_task_status = ExtractTaskStatus()


@app.get("/health")
def health() -> dict:
    _append_log("Health check requested.")
    return {"status": "ok"}


@app.get("/logs/files")
def list_log_files() -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(LOGS_DIR.glob("*.log"), reverse=True):
        stat = path.stat()
        files.append(
            {
                "file_name": path.name,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return {"count": len(files), "files": files}


@app.get("/logs/file/{file_name}")
def get_log_file(file_name: str, tail: int = 300) -> dict:
    path = _resolve_log_file(file_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Log file not found.")

    lines = path.read_text(encoding="utf-8").splitlines()
    clipped = lines[-tail:] if tail > 0 else lines
    return {"file_name": file_name, "line_count": len(lines), "lines": clipped}

@app.get("/logs/stream")
async def stream_logs(file_name: str = "latest.log", follow: bool = True) -> StreamingResponse:
    path = _resolve_log_file(file_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Log file not found.")

    async def event_stream() -> asyncio.AsyncIterator[str]:
        position = 0
        while True:
            if not path.exists():
                payload = {"line": "Log file no longer exists."}
                yield f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"
                return

            with path.open("r", encoding="utf-8") as handle:
                handle.seek(position)
                chunk = handle.read()
                position = handle.tell()

            if chunk:
                for line in chunk.splitlines():
                    payload = {"line": line}
                    yield f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"
            elif not follow:
                return

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/logs/recent")
def get_recent_logs(limit: int = 100) -> dict:
    path = _resolve_log_file("latest.log")
    if not path.exists():
        return {"lines": [], "total": 0, "file_name": "latest.log"}

    all_lines = path.read_text(encoding="utf-8").splitlines()
    recent = all_lines[-max(1, limit):] if limit > 0 else all_lines

    parsed_lines = []
    for line in recent:
        level = "INFO"
        if "ERROR" in line:
            level = "ERROR"
        elif "WARN" in line:
            level = "WARN"
        elif "===" in line:
            level = "SECTION"

        parsed_lines.append({"level": level, "message": line})

    return {"lines": parsed_lines, "total": len(all_lines), "file_name": "latest.log"}


@app.post("/crawl")
async def crawl(payload: CrawlRequest) -> dict:
    _reset_data_directory()
    _start_run_log("crawl")
    _append_log("Data directory reset for new crawl request.")

    query = payload.query.strip()
    question = payload.question.strip()

    if not query and not question:
        _append_log("Crawl request rejected: missing query and question.", level="ERROR")
        raise HTTPException(status_code=422, detail="Provide either query or question.")

    try:
        if question:
            _append_log("Starting crawl from natural-language question.")
            summary = await orchestrator.run_from_question(
                question=question,
                limit_per_source=payload.limit_per_source,
                max_papers=payload.max_papers,
                concurrency=payload.concurrency,
                topic_count=payload.topic_count,
            )
        else:
            _append_log("Starting crawl from direct query.")
            summary = await orchestrator.run(
                query=query,
                limit_per_source=payload.limit_per_source,
                max_papers=payload.max_papers,
                concurrency=payload.concurrency,
            )
    except Exception as exc:  # noqa: BLE001
        _append_log(f"Crawl failed: {exc}", level="ERROR")
        raise

    _append_log(f"Search topics used: {', '.join(summary.topics)}")
    _append_log(
        "Crawl completed: "
        f"discovered={summary.discovered}, deduped={summary.deduped}, "
        f"saved={summary.saved}, skipped={summary.skipped}, failed={summary.failed}."
    )
    _log_download_results(summary.results)
    _log_crawl_summary(summary.results)

    return {
        "query": summary.query,
        "topics": summary.topics,
        "discovered": summary.discovered,
        "deduped": summary.deduped,
        "attempted": summary.attempted,
        "saved": summary.saved,
        "skipped": summary.skipped,
        "failed": summary.failed,
        "results": [
            {
                "paper_id": item.paper_id,
                "status": item.status,
                "reason": item.reason,
                "pdf_path": str(item.pdf_path) if item.pdf_path else "",
                "metadata_path": str(item.metadata_path) if item.metadata_path else "",
            }
            for item in summary.results
        ],
    }


@app.get("/papers")
def list_papers() -> dict:
    _append_log("Listing papers from metadata directory.")
    papers = []
    for path in sorted(METADATA_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        paper_id = path.stem
        papers.append(
            {
                "paper_id": paper_id,
                "title": payload.get("title", ""),
                "year": payload.get("year", ""),
                "source": payload.get("source", ""),
                "pdf_path": payload.get("pdf_path", ""),
                "metadata_path": str(path),
            }
        )
    return {"count": len(papers), "papers": papers}


@app.post("/extract/{paper_id}")
def extract_paper(paper_id: str) -> dict:
    _append_log(f"Extract single paper requested: {paper_id}")
    metadata_path = METADATA_DIR / f"{paper_id}.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata file not found.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pdf_path = Path(metadata.get("pdf_path", ""))
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found.")

    extracted = extract_from_pdf(
        paper_id=paper_id,
        pdf_path=pdf_path,
        output_dir=EXTRACTED_DIR,
        metadata=metadata,
    )

    return {
        "paper_id": paper_id,
        "extracted_path": str(EXTRACTED_DIR / f"{paper_id}.json"),
        "claim_count": len(extracted.get("claims", [])),
        "method_count": len(extracted.get("methods", [])),
        "dataset_count": len(extracted.get("datasets", [])),
    }


@app.post("/extract-all")
def extract_all_papers() -> dict:
    _append_log("Extract-all sync requested.")
    return _extract_all_papers_sync()


@app.post("/extract-all/background")
def extract_all_papers_background(background_tasks: BackgroundTasks) -> dict:
    _append_log("Extract-all background requested.")
    if extract_all_task_status.status == "running":
        _append_log("Extract-all background rejected: task already running.", level="ERROR")
        raise HTTPException(status_code=409, detail="extract-all background task is already running")

    extract_all_task_status.status = "running"
    extract_all_task_status.error = ""
    extract_all_task_status.started_at = _now_iso()
    extract_all_task_status.finished_at = ""
    extract_all_task_status.processed_count = 0
    extract_all_task_status.skipped_count = 0
    extract_all_task_status.processed = []
    extract_all_task_status.skipped = []

    background_tasks.add_task(_run_extract_all_background)
    return {"status": "accepted", "message": "extract-all started in background"}


@app.get("/extract-all/status")
def extract_all_status() -> dict:
    _append_log("Extract-all status requested.")
    return extract_all_task_status.model_dump()


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict:
    _append_log("Analyze requested.")
    report = build_report(EXTRACTED_DIR, paper_ids=payload.paper_ids or None)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "latest_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    return {
        "report_path": str(output_path),
        "paper_count": report["paper_count"],
        "claim_count": report["claim_count"],
        "contradiction_count": len(report["contradictions"]),
        "gaps": report["gaps"],
        "detailed_report": report.get("detailed_report", {}),
        "report": report,
    }


@app.get("/health/crawler")
def health_crawler() -> dict:
    crawl4ai_available = False
    try:
        from crawl4ai import AsyncWebCrawler  # type: ignore
        crawl4ai_available = True
    except Exception:
        pass

    crawl4ai_enabled = os.getenv("CRAWL4AI_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}

    return {
        "crawler_status": "ok",
        "crawl4ai_installed": crawl4ai_available,
        "crawl4ai_enabled": crawl4ai_enabled,
        "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "data_dir": str(DATA_DIR),
    }


@app.get("/report")
def get_latest_report() -> dict:
    _append_log("Latest report requested.")
    path = REPORTS_DIR / "latest_report.json"
    if not path.exists():
        extracted_files = list(EXTRACTED_DIR.glob("*.json"))
        if not extracted_files:
            raise HTTPException(
                status_code=404,
                detail="No report generated yet and no extracted papers found. Run /extract-all first.",
            )

        _append_log("latest_report.json missing; auto-generating from extracted papers.")
        report = build_report(EXTRACTED_DIR)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        "detailed_report" not in payload
        or "executive_summary_json" not in payload
        or "executive_summary_markdown" not in payload
    ):
        _append_log(
            "latest_report.json missing enhanced summary fields; rebuilding report payload.",
            level="WARN",
        )
        payload = build_report(EXTRACTED_DIR)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    return payload


@app.post("/final-report")
def final_report(payload: FinalReportRequest) -> dict:
    _append_log("Final report requested.")
    report = build_final_report(
        extracted_dir=EXTRACTED_DIR,
        target_research_finding=payload.target_research_finding.strip(),
        top_k=payload.top_k,
        paper_ids=payload.paper_ids or None,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "final_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    return {
        "report_path": str(output_path),
        "report": report,
    }


@app.post("/crawl-report")
async def crawl_report(payload: CrawlAndReportRequest) -> dict:
    _reset_data_directory()
    _start_run_log("crawl_report")
    _append_log("Data directory reset for new crawl-report request.")

    query = payload.query.strip()
    question = payload.question.strip()

    if not query and not question:
        _append_log("Crawl-report request rejected: missing query and question.", level="ERROR")
        raise HTTPException(status_code=422, detail="Provide either query or question.")

    try:
        if question:
            _append_log("Step 1/4: crawl from question started.")
            crawl_summary = await orchestrator.run_from_question(
                question=question,
                limit_per_source=payload.limit_per_source,
                max_papers=payload.max_papers,
                concurrency=payload.concurrency,
                topic_count=payload.topic_count,
            )
        else:
            _append_log("Step 1/4: crawl from query started.")
            crawl_summary = await orchestrator.run(
                query=query,
                limit_per_source=payload.limit_per_source,
                max_papers=payload.max_papers,
                concurrency=payload.concurrency,
            )
        _append_log("Step 1/4: crawl completed.")
        _log_download_results(crawl_summary.results)

        _append_log("Step 2/4: extraction started.")
        extract_summary = await asyncio.to_thread(_extract_all_papers_sync)
        _append_log("Step 2/4: extraction completed.")

        _append_log("Step 3/4: analysis started.")
        analyze_report = build_report(EXTRACTED_DIR)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        latest_path = REPORTS_DIR / "latest_report.json"
        latest_path.write_text(json.dumps(analyze_report, ensure_ascii=True, indent=2), encoding="utf-8")
        _append_log("Step 3/4: analysis completed.")

        _append_log("Step 4/4: final report started.")
        final_payload = build_final_report(
            extracted_dir=EXTRACTED_DIR,
            target_research_finding=payload.target_research_finding.strip(),
            top_k=payload.top_k,
        )
        final_path = REPORTS_DIR / "final_report.json"
        final_path.write_text(json.dumps(final_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        _append_log("Step 4/4: final report completed.")
    except Exception as exc:  # noqa: BLE001
        _append_log(f"Crawl-report failed: {exc}", level="ERROR")
        raise

    return {
        "crawl": {
            "query": crawl_summary.query,
            "topics": crawl_summary.topics,
            "discovered": crawl_summary.discovered,
            "deduped": crawl_summary.deduped,
            "attempted": crawl_summary.attempted,
            "saved": crawl_summary.saved,
            "skipped": crawl_summary.skipped,
            "failed": crawl_summary.failed,
        },
        "extract": extract_summary,
        "analyze": {
            "report_path": str(latest_path),
            "paper_count": analyze_report["paper_count"],
            "claim_count": analyze_report["claim_count"],
            "contradiction_count": len(analyze_report["contradictions"]),
            "gaps": analyze_report["gaps"],
            "detailed_report": analyze_report.get("detailed_report", {}),
            "report": analyze_report,
        },
        "final_report": {
            "report_path": str(final_path),
            "executive_summary": final_payload.get("executive_summary", {}),
        },
    }


def _extract_all_papers_sync() -> dict:
    _append_log("Extract-all sync execution started.")
    processed = []
    skipped = []

    for metadata_path in sorted(METADATA_DIR.glob("*.json")):
        paper_id = metadata_path.stem
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pdf_path = Path(metadata.get("pdf_path", ""))

        if not pdf_path.exists():
            _append_log(f"Extract skipped for {paper_id}: PDF file not found.", level="WARN")
            skipped.append({"paper_id": paper_id, "reason": "PDF file not found"})
            continue

        extracted = extract_from_pdf(
            paper_id=paper_id,
            pdf_path=pdf_path,
            output_dir=EXTRACTED_DIR,
            metadata=metadata,
        )

        processed.append(
            {
                "paper_id": paper_id,
                "claim_count": len(extracted.get("claims", [])),
                "method_count": len(extracted.get("methods", [])),
                "dataset_count": len(extracted.get("datasets", [])),
            }
        )
        _append_log(f"Extract completed for {paper_id}.")

    result = {
        "processed_count": len(processed),
        "skipped_count": len(skipped),
        "processed": processed,
        "skipped": skipped,
    }
    _append_log(
        f"Extract-all sync finished: processed={result['processed_count']}, skipped={result['skipped_count']}."
    )
    return result


def _run_extract_all_background() -> None:
    _append_log("Extract-all background worker started.")
    try:
        result = _extract_all_papers_sync()
        extract_all_task_status.status = "completed"
        extract_all_task_status.processed_count = result["processed_count"]
        extract_all_task_status.skipped_count = result["skipped_count"]
        extract_all_task_status.processed = result["processed"]
        extract_all_task_status.skipped = result["skipped"]
    except Exception as exc:  # noqa: BLE001
        extract_all_task_status.status = "failed"
        extract_all_task_status.error = str(exc)
        _append_log(f"Extract-all background failed: {exc}", level="ERROR")
    finally:
        extract_all_task_status.finished_at = _now_iso()
        _append_log("Extract-all background worker finished.")


def _reset_data_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for child in DATA_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
    for path in [PDF_DIR, METADATA_DIR, EXTRACTED_DIR, REPORTS_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _start_run_log(run_name: str) -> None:
    global _active_log_path
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _active_log_path = LOGS_DIR / f"{run_name}_{stamp}.log"
    _append_log(f"Run started: {run_name}")


def _resolve_log_file(file_name: str) -> Path:
    if not file_name or "/" in file_name or "\\" in file_name or ".." in file_name:
        raise HTTPException(status_code=400, detail="Invalid file_name.")
    return LOGS_DIR / file_name


def _append_log(message: str, level: str = "INFO") -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{_now_iso()} | {level.upper()} | {message}"
    latest_path = LOGS_DIR / "latest.log"

    with _log_lock:
        targets = {latest_path}
        if _active_log_path is not None:
            targets.add(_active_log_path)

        for path in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _log_download_results(results: list) -> None:
    if not results:
        _append_log("No download results produced by crawl.", level="WARN")
        return

    for item in results:
        paper_id = str(getattr(item, "paper_id", "")).strip() or "unknown"
        status = str(getattr(item, "status", "unknown")).strip() or "unknown"
        reason = str(getattr(item, "reason", "")).strip()

        if reason:
            _append_log(f"Crawl result: paper_id={paper_id}, status={status}, reason={reason}")
        else:
            _append_log(f"Crawl result: paper_id={paper_id}, status={status}")


def _log_crawl_summary(results: list) -> None:
    if not results:
        return

    skip_reasons: dict[str, int] = {}
    fail_reasons: dict[str, int] = {}

    for item in results:
        status = str(getattr(item, "status", "")).strip()
        reason = str(getattr(item, "reason", "")).strip() or "unknown"
        if status == "skipped":
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        elif status == "failed":
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    if skip_reasons:
        _append_log("=== Skip Summary ===")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            _append_log(f"  {reason}: {count}")
            if "pdf url not resolved" in reason.lower():
                _append_log("    → Enable crawl4ai (set CRAWL4AI_ENABLED=1) for better link discovery")
            elif "paywall" in reason.lower():
                _append_log("    → Publisher paywall; need institutional access or use open-access mirrors")

    if fail_reasons:
        _append_log("=== Failure Summary ===")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            _append_log(f"  {reason}: {count}")
            if "non-pdf content" in reason.lower():
                _append_log("    → PDF resolver found HTML; enable crawl4ai or use /pdf/search endpoints")
            elif "timeout" in reason.lower():
                _append_log("    → Network timeout; increase timeout (set PDF_DOWNLOAD_TIMEOUT env var)")
            elif "404" in reason.lower():
                _append_log("    → Dead/moved URL; check source availability")
            elif "paywall" in reason.lower() or "403" in reason.lower():
                _append_log("    → Access denied; paper may require institutional subscription")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
