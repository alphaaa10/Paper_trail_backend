from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from research_crawler.models import DownloadResult, PaperRecord, sanitize_record, to_metadata_json
from research_crawler.utils import build_paper_id, normalize_doi, with_retries

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENALEX_URL = "https://api.openalex.org/works"
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DUCK_HTML_SEARCH = "https://duckduckgo.com/html/"
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
ARXIV_ABS_PATTERN = re.compile(r"https?://arxiv\.org/abs/([^/?#]+)", re.IGNORECASE)
_GROQ_KEY_CURSOR = 0


@dataclass(slots=True)
class CrawlSummary:
    query: str
    topics: list[str]
    discovered: int
    deduped: int
    attempted: int
    saved: int
    skipped: int
    failed: int
    results: list[DownloadResult]


class TopicPlanner:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    async def extract_topics(
        self,
        client: httpx.AsyncClient,
        question: str,
        topic_count: int = 4,
    ) -> list[str]:
        if not _groq_api_keys():
            return [question.strip()]

        prompt = (
            "Given a user research question, return short paper-search topics as JSON only. "
            "Keep each topic concise and publication-search friendly.\n"
            f"Return JSON object schema: {{\"topics\": [string, ...]}} with up to {max(1, topic_count)} topics.\n"
            f"Question: {question}"
        )

        async def _request() -> dict:
            payload = {
                "model": self.model,
                "max_tokens": 300,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            }
            response_payload = await _groq_chat_with_rotation(client=client, payload=payload)
            if not response_payload:
                raise RuntimeError("groq request failed")
            return response_payload

        try:
            payload = await with_retries(_request, retries=2, base_delay=0.6)
        except Exception:  # noqa: BLE001
            return [question.strip()]

        text = self._extract_text(payload)
        topics = self._parse_topics(text, topic_count=topic_count)
        if not topics:
            return [question.strip()]
        return topics

    @staticmethod
    def _extract_text(payload: dict) -> str:
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message", {})
        if not isinstance(message, dict):
            return ""
        content = message.get("content", "")
        return content if isinstance(content, str) else ""

    @staticmethod
    def _parse_topics(text: str, topic_count: int) -> list[str]:
        if not text:
            return []

        data = TopicPlanner._try_json(text)
        if not data:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                data = TopicPlanner._try_json(match.group(0))

        if not isinstance(data, dict):
            return []

        topics_obj = data.get("topics", [])
        if not isinstance(topics_obj, list):
            return []

        clean: list[str] = []
        for item in topics_obj:
            if not isinstance(item, str):
                continue
            topic = " ".join(item.strip().split())
            if topic and topic not in clean:
                clean.append(topic)
            if len(clean) >= max(1, topic_count):
                break
        return clean

    @staticmethod
    def _try_json(value: str) -> dict | None:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


def _groq_api_keys() -> list[str]:
    pooled = os.getenv("GROQ_API_KEYS", "")
    keys = [item.strip() for item in pooled.split(",") if item.strip()]

    single = os.getenv("GROQ_API_KEY", "").strip()
    if single and single not in keys:
        keys.append(single)

    return keys


def _next_groq_api_key(keys: list[str]) -> str:
    global _GROQ_KEY_CURSOR
    key = keys[_GROQ_KEY_CURSOR % len(keys)]
    _GROQ_KEY_CURSOR += 1
    return key


async def _groq_chat_with_rotation(client: httpx.AsyncClient, payload: dict) -> dict | None:
    keys = _groq_api_keys()
    if not keys:
        return None

    for _ in range(len(keys)):
        api_key = _next_groq_api_key(keys)
        try:
            response = await client.post(
                GROQ_URL,
                headers={
                    "authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
                json=payload,
                timeout=25.0,
            )
            if response.status_code == 429 or response.status_code >= 500:
                continue
            response.raise_for_status()
            result = response.json()
            if isinstance(result, dict):
                return result
        except Exception:  # noqa: BLE001
            continue

    return None


async def search_openalex(
    client: httpx.AsyncClient,
    query: str,
    limit: int = 10,
) -> list[PaperRecord]:
    async def _request() -> dict:
        response = await client.get(
            OPENALEX_URL,
            params={"search": query, "per-page": limit},
            timeout=20.0,
        )
        response.raise_for_status()
        return response.json()

    payload = await with_retries(_request)
    works = payload.get("results", [])
    records: list[PaperRecord] = []

    for work in works:
        title = (work.get("title") or "").strip()
        if not title:
            continue

        authors = [
            authorship.get("author", {}).get("display_name", "").strip()
            for authorship in work.get("authorships", [])
            if authorship.get("author", {}).get("display_name")
        ]
        doi = normalize_doi(work.get("doi") or "")
        year = str(work.get("publication_year") or "")

        primary_location = work.get("primary_location") or {}
        open_access = work.get("open_access") or {}
        paper_url = (primary_location.get("landing_page_url") or "").strip()
        pdf_url = (open_access.get("oa_url") or "").strip()

        records.append(
            PaperRecord(
                title=title,
                authors=authors,
                doi=doi,
                year=year,
                source="openalex",
                paper_url=paper_url,
                pdf_url=pdf_url,
            )
        )

    return records


async def search_semantic_scholar(
    client: httpx.AsyncClient,
    query: str,
    limit: int = 10,
) -> list[PaperRecord]:
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    headers = {"x-api-key": api_key} if api_key else {}
    fields = "title,authors,year,externalIds,url,openAccessPdf"

    async def _request() -> dict:
        response = await client.get(
            S2_URL,
            params={"query": query, "limit": limit, "fields": fields},
            headers=headers,
            timeout=20.0,
        )
        response.raise_for_status()
        return response.json()

    payload = await with_retries(_request)
    papers = payload.get("data", [])
    records: list[PaperRecord] = []

    for paper in papers:
        title = (paper.get("title") or "").strip()
        if not title:
            continue

        authors = [
            author.get("name", "").strip()
            for author in paper.get("authors", [])
            if author.get("name")
        ]

        external_ids = paper.get("externalIds") or {}
        doi = normalize_doi(external_ids.get("DOI") or "")
        year = str(paper.get("year") or "")
        paper_url = (paper.get("url") or "").strip()
        pdf_url = ((paper.get("openAccessPdf") or {}).get("url") or "").strip()

        records.append(
            PaperRecord(
                title=title,
                authors=authors,
                doi=doi,
                year=year,
                source="semantic_scholar",
                paper_url=paper_url,
                pdf_url=pdf_url,
            )
        )

    return records


def _extract_actual_url(raw_href: str) -> str:
    parsed = urlparse(raw_href)
    if parsed.path == "/l/":
        uddg = parse_qs(parsed.query).get("uddg", [""])[0]
        return uddg or raw_href
    return raw_href


async def search_duckduckgo(
    client: httpx.AsyncClient,
    query: str,
    limit: int = 10,
) -> list[PaperRecord]:
    async def _request() -> str:
        response = await client.post(
            DUCK_HTML_SEARCH,
            data={"q": query},
            timeout=20.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text

    html = await with_retries(_request)
    soup = BeautifulSoup(html, "html.parser")
    records: list[PaperRecord] = []

    result_nodes = soup.select("div.result")
    for node in result_nodes:
        if len(records) >= limit:
            break

        title_link = node.select_one("a.result__a")
        if not title_link:
            continue

        title = title_link.get_text(" ", strip=True)
        href_val = title_link.get("href")
        href = href_val if isinstance(href_val, str) else ""
        paper_url = _extract_actual_url(href.strip())
        snippet = node.select_one("a.result__snippet") or node.select_one("div.result__snippet")
        snippet_text = snippet.get_text(" ", strip=True) if snippet else ""

        doi_match = DOI_PATTERN.search(f"{title} {snippet_text} {paper_url}")
        year_match = YEAR_PATTERN.search(f"{title} {snippet_text}")

        records.append(
            PaperRecord(
                title=title,
                authors=[],
                doi=normalize_doi(doi_match.group(0) if doi_match else ""),
                year=year_match.group(0) if year_match else "",
                source="duckduckgo",
                paper_url=paper_url,
                pdf_url="",
            )
        )

    return records


class PDFResolver:
    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout

    async def resolve_pdf_url(self, client: httpx.AsyncClient, record: PaperRecord) -> str:
        if self._looks_like_pdf(record.pdf_url):
            return record.pdf_url

        arxiv_pdf = self._arxiv_pdf_url(record.paper_url) or self._arxiv_from_doi(record.doi)
        if arxiv_pdf:
            return arxiv_pdf

        if self._looks_like_pdf(record.paper_url):
            return record.paper_url

        if not record.paper_url:
            return ""

        if self._crawl4ai_enabled():
            crawl4ai_link = await self._discover_with_crawl4ai(record.paper_url)
            if crawl4ai_link:
                return crawl4ai_link

        return await self._discover_from_html(client, record.paper_url)

    async def _discover_from_html(self, client: httpx.AsyncClient, page_url: str) -> str:
        async def _request() -> str:
            response = await client.get(page_url, timeout=self.timeout, follow_redirects=True)
            response.raise_for_status()
            return response.text

        try:
            html = await with_retries(_request)
        except Exception:  # noqa: BLE001
            return ""

        soup = BeautifulSoup(html, "html.parser")
        anchors = soup.select("a[href]")
        candidates: list[str] = []

        for anchor in anchors:
            href_val = anchor.get("href")
            href = href_val if isinstance(href_val, str) else ""
            href = href.strip()
            if not href:
                continue
            full = urljoin(page_url, href)
            if self._looks_like_pdf(full):
                candidates.append(full)
                continue
            if "pdf" in full.lower() and full.startswith("http"):
                candidates.append(full)

        return self._pick_best(candidates)

    async def _discover_with_crawl4ai(self, page_url: str) -> str:
        try:
            from crawl4ai import AsyncWebCrawler  # type: ignore
        except Exception:  # noqa: BLE001
            return ""

        try:
            async def _crawl() -> object:
                async with AsyncWebCrawler(verbose=False) as crawler:
                    return await crawler.arun(url=page_url)

            result = await with_retries(_crawl, retries=2, base_delay=0.5)
        except Exception:  # noqa: BLE001
            return ""

        candidates: list[str] = []
        links_obj = getattr(result, "links", None)

        if isinstance(links_obj, dict):
            for value in links_obj.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            url = (item.get("href") or item.get("url") or "").strip()
                            if url:
                                candidates.append(url)
                        elif isinstance(item, str):
                            candidates.append(item.strip())
        elif isinstance(links_obj, list):
            for item in links_obj:
                if isinstance(item, dict):
                    url = (item.get("href") or item.get("url") or "").strip()
                    if url:
                        candidates.append(url)
                elif isinstance(item, str):
                    candidates.append(item.strip())

        markdown_text = getattr(result, "markdown", "")
        if isinstance(markdown_text, str) and markdown_text:
            candidates.extend(re.findall(r"https?://[^\s)\]]+", markdown_text))

        return self._pick_best(candidates)

    @staticmethod
    def _pick_best(candidates: list[str]) -> str:
        cleaned = [c.strip() for c in candidates if c and c.strip().startswith("http")]
        if not cleaned:
            return ""

        for candidate in cleaned:
            if candidate.lower().endswith(".pdf"):
                return candidate
        for candidate in cleaned:
            if "arxiv.org/pdf/" in candidate.lower():
                return candidate
        for candidate in cleaned:
            if "pdf" in candidate.lower():
                return candidate

        return ""

    @staticmethod
    def _crawl4ai_enabled() -> bool:
        return os.getenv("CRAWL4AI_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _looks_like_pdf(url: str) -> bool:
        if not url:
            return False
        lowered = url.lower()
        return lowered.endswith(".pdf") or "/pdf/" in lowered or "type=pdf" in lowered

    @staticmethod
    def _arxiv_pdf_url(url: str) -> str:
        if not url:
            return ""
        match = ARXIV_ABS_PATTERN.match(url)
        if not match:
            return ""
        return f"https://arxiv.org/pdf/{match.group(1)}.pdf"

    @staticmethod
    def _arxiv_from_doi(doi: str) -> str:
        if not doi:
            return ""
        lowered = doi.lower()
        if lowered.startswith("10.48550/arxiv."):
            identifier = lowered.split("10.48550/arxiv.", maxsplit=1)[1]
            return f"https://arxiv.org/pdf/{identifier}.pdf"
        return ""


class PDFDownloader:
    def __init__(
        self,
        pdf_dir: Path,
        metadata_dir: Path,
        resolver: PDFResolver,
        timeout: float = 40.0,
    ) -> None:
        self.pdf_dir = pdf_dir
        self.metadata_dir = metadata_dir
        self.resolver = resolver
        self.timeout = timeout
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    async def save_one(self, client: httpx.AsyncClient, record: PaperRecord) -> DownloadResult:
        record = sanitize_record(record)
        if not record.paper_id:
            return DownloadResult(paper_id="", status="failed", reason="missing paper_id")

        resolved_pdf = await self.resolver.resolve_pdf_url(client, record)
        if not resolved_pdf:
            return DownloadResult(paper_id=record.paper_id, status="skipped", reason="pdf url not resolved")

        record.pdf_url = resolved_pdf
        pdf_path = self.pdf_dir / f"{record.paper_id}.pdf"
        metadata_path = self.metadata_dir / f"{record.paper_id}.json"

        async def _download() -> bytes:
            response = await client.get(resolved_pdf, timeout=self.timeout, follow_redirects=True)
            response.raise_for_status()
            return response.content

        try:
            pdf_bytes = await with_retries(_download, retries=3, base_delay=0.8)
        except Exception as exc:  # noqa: BLE001
            return DownloadResult(paper_id=record.paper_id, status="failed", reason=str(exc))

        if not self._looks_like_pdf(pdf_bytes):
            return DownloadResult(paper_id=record.paper_id, status="failed", reason="non-pdf content")

        pdf_path.write_bytes(pdf_bytes)
        metadata_path.write_text(
            json.dumps(to_metadata_json(record, pdf_path), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        return DownloadResult(
            paper_id=record.paper_id,
            status="saved",
            pdf_path=pdf_path,
            metadata_path=metadata_path,
        )

    async def save_batch(
        self,
        client: httpx.AsyncClient,
        records: list[PaperRecord],
        concurrency: int = 6,
    ) -> list[DownloadResult]:
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def _wrapped(record: PaperRecord) -> DownloadResult:
            async with semaphore:
                return await self.save_one(client, record)

        tasks = [asyncio.create_task(_wrapped(record)) for record in records]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _looks_like_pdf(data: bytes) -> bool:
        return bool(data and data[:5] == b"%PDF-")


class ResearchOrchestrator:
    def __init__(self, pdf_dir: Path, metadata_dir: Path) -> None:
        self.pdf_dir = pdf_dir
        self.metadata_dir = metadata_dir
        self.topic_planner = TopicPlanner()

    async def run(
        self,
        query: str,
        limit_per_source: int = 10,
        max_papers: int = 20,
        concurrency: int = 6,
    ) -> CrawlSummary:
        return await self._run_for_topics(
            query=query,
            topics=[query],
            limit_per_source=limit_per_source,
            max_papers=max_papers,
            concurrency=concurrency,
        )

    async def run_from_question(
        self,
        question: str,
        limit_per_source: int = 10,
        max_papers: int = 20,
        concurrency: int = 6,
        topic_count: int = 4,
    ) -> CrawlSummary:
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            topics = await self.topic_planner.extract_topics(
                client,
                question=question,
                topic_count=topic_count,
            )

        return await self._run_for_topics(
            query=question,
            topics=topics,
            limit_per_source=limit_per_source,
            max_papers=max_papers,
            concurrency=concurrency,
        )

    async def _run_for_topics(
        self,
        query: str,
        topics: list[str],
        limit_per_source: int,
        max_papers: int,
        concurrency: int,
    ) -> CrawlSummary:
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            discovered = await self._search_many(client, topics, limit_per_source)
            merged = self._merge_dedupe(discovered)
            selected = merged[: max(0, max_papers)]

            downloader = PDFDownloader(
                pdf_dir=self.pdf_dir,
                metadata_dir=self.metadata_dir,
                resolver=PDFResolver(),
            )
            results = await downloader.save_batch(client, selected, concurrency=concurrency)

        saved = sum(1 for item in results if item.status == "saved")
        skipped = sum(1 for item in results if item.status == "skipped")
        failed = sum(1 for item in results if item.status == "failed")

        return CrawlSummary(
            query=query,
            topics=topics,
            discovered=len(discovered),
            deduped=len(merged),
            attempted=len(selected),
            saved=saved,
            skipped=skipped,
            failed=failed,
            results=results,
        )

    async def _search_many(
        self,
        client: httpx.AsyncClient,
        topics: list[str],
        limit_per_source: int,
    ) -> list[PaperRecord]:
        unique_topics: list[str] = []
        for topic in topics:
            value = topic.strip()
            if value and value not in unique_topics:
                unique_topics.append(value)

        tasks = [
            asyncio.create_task(self._search_all(client, topic, limit_per_source))
            for topic in unique_topics
        ]
        settled = await asyncio.gather(*tasks, return_exceptions=True)

        records: list[PaperRecord] = []
        for chunk in settled:
            if isinstance(chunk, Exception):
                continue
            if isinstance(chunk, list):
                records.extend(chunk)
        return records

    async def _search_all(
        self,
        client: httpx.AsyncClient,
        query: str,
        limit_per_source: int,
    ) -> list[PaperRecord]:
        tasks = [
            asyncio.create_task(search_openalex(client, query, limit=limit_per_source)),
            asyncio.create_task(search_semantic_scholar(client, query, limit=limit_per_source)),
        ]
        settled = await asyncio.gather(*tasks, return_exceptions=True)

        records: list[PaperRecord] = []
        for chunk in settled:
            if isinstance(chunk, Exception):
                continue
            if isinstance(chunk, list):
                records.extend(chunk)
        return records

    def _merge_dedupe(self, records: list[PaperRecord]) -> list[PaperRecord]:
        merged: dict[str, PaperRecord] = {}

        for record in records:
            if not record.title.strip():
                continue

            record.paper_id = build_paper_id(record.doi, record.title)
            existing = merged.get(record.paper_id)
            if existing is None:
                merged[record.paper_id] = record
                continue

            if self._score(record) > self._score(existing):
                merged[record.paper_id] = record

        return sorted(
            merged.values(),
            key=lambda item: (
                self._score(item),
                item.year or "0000",
                item.title.lower(),
            ),
            reverse=True,
        )

    @staticmethod
    def _score(record: PaperRecord) -> int:
        source_weight = {
            "openalex": 3,
            "semantic_scholar": 2,
            "duckduckgo": 1,
        }.get(record.source.lower(), 0)

        return (
            source_weight
            + (3 if bool(record.doi) else 0)
            + (2 if bool(record.pdf_url) else 0)
            + (1 if bool(record.authors) else 0)
            + (1 if bool(record.year) else 0)
        )
