from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
from fastapi import APIRouter

from council_api.extraction import _next_groq_api_key as _next_groq_key

router = APIRouter(prefix="/timeline", tags=["timeline"])

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
METADATA_DIR = DATA_DIR / "metadata"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def _groq_keys() -> list[str]:
    pooled = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
    single = os.getenv("GROQ_API_KEY", "").strip()
    if single and single not in pooled:
        pooled.append(single)
    return pooled


def read_papers() -> list[dict]:
    papers: list[dict] = []
    metadata = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(METADATA_DIR.glob("*.json"))
    }
    for path in sorted(EXTRACTED_DIR.glob("*.json")):
        extracted = json.loads(path.read_text(encoding="utf-8"))
        paper_id = str(extracted.get("paper_id") or path.stem).strip()
        meta = metadata.get(paper_id, {})
        claims = [str(c).strip() for c in extracted.get("claims", []) if str(c).strip()]
        methods = [str(m).strip() for m in extracted.get("methods", []) if str(m).strip()]
        year = str(extracted.get("year") or meta.get("year") or "").strip() or "Unknown"
        papers.append(
            {
                "paper_id": paper_id,
                "title": str(extracted.get("title") or meta.get("title") or "").strip(),
                "year": year,
                "claims": claims,
                "methods": methods,
            }
        )
    return papers


def _timeline_prompt(papers: list[dict]) -> str:
    compact = [
        {
            "paper_id": p["paper_id"],
            "year": p["year"],
            "title": p["title"],
            "claims": p["claims"][:3],
        }
        for p in papers
    ]
    return (
        "You are a research historian. Given these papers and their claims, "
        "write a one-sentence contribution for each paper explaining what it "
        "added to the evolution of this topic. Be specific - reference the "
        "actual claim or method. Return JSON only:\n"
        "{\n"
        "  contributions: [\n"
        "    { paper_id: string, year: string, contribution: string }\n"
        "  ]\n"
        "}\n\n"
        f"papers: {json.dumps(compact, ensure_ascii=True)}"
    )


def _parse_contributions(parsed: dict) -> dict[str, str]:
    contributions = parsed.get("contributions", []) if isinstance(parsed, dict) else []
    return {
        str(item.get("paper_id", "")).strip(): str(item.get("contribution", "")).strip()
        for item in contributions
        if isinstance(item, dict) and str(item.get("paper_id", "")).strip()
    }


def call_groq(papers: list[dict]) -> dict[str, str]:
    keys = _groq_keys()
    if not papers or not keys:
        return {}

    body = {
        "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "temperature": 0,
        "max_tokens": 1800,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": _timeline_prompt(papers)}],
    }

    try:
        with httpx.Client(timeout=45.0) as client:
            response = client.post(
                GROQ_URL,
                headers={
                    "authorization": f"Bearer {_next_groq_key(keys)}",
                    "content-type": "application/json",
                },
                json=body,
            )
        if response.status_code >= 400:
            return {}
        payload = response.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = json.loads(content) if isinstance(content, str) and content else {}
    except Exception:
        return {}

    return _parse_contributions(parsed)


def _sort_years(values: list[str]) -> list[str]:
    unique = list(dict.fromkeys(values))

    def _rank(year: str) -> tuple[int, int, str]:
        if year == "Unknown":
            return (2, 0, year)
        if year.isdigit():
            return (0, int(year), year)
        match = re.search(r"(19|20)\d{2}", year)
        if match:
            return (1, int(match.group(0)), year)
        return (1, 0, year)

    return sorted(unique, key=_rank)


@router.get("")
def get_timeline() -> dict:
    papers = read_papers()
    if not papers:
        return {
            "years": [],
            "papers": [],
            "papers_by_year": {},
            "total_papers": 0,
            "year_range": {"earliest": "Unknown", "latest": "Unknown"},
        }

    contributions = call_groq(papers)
    grouped: dict[str, list[dict]] = {}

    for paper in papers:
        fallback = paper["claims"][0] if paper["claims"] else ""
        grouped.setdefault(paper["year"], []).append(
            {
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "year": paper["year"],
                "contribution": contributions.get(paper["paper_id"]) or fallback,
                "claims": paper["claims"],
                "methods": paper["methods"],
            }
        )

    years = _sort_years(list(grouped.keys()))
    known_years = [year for year in years if year != "Unknown"]
    papers_flat = [item for year in years for item in grouped[year]]

    return {
        "years": years,
        "papers": papers_flat,
        "papers_by_year": {year: grouped[year] for year in years},
        "total_papers": len(papers),
        "year_range": {
            "earliest": known_years[0] if known_years else "Unknown",
            "latest": known_years[-1] if known_years else "Unknown",
        },
    }
