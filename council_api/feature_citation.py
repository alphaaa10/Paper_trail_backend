from __future__ import annotations

import json
import re
from urllib.parse import quote
from pathlib import Path
from typing import Any

import fitz
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/feature", tags=["feature"])

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
METADATA_DIR = DATA_DIR / "metadata"


class CitationRequest(BaseModel):
    paper_id: str = Field(..., min_length=1)
    claim_text: str = Field(..., min_length=1)


class CitationResponse(BaseModel):
    paper_id: str
    paper_display_name: str = ""
    claim_text: str
    pdf_path: str
    pdf_url: str = ""
    page_number: int
    bbox: list[float]


def _resolve_metadata_path(paper_id: str) -> Path:
    direct = METADATA_DIR / f"{paper_id}.json"
    if direct.exists():
        return direct

    matches = list(METADATA_DIR.glob(f"*{paper_id}*.json"))
    if matches:
        return matches[0]

    raise HTTPException(status_code=404, detail=f"Metadata not found for paper_id '{paper_id}'")


def _extract_pdf_path(metadata: dict[str, Any]) -> str:
    if isinstance(metadata.get("pdf_path"), str) and metadata["pdf_path"].strip():
        return metadata["pdf_path"].strip()

    paper_obj = metadata.get("paper")
    if isinstance(paper_obj, dict):
        candidate = paper_obj.get("pdf_path")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    raise HTTPException(status_code=422, detail="pdf_path missing in metadata")


def _first_match_bbox(pdf_path: Path, claim_text: str) -> tuple[int, list[float]]:
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        candidates = _candidate_queries(claim_text)
        for page_index in range(len(doc)):
            page = doc[page_index]

            for query in candidates:
                hits = page.search_for(query)
                if hits:
                    rect = hits[0]
                    return page_index + 1, [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]

        # Fuzzy fallback: pick the page with highest token overlap and try phrase anchors.
        claim_tokens = _tokenize(claim_text)
        if claim_tokens:
            best_index = -1
            best_score = 0.0
            for page_index in range(len(doc)):
                page_text = doc[page_index].get_text("text")
                page_tokens = _tokenize(page_text)
                if not page_tokens:
                    continue
                overlap = len(claim_tokens.intersection(page_tokens))
                score = overlap / max(1, len(claim_tokens))
                if score > best_score:
                    best_score = score
                    best_index = page_index

            if best_index >= 0 and best_score >= 0.25:
                best_page = doc[best_index]
                for anchor in _anchor_phrases(claim_text):
                    hits = best_page.search_for(anchor)
                    if hits:
                        rect = hits[0]
                        return best_index + 1, [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]

                # Last resort: return page bounds so frontend can at least jump to the right page.
                rect = best_page.rect
                return best_index + 1, [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
    finally:
        doc.close()

    raise HTTPException(status_code=404, detail="claim_text not found in PDF after fuzzy matching")


def _candidate_queries(claim_text: str) -> list[str]:
    compact = " ".join(claim_text.split()).strip()
    if not compact:
        return []

    variants = [
        compact,
        compact.strip('"\''),
        compact[:220],
    ]
    sentence_parts = [part.strip() for part in re.split(r"[.;]\s+", compact) if part.strip()]
    variants.extend(sentence_parts[:3])

    for anchor in _anchor_phrases(compact):
        variants.append(anchor)

    deduped: list[str] = []
    for item in variants:
        normalized = " ".join(item.split()).strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped[:20]


def _anchor_phrases(text: str) -> list[str]:
    words = [word for word in re.findall(r"[A-Za-z0-9\-]+", text) if len(word) >= 3]
    if len(words) < 5:
        return [" ".join(words)] if words else []

    anchors: list[str] = []
    for window in (10, 8, 6):
        if len(words) < window:
            continue
        for start in range(0, min(4, len(words) - window + 1)):
            anchors.append(" ".join(words[start : start + window]))

    deduped: list[str] = []
    for anchor in anchors:
        if anchor not in deduped:
            deduped.append(anchor)
    return deduped[:12]


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{4,}", text.lower())}


@router.post("/citation", response_model=CitationResponse)
def citation_jump(payload: CitationRequest, request: Request) -> CitationResponse:
    resolved_paper_id = _resolve_paper_id(payload.paper_id)
    metadata_path = _resolve_metadata_path(resolved_paper_id)

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid metadata JSON: {metadata_path}") from exc

    pdf_path_str = _extract_pdf_path(metadata)
    pdf_path = Path(pdf_path_str)
    if not pdf_path.is_absolute():
        pdf_path = ROOT_DIR / pdf_path

    page_number, bbox = _first_match_bbox(pdf_path, payload.claim_text)

    return CitationResponse(
        paper_id=resolved_paper_id,
        paper_display_name=_paper_display_name(metadata=metadata, fallback_id=resolved_paper_id),
        claim_text=payload.claim_text,
        pdf_path=str(pdf_path),
        pdf_url=_build_pdf_url(pdf_path=pdf_path, request=request),
        page_number=page_number,
        bbox=bbox,
    )


@router.get("/citation/open")
def citation_open(
    request: Request,
    paper_id: str = Query(..., min_length=1),
    claim_text: str = Query(..., min_length=1),
) -> RedirectResponse:
    citation = citation_jump(CitationRequest(paper_id=paper_id, claim_text=claim_text), request)
    if not citation.pdf_url:
        raise HTTPException(status_code=404, detail="Unable to resolve a PDF URL for this paper")

    return RedirectResponse(url=f"{citation.pdf_url}#page={citation.page_number}", status_code=307)


def _build_pdf_url(pdf_path: Path, request: Request) -> str:
    pdf_root = ROOT_DIR / "data" / "pdf"
    try:
        relative = pdf_path.resolve().relative_to(pdf_root.resolve())
    except Exception:
        return ""

    relative_url_path = quote(relative.as_posix())
    return str(request.base_url).rstrip("/") + "/pdf/" + relative_url_path


def _resolve_paper_id(identifier: str) -> str:
    candidate = identifier.strip()
    if not candidate:
        raise HTTPException(status_code=422, detail="Paper identifier cannot be empty.")

    direct = METADATA_DIR / f"{candidate}.json"
    if direct.exists():
        return candidate

    key = _normalize_lookup_key(candidate)
    matches: list[str] = []
    for path in sorted(METADATA_DIR.glob("*.json")):
        paper_id = path.stem
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        title = str(metadata.get("title", "")).strip()
        title_key = _normalize_lookup_key(title)

        if key == _normalize_lookup_key(paper_id) or (title_key and key == title_key):
            return paper_id
        if title_key and key in title_key:
            matches.append(paper_id)

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    if len(unique_matches) > 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "Paper identifier is ambiguous. Matched paper_ids: "
                + ", ".join(unique_matches[:8])
                + ". Please provide a specific paper_id."
            ),
        )

    raise HTTPException(
        status_code=404,
        detail=f"No paper matched identifier '{candidate}'. Use paper_id or an exact/unique title.",
    )


def _normalize_lookup_key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _paper_display_name(metadata: dict[str, Any], fallback_id: str) -> str:
    title = " ".join(str(metadata.get("title", "")).split())
    year = " ".join(str(metadata.get("year", "")).split())
    if title and year:
        return f"{title} ({year})"
    if title:
        return title
    return fallback_id
