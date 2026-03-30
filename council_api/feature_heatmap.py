from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from council_api.extraction import extract_from_pdf

router = APIRouter(tags=["feature-heatmap"])

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
METADATA_DIR = DATA_DIR / "metadata"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
POSITIVE_WORDS = ["improve", "better", "outperform", "increase", "gain"]
NEGATIVE_WORDS = ["worse", "fails", "decrease", "does not improve", "no improvement"]


class HeatmapRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=2)


@router.post("/feature/heatmap")
def contradiction_heatmap(payload: HeatmapRequest) -> dict:
    requested_ids = [paper_id.strip() for paper_id in payload.paper_ids if paper_id.strip()]
    if len(requested_ids) < 2:
        raise HTTPException(status_code=422, detail="Provide at least two paper_ids.")

    papers = [_load_extracted(paper_id) for paper_id in requested_ids]
    canonical_ids = [str(item.get("paper_id", "")).strip() for item in papers]
    display_names = [
        {"paper_id": str(item.get("paper_id", "")).strip(), "display_name": _paper_display_name(item)}
        for item in papers
    ]

    llm_matrix = _heatmap_with_groq(papers)
    if llm_matrix:
        return {
            "paper_ids": canonical_ids,
            "paper_display_names": display_names,
            "matrix": llm_matrix,
            "mode": "groq",
        }

    return {
        "paper_ids": canonical_ids,
        "paper_display_names": display_names,
        "matrix": _heatmap_fallback(papers),
        "mode": "heuristic",
    }


def _load_extracted(paper_id: str) -> dict:
    resolved_paper_id = _resolve_paper_id(paper_id)
    path = EXTRACTED_DIR / f"{resolved_paper_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    # Auto-extract on demand so heatmap works even when extract-all wasn't run.
    metadata_path = METADATA_DIR / f"{resolved_paper_id}.json"
    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Extracted JSON not found for {resolved_paper_id}, and metadata is missing. "
                "Run /crawl first or provide a valid paper identifier (id or title)."
            ),
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pdf_path = Path(metadata.get("pdf_path", ""))
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Extracted JSON not found for {paper_id}, and source PDF is missing. "
                "Run /crawl again or verify metadata/pdf paths."
            ),
        )

    try:
        extracted = extract_from_pdf(
            paper_id=resolved_paper_id,
            pdf_path=pdf_path,
            output_dir=EXTRACTED_DIR,
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Failed to auto-extract {resolved_paper_id} for heatmap: {exc}",
        ) from exc

    return extracted


def _resolve_paper_id(identifier: str) -> str:
    candidate = identifier.strip()
    if not candidate:
        raise HTTPException(status_code=422, detail="Paper identifier cannot be empty.")

    path = EXTRACTED_DIR / f"{candidate}.json"
    if path.exists():
        return candidate

    metadata_path = METADATA_DIR / f"{candidate}.json"
    if metadata_path.exists():
        return candidate

    key = _normalize_lookup_key(candidate)
    matches: list[str] = []
    for item in _all_paper_summaries():
        paper_id = str(item.get("paper_id", "")).strip()
        title = str(item.get("title", "")).strip()
        normalized_title = _normalize_lookup_key(title)
        if not paper_id:
            continue
        if key == _normalize_lookup_key(paper_id) or (normalized_title and key == normalized_title):
            return paper_id
        if normalized_title and key in normalized_title:
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


def _all_paper_summaries() -> list[dict]:
    summaries: list[dict] = []
    for path in sorted(METADATA_DIR.glob("*.json")):
        paper_id = path.stem
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summaries.append({"paper_id": paper_id, "title": metadata.get("title", "")})
    return summaries


def _normalize_lookup_key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _paper_display_name(paper: dict) -> str:
    title = " ".join(str(paper.get("title", "")).split())
    year = " ".join(str(paper.get("year", "")).split())
    if title and year:
        return f"{title} ({year})"
    if title:
        return title
    return str(paper.get("paper_id", "Unknown Paper"))


def _heatmap_with_groq(papers: list[dict]) -> list[list[dict]]:
    keys = _groq_api_keys()
    if not keys:
        return []

    prompt_payload = []
    for item in papers:
        prompt_payload.append(
            {
                "paper_id": item.get("paper_id", ""),
                "title": item.get("title", ""),
                "claims": _short_claims(item),
            }
        )

    prompt = (
        "Given these papers and extracted claims, identify semantic contradictions between each pair. "
        "Return JSON only with schema: "
        "{\"matrix\": [[{\"from\": string, \"to\": string, \"contradicts\": boolean, "
        "\"contradictions\": [string]}]]}. "
        "Matrix must be NxN in the same paper order. Diagonal must be contradicts=false and empty contradictions."
        " Keep contradiction texts brief and grounded in the provided claims.\n"
        f"Papers: {json.dumps(prompt_payload, ensure_ascii=True)}"
    )

    body = {
        "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "temperature": 0,
        "max_tokens": 1800,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    }

    with httpx.Client(timeout=35.0) as client:
        for api_key in keys:
            try:
                response = client.post(
                    GROQ_URL,
                    headers={
                        "authorization": f"Bearer {api_key}",
                        "content-type": "application/json",
                    },
                    json=body,
                )
                if response.status_code >= 400:
                    continue
                data = response.json()
                content = _extract_chat_content(data)
                parsed = _parse_json_object(content)
                if not parsed:
                    continue
                matrix = parsed.get("matrix", [])
                return _sanitize_matrix(matrix, papers)
            except Exception:  # noqa: BLE001
                continue

    return []


def _heatmap_fallback(papers: list[dict]) -> list[list[dict]]:
    matrix: list[list[dict]] = []
    for left in papers:
        row: list[dict] = []
        for right in papers:
            if left.get("paper_id") == right.get("paper_id"):
                row.append(
                    {
                        "from": left.get("paper_id", ""),
                        "to": right.get("paper_id", ""),
                        "contradicts": False,
                        "contradictions": [],
                    }
                )
                continue

            contradictions = _pairwise_contradictions(left, right)
            row.append(
                {
                    "from": left.get("paper_id", ""),
                    "to": right.get("paper_id", ""),
                    "contradicts": len(contradictions) > 0,
                    "contradictions": contradictions[:5],
                }
            )
        matrix.append(row)

    return matrix


def _pairwise_contradictions(left: dict, right: dict) -> list[str]:
    matches: list[str] = []
    for left_claim in _short_claims(left, max_items=8):
        left_low = left_claim.lower()
        left_positive = any(word in left_low for word in POSITIVE_WORDS)
        left_negative = any(word in left_low for word in NEGATIVE_WORDS)
        if not (left_positive or left_negative):
            continue

        for right_claim in _short_claims(right, max_items=8):
            right_low = right_claim.lower()
            if not _claims_are_related(left_low, right_low):
                continue

            right_positive = any(word in right_low for word in POSITIVE_WORDS)
            right_negative = any(word in right_low for word in NEGATIVE_WORDS)
            if (left_positive and right_negative) or (left_negative and right_positive):
                matches.append(f"{left_claim} <-> {right_claim}")
                if len(matches) >= 5:
                    return matches

    return matches


def _short_claims(payload: dict, max_items: int = 10, max_chars: int = 240) -> list[str]:
    claims: list[str] = []
    for claim in payload.get("claims", []):
        text = " ".join(str(claim).split())
        if not text:
            continue
        claims.append(text[:max_chars])
        if len(claims) >= max_items:
            break
    return claims


def _sanitize_matrix(raw: list, papers: list[dict]) -> list[list[dict]]:
    ids = [str(item.get("paper_id", "")) for item in papers]
    if not isinstance(raw, list) or len(raw) != len(ids):
        return []

    matrix: list[list[dict]] = []
    for row_index, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != len(ids):
            return []
        clean_row: list[dict] = []
        for col_index, cell in enumerate(row):
            if not isinstance(cell, dict):
                return []
            contradictions = cell.get("contradictions", [])
            if not isinstance(contradictions, list):
                contradictions = []
            clean_row.append(
                {
                    "from": ids[row_index],
                    "to": ids[col_index],
                    "contradicts": bool(cell.get("contradicts", False)) if row_index != col_index else False,
                    "contradictions": [" ".join(str(item).split())[:300] for item in contradictions][:6]
                    if row_index != col_index
                    else [],
                }
            )
        matrix.append(clean_row)

    return matrix


def _groq_api_keys() -> list[str]:
    pooled = os.getenv("GROQ_API_KEYS", "")
    keys = [item.strip() for item in pooled.split(",") if item.strip()]

    single = os.getenv("GROQ_API_KEY", "").strip()
    if single and single not in keys:
        keys.append(single)

    return keys


def _extract_chat_content(payload: dict) -> str:
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        return ""

    content = message.get("content", "")
    return content if isinstance(content, str) else ""


def _parse_json_object(text: str) -> dict | None:
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None


def _claims_are_related(left: str, right: str) -> bool:
    left_tokens = set(re.findall(r"[a-z]{5,}", left))
    right_tokens = set(re.findall(r"[a-z]{5,}", right))
    return len(left_tokens.intersection(right_tokens)) > 0
