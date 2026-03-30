from __future__ import annotations

import json
import os
import re
from itertools import combinations
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from council_api.extraction import extract_from_pdf

router = APIRouter(tags=["feature-debate"])

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
METADATA_DIR = DATA_DIR / "metadata"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

DEBATE_AXES = [
    "problem_framing",
    "literature_review",
    "methodology",
    "data_dataset",
    "results_metrics",
    "reproducibility",
    "limitations",
    "ethical_impact",
    "novelty",
    "practical_applicability",
]

AXIS_DESCRIPTIONS = {
    "problem_framing": "Problem Definition & Relevance",
    "literature_review": "Literature Coverage & Positioning",
    "methodology": "Methodological Validity",
    "data_dataset": "Data Quality & Credibility",
    "results_metrics": "Results & Metrics Appropriateness",
    "reproducibility": "Reproducibility & Transparency",
    "limitations": "Honest Limitations Statement",
    "ethical_impact": "Ethical & Social Considerations",
    "novelty": "Novelty vs Incremental Work",
    "practical_applicability": "Real-World Deployment Potential",
}


class DebateRequest(BaseModel):
    paper_id_A: str = Field(default="")
    paper_id_B: str = Field(default="")
    paper_ids: list[str] = Field(default_factory=list)
    include_live_debate: bool = Field(default=True)


@router.post("/feature/debate")
async def live_debate(payload: DebateRequest) -> StreamingResponse:
    paper_a = _load_extracted(payload.paper_id_A.strip())
    paper_b = _load_extracted(payload.paper_id_B.strip())

    prompt = _build_debate_prompt(paper_a=paper_a, paper_b=paper_b)

    async def event_stream() -> AsyncIterator[str]:
        async for chunk in _stream_groq_debate(prompt):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/feature/structured-debate")
def structured_debate(payload: DebateRequest) -> dict:
    """Analyzes either a pair or a set of papers with pairwise debate outputs."""
    requested_ids = [paper_id.strip() for paper_id in payload.paper_ids if paper_id.strip()]
    if len(requested_ids) >= 2:
        return _structured_debate_multi(requested_ids, include_live_debate=payload.include_live_debate)

    if not payload.paper_id_A.strip() or not payload.paper_id_B.strip():
        raise HTTPException(
            status_code=422,
            detail="Provide paper_id_A and paper_id_B, or provide paper_ids with at least two items.",
        )

    paper_a = _load_extracted(payload.paper_id_A.strip())
    paper_b = _load_extracted(payload.paper_id_B.strip())
    result = _structured_debate_pair(
        paper_a=paper_a,
        paper_b=paper_b,
        include_live_debate=payload.include_live_debate,
    )
    _save_debate_result(result)
    return result


@router.post("/feature/structured-debate-multi")
def structured_debate_multi(payload: DebateRequest) -> dict:
    """Explicit multi-paper endpoint for pairwise debate analysis."""
    requested_ids = [paper_id.strip() for paper_id in payload.paper_ids if paper_id.strip()]
    if len(requested_ids) < 2:
        requested_ids = [payload.paper_id_A.strip(), payload.paper_id_B.strip()]
    if len(requested_ids) < 2 or not requested_ids[0] or not requested_ids[1]:
        raise HTTPException(status_code=422, detail="Provide at least two paper_ids.")
    return _structured_debate_multi(requested_ids, include_live_debate=payload.include_live_debate)


@router.get("/feature/debates")
def list_debates() -> list:
    """List all saved debate results."""
    debates_dir = DATA_DIR / "debates"
    if not debates_dir.exists():
        return []

    debates = []
    for debate_file in sorted(debates_dir.glob("*.json")):
        try:
            debate_data = json.loads(debate_file.read_text(encoding="utf-8"))
            debates.append({
                "filename": debate_file.name,
                "paper_A": debate_data.get("paper_A", {}),
                "paper_B": debate_data.get("paper_B", {}),
                "winner": debate_data.get("verdict_card", {}).get("winner"),
            })
        except Exception:
            pass

    return debates


@router.get("/feature/debates/{debate_id}")
def get_debate(debate_id: str) -> dict:
    """Retrieve a saved debate result."""
    debate_path = DATA_DIR / "debates" / f"{debate_id}.json"
    if not debate_path.exists():
        raise HTTPException(status_code=404, detail=f"Debate {debate_id} not found.")
    return json.loads(debate_path.read_text(encoding="utf-8"))


def _load_extracted(paper_id: str) -> dict:
    resolved_paper_id = _resolve_paper_id(paper_id)
    path = EXTRACTED_DIR / f"{resolved_paper_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    # Auto-extract on demand to make /feature/debate resilient when extract-all
    # has not been run yet.
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
                f"Extracted JSON not found for {resolved_paper_id}, and source PDF is missing. "
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
            detail=f"Failed to auto-extract {resolved_paper_id} for debate: {exc}",
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
    if not key:
        raise HTTPException(status_code=422, detail="Paper identifier cannot be blank.")

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


def _build_debate_prompt(paper_a: dict, paper_b: dict) -> str:
    claims_a = _short_claims(paper_a)
    claims_b = _short_claims(paper_b)

    return (
        "You are simulating a strict two-agent scientific debate.\n"
        "Agent A defends Paper A claims. Agent B attacks Paper A claims using Paper B.\n"
        "Output as plain text with alternating turns: A1, B1, A2, B2, A3, B3, then Verdict.\n"
        "Each turn must be concise, evidence-grounded, and cite claim snippets in quotes.\n\n"
        f"Paper A (paper_id={paper_a.get('paper_id', '')}, title={paper_a.get('title', '')}) claims:\n"
        + "\n".join(f"- {item}" for item in claims_a)
        + "\n\n"
        + f"Paper B (paper_id={paper_b.get('paper_id', '')}, title={paper_b.get('title', '')}) claims:\n"
        + "\n".join(f"- {item}" for item in claims_b)
        + "\n"
    )


def _short_claims(payload: dict, max_items: int = 8, max_chars: int = 260) -> list[str]:
    claims: list[str] = []
    for claim in payload.get("claims", []):
        text = " ".join(str(claim).split())
        if not text:
            continue
        claims.append(text[:max_chars])
        if len(claims) >= max_items:
            break
    return claims or ["No extracted claims available."]


async def _stream_groq_debate(prompt: str) -> AsyncIterator[str]:
    keys = _groq_api_keys()
    if not keys:
        yield _sse_data({"error": "No Groq API key found in environment."})
        return

    payload = {
        "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "temperature": 0.2,
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert scientific debate assistant. Stay factual and concise.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        for api_key in keys:
            try:
                async with client.stream(
                    "POST",
                    GROQ_URL,
                    headers={
                        "authorization": f"Bearer {api_key}",
                        "content-type": "application/json",
                    },
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        continue

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue

                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            yield "event: done\ndata: [DONE]\n\n"
                            return

                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta:
                            yield _sse_data({"token": delta})

                    yield "event: done\ndata: [DONE]\n\n"
                    return
            except Exception:  # noqa: BLE001
                continue

    yield _sse_data({"error": "Failed to stream debate from Groq."})


def _groq_api_keys() -> list[str]:
    pooled = os.getenv("GROQ_API_KEYS", "")
    keys = [item.strip() for item in pooled.split(",") if item.strip()]

    single = os.getenv("GROQ_API_KEY", "").strip()
    if single and single not in keys:
        keys.append(single)

    return keys


def _sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=True)}\\n\\n"


def _structured_debate_pair(paper_a: dict, paper_b: dict, include_live_debate: bool = True) -> dict:
    axes_analysis = {}
    verdict_scores = {"A": 0, "B": 0}

    for axis in DEBATE_AXES:
        analysis = _analyze_axis(axis, paper_a, paper_b)
        axes_analysis[axis] = analysis

        if analysis["winner"] == "A":
            verdict_scores["A"] += int(analysis.get("score_award", 0))
        elif analysis["winner"] == "B":
            verdict_scores["B"] += int(analysis.get("score_award", 0))

    contradiction_report = _pair_contradiction_report(paper_a=paper_a, paper_b=paper_b, axes_analysis=axes_analysis)
    live_debate = _generate_pair_debate_text(
        paper_a=paper_a,
        paper_b=paper_b,
        contradiction_report=contradiction_report,
    ) if include_live_debate else {"mode": "disabled", "text": ""}
    verdict_card = _generate_verdict_card(paper_a, paper_b, axes_analysis, verdict_scores)

    return {
        "paper_A": {
            "id": paper_a.get("paper_id", ""),
            "title": paper_a.get("title", ""),
            "display_name": _paper_display_name(paper_a),
        },
        "paper_B": {
            "id": paper_b.get("paper_id", ""),
            "title": paper_b.get("title", ""),
            "display_name": _paper_display_name(paper_b),
        },
        "axes_analysis": axes_analysis,
        "contradiction_report": contradiction_report,
        "live_debate": live_debate,
        "verdict_card": verdict_card,
    }


def _structured_debate_multi(paper_ids: list[str], include_live_debate: bool = True) -> dict:
    unique_ids: list[str] = []
    for paper_id in paper_ids:
        if paper_id and paper_id not in unique_ids:
            unique_ids.append(paper_id)

    if len(unique_ids) < 2:
        raise HTTPException(status_code=422, detail="Provide at least two unique paper_ids.")

    papers = {}
    canonical_ids: list[str] = []
    for identifier in unique_ids:
        paper = _load_extracted(identifier)
        paper_id = str(paper.get("paper_id", "")).strip() or identifier
        if paper_id in papers:
            continue
        papers[paper_id] = paper
        canonical_ids.append(paper_id)

    unique_ids = canonical_ids

    pair_debates: list[dict] = []
    contradiction_total = 0
    for left_id, right_id in combinations(unique_ids, 2):
        pair_result = _structured_debate_pair(
            papers[left_id],
            papers[right_id],
            include_live_debate=include_live_debate,
        )
        pair_result["pair_key"] = f"{left_id}__vs__{right_id}"
        contradiction_total += int(pair_result.get("contradiction_report", {}).get("contradiction_count", 0))
        pair_debates.append(pair_result)
        _save_debate_result(pair_result)

    strongest_pairs = sorted(
        pair_debates,
        key=lambda item: int(item.get("contradiction_report", {}).get("contradiction_count", 0)),
        reverse=True,
    )[:5]

    return {
        "paper_ids": unique_ids,
        "paper_display_names": [
            {"paper_id": paper_id, "display_name": _paper_display_name(papers[paper_id])}
            for paper_id in unique_ids
        ],
        "pair_count": len(pair_debates),
        "pairs_evaluated": [item.get("pair_key", "") for item in pair_debates],
        "pair_debates": pair_debates,
        "summary": {
            "papers_considered": len(unique_ids),
            "total_pairwise_contradictions": contradiction_total,
            "live_debate_mode": "groq" if any(
                item.get("live_debate", {}).get("mode") == "groq" for item in pair_debates
            ) else "fallback",
            "most_conflicting_pairs": [
                {
                    "pair_key": item.get("pair_key", ""),
                    "paper_A": item.get("paper_A", {}).get("id", ""),
                    "paper_B": item.get("paper_B", {}).get("id", ""),
                    "paper_A_name": item.get("paper_A", {}).get("display_name", ""),
                    "paper_B_name": item.get("paper_B", {}).get("display_name", ""),
                    "contradiction_count": item.get("contradiction_report", {}).get("contradiction_count", 0),
                }
                for item in strongest_pairs
            ],
        },
    }


def _analyze_axis(axis: str, paper_a: dict, paper_b: dict) -> dict:
    """Analyze a single axis for both papers using heuristic scoring."""
    axis_desc = AXIS_DESCRIPTIONS.get(axis, axis)

    if axis == "problem_framing":
        score_a, reasoning_a = _score_problem_framing(paper_a)
        score_b, reasoning_b = _score_problem_framing(paper_b)
    elif axis == "literature_review":
        score_a, reasoning_a = _score_literature_review(paper_a)
        score_b, reasoning_b = _score_literature_review(paper_b)
    elif axis == "methodology":
        score_a, reasoning_a = _score_methodology(paper_a)
        score_b, reasoning_b = _score_methodology(paper_b)
    elif axis == "data_dataset":
        score_a, reasoning_a = _score_data_dataset(paper_a)
        score_b, reasoning_b = _score_data_dataset(paper_b)
    elif axis == "results_metrics":
        score_a, reasoning_a = _score_results_metrics(paper_a)
        score_b, reasoning_b = _score_results_metrics(paper_b)
    elif axis == "reproducibility":
        score_a, reasoning_a = _score_reproducibility(paper_a)
        score_b, reasoning_b = _score_reproducibility(paper_b)
    elif axis == "limitations":
        score_a, reasoning_a = _score_limitations(paper_a)
        score_b, reasoning_b = _score_limitations(paper_b)
    elif axis == "ethical_impact":
        score_a, reasoning_a = _score_ethical_impact(paper_a)
        score_b, reasoning_b = _score_ethical_impact(paper_b)
    elif axis == "novelty":
        score_a, reasoning_a = _score_novelty(paper_a)
        score_b, reasoning_b = _score_novelty(paper_b)
    elif axis == "practical_applicability":
        score_a, reasoning_a = _score_practical_applicability(paper_a)
        score_b, reasoning_b = _score_practical_applicability(paper_b)
    else:
        score_a, reasoning_a = 5, "Unknown axis."
        score_b, reasoning_b = 5, "Unknown axis."

    # Determine winner
    if score_a > score_b:
        winner = "A"
        diff = score_a - score_b
    elif score_b > score_a:
        winner = "B"
        diff = score_b - score_a
    else:
        winner = "Tie"
        diff = 0

    evidence_a = _claims_for_axis(paper_a, axis)
    evidence_b = _claims_for_axis(paper_b, axis)
    logical_reasoning = _build_logical_reasoning(
        axis=axis,
        paper_a=paper_a,
        paper_b=paper_b,
        evidence_a=evidence_a,
        evidence_b=evidence_b,
        winner=winner,
        score_a=score_a,
        score_b=score_b,
    )

    return {
        "axis": axis,
        "description": axis_desc,
        "paper_A": {"score": score_a, "reasoning": reasoning_a},
        "paper_B": {"score": score_b, "reasoning": reasoning_b},
        "winner": winner,
        "score_diff": diff,
        "score_award": max(score_a, score_b),
        "evidence": {
            "paper_A_claims": evidence_a,
            "paper_B_claims": evidence_b,
            "paper_A_claim_cards": _claim_cards_for_axis(paper_a, evidence_a),
            "paper_B_claim_cards": _claim_cards_for_axis(paper_b, evidence_b),
        },
        "logical_reasoning": logical_reasoning,
    }


def _claims_for_axis(paper: dict, axis: str, max_items: int = 3) -> list[str]:
    claims = _short_claims(paper, max_items=12)
    axis_keywords = {
        "problem_framing": ["problem", "challenge", "gap", "objective"],
        "literature_review": ["prior", "previous", "related", "literature"],
        "methodology": ["method", "architecture", "algorithm", "approach"],
        "data_dataset": ["dataset", "benchmark", "corpus", "data"],
        "results_metrics": ["result", "accuracy", "f1", "auc", "baseline", "improve"],
        "reproducibility": ["code", "reproduce", "configuration", "hyperparameter"],
        "limitations": ["limitation", "future work", "does not", "challenge"],
        "ethical_impact": ["bias", "fairness", "privacy", "ethical", "impact"],
        "novelty": ["novel", "new", "innovative", "first"],
        "practical_applicability": ["deployment", "real-world", "scalable", "practical"],
    }
    keywords = axis_keywords.get(axis, [])

    picked = [
        claim for claim in claims if any(keyword in claim.lower() for keyword in keywords)
    ]
    if not picked:
        picked = claims
    return picked[:max_items]


def _claim_cards_for_axis(paper: dict, claims: list[str]) -> list[dict]:
    paper_id = " ".join(str(paper.get("paper_id", "")).split())
    display_name = _paper_display_name(paper)
    pdf_url = _paper_pdf_url(paper_id) if paper_id else ""

    cards: list[dict] = []
    for claim in claims:
        claim_text = " ".join(str(claim).split())
        if not claim_text:
            continue

        cards.append(
            {
                "paper_id": paper_id,
                "paper_display_name": display_name,
                "claim_text": claim_text,
                "pdf_url": pdf_url,
                "citation_endpoint": "/feature/citation",
                "citation_open_url": _citation_open_url(paper_id=paper_id, claim_text=claim_text),
                "citation_request": {
                    "paper_id": paper_id,
                    "claim_text": claim_text,
                },
            }
        )

    return cards


def _citation_open_url(paper_id: str, claim_text: str) -> str:
    return (
        "/feature/citation/open?paper_id="
        + quote(paper_id, safe="")
        + "&claim_text="
        + quote(claim_text, safe="")
    )


def _paper_pdf_url(paper_id: str) -> str:
    metadata_path = METADATA_DIR / f"{paper_id}.json"
    if not metadata_path.exists():
        return ""

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    pdf_candidate = metadata.get("pdf_path", "")
    if not isinstance(pdf_candidate, str) or not pdf_candidate.strip():
        paper_obj = metadata.get("paper")
        if isinstance(paper_obj, dict):
            fallback_candidate = paper_obj.get("pdf_path", "")
            if isinstance(fallback_candidate, str):
                pdf_candidate = fallback_candidate

    if not isinstance(pdf_candidate, str) or not pdf_candidate.strip():
        return ""

    pdf_path = Path(pdf_candidate.strip())
    if not pdf_path.is_absolute():
        pdf_path = ROOT_DIR / pdf_path

    pdf_root = ROOT_DIR / "data" / "pdf"
    try:
        relative = pdf_path.resolve().relative_to(pdf_root.resolve())
    except Exception:
        return ""

    return "/pdf/" + quote(relative.as_posix())


def _build_logical_reasoning(
    axis: str,
    paper_a: dict,
    paper_b: dict,
    evidence_a: list[str],
    evidence_b: list[str],
    winner: str,
    score_a: int,
    score_b: int,
) -> dict:
    premise_a = evidence_a[0] if evidence_a else "No direct supporting claim extracted."
    premise_b = evidence_b[0] if evidence_b else "No direct supporting claim extracted."
    score_gap = abs(score_a - score_b)

    if winner == "A":
        inference = (
            f"On axis '{axis}', Paper A provides comparatively stronger evidence quality and relevance "
            f"than Paper B (score gap: {score_gap})."
        )
    elif winner == "B":
        inference = (
            f"On axis '{axis}', Paper B provides comparatively stronger evidence quality and relevance "
            f"than Paper A (score gap: {score_gap})."
        )
    else:
        inference = (
            f"On axis '{axis}', both papers present similar evidence strength, yielding a tie."
        )

    confidence = "high" if score_gap >= 3 else "medium" if score_gap >= 1 else "low"
    return {
        "premise_A": premise_a,
        "premise_B": premise_b,
        "inference": inference,
        "confidence": confidence,
    }


def _generate_pair_debate_text(paper_a: dict, paper_b: dict, contradiction_report: dict) -> dict:
    keys = _groq_api_keys()
    if not keys:
        return {
            "mode": "fallback",
            "text": _fallback_debate_text(paper_a, paper_b, contradiction_report),
        }

    prompt = _build_debate_prompt(paper_a=paper_a, paper_b=paper_b)
    contradiction_hint = json.dumps(
        contradiction_report.get("claim_level_contradictions", [])[:4],
        ensure_ascii=True,
    )
    user_prompt = (
        prompt
        + "\nUse the listed claim contradictions as key battleground points when relevant:\n"
        + contradiction_hint
        + "\n\nReturn concise structured text with sections: A1, B1, A2, B2, A3, B3, Verdict."
    )

    payload = {
        "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "temperature": 0.2,
        "max_tokens": 900,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert scientific debate assistant. Stay factual, compare evidence, "
                    "and avoid generic claims."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    }

    with httpx.Client(timeout=45.0) as client:
        for api_key in keys:
            try:
                response = client.post(
                    GROQ_URL,
                    headers={
                        "authorization": f"Bearer {api_key}",
                        "content-type": "application/json",
                    },
                    json=payload,
                )
                if response.status_code >= 400:
                    continue
                content = _extract_chat_content(response.json())
                text = "\n".join(line.rstrip() for line in content.splitlines() if line.strip())
                if text:
                    return {"mode": "groq", "text": text[:6000]}
            except Exception:
                continue

    return {
        "mode": "fallback",
        "text": _fallback_debate_text(paper_a, paper_b, contradiction_report),
    }


def _extract_chat_content(payload: dict) -> str:
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    return content if isinstance(content, str) else ""


def _fallback_debate_text(paper_a: dict, paper_b: dict, contradiction_report: dict) -> str:
    a_name = _paper_display_name(paper_a)
    b_name = _paper_display_name(paper_b)
    contradictions = contradiction_report.get("claim_level_contradictions", [])
    top_conflict = contradictions[0] if contradictions else {}
    a_claim = str(top_conflict.get("paper_A_claim", "No direct contradiction claim extracted."))
    b_claim = str(top_conflict.get("paper_B_claim", "No direct contradiction claim extracted."))

    return "\n".join(
        [
            f"A1: {a_name} argues that its evidence supports stronger outcomes in the target problem setting.",
            f"B1: {b_name} challenges that conclusion by citing alternative framing and assumptions.",
            f"A2: Key claim from Paper A: \"{a_claim[:260]}\".",
            f"B2: Counter-claim from Paper B: \"{b_claim[:260]}\".",
            "A3: Paper A maintains applicability based on its stated evaluation evidence.",
            "B3: Paper B argues that broader methodological context weakens Paper A's generalization.",
            "Verdict: Both papers contribute, but decisions should prioritize directly validated claims and unresolved contradictions.",
        ]
    )


def _pair_contradiction_report(paper_a: dict, paper_b: dict, axes_analysis: dict) -> dict:
    contradictions = _claim_level_contradictions(paper_a=paper_a, paper_b=paper_b)
    axis_tensions = [
        {
            "axis": analysis.get("axis", ""),
            "description": analysis.get("description", ""),
            "winner": analysis.get("winner", "Tie"),
            "score_diff": analysis.get("score_diff", 0),
            "logical_reasoning": analysis.get("logical_reasoning", {}),
        }
        for analysis in axes_analysis.values()
        if analysis.get("score_diff", 0) >= 2
    ]

    return {
        "contradiction_count": len(contradictions),
        "claim_level_contradictions": contradictions,
        "axis_tensions": axis_tensions,
    }


def _claim_level_contradictions(paper_a: dict, paper_b: dict, max_items: int = 8) -> list[dict]:
    positives = ["improve", "better", "outperform", "increase", "gain", "higher"]
    negatives = ["worse", "fails", "decrease", "no improvement", "lower", "does not"]
    matches: list[dict] = []

    for claim_a in _short_claims(paper_a, max_items=10, max_chars=320):
        low_a = claim_a.lower()
        a_positive = any(word in low_a for word in positives)
        a_negative = any(word in low_a for word in negatives)
        if not (a_positive or a_negative):
            continue

        tokens_a = set(re.findall(r"[a-z]{5,}", low_a))
        for claim_b in _short_claims(paper_b, max_items=10, max_chars=320):
            low_b = claim_b.lower()
            tokens_b = set(re.findall(r"[a-z]{5,}", low_b))
            overlap = sorted(tokens_a.intersection(tokens_b))
            if not overlap:
                continue

            b_positive = any(word in low_b for word in positives)
            b_negative = any(word in low_b for word in negatives)
            if not ((a_positive and b_negative) or (a_negative and b_positive)):
                continue

            matches.append(
                {
                    "paper_A_claim": claim_a,
                    "paper_B_claim": claim_b,
                    "logical_reasoning": (
                        "Claims discuss overlapping concepts but assert opposite directional outcomes. "
                        f"Shared terms: {', '.join(overlap[:6])}."
                    ),
                    "confidence": "high" if len(overlap) >= 3 else "medium",
                }
            )
            if len(matches) >= max_items:
                return matches

    return matches


def _score_problem_framing(paper: dict) -> tuple[int, str]:
    """Score clarity of problem statement and relevance."""
    abstract = paper.get("sections", {}).get("abstract", "")
    intro = paper.get("sections", {}).get("introduction", "")

    relevance_keywords = ["novel", "important", "critical", "significant", "challenge", "gap"]
    relevance_hits = sum(
        1 for kw in relevance_keywords if kw.lower() in (abstract + intro).lower()
    )

    clarity_score = min(3, relevance_hits)
    problem_score = 2 if len(abstract) > 300 else 1

    score = 1 + clarity_score + problem_score
    reasoning = (
        f"Clear problem statement (score: {clarity_score}/3) with {relevance_hits} relevance signals. "
        f"Abstract length: {len(abstract)} chars."
    )

    return min(10, score), reasoning


def _score_literature_review(paper: dict) -> tuple[int, str]:
    """Score coverage of prior work and positioning."""
    references = paper.get("references", [])
    intro = paper.get("sections", {}).get("introduction", "")

    ref_count = len(references)
    coverage_score = min(4, ref_count // 10)

    positioning_keywords = [
        "improve upon",
        "extends",
        "build upon",
        "unlike",
        "previous work",
        "gap",
    ]
    positioning_hits = sum(1 for kw in positioning_keywords if kw.lower() in intro.lower())

    score = 2 + coverage_score + min(3, positioning_hits)
    reasoning = (
        f"Referenced {ref_count} sources with {positioning_hits} positioning statements. "
        f"Coverage score: {coverage_score}/4."
    )

    return min(10, score), reasoning


def _score_methodology(paper: dict) -> tuple[int, str]:
    """Score methodological validity and appropriateness."""
    methods = paper.get("methods", [])
    methodology_section = paper.get("sections", {}).get("methodology", "")

    method_diversity = len(set(methods))
    detail_score = 2 if len(methodology_section) > 500 else 1

    valid_keywords = ["algorithm", "architecture", "procedure", "implemented", "parameter"]
    validity_hits = sum(1 for kw in valid_keywords if kw.lower() in methodology_section.lower())

    score = 2 + min(3, method_diversity) + detail_score + min(3, validity_hits)
    reasoning = (
        f"Method diversity: {method_diversity} unique methods. "
        f"Methodology detail score: {detail_score}/2. Validity signals: {validity_hits}."
    )

    return min(10, score), reasoning


def _score_data_dataset(paper: dict) -> tuple[int, str]:
    """Score data quality, credibility, and diversity."""
    datasets = paper.get("datasets", [])
    results_section = paper.get("sections", {}).get("results", "")

    dataset_count = len(set(datasets))
    credibility_score = min(3, dataset_count)

    trusted_datasets = [
        "imagenet",
        "cifar",
        "mnist",
        "squad",
        "coco",
        "wikitext",
        "openwebtext",
    ]
    trusted_hits = sum(
        1 for ds in datasets if any(trusted in ds.lower() for trusted in trusted_datasets)
    )

    scale_keywords = ["large", "comprehensive", "extensive", "benchmark", "diverse"]
    scale_signals = sum(1 for kw in scale_keywords if kw.lower() in results_section.lower())

    score = 2 + credibility_score + min(2, trusted_hits) + min(2, scale_signals)
    reasoning = (
        f"Dataset diversity: {dataset_count} unique datasets ({trusted_hits} trusted). "
        f"Scale signals: {scale_signals}."
    )

    return min(10, score), reasoning


def _score_results_metrics(paper: dict) -> tuple[int, str]:
    """Score appropriateness and rigor of metrics and baselines."""
    results_section = paper.get("sections", {}).get("results", "")
    claims = paper.get("claims", [])

    metric_keywords = ["accuracy", "f1", "auc", "precision", "recall", "improvement", "baseline"]
    metric_count = sum(1 for kw in metric_keywords if kw.lower() in results_section.lower())
    metric_score = min(3, metric_count // 2)

    comparison_keywords = ["compared", "versus", "baseline", "sota", "state-of-the-art"]
    comparison_count = sum(1 for kw in comparison_keywords if kw.lower() in results_section.lower())
    comparison_score = min(2, comparison_count)

    improvement_claims = sum(1 for claim in claims if "improve" in claim.lower())
    evidence_score = min(2, improvement_claims)

    score = 2 + metric_score + comparison_score + evidence_score
    reasoning = (
        f"Metric rigor: {metric_count} signals, {metric_score}/3 score. "
        f"Comparison signals: {comparison_count}. Evidence-backed claims: {improvement_claims}."
    )

    return min(10, score), reasoning


def _score_reproducibility(paper: dict) -> tuple[int, str]:
    """Score code/data availability and clarity of procedures."""
    sections = paper.get("sections", {})
    combined = " ".join(sections.values())

    repo_keywords = ["github", "pytorch", "tensorflow", "code available", "supplementary", "appendix"]
    repo_signals = sum(1 for kw in repo_keywords if kw.lower() in combined.lower())

    methodology_length = len(sections.get("methodology", ""))
    detail_score = 2 if methodology_length > 400 else 1

    param_keywords = ["learning rate", "batch size", "epochs", "hyperparameter", "configuration"]
    param_signals = sum(1 for kw in param_keywords if kw.lower() in combined.lower())

    score = 2 + min(3, repo_signals) + detail_score + min(2, param_signals)
    reasoning = (
        f"Code/resource signals: {repo_signals}. "
        f"Implementation detail score: {detail_score}/2. Parameter transparency: {param_signals}."
    )

    return min(10, score), reasoning


def _score_limitations(paper: dict) -> tuple[int, str]:
    """Score honesty and depth of stated limitations."""
    sections = paper.get("sections", {})
    combined = " ".join(sections.values())

    limitation_keywords = [
        "limitation",
        "limitation",
        "challenge",
        "difficulty",
        "unable to",
        "future work",
        "does not generalize",
    ]
    limitation_signals = sum(1 for kw in limitation_keywords if kw.lower() in combined.lower())
    limitation_score = min(3, limitation_signals)

    conclusion = sections.get("conclusion", "")
    conclusion_length = len(conclusion)
    conclusion_score = 2 if conclusion_length > 300 else 1

    negative_keywords = ["fail", "insufficient", "does not", "lack", "insufficient"]
    honesty_signals = sum(1 for kw in negative_keywords if kw.lower() in conclusion.lower())

    score = 2 + limitation_score + conclusion_score + min(2, honesty_signals)
    reasoning = (
        f"Limitation depth: {limitation_signals} signals. "
        f"Conclusion quality: {conclusion_score}/2. Honesty signals: {honesty_signals}."
    )

    return min(10, score), reasoning


def _score_ethical_impact(paper: dict) -> tuple[int, str]:
    """Score consideration of ethical and social impacts."""
    sections = paper.get("sections", {})
    combined = " ".join(sections.values())

    ethical_keywords = [
        "bias",
        "fairness",
        "ethical",
        "responsible",
        "privacy",
        "security",
        "misuse",
        "social impact",
    ]
    ethical_signals = sum(1 for kw in ethical_keywords if kw.lower() in combined.lower())

    application_keywords = ["application", "deployment", "real-world", "impact", "effect"]
    application_signals = sum(1 for kw in application_keywords if kw.lower() in combined.lower())

    score = 2 + min(4, ethical_signals) + min(2, application_signals)
    reasoning = (
        f"Ethical consideration signals: {ethical_signals}. "
        f"Real-world impact discussion: {application_signals}."
    )

    return min(10, score), reasoning


def _score_novelty(paper: dict) -> tuple[int, str]:
    """Score novelty vs incremental contributions."""
    claims = paper.get("claims", [])
    methods = paper.get("methods", [])
    methodology_section = paper.get("sections", {}).get("methodology", "")

    novelty_keywords = ["novel", "first", "new approach", "original", "innovative", "unique"]
    novelty_signals = sum(
        1 for claim in claims if any(kw.lower() in claim.lower() for kw in novelty_keywords)
    )

    incremental_keywords = ["extend", "improve", "enhance", "adaptation", "application"]
    incremental_signals = sum(
        1 for claim in claims if any(kw.lower() in claim.lower() for kw in incremental_keywords)
    )

    method_combinations = len(methods)
    combination_score = min(2, method_combinations - 1) if method_combinations > 1 else 1

    score = 3 + min(3, novelty_signals) + (0 if incremental_signals > novelty_signals else 1) + (
        combination_score
    )
    reasoning = (
        f"Novelty claims: {novelty_signals}. Incremental signals: {incremental_signals}. "
        f"Method combinations: {method_combinations}."
    )

    return min(10, score), reasoning


def _score_practical_applicability(paper: dict) -> tuple[int, str]:
    """Score real-world deployment potential and scalability."""
    sections = paper.get("sections", {})
    combined = " ".join(sections.values())

    scalability_keywords = ["scalable", "efficient", "performance", "real-time", "large-scale", "production"]
    scalability_signals = sum(1 for kw in scalability_keywords if kw.lower() in combined.lower())

    deployment_keywords = ["deployment", "implementation", "system", "integrated", "practical", "case study"]
    deployment_signals = sum(1 for kw in deployment_keywords if kw.lower() in combined.lower())

    limitation_signals = sum(
        1 for kw in ["limited to", "only works with", "not applicable"] if kw.lower() in combined.lower()
    )

    score = 2 + min(3, scalability_signals) + min(3, deployment_signals) - min(2, limitation_signals)
    reasoning = (
        f"Scalability signals: {scalability_signals}. Deployment evidence: {deployment_signals}. "
        f"Applicability limitations: {limitation_signals}."
    )

    return min(10, max(1, score)), reasoning


def _save_debate_result(debate_result: dict) -> None:
    """Save debate result to disk with timestamp."""
    from datetime import datetime, timezone

    debates_dir = DATA_DIR / "debates"
    debates_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat().replace(":", "-").split(".")[0]
    paper_a_id = debate_result.get("paper_A", {}).get("id", "unknown_a")
    paper_b_id = debate_result.get("paper_B", {}).get("id", "unknown_b")
    filename = f"{timestamp}_{paper_a_id}_vs_{paper_b_id}.json"

    debate_path = debates_dir / filename
    debate_path.write_text(json.dumps(debate_result, ensure_ascii=True, indent=2), encoding="utf-8")


def _generate_verdict_card(paper_a: dict, paper_b: dict, axes_analysis: dict, verdict_scores: dict) -> dict:
    """Generate a comprehensive verdict card summarizing the debate."""
    paper_a_id = paper_a.get("paper_id", "Paper A")
    paper_b_id = paper_b.get("paper_id", "Paper B")
    paper_a_title = paper_a.get("title", "Untitled")
    paper_b_title = paper_b.get("title", "Untitled")
    paper_a_name = _paper_display_name(paper_a)
    paper_b_name = _paper_display_name(paper_b)

    total_score_a = verdict_scores["A"]
    total_score_b = verdict_scores["B"]

    if total_score_a > total_score_b:
        overall_winner = "A"
        winner_name = paper_a_id
        winner_title = paper_a_title
        margin = total_score_a - total_score_b
    elif total_score_b > total_score_a:
        overall_winner = "B"
        winner_name = paper_b_id
        winner_title = paper_b_title
        margin = total_score_b - total_score_a
    else:
        overall_winner = "Tie"
        winner_name = "Both Papers"
        winner_title = "Equally competitive"
        margin = 0

    # Count wins per paper
    a_wins = sum(1 for analysis in axes_analysis.values() if analysis["winner"] == "A")
    b_wins = sum(1 for analysis in axes_analysis.values() if analysis["winner"] == "B")

    # Identify strengths and weaknesses
    a_strongest = max(
        (analysis for analysis in axes_analysis.values() if analysis["winner"] == "A"),
        key=lambda x: x["score_diff"],
        default=None,
    )
    b_strongest = max(
        (analysis for analysis in axes_analysis.values() if analysis["winner"] == "B"),
        key=lambda x: x["score_diff"],
        default=None,
    )

    verdict_narrative = _generate_verdict_narrative(
        overall_winner,
        paper_a_id,
        paper_a_title,
        paper_b_id,
        paper_b_title,
        a_wins,
        b_wins,
        margin,
        a_strongest,
        b_strongest,
    )

    return {
        "winner": overall_winner,
        "winner_id": winner_name,
        "winner_display_name": paper_a_name if overall_winner == "A" else paper_b_name if overall_winner == "B" else "Tie",
        "winner_title": winner_title,
        "total_score_A": total_score_a,
        "total_score_B": total_score_b,
        "score_margin": margin,
        "axes_won": {"A": a_wins, "B": b_wins, "tie": 10 - a_wins - b_wins},
        "paper_A": {
            "id": paper_a_id,
            "title": paper_a_title,
            "display_name": paper_a_name,
            "strongest_axis": a_strongest["description"] if a_strongest else "N/A",
        },
        "paper_B": {
            "id": paper_b_id,
            "title": paper_b_title,
            "display_name": paper_b_name,
            "strongest_axis": b_strongest["description"] if b_strongest else "N/A",
        },
        "narrative": verdict_narrative,
    }


def _paper_display_name(paper: dict) -> str:
    title = " ".join(str(paper.get("title", "")).split())
    year = " ".join(str(paper.get("year", "")).split())
    if title and year:
        return f"{title} ({year})"
    if title:
        return title
    return str(paper.get("paper_id", "Unknown Paper"))


def _generate_verdict_narrative(
    winner: str,
    paper_a_id: str,
    paper_a_title: str,
    paper_b_id: str,
    paper_b_title: str,
    a_wins: int,
    b_wins: int,
    margin: int,
    a_strongest: dict | None,
    b_strongest: dict | None,
) -> str:
    """Generate human-readable verdict narrative."""
    lines = []

    if winner == "A":
        lines.append(
            f"📊 VERDICT: {paper_a_id} dominates across {a_wins} of 10 axes. "
            f"({paper_a_title})"
        )
        lines.append(f"🏆 Winner by score margin: {margin} points")
        lines.append(
            f"💪 {paper_a_id}'s strongest axis: {a_strongest['description'] if a_strongest else 'Multiple'}"
        )
        if b_strongest:
            lines.append(
                f"📌 {paper_b_id}'s strongest showing: {b_strongest['description']}"
            )
    elif winner == "B":
        lines.append(
            f"📊 VERDICT: {paper_b_id} dominates across {b_wins} of 10 axes. "
            f"({paper_b_title})"
        )
        lines.append(f"🏆 Winner by score margin: {margin} points")
        lines.append(
            f"💪 {paper_b_id}'s strongest axis: {b_strongest['description'] if b_strongest else 'Multiple'}"
        )
        if a_strongest:
            lines.append(
                f"📌 {paper_a_id}'s strongest showing: {a_strongest['description']}"
            )
    else:
        lines.append("📊 VERDICT: Highly competitive papers with matched strengths.")
        lines.append(f"⚖️  {paper_a_id} and {paper_b_id} each excel in different areas.")
        if a_strongest:
            lines.append(f"• {paper_a_id} leads in {a_strongest['description']}")
        if b_strongest:
            lines.append(f"• {paper_b_id} leads in {b_strongest['description']}")

    lines.append("")
    lines.append("📈 Detailed breakdown by axis included in axes_analysis.")
    lines.append("✅ Use for literature review, grant decisions, or research direction.")

    return "\n".join(lines)
