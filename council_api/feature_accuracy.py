from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/feature", tags=["feature"])

ROOT_DIR = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = ROOT_DIR / "data" / "extracted"
POSITIVE_WORDS = ["improve", "better", "outperform", "increase", "gain"]
NEGATIVE_WORDS = ["worse", "fails", "decrease", "does not improve", "no improvement"]


class MostAccurateRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)
    question: str = Field(default="", description="Optional context question")


@router.post("/most-accurate")
def most_accurate_paper(payload: MostAccurateRequest) -> dict:
    papers = _load_extracted(payload.paper_ids)
    if not papers:
        raise HTTPException(status_code=404, detail="No extracted papers found.")

    contradiction_counts = _pairwise_contradiction_counts(papers)

    ranking = []
    for paper in papers:
        paper_id = str(paper.get("paper_id", "")).strip()
        claims = _short_claims(paper, max_items=20)
        methods = paper.get("methods", [])
        datasets = paper.get("datasets", [])

        support_score = len(claims) + len(methods) + len(datasets)
        contradiction_penalty = contradiction_counts.get(paper_id, 0) * 2
        accuracy_score = support_score - contradiction_penalty

        ranking.append(
            {
                "paper_id": paper_id,
                "title": str(paper.get("title", "")).strip(),
                "support_score": support_score,
                "contradiction_count": contradiction_counts.get(paper_id, 0),
                "accuracy_score": accuracy_score,
                "evidence": {
                    "claim_count": len(claims),
                    "method_count": len(methods),
                    "dataset_count": len(datasets),
                },
            }
        )

    ranking.sort(key=lambda item: item["accuracy_score"], reverse=True)
    top = ranking[0]

    return {
        "question": payload.question.strip(),
        "evaluated_papers": len(ranking),
        "most_accurate": top,
        "ranking": ranking,
        "mode": "heuristic",
        "notes": [
            "Reads extracted JSON from data/extracted.",
            "Scores papers by support evidence minus contradiction penalties.",
            "Use as directional signal; manual validation is still recommended.",
        ],
    }


def _load_extracted(paper_ids: list[str]) -> list[dict]:
    if not EXTRACTED_DIR.exists():
        return []

    selected = {paper_id.strip() for paper_id in paper_ids if paper_id.strip()}
    items = []
    for path in sorted(EXTRACTED_DIR.glob("*.json")):
        if selected and path.stem not in selected:
            continue
        items.append(json.loads(path.read_text(encoding="utf-8")))
    return items


def _pairwise_contradiction_counts(papers: list[dict]) -> dict[str, int]:
    counts = {str(paper.get("paper_id", "")).strip(): 0 for paper in papers}

    for left_index, left in enumerate(papers):
        left_id = str(left.get("paper_id", "")).strip()
        left_claims = _short_claims(left, max_items=10)

        for right in papers[left_index + 1 :]:
            right_id = str(right.get("paper_id", "")).strip()
            right_claims = _short_claims(right, max_items=10)

            if _has_contradiction(left_claims, right_claims):
                counts[left_id] = counts.get(left_id, 0) + 1
                counts[right_id] = counts.get(right_id, 0) + 1

    return counts


def _has_contradiction(left_claims: list[str], right_claims: list[str]) -> bool:
    for left_claim in left_claims:
        left_low = left_claim.lower()
        left_positive = any(word in left_low for word in POSITIVE_WORDS)
        left_negative = any(word in left_low for word in NEGATIVE_WORDS)
        if not (left_positive or left_negative):
            continue

        for right_claim in right_claims:
            right_low = right_claim.lower()
            if not _claims_are_related(left_low, right_low):
                continue

            right_positive = any(word in right_low for word in POSITIVE_WORDS)
            right_negative = any(word in right_low for word in NEGATIVE_WORDS)
            if (left_positive and right_negative) or (left_negative and right_positive):
                return True

    return False


def _short_claims(payload: dict, max_items: int, max_chars: int = 250) -> list[str]:
    claims = []
    for claim in payload.get("claims", []):
        text = " ".join(str(claim).split())
        if not text:
            continue
        claims.append(text[:max_chars])
        if len(claims) >= max_items:
            break
    return claims


def _claims_are_related(left: str, right: str) -> bool:
    left_tokens = set(re.findall(r"[a-z]{5,}", left))
    right_tokens = set(re.findall(r"[a-z]{5,}", right))
    return len(left_tokens.intersection(right_tokens)) > 0
