"""
Citation-Aware Chatbot System (Vectorless)

Matches questions against extracted claims/sections deterministically.
Each answer statement is backed by specific paper citations.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/feature", tags=["feature"])

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


class CitationAwareChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    paper_ids: list[str] = Field(default_factory=list)
    require_citations: bool = Field(default=True, description="Only accept answers with citations")


class Citation(BaseModel):
    paper_id: str
    claim_text: str
    section: str = "unknown"
    relevance_score: float = Field(ge=0, le=1)


class CitationAwareChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    answer_with_citations: str  # Markdown formatted with [1], [2] markers
    papers_considered: int
    mode: str = "citation-aware"


@router.post("/citation-chat")
def citation_aware_chat(payload: CitationAwareChatRequest) -> CitationAwareChatResponse:
    """
    Answer research questions with full citation traceability.
    No vectors - deterministic claim matching for reproducibility.
    """
    question = " ".join(payload.question.split())
    papers = _load_extracted(payload.paper_ids)
    
    if not papers:
        raise HTTPException(status_code=404, detail="No extracted papers found.")

    # Find relevant claims/sections
    relevant_claims = _find_relevant_claims(question, papers)
    
    if not relevant_claims and payload.require_citations:
        return CitationAwareChatResponse(
            question=question,
            answer="No relevant claims found in the provided papers to answer this question.",
            citations=[],
            answer_with_citations="No relevant claims found in the provided papers to answer this question.",
            papers_considered=len(papers),
            mode="no-match",
        )

    # Generate answer with LLM, constrained by citations
    answer_result = _generate_cited_answer(question, relevant_claims)

    if answer_result:
        formatted_answer = _format_answer_with_citations(
            answer_result["answer"],
            answer_result["citations"]
        )
        
        return CitationAwareChatResponse(
            question=question,
            answer=answer_result["answer"],
            citations=answer_result["citations"],
            answer_with_citations=formatted_answer,
            papers_considered=len(papers),
            mode="groq-cited",
        )

    # Fallback: summarize top claims
    fallback_citations = relevant_claims[:5]
    fallback_answer = _build_fallback_answer(question, fallback_citations)
    formatted_fallback = _format_answer_with_citations(fallback_answer, fallback_citations)

    return CitationAwareChatResponse(
        question=question,
        answer=fallback_answer,
        citations=fallback_citations,
        answer_with_citations=formatted_fallback,
        papers_considered=len(papers),
        mode="fallback-cited",
    )


def _load_extracted(paper_ids: list[str]) -> list[dict]:
    if not EXTRACTED_DIR.exists():
        return []

    selected = {paper_id.strip() for paper_id in paper_ids if paper_id.strip()}
    items: list[dict] = []
    for path in sorted(EXTRACTED_DIR.glob("*.json")):
        if selected and path.stem not in selected:
            continue
        items.append(json.loads(path.read_text(encoding="utf-8")))
    return items


def _find_relevant_claims(question: str, papers: list[dict]) -> list[Citation]:
    """
    Find claims/sections relevant to question using deterministic matching.
    Vectorless approach: keyword overlap + heuristic ranking.
    """
    question_tokens = _tokenize(question)
    relevant: list[tuple[Citation, float]] = []

    for paper in papers:
        paper_id = str(paper.get("paper_id", "")).strip()
        
        # Check claims
        for claim in paper.get("claims", []):
            claim_text = str(claim).strip()
            if not claim_text:
                continue
            
            score = _claim_relevance(question_tokens, claim_text)
            if score > 0.2:  # Relevance threshold
                citation = Citation(
                    paper_id=paper_id,
                    claim_text=claim_text,
                    section="claims",
                    relevance_score=score,
                )
                relevant.append((citation, score))
        
        # Check abstract/introduction for key concepts
        abstract = str(paper.get("sections", {}).get("abstract", "")).strip()
        if abstract:
            for sentence in _split_sentences(abstract)[:3]:
                score = _claim_relevance(question_tokens, sentence)
                if score > 0.3:  # Higher threshold for abstract
                    citation = Citation(
                        paper_id=paper_id,
                        claim_text=sentence[:300],  # Truncate long sentences
                        section="abstract",
                        relevance_score=score,
                    )
                    relevant.append((citation, score))

    # Sort by relevance and return top 10
    relevant.sort(key=lambda x: x[1], reverse=True)
    return [cit for cit, _ in relevant[:10]]


def _tokenize(text: str) -> set[str]:
    """Extract meaningful tokens from text."""
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    # Filter out very common words
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "by", "with", "is", "are", "was", "were", "be", "been",
        "what", "which", "who", "how", "why", "when", "where", "do", "does", "can", "could"
    }
    return {t for t in tokens if t not in stopwords and len(t) > 2}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]


def _claim_relevance(question_tokens: set[str], claim_text: str) -> float:
    """
    Calculate relevance score (0-1) between question and claim.
    Based on token overlap and semantic heuristics.
    """
    claim_tokens = _tokenize(claim_text)
    if not claim_tokens:
        return 0.0
    
    # Token overlap
    overlap = question_tokens.intersection(claim_tokens)
    overlap_score = len(overlap) / max(len(question_tokens), 1)
    
    # Length penalty (prefer concise claims)
    length_score = min(1.0, 300 / len(claim_text)) if claim_text else 0.0
    
    # Keyword boost for research-related terms
    research_terms = {"propose", "present", "show", "demonstrate", "improve", "outperform",
                     "method", "approach", "model", "algorithm", "dataset", "result"}
    research_boost = 0.1 if any(term in claim_text.lower() for term in research_terms) else 0.0
    
    return min(1.0, overlap_score * 0.7 + (length_score * 0.2) + research_boost)


def _generate_cited_answer(question: str, citations: list[Citation]) -> dict | None:
    """
    Generate answer from LLM using citations as grounding.
    LLM is instructed to cite specific claims.
    """
    keys = _groq_api_keys()
    if not keys:
        return None

    # Build citation context
    citation_context = "\n".join([
        f"[{i+1}] (Paper {c.paper_id}, {c.section}): {c.claim_text}"
        for i, c in enumerate(citations)
    ])

    prompt = (
        "Answer the research question using ONLY the provided citations. "
        "Each statement must be backed by at least one citation. "
        "Format your answer as plain text, referencing citations by number like [1], [2]. "
        "Return JSON: {\"answer\": string, \"used_citation_indices\": [int]}.\n\n"
        f"Question: {question}\n\n"
        f"Citations:\n{citation_context}"
    )

    payload = {
        "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "temperature": 0.2,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a research assistant. Return valid JSON only."},
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
                    json=payload,
                )
                if response.status_code >= 400:
                    continue
                
                parsed = _parse_json_response(_extract_chat_content(response.json()))
                if not parsed:
                    continue

                answer = " ".join(str(parsed.get("answer", "")).split())
                used_indices = parsed.get("used_citation_indices", [])
                
                if not isinstance(used_indices, list):
                    used_indices = []
                
                # Map indices to citations
                used_citations = []
                for idx in used_indices:
                    if isinstance(idx, int) and 0 <= idx - 1 < len(citations):
                        used_citations.append(citations[idx - 1])
                
                if answer and used_citations:
                    return {
                        "answer": answer,
                        "citations": used_citations,
                    }
            except Exception:
                continue

    return None


def _build_fallback_answer(question: str, citations: list[Citation]) -> str:
    """Build fallback answer from top citations."""
    if not citations:
        return "Unable to find relevant information."
    
    lines = [
        f"Based on the extracted papers, here are relevant findings:\n"
    ]
    
    paper_groups = {}
    for cit in citations:
        if cit.paper_id not in paper_groups:
            paper_groups[cit.paper_id] = []
        paper_groups[cit.paper_id].append(cit.claim_text)
    
    for paper_id, claims in paper_groups.items():
        lines.append(f"\n**{paper_id}:**")
        for claim in claims[:2]:
            lines.append(f"- {claim[:200]}")
    
    return "\n".join(lines)


def _format_answer_with_citations(answer: str, citations: list[Citation]) -> str:
    """Format answer as markdown with citation references."""
    if not citations:
        return answer
    
    # Build citation footnotes
    citation_section = "\n\n---\n\n### 📚 Citations\n"
    for i, cit in enumerate(citations, 1):
        citation_section += (
            f"[{i}] **{cit.paper_id}** ({cit.section})\n"
            f"> {cit.claim_text[:200]}\n"
        )
    
    return answer + citation_section


def _groq_api_keys() -> list[str]:
    keys: list[str] = []

    pooled = os.getenv("GROQ_API_KEYS", "")
    keys.extend(item.strip() for item in pooled.split(",") if item.strip())

    single = os.getenv("GROQ_API_KEY", "").strip()
    if single:
        keys.append(single)

    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        file_env = dotenv_values(env_path)

        file_pooled = str(file_env.get("GROQ_API_KEYS", "") or "")
        keys.extend(item.strip() for item in file_pooled.split(",") if item.strip())

        file_single = str(file_env.get("GROQ_API_KEY", "") or "").strip()
        if file_single:
            keys.append(file_single)

    unique_keys: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        unique_keys.append(key)

    return unique_keys


def _extract_chat_content(payload: dict) -> str:
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        return ""

    content = message.get("content", "")
    return content if isinstance(content, str) else ""


def _parse_json_response(text: str) -> dict | None:
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
