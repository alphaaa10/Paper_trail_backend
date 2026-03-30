from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/feature", tags=["feature"])

ROOT_DIR = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = ROOT_DIR / "data" / "extracted"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    paper_ids: list[str] = Field(default_factory=list)


@router.post("/ask")
def ask_question(payload: AskRequest) -> dict:
    question = " ".join(payload.question.split())
    papers = _load_extracted(payload.paper_ids)
    if not papers:
        raise HTTPException(status_code=404, detail="No extracted papers found.")

    context = _build_context(papers)
    llm_answer = _answer_with_groq(question=question, context=context)

    if llm_answer:
        return {
            "question": question,
            "answer": llm_answer["answer"],
            "cited_paper_ids": llm_answer["cited_paper_ids"],
            "papers_considered": len(papers),
            "context_paper_ids": [str(p.get("paper_id", "")) for p in papers],
            "mode": "groq",
        }

    top = papers[0]
    fallback_answer = (
        "I could not reach the LLM provider, so this is a fallback summary. "
        f"Top available paper in this context is '{top.get('title', '')}' "
        f"(paper_id={top.get('paper_id', '')})."
    )
    return {
        "question": question,
        "answer": fallback_answer,
        "cited_paper_ids": [str(top.get("paper_id", ""))],
        "papers_considered": len(papers),
        "context_paper_ids": [str(p.get("paper_id", "")) for p in papers],
        "mode": "fallback",
    }


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


def _build_context(papers: list[dict]) -> str:
    chunks: list[str] = []
    for paper in papers[:25]:
        paper_id = str(paper.get("paper_id", "")).strip()
        title = str(paper.get("title", "")).strip()
        claims = _short_list(paper.get("claims", []), max_items=8, max_chars=220)
        methods = _short_list(paper.get("methods", []), max_items=6, max_chars=80)
        datasets = _short_list(paper.get("datasets", []), max_items=6, max_chars=80)

        chunk = (
            f"paper_id: {paper_id}\n"
            f"title: {title}\n"
            f"claims: {json.dumps(claims, ensure_ascii=True)}\n"
            f"methods: {json.dumps(methods, ensure_ascii=True)}\n"
            f"datasets: {json.dumps(datasets, ensure_ascii=True)}\n"
        )
        chunks.append(chunk)

    return "\n---\n".join(chunks)


def _short_list(values: list, max_items: int, max_chars: int) -> list[str]:
    out: list[str] = []
    for value in values:
        text = " ".join(str(value).split())
        if not text:
            continue
        out.append(text[:max_chars])
        if len(out) >= max_items:
            break
    return out


def _answer_with_groq(question: str, context: str) -> dict | None:
    keys = _groq_api_keys()
    if not keys:
        return None

    prompt = (
        "You answer research questions using only provided paper context. "
        "The question can ask about one paper or compare multiple papers. "
        "Return JSON only with schema {\"answer\": string, \"cited_paper_ids\": [string]}. "
        "Keep answer concise and evidence-grounded.\n\n"
        f"Question: {question}\n\n"
        f"Paper context:\n{context[:38000]}"
    )

    payload = {
        "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "temperature": 0,
        "max_tokens": 900,
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
                    json=payload,
                )
                if response.status_code >= 400:
                    continue
                parsed = _parse_json_object(_extract_chat_content(response.json()))
                if not parsed:
                    continue

                answer = " ".join(str(parsed.get("answer", "")).split())
                cited = parsed.get("cited_paper_ids", [])
                if not isinstance(cited, list):
                    cited = []
                cited_ids = [" ".join(str(item).split()) for item in cited if str(item).strip()][:10]
                if not answer:
                    continue
                return {"answer": answer, "cited_paper_ids": cited_ids}
            except Exception:  # noqa: BLE001
                continue

    return None


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
