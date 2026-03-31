from __future__ import annotations

import json
import os
import re
from pathlib import Path
from collections import defaultdict

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from council_api.extraction import build_report, extract_from_pdf

router = APIRouter(tags=["feature-heatmap"])

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
METADATA_DIR = DATA_DIR / "metadata"
REPORTS_DIR = DATA_DIR / "reports"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
POSITIVE_WORDS = ["improve", "better", "outperform", "increase", "gain"]
NEGATIVE_WORDS = ["worse", "fails", "decrease", "does not improve", "no improvement"]


class HeatmapRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=2)
    save_files: bool = True


@router.post("/feature/heatmap")
def contradiction_heatmap(payload: HeatmapRequest) -> dict:
    generation_log: list[str] = []
    requested_ids = [paper_id.strip() for paper_id in payload.paper_ids if paper_id.strip()]
    if len(requested_ids) < 2:
        raise HTTPException(status_code=422, detail="Provide at least two paper_ids.")
    generation_log.append(f"Received request for {len(requested_ids)} papers.")

    papers = [_load_extracted(paper_id) for paper_id in requested_ids]
    canonical_ids = [str(item.get("paper_id", "")).strip() for item in papers]
    generation_log.append(f"Resolved to {len(canonical_ids)} canonical paper ids.")
    display_names = [
        {"paper_id": str(item.get("paper_id", "")).strip(), "display_name": _paper_display_name(item)}
        for item in papers
    ]

    base_report = build_report(EXTRACTED_DIR, paper_ids=canonical_ids)

    graph_payload = _build_knowledge_graph(papers=papers, report=base_report)
    reactflow_payload = _to_reactflow(graph=graph_payload)
    generation_log.append(
        f"Built graph with {len(graph_payload.get('nodes', []))} nodes and {len(graph_payload.get('edges', []))} edges."
    )

    saved_files = {}
    if payload.save_files:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        knowledge_path = REPORTS_DIR / "knowledge_graph.json"
        reactflow_path = REPORTS_DIR / "reactflow_graph.json"
        payload_log_path = REPORTS_DIR / "knowledge_graph_payload_log.json"
        knowledge_path.write_text(json.dumps(graph_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        reactflow_path.write_text(json.dumps(reactflow_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        payload_log_path.write_text(
            json.dumps(
                {
                    "paper_ids_requested": requested_ids,
                    "paper_ids_resolved": canonical_ids,
                    "generation_log": generation_log,
                    "knowledge_graph": graph_payload,
                    "reactflow_graph": reactflow_payload,
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        saved_files = {
            "knowledge_graph_json": str(knowledge_path),
            "reactflow_graph_json": str(reactflow_path),
            "payload_log_json": str(payload_log_path),
        }

    return {
        "paper_ids": canonical_ids,
        "paper_display_names": display_names,
        "knowledge_graph": graph_payload,
        "reactflow_graph": reactflow_payload,
        "generation_log": generation_log,
        "saved_files": saved_files,
        "mode": "knowledge-graph",
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


def _build_knowledge_graph(papers: list[dict], report: dict) -> dict:
    contradiction_pairs = (
        report.get("detailed_report", {})
        .get("contradiction_analysis", {})
        .get("pairwise_contradictions", [])
    )
    conflict_keys = {
        tuple(sorted([str(item.get("paper_a_id", "")), str(item.get("paper_b_id", ""))]))
        for item in contradiction_pairs
    }

    nodes = []
    method_nodes: dict[str, dict] = {}
    dataset_nodes: dict[str, dict] = {}
    theme_nodes: dict[str, dict] = {}
    for item in papers:
        methods = [str(m).strip() for m in item.get("methods", []) if str(m).strip()]
        datasets = [str(d).strip() for d in item.get("datasets", []) if str(d).strip()]
        claims = [str(c).strip() for c in item.get("claims", []) if str(c).strip()]
        sections = item.get("sections", {}) if isinstance(item.get("sections", {}), dict) else {}
        abstract = " ".join(str(sections.get("abstract", "")).split())[:420]
        intro = " ".join(str(sections.get("introduction", "")).split())[:260]
        topic_keywords = sorted(list(_topic_tokens(item)))[:10]
        paper_id = str(item.get("paper_id", "")).strip()

        nodes.append(
            {
                "id": paper_id,
                "node_type": "paper",
                "title": str(item.get("title", "")).strip(),
                "year": str(item.get("year", "")).strip(),
                "source": str(item.get("source", "")).strip(),
                "key_methodology": methods[0] if methods else "unspecified",
                "main_contribution": claims[0][:260] if claims else "No concise contribution extracted.",
                "problem_statement": abstract or intro,
                "top_claims": claims[:5],
                "top_methods": methods[:5],
                "top_datasets": datasets[:5],
                "topic_keywords": topic_keywords,
                "claim_count": len(claims),
                "method_count": len(methods),
                "dataset_count": len(datasets),
                "tags": _infer_tags(item),
            }
        )

        for method in methods[:6]:
            method_id = f"method:{_slug(method)}"
            if method_id not in method_nodes:
                method_nodes[method_id] = {
                    "id": method_id,
                    "node_type": "method",
                    "title": method,
                    "description": f"Method referenced across papers: {method}",
                }
        for dataset in datasets[:6]:
            dataset_id = f"dataset:{_slug(dataset)}"
            if dataset_id not in dataset_nodes:
                dataset_nodes[dataset_id] = {
                    "id": dataset_id,
                    "node_type": "dataset",
                    "title": dataset,
                    "description": f"Dataset/benchmark entity: {dataset}",
                }
        for keyword in topic_keywords[:5]:
            theme_id = f"theme:{_slug(keyword)}"
            if theme_id not in theme_nodes:
                theme_nodes[theme_id] = {
                    "id": theme_id,
                    "node_type": "theme",
                    "title": keyword,
                    "description": f"Shared topical concept: {keyword}",
                }

    nodes.extend(method_nodes.values())
    nodes.extend(dataset_nodes.values())
    nodes.extend(theme_nodes.values())

    edges: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    by_id = {str(item.get("paper_id", "")).strip(): item for item in papers}
    ordered_ids = [str(item.get("paper_id", "")).strip() for item in papers]

    def add_edge(
        source: str,
        target: str,
        relation: str,
        explanation: str,
        confidence: float,
        evidence: dict | None = None,
    ) -> None:
        key = (source, target, relation)
        if not source or not target or source == target or key in seen:
            return
        seen.add(key)
        edges.append(
            {
                "source": source,
                "target": target,
                "relationship": relation,
                "explanation": " ".join(explanation.split())[:260],
                "confidence": max(0.0, min(1.0, confidence)),
                "evidence": evidence or {},
            }
        )

    for item in papers:
        paper_id = str(item.get("paper_id", "")).strip()
        methods = [str(m).strip() for m in item.get("methods", []) if str(m).strip()]
        datasets = [str(d).strip() for d in item.get("datasets", []) if str(d).strip()]
        keywords = sorted(list(_topic_tokens(item)))[:5]

        for method in methods[:6]:
            add_edge(
                paper_id,
                f"method:{_slug(method)}",
                "USES_METHOD",
                "Paper uses this method or analytical approach.",
                0.85,
                {"method": method},
            )
        for dataset in datasets[:6]:
            add_edge(
                paper_id,
                f"dataset:{_slug(dataset)}",
                "USES_DATASET",
                "Paper reports evidence using this dataset/benchmark.",
                0.84,
                {"dataset": dataset},
            )
        for keyword in keywords:
            add_edge(
                paper_id,
                f"theme:{_slug(keyword)}",
                "FOCUSES_ON_THEME",
                "Paper discusses this recurring concept/theme.",
                0.73,
                {"keyword": keyword},
            )

    for index, left_id in enumerate(ordered_ids):
        left = by_id.get(left_id, {})
        left_methods = {str(m).strip().lower() for m in left.get("methods", []) if str(m).strip()}
        left_topic = _topic_tokens(left)
        left_claim_blob = " ".join(str(c).lower() for c in left.get("claims", []))
        left_year = _safe_year(left.get("year", ""))

        for right_id in ordered_ids[index + 1 :]:
            right = by_id.get(right_id, {})
            right_methods = {str(m).strip().lower() for m in right.get("methods", []) if str(m).strip()}
            right_topic = _topic_tokens(right)
            right_claim_blob = " ".join(str(c).lower() for c in right.get("claims", []))
            right_year = _safe_year(right.get("year", ""))

            method_overlap = left_methods.intersection(right_methods)
            topic_overlap = left_topic.intersection(right_topic)
            conflict_key = tuple(sorted([left_id, right_id]))

            if method_overlap:
                add_edge(
                    left_id,
                    right_id,
                    "USES_SIMILAR_METHOD",
                    f"Both papers rely on related methods such as {', '.join(sorted(method_overlap)[:3])}.",
                    0.82,
                    {"shared_methods": sorted(list(method_overlap))[:6]},
                )
                add_edge(
                    right_id,
                    left_id,
                    "USES_SIMILAR_METHOD",
                    f"Both papers rely on related methods such as {', '.join(sorted(method_overlap)[:3])}.",
                    0.82,
                    {"shared_methods": sorted(list(method_overlap))[:6]},
                )

            if len(topic_overlap) >= 2:
                add_edge(
                    left_id,
                    right_id,
                    "FOCUSES_ON_SIMILAR_PROBLEM",
                    "The papers share overlapping problem vocabulary and discuss similar topic scope.",
                    0.78,
                    {"shared_topics": sorted(list(topic_overlap))[:8]},
                )
                add_edge(
                    right_id,
                    left_id,
                    "FOCUSES_ON_SIMILAR_PROBLEM",
                    "The papers share overlapping problem vocabulary and discuss similar topic scope.",
                    0.78,
                    {"shared_topics": sorted(list(topic_overlap))[:8]},
                )

            if conflict_key in conflict_keys:
                add_edge(
                    left_id,
                    right_id,
                    "CONTRADICTS",
                    "Claim directions conflict across this pair in the contradiction analysis.",
                    0.9,
                    {
                        "left_claim_sample": _claim_sample(left),
                        "right_claim_sample": _claim_sample(right),
                    },
                )
                add_edge(
                    right_id,
                    left_id,
                    "CONTRADICTS",
                    "Claim directions conflict across this pair in the contradiction analysis.",
                    0.9,
                    {
                        "left_claim_sample": _claim_sample(right),
                        "right_claim_sample": _claim_sample(left),
                    },
                )
            elif method_overlap or len(topic_overlap) >= 2:
                add_edge(
                    left_id,
                    right_id,
                    "VALIDATES",
                    "No direct contradiction was detected and the papers share evidence context.",
                    0.68,
                    {
                        "shared_methods": sorted(list(method_overlap))[:4],
                        "shared_topics": sorted(list(topic_overlap))[:6],
                    },
                )
                add_edge(
                    right_id,
                    left_id,
                    "VALIDATES",
                    "No direct contradiction was detected and the papers share evidence context.",
                    0.68,
                    {
                        "shared_methods": sorted(list(method_overlap))[:4],
                        "shared_topics": sorted(list(topic_overlap))[:6],
                    },
                )

            if left_year != 9999 and right_year != 9999 and left_year != right_year and len(topic_overlap) >= 2:
                newer_id = left_id if left_year > right_year else right_id
                older_id = right_id if left_year > right_year else left_id
                newer_blob = left_claim_blob if newer_id == left_id else right_claim_blob

                if any(term in newer_blob for term in ["improve", "outperform", "better"]):
                    add_edge(
                        newer_id,
                        older_id,
                        "IMPROVES_UPON",
                        "Newer paper reports stronger performance while addressing similar problem settings.",
                        0.73,
                        {"newer_year": max(left_year, right_year), "older_year": min(left_year, right_year)},
                    )
                else:
                    add_edge(
                        newer_id,
                        older_id,
                        "EXTENDS",
                        "Newer paper appears to continue prior work with additional context or scope.",
                        0.64,
                        {"newer_year": max(left_year, right_year), "older_year": min(left_year, right_year)},
                    )

            if _references_target(left, right):
                add_edge(
                    left_id,
                    right_id,
                    "CITES",
                    "Reference text in source paper appears to mention the target paper title.",
                    0.7,
                    {"source": left_id, "target": right_id},
                )
            if _references_target(right, left):
                add_edge(
                    right_id,
                    left_id,
                    "CITES",
                    "Reference text in source paper appears to mention the target paper title.",
                    0.7,
                    {"source": right_id, "target": left_id},
                )

    graph_stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "paper_nodes": len([n for n in nodes if n.get("node_type") == "paper"]),
        "method_nodes": len([n for n in nodes if n.get("node_type") == "method"]),
        "dataset_nodes": len([n for n in nodes if n.get("node_type") == "dataset"]),
        "theme_nodes": len([n for n in nodes if n.get("node_type") == "theme"]),
        "relationship_counts": _relationship_counts(edges),
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": graph_stats,
    }


def _to_reactflow(graph: dict) -> dict:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    conflicts: set[str] = set()
    for edge in edges:
        if edge.get("relationship") == "CONTRADICTS":
            conflicts.add(str(edge.get("source", "")))
            conflicts.add(str(edge.get("target", "")))

    clusters: dict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        node_type = str(node.get("node_type", "paper")).strip() or "paper"
        if node_type == "paper":
            cluster = str(node.get("key_methodology", "unspecified")).strip() or "unspecified"
        else:
            cluster = node_type
        clusters[cluster].append(node)

    react_nodes = []
    sorted_clusters = sorted(clusters.items(), key=lambda item: item[0].lower())
    for cluster_index, (cluster_name, cluster_nodes) in enumerate(sorted_clusters):
        for row, node in enumerate(cluster_nodes):
            node_id = str(node.get("id", ""))
            tags = ", ".join(node.get("tags", [])[:4])
            conflict_prefix = "[CONFLICT] " if node_id in conflicts else ""
            node_type = str(node.get("node_type", "paper"))
            base_description = str(node.get("description", "")).strip()
            if not base_description:
                base_description = (
                    f"Method: {cluster_name}; Contribution: {node.get('main_contribution', '')}; "
                    f"Tags: {tags}; Type: {node_type}"
                )
            react_nodes.append(
                {
                    "id": node_id,
                    "data": {
                        "label": f"{conflict_prefix}{node.get('title', 'Untitled Node')}",
                        "description": base_description[:460],
                        "node_type": node_type,
                        "tags": node.get("tags", []),
                    },
                    "position": {
                        "x": 120 + (cluster_index * 430),
                        "y": 80 + (row * 190),
                    },
                }
            )

    react_edges = []
    for index, edge in enumerate(edges):
        relation = str(edge.get("relationship", "")).strip()
        label = "CONTRADICTS (!)" if relation == "CONTRADICTS" else relation
        react_edges.append(
            {
                "id": f"e{index}",
                "source": str(edge.get("source", "")),
                "target": str(edge.get("target", "")),
                "label": label,
                "data": {
                    "confidence": edge.get("confidence", 0.0),
                    "explanation": edge.get("explanation", ""),
                    "relationship": relation,
                    "evidence": edge.get("evidence", {}),
                },
            }
        )

    return {
        "nodes": react_nodes,
        "edges": react_edges,
    }


def _infer_tags(paper: dict) -> list[str]:
    text = " ".join(
        [
            str(paper.get("title", "")),
            " ".join(str(x) for x in paper.get("methods", [])),
            " ".join(str(x) for x in paper.get("datasets", [])),
        ]
    ).lower()
    tags: list[str] = []
    mapping = {
        "ML": ["learning", "model", "neural", "regression", "classification"],
        "Health": ["clinical", "obesity", "metabolic", "diet", "diabetes", "patient"],
        "NLP": ["language", "text", "retrieval", "transformer", "bert"],
        "CV": ["image", "vision", "object", "detection", "imagenet", "cifar"],
    }
    for tag, signals in mapping.items():
        if any(signal in text for signal in signals):
            tags.append(tag)
    return tags or ["Research"]


def _topic_tokens(paper: dict) -> set[str]:
    abstract = str(paper.get("sections", {}).get("abstract", ""))
    intro = str(paper.get("sections", {}).get("introduction", ""))
    title = str(paper.get("title", ""))
    tokens = re.findall(r"[a-z]{5,}", f"{title} {abstract} {intro}".lower())
    return {token for token in tokens}


def _references_target(source: dict, target: dict) -> bool:
    target_title = " ".join(re.findall(r"[a-z0-9]+", str(target.get("title", "")).lower()))
    if not target_title:
        return False

    title_tokens = target_title.split()
    if len(title_tokens) < 3:
        return False

    target_signature = " ".join(title_tokens[:4])
    for ref in source.get("references", []):
        ref_clean = " ".join(re.findall(r"[a-z0-9]+", str(ref).lower()))
        if target_signature and target_signature in ref_clean:
            return True
    return False


def _safe_year(raw: str) -> int:
    try:
        return int(str(raw).strip())
    except Exception:
        return 9999


def _slug(value: str) -> str:
    compact = "-".join(re.findall(r"[a-z0-9]+", value.lower()))
    return compact[:90] if compact else "unknown"


def _claim_sample(paper: dict) -> str:
    for claim in paper.get("claims", []):
        text = " ".join(str(claim).split())
        if text:
            return text[:220]
    return ""


def _relationship_counts(edges: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for edge in edges:
        relation = str(edge.get("relationship", "")).strip() or "UNKNOWN"
        counts[relation] = counts.get(relation, 0) + 1
    return counts
