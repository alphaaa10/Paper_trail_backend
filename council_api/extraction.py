from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

SECTION_ORDER = ["abstract", "introduction", "methodology", "results", "conclusion"]
SECTION_ALIASES = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "1 introduction"],
    "methodology": ["methodology", "methods", "materials and methods"],
    "results": ["results", "experiments", "evaluation", "discussion"],
    "conclusion": ["conclusion", "conclusions", "future work"],
}
CLAIM_HINTS = [
    "we propose",
    "we present",
    "we show",
    "we demonstrate",
    "our results",
    "outperform",
    "improve",
    "state-of-the-art",
    "significant",
]
METHOD_HINTS = [
    "transformer",
    "bert",
    "cnn",
    "rnn",
    "diffusion",
    "reinforcement learning",
    "ablation",
    "benchmark",
]
DATASET_HINTS = [
    "dataset",
    "corpus",
    "imagenet",
    "cifar",
    "squad",
    "ms coco",
    "wikitext",
]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
_GROQ_KEY_CURSOR = 0
STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "been",
    "between",
    "both",
    "from",
    "have",
    "into",
    "more",
    "most",
    "only",
    "other",
    "over",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "those",
    "very",
    "with",
}


def extract_from_pdf(paper_id: str, pdf_path: Path, output_dir: Path, metadata: dict) -> dict:
    text = _read_pdf_text(pdf_path)
    sections = _split_sections(text)
    abstract_intro = "\n".join(
        [
            sections.get("abstract", ""),
            sections.get("introduction", ""),
        ]
    ).strip()
    llm_extraction = _extract_with_groq(
        title=metadata.get("title", ""),
        text=abstract_intro,
    )
    combined_text = "\n".join(sections.values())

    claims = llm_extraction.get("claims", []) or _extract_claims(combined_text)
    methods = llm_extraction.get("methods", []) or _extract_keywords(combined_text, METHOD_HINTS)
    datasets = llm_extraction.get("datasets", []) or _extract_keywords(combined_text, DATASET_HINTS)
    references = _extract_reference_candidates(text)

    payload = {
        "paper_id": paper_id,
        "title": metadata.get("title", ""),
        "source": metadata.get("source", ""),
        "year": metadata.get("year", ""),
        "sections": sections,
        "claims": claims,
        "methods": methods,
        "datasets": datasets,
        "references": references,
        "extraction_model": llm_extraction.get("model", ""),
        "extraction_mode": llm_extraction.get("mode", "heuristic"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{paper_id}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return payload


def extract_from_html(paper_id: str, html: str, output_dir: Path, metadata: dict) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    sections = _split_sections(text)
    full_text = "\n".join(sections.values())

    groq_result = _extract_with_groq(metadata.get("title", ""), full_text)
    if groq_result:
        claims = groq_result.get("claims", [])
        methods = groq_result.get("methods", [])
        datasets = groq_result.get("datasets", [])
    else:
        claims = _extract_claims(full_text)
        methods = _extract_keywords(full_text, METHOD_HINTS)
        datasets = _extract_keywords(full_text, DATASET_HINTS)

    sentences: dict[str, list[str]] = {}
    for section_name, section_text in sections.items():
        raw = re.split(r"(?<=[.!?])\s+", section_text)
        sentences[section_name] = [s.strip() for s in raw if len(s.strip()) > 40]

    payload = {
        "paper_id": paper_id,
        "title": metadata.get("title", ""),
        "source": metadata.get("source", ""),
        "year": metadata.get("year", ""),
        "sections": sections,
        "sentences": sentences,
        "claims": claims,
        "methods": methods,
        "datasets": datasets,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{paper_id}.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return payload


def build_report(extracted_dir: Path, paper_ids: list[str] | None = None) -> dict:
    extracted = []
    for path in sorted(extracted_dir.glob("*.json")):
        if paper_ids and path.stem not in set(paper_ids):
            continue
        extracted.append(json.loads(path.read_text(encoding="utf-8")))

    all_claims = []
    for item in extracted:
        for claim in item.get("claims", []):
            all_claims.append({"paper_id": item.get("paper_id", ""), "claim": claim})

    contradictions = _find_contradictions(all_claims)
    method_counter = _count_items([m for item in extracted for m in item.get("methods", [])])
    dataset_counter = _count_items([d for item in extracted for d in item.get("datasets", [])])
    gaps = _suggest_gaps(method_counter, dataset_counter, len(contradictions))
    detailed_report = _build_detailed_report(
        extracted_items=extracted,
        contradictions=contradictions,
        top_methods=method_counter,
        top_datasets=dataset_counter,
        gaps=gaps,
    )

    base_payload = {
        "paper_count": len(extracted),
        "claim_count": len(all_claims),
        "top_methods": method_counter[:10],
        "top_datasets": dataset_counter[:10],
        "contradictions": contradictions,
        "gaps": gaps,
        "detailed_report": detailed_report,
    }

    comprehensive_summary = _build_comprehensive_executive_summary(
        extracted_items=extracted,
        base_report=base_payload,
    )
    recent_works = _recent_works(extracted)
    base_payload["executive_summary"] = {
        "papers_considered": base_payload["paper_count"],
        "unanswered_question_count": len(base_payload["gaps"]),
        "decision_topic_count": len(base_payload["top_methods"]),
        "contradicting_paper_count": len(base_payload["contradictions"]),
        "recent_works_count": len(recent_works),
        "individual_paper_summaries": comprehensive_summary["individual_paper_summaries"],
        "cross_paper_analysis": comprehensive_summary["cross_paper_analysis"],
        "critical_insights": comprehensive_summary["critical_insights"],
    }
    base_payload["executive_summary_json"] = comprehensive_summary
    base_payload["executive_summary_markdown"] = _render_executive_summary_markdown(comprehensive_summary)

    return base_payload


def build_final_report(
    extracted_dir: Path,
    target_research_finding: str,
    top_k: int = 10,
    paper_ids: list[str] | None = None,
) -> dict:
    base = build_report(extracted_dir=extracted_dir, paper_ids=paper_ids)
    extracted_items = _load_extracted_items(extracted_dir=extracted_dir, paper_ids=paper_ids)
    comprehensive_summary = base.get("executive_summary_json", {})
    executive_summary_markdown = base.get("executive_summary_markdown", "")
    references = _extract_reference_lines_from_items(extracted_items)
    reference_counter = _count_items(references)
    recent_works = _recent_works(extracted_items)
    citation_edges = _build_in_corpus_edges(extracted_items)
    decision_recommendations = _build_decision_recommendations(
        contradiction_count=len(base["contradictions"]),
        gap_count=len(base["gaps"]),
        method_count=len(base["top_methods"]),
        recent_count=len(recent_works),
    )

    return {
        "executive_summary": {
            "papers_considered": base["paper_count"],
            "unanswered_question_count": len(base["gaps"]),
            "decision_topic_count": len(base["top_methods"]),
            "contradicting_paper_count": len(base["contradictions"]),
            "recent_works_count": len(recent_works),
            "individual_paper_summaries": comprehensive_summary.get("individual_paper_summaries", []),
            "cross_paper_analysis": comprehensive_summary.get("cross_paper_analysis", {}),
            "critical_insights": comprehensive_summary.get("critical_insights", {}),
        },
        "executive_summary_json": comprehensive_summary,
        "executive_summary_markdown": executive_summary_markdown,
        "which_question_has_been_circling_but_unanswered": base["gaps"],
        "topics_to_search_for_further_decision": [item["name"] for item in base["top_methods"][:top_k]],
        "contradicting_papers_defying_targeted_research": base["contradictions"][:top_k],
        "recent_works_last_2_years": recent_works,
        "citation_trail": {
            "top_cited_references": [
                {"reference": item["name"], "citation_count": item["count"]}
                for item in reference_counter[:top_k]
            ],
            "in_corpus_citation_edges": citation_edges[:top_k],
            "citation_links": citation_edges,
            "total_citation_links": len(citation_edges),
        },
        "seminal_papers": [
            {
                "reference": item["name"],
                "signal": "high_citation_frequency_in_collected_corpus",
                "citation_count": item["count"],
            }
            for item in reference_counter[:top_k]
        ],
        "established_findings": [{"title": "Target finding", "finding": target_research_finding}],
        "critical_additional_findings": [
            {
                "title": "Method concentration",
                "finding": {
                    "top_methods": base["top_methods"][:top_k],
                    "interpretation": "High concentration can indicate fragility in method diversity.",
                },
            },
            {
                "title": "Priority unresolved gaps",
                "finding": {
                    "count": len(base["gaps"]),
                    "examples": base["gaps"][:top_k],
                },
            },
            {
                "title": "Contradiction pressure",
                "finding": {
                    "total_contradictions": len(base["contradictions"]),
                    "critical_or_high": 0,
                },
            },
        ],
        "decision_recommendations": decision_recommendations,
        "detailed_report": base.get("detailed_report", {}),
        "structured_debates": _load_recent_debates(),
    }


def _build_comprehensive_executive_summary(extracted_items: list[dict], base_report: dict) -> dict:
    paper_summaries = [_single_paper_summary(item) for item in extracted_items]
    contradiction_pairs = (
        base_report.get("detailed_report", {})
        .get("contradiction_analysis", {})
        .get("pairwise_contradictions", [])
    )

    top_methods = [item.get("name", "") for item in base_report.get("top_methods", []) if item.get("name")]
    top_datasets = [item.get("name", "") for item in base_report.get("top_datasets", []) if item.get("name")]
    methodology_differences = []
    for item in paper_summaries:
        unique_methods = item.get("methodology_used", [])
        methodology_differences.append(
            {
                "paper_id": item.get("paper_id", ""),
                "title": item.get("title", ""),
                "notable_method_choices": unique_methods[:4],
            }
        )

    conflicts = []
    for pair in contradiction_pairs[:12]:
        conflicts.append(
            {
                "paper_a": pair.get("paper_a_id", ""),
                "paper_b": pair.get("paper_b_id", ""),
                "conflict_count": pair.get("conflict_count", 0),
                "explanation": (pair.get("reasons", []) or ["Conflicting claim directions detected."])[0],
            }
        )

    support_links = _supporting_pairs(extracted_items=extracted_items, contradiction_pairs=contradiction_pairs)
    timeline = _timeline_progression(extracted_items)

    effective_approaches = []
    for method in base_report.get("top_methods", [])[:8]:
        name = method.get("name", "")
        count = method.get("count", 0)
        if not name:
            continue
        effective_approaches.append(
            {
                "approach": name,
                "supporting_papers": count,
                "why_effective": "Appears repeatedly across the corpus and is often linked to key claims/results.",
            }
        )

    gaps = base_report.get("detailed_report", {}).get("gaps_between_papers", [])
    gap_lines = [item.get("gap", "") for item in gaps if item.get("gap")]
    future_steps = base_report.get("detailed_report", {}).get("future_steps", [])

    return {
        "individual_paper_summaries": paper_summaries,
        "cross_paper_analysis": {
            "common_themes": [
                {
                    "theme": "Methodological overlap",
                    "evidence": top_methods[:6],
                },
                {
                    "theme": "Dataset convergence",
                    "evidence": top_datasets[:6],
                },
            ],
            "methodology_differences": methodology_differences,
            "conflicting_conclusions": conflicts,
            "supporting_or_validating_pairs": support_links,
            "evolution_of_ideas": timeline,
        },
        "critical_insights": {
            "most_effective_approaches": effective_approaches,
            "research_gaps_identified": gap_lines or ["No major structural gaps detected in current extraction snapshot."],
            "opportunities_for_future_work": future_steps,
        },
    }


def _single_paper_summary(item: dict) -> dict:
    paper_id = str(item.get("paper_id", "")).strip()
    title = str(item.get("title", "")).strip()
    sections = item.get("sections", {}) if isinstance(item.get("sections", {}), dict) else {}

    abstract = " ".join(str(sections.get("abstract", "")).split())
    intro = " ".join(str(sections.get("introduction", "")).split())
    methodology_section = " ".join(str(sections.get("methodology", "")).split())
    results_section = " ".join(str(sections.get("results", "")).split())
    conclusion_section = " ".join(str(sections.get("conclusion", "")).split())

    problem_statement = _first_sentence(abstract or intro, fallback="Problem statement was not explicitly extracted.")
    methods = [" ".join(str(x).split()) for x in item.get("methods", []) if str(x).strip()]
    contributions = _top_distinct_lines(item.get("claims", []), max_items=4)
    findings = _top_distinct_lines(item.get("claims", []), max_items=3)
    if results_section:
        findings = ([ _first_sentence(results_section, fallback="") ] + findings)[:4]

    limitations = _extract_limitations(conclusion_section)
    if not limitations:
        limitations = ["Limitations are not explicitly stated in extracted sections."]

    return {
        "paper_id": paper_id,
        "title": title,
        "problem_statement": problem_statement,
        "methodology_used": methods[:5] if methods else [_first_sentence(methodology_section, fallback="Method details were not clearly extracted.")],
        "key_contributions": contributions,
        "results_findings": findings,
        "limitations": limitations,
        "tags": _paper_tags(item),
    }


def _supporting_pairs(extracted_items: list[dict], contradiction_pairs: list[dict]) -> list[dict]:
    conflict_keys = {
        tuple(sorted([str(item.get("paper_a_id", "")), str(item.get("paper_b_id", ""))]))
        for item in contradiction_pairs
    }

    supports: list[dict] = []
    for i, left in enumerate(extracted_items):
        left_id = str(left.get("paper_id", "")).strip()
        left_methods = {str(x).strip().lower() for x in left.get("methods", []) if str(x).strip()}
        left_tokens = _topic_tokens(left)
        for right in extracted_items[i + 1 :]:
            right_id = str(right.get("paper_id", "")).strip()
            if not left_id or not right_id:
                continue
            if tuple(sorted([left_id, right_id])) in conflict_keys:
                continue

            right_methods = {str(x).strip().lower() for x in right.get("methods", []) if str(x).strip()}
            right_tokens = _topic_tokens(right)
            if not (left_methods.intersection(right_methods) or left_tokens.intersection(right_tokens)):
                continue

            supports.append(
                {
                    "paper_a": left_id,
                    "paper_b": right_id,
                    "support_signal": "Shared methods/topics with no direct contradiction signal.",
                }
            )
            if len(supports) >= 12:
                return supports
    return supports


def _timeline_progression(extracted_items: list[dict]) -> list[dict]:
    timeline = []
    sorted_items = sorted(extracted_items, key=lambda item: _safe_year(item.get("year", "")))
    for item in sorted_items:
        timeline.append(
            {
                "year": str(item.get("year", "Unknown")),
                "paper_id": str(item.get("paper_id", "")),
                "title": str(item.get("title", "")),
                "progression_note": _first_sentence(
                    " ".join(str(item.get("sections", {}).get("conclusion", "")).split()),
                    fallback="Contributes additional evidence to the evolving topic.",
                ),
            }
        )
    return timeline


def _render_executive_summary_markdown(summary: dict) -> str:
    lines = ["# Executive Summary", "", "## Individual Paper Summaries"]
    for paper in summary.get("individual_paper_summaries", []):
        lines.append(f"### {paper.get('title', 'Untitled')}")
        lines.append(f"- Problem Statement: {paper.get('problem_statement', '')}")
        lines.append(f"- Methodology Used: {', '.join(paper.get('methodology_used', [])[:4])}")
        lines.append(f"- Key Contributions: {'; '.join(paper.get('key_contributions', [])[:3])}")
        lines.append(f"- Results/Findings: {'; '.join(paper.get('results_findings', [])[:3])}")
        lines.append(f"- Limitations: {'; '.join(paper.get('limitations', [])[:2])}")
        lines.append("")

    cross = summary.get("cross_paper_analysis", {})
    lines.append("## Cross-Paper Analysis")
    lines.append("- Common Themes: " + "; ".join(
        [f"{item.get('theme', '')}: {', '.join(item.get('evidence', [])[:4])}" for item in cross.get("common_themes", [])]
    ))
    lines.append("- Methodology Differences: " + "; ".join(
        [f"{item.get('paper_id', '')} -> {', '.join(item.get('notable_method_choices', [])[:3])}" for item in cross.get("methodology_differences", [])[:6]]
    ))
    lines.append("- Conflicting Conclusions: " + "; ".join(
        [f"{item.get('paper_a', '')} vs {item.get('paper_b', '')}: {item.get('explanation', '')}" for item in cross.get("conflicting_conclusions", [])[:5]]
    ))
    lines.append("- Supporting Papers: " + "; ".join(
        [f"{item.get('paper_a', '')} and {item.get('paper_b', '')}" for item in cross.get("supporting_or_validating_pairs", [])[:8]]
    ))
    lines.append("- Evolution of Ideas: " + "; ".join(
        [f"{item.get('year', '')} {item.get('paper_id', '')}: {item.get('progression_note', '')}" for item in cross.get("evolution_of_ideas", [])[:8]]
    ))

    critical = summary.get("critical_insights", {})
    lines.append("")
    lines.append("## Critical Insights")
    lines.append("- Most Effective Approaches: " + "; ".join(
        [f"{item.get('approach', '')} ({item.get('supporting_papers', 0)} papers)" for item in critical.get("most_effective_approaches", [])[:6]]
    ))
    lines.append("- Research Gaps: " + "; ".join(critical.get("research_gaps_identified", [])[:8]))
    lines.append("- Opportunities for Future Work: " + "; ".join(critical.get("opportunities_for_future_work", [])[:8]))
    return "\n".join(lines)


def _first_sentence(text: str, fallback: str) -> str:
    compact = " ".join(str(text).split())
    if not compact:
        return fallback
    parts = re.split(r"(?<=[.!?])\s+", compact)
    first = parts[0].strip() if parts else ""
    if not first:
        return fallback
    return first[:320]


def _top_distinct_lines(lines: list, max_items: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        compact = " ".join(str(raw).split())
        if not compact:
            continue
        normalized = compact.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(compact[:320])
        if len(output) >= max_items:
            break
    return output or ["Not enough extracted evidence for a specific summary line."]


def _extract_limitations(conclusion_text: str) -> list[str]:
    if not conclusion_text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", conclusion_text)
    signals = []
    for sentence in sentences:
        compact = " ".join(sentence.split())
        low = compact.lower()
        if any(token in low for token in ["limitation", "future", "however", "challenge", "uncertain"]):
            signals.append(compact[:280])
        if len(signals) >= 3:
            break
    return signals


def _paper_tags(item: dict) -> list[str]:
    tags: list[str] = []
    text_blob = " ".join(
        [
            str(item.get("title", "")),
            " ".join(str(x) for x in item.get("methods", [])),
            " ".join(str(x) for x in item.get("datasets", [])),
        ]
    ).lower()
    mapping = {
        "ml": ["learning", "neural", "transformer", "cnn", "rnn", "bert"],
        "health": ["clinical", "diet", "obesity", "metabolic", "diabetes", "patient"],
        "nlp": ["language", "text", "bert", "retrieval", "question answering"],
        "cv": ["image", "vision", "cifar", "imagenet", "object detection"],
    }
    for tag, keywords in mapping.items():
        if any(keyword in text_blob for keyword in keywords):
            tags.append(tag)
    if not tags:
        tags.append("research")
    return tags


def _topic_tokens(item: dict) -> set[str]:
    source = " ".join(
        [
            str(item.get("title", "")),
            str(item.get("sections", {}).get("abstract", "")),
            str(item.get("sections", {}).get("introduction", "")),
        ]
    ).lower()
    return {token for token in re.findall(r"[a-z]{5,}", source) if token not in STOPWORDS}


def _safe_year(raw_year: str) -> int:
    try:
        return int(str(raw_year).strip())
    except Exception:
        return 9999


def _read_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _split_sections(text: str) -> dict[str, str]:
    normalized_lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections = {name: [] for name in SECTION_ORDER}
    current = "abstract"

    for line in normalized_lines:
        lowered = line.lower()
        next_section = _match_section(lowered)
        if next_section:
            current = next_section
            continue
        sections[current].append(line)

    compact = {}
    for name in SECTION_ORDER:
        compact[name] = " ".join(sections[name])[:10000]
    return compact


def _match_section(line: str) -> str:
    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            if line == alias or line.startswith(alias + " "):
                return section
    return ""


def _extract_claims(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    claims = []
    for sentence in sentences:
        clean = " ".join(sentence.split())
        if len(clean) < 60 or len(clean) > 400:
            continue
        lowered = clean.lower()
        if any(hint in lowered for hint in CLAIM_HINTS):
            claims.append(clean)
        if len(claims) >= 25:
            break
    return claims


def _extract_keywords(text: str, hints: list[str]) -> list[str]:
    lowered = text.lower()
    found = []
    for hint in hints:
        if hint in lowered and hint not in found:
            found.append(hint)
    return found


def _count_items(items: list[str]) -> list[dict]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda value: value[1], reverse=True)
    return [{"name": name, "count": count} for name, count in sorted_items]


def _find_contradictions(claims: list[dict]) -> list[dict]:
    llm_items = _find_contradictions_with_groq(claims)
    if llm_items:
        return llm_items
    return _find_simple_contradictions(claims)


def _find_simple_contradictions(claims: list[dict]) -> list[dict]:
    contradictions = []
    positive_words = ["improve", "better", "outperform", "increase"]
    negative_words = ["worse", "fails", "decrease", "does not improve"]

    for index, left in enumerate(claims):
        left_text = left["claim"].lower()
        left_positive = any(word in left_text for word in positive_words)
        left_negative = any(word in left_text for word in negative_words)
        if not (left_positive or left_negative):
            continue

        for right in claims[index + 1 :]:
            if left["paper_id"] == right["paper_id"]:
                continue
            right_text = right["claim"].lower()
            right_positive = any(word in right_text for word in positive_words)
            right_negative = any(word in right_text for word in negative_words)

            if left_positive and right_negative:
                contradictions.append(
                    {
                        "paper_a": left,
                        "paper_b": right,
                        "reason": _heuristic_contradiction_reason(
                            left_claim=left.get("claim", ""),
                            right_claim=right.get("claim", ""),
                            direction="positive_vs_negative",
                        ),
                        "confidence": "medium",
                        "mode": "heuristic",
                    }
                )
            if left_negative and right_positive:
                contradictions.append(
                    {
                        "paper_a": left,
                        "paper_b": right,
                        "reason": _heuristic_contradiction_reason(
                            left_claim=left.get("claim", ""),
                            right_claim=right.get("claim", ""),
                            direction="negative_vs_positive",
                        ),
                        "confidence": "medium",
                        "mode": "heuristic",
                    }
                )

            if len(contradictions) >= 20:
                return contradictions
    return contradictions


def _heuristic_contradiction_reason(left_claim: str, right_claim: str, direction: str) -> str:
    left_clean = " ".join(str(left_claim).split())[:220]
    right_clean = " ".join(str(right_claim).split())[:220]

    if direction == "positive_vs_negative":
        summary = "One claim reports improvement while the other reports degradation or no improvement on overlapping terms."
    else:
        summary = "One claim reports degradation or no improvement while the other reports improvement on overlapping terms."

    return (
        summary
        + f" Claim A: '{left_clean}'. Claim B: '{right_clean}'."
    )


def _suggest_gaps(methods: list[dict], datasets: list[dict], contradiction_count: int) -> list[str]:
    gaps = []
    if not methods:
        gaps.append("Method details are sparse across extracted papers.")
    if not datasets:
        gaps.append("Dataset mentions are sparse; benchmark comparison may be weak.")
    if contradiction_count > 5:
        gaps.append("Several contradictory claims need manual verification.")
    if not gaps:
        gaps.append("No major structural gap detected from the extracted snapshot.")
    return gaps


def _build_detailed_report(
    extracted_items: list[dict],
    contradictions: list[dict],
    top_methods: list[dict],
    top_datasets: list[dict],
    gaps: list[str],
) -> dict:
    papers = _paper_profiles(extracted_items)
    contradiction_analysis = _build_contradiction_analysis(contradictions=contradictions, papers=papers)
    unresolved_areas = _build_unresolved_areas(
        papers=papers,
        top_methods=top_methods,
        top_datasets=top_datasets,
        gaps=gaps,
        contradiction_pair_count=len(contradiction_analysis["pairwise_contradictions"]),
    )
    future_steps = _build_future_steps(
        contradiction_pair_count=len(contradiction_analysis["pairwise_contradictions"]),
        unresolved_areas=unresolved_areas,
    )
    recommended_actions = _build_recommended_actions(
        contradiction_pair_count=len(contradiction_analysis["pairwise_contradictions"]),
        unresolved_areas=unresolved_areas,
    )

    return {
        "paper_profiles": papers,
        "contradiction_analysis": contradiction_analysis,
        "gaps_between_papers": unresolved_areas,
        "future_steps": future_steps,
        "recommended_actions": recommended_actions,
        "not_addressed": [area["gap"] for area in unresolved_areas],
        "what_to_work_on_next": [
            {
                "priority": action["priority"],
                "focus": action["action"],
                "why": action["rationale"],
            }
            for action in recommended_actions
        ],
    }


def _paper_profiles(extracted_items: list[dict]) -> list[dict]:
    profiles: list[dict] = []
    for item in extracted_items:
        paper_id = str(item.get("paper_id", "")).strip()
        if not paper_id:
            continue
        profiles.append(
            {
                "paper_id": paper_id,
                "title": str(item.get("title", "")).strip(),
                "year": str(item.get("year", "")).strip(),
                "claim_count": len(item.get("claims", [])),
                "method_count": len(item.get("methods", [])),
                "dataset_count": len(item.get("datasets", [])),
            }
        )
    return profiles


def _build_contradiction_analysis(contradictions: list[dict], papers: list[dict]) -> dict:
    paper_index: dict[str, dict] = {}
    for paper in papers:
        paper_id = paper["paper_id"]
        paper_index[paper_id] = {
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "year": paper.get("year", ""),
            "contradiction_count": 0,
            "contradicts_with": [],
            "examples": [],
        }

    pair_map: dict[str, dict] = {}
    for item in contradictions:
        left = item.get("paper_a", {})
        right = item.get("paper_b", {})
        left_id = str(left.get("paper_id", "")).strip()
        right_id = str(right.get("paper_id", "")).strip()
        if not left_id or not right_id or left_id == right_id:
            continue

        reason = " ".join(str(item.get("reason", "")).split())[:300]
        claim_a = " ".join(str(left.get("claim", "")).split())[:280]
        claim_b = " ".join(str(right.get("claim", "")).split())[:280]
        if not reason:
            reason = _heuristic_contradiction_reason(
                left_claim=claim_a,
                right_claim=claim_b,
                direction="positive_vs_negative",
            )[:300]

        for source_id, target_id in [(left_id, right_id), (right_id, left_id)]:
            if source_id not in paper_index:
                paper_index[source_id] = {
                    "paper_id": source_id,
                    "title": "",
                    "year": "",
                    "contradiction_count": 0,
                    "contradicts_with": [],
                    "examples": [],
                }
            profile = paper_index[source_id]
            profile["contradiction_count"] += 1
            if target_id not in profile["contradicts_with"]:
                profile["contradicts_with"].append(target_id)

        example = {
            "paper_a_id": left_id,
            "paper_b_id": right_id,
            "claim_a": claim_a,
            "claim_b": claim_b,
            "reason": reason,
            "confidence": str(item.get("confidence", "medium")),
            "mode": str(item.get("mode", "unknown")),
        }
        if len(paper_index[left_id]["examples"]) < 3:
            paper_index[left_id]["examples"].append(example)
        if len(paper_index[right_id]["examples"]) < 3:
            paper_index[right_id]["examples"].append(example)

        key = "::".join(sorted([left_id, right_id]))
        if key not in pair_map:
            pair_map[key] = {
                "paper_a_id": min(left_id, right_id),
                "paper_b_id": max(left_id, right_id),
                "conflict_count": 0,
                "reasons": [],
                "examples": [],
            }
        pair_map[key]["conflict_count"] += 1
        if reason and reason not in pair_map[key]["reasons"]:
            pair_map[key]["reasons"].append(reason)
        if len(pair_map[key]["examples"]) < 3:
            pair_map[key]["examples"].append(example)

    per_paper = sorted(
        paper_index.values(),
        key=lambda item: (item.get("contradiction_count", 0), item.get("paper_id", "")),
        reverse=True,
    )
    pairwise = sorted(
        pair_map.values(),
        key=lambda item: (item.get("conflict_count", 0), item.get("paper_a_id", ""), item.get("paper_b_id", "")),
        reverse=True,
    )

    return {
        "total_contradiction_items": len(contradictions),
        "total_contradicting_pairs": len(pairwise),
        "pairwise_contradictions": pairwise,
        "per_paper_contradictions": per_paper,
    }


def _build_unresolved_areas(
    papers: list[dict],
    top_methods: list[dict],
    top_datasets: list[dict],
    gaps: list[str],
    contradiction_pair_count: int,
) -> list[dict]:
    unresolved: list[dict] = []

    if contradiction_pair_count > 0:
        unresolved.append(
            {
                "gap": "Cross-paper contradictions remain unresolved.",
                "impact": "high",
                "evidence": f"{contradiction_pair_count} contradictory paper pairs detected.",
            }
        )

    sparse_methods = [paper["paper_id"] for paper in papers if paper.get("method_count", 0) == 0]
    if sparse_methods:
        unresolved.append(
            {
                "gap": "Some papers do not clearly report methods.",
                "impact": "medium",
                "evidence": f"Missing method details in {len(sparse_methods)} papers.",
            }
        )

    sparse_datasets = [paper["paper_id"] for paper in papers if paper.get("dataset_count", 0) == 0]
    if sparse_datasets:
        unresolved.append(
            {
                "gap": "Dataset coverage is incomplete across papers.",
                "impact": "medium",
                "evidence": f"Missing dataset references in {len(sparse_datasets)} papers.",
            }
        )

    if len(top_methods) <= 2 and papers:
        unresolved.append(
            {
                "gap": "Method diversity appears limited.",
                "impact": "medium",
                "evidence": f"Only {len(top_methods)} prominent methods extracted.",
            }
        )

    if len(top_datasets) <= 2 and papers:
        unresolved.append(
            {
                "gap": "Benchmark diversity appears limited.",
                "impact": "medium",
                "evidence": f"Only {len(top_datasets)} prominent datasets extracted.",
            }
        )

    for gap in gaps:
        if "no major structural gap" in gap.lower():
            continue
        unresolved.append(
            {
                "gap": gap,
                "impact": "medium",
                "evidence": "Detected from aggregate extraction summary.",
            }
        )

    if not unresolved:
        unresolved.append(
            {
                "gap": "No major unresolved area identified from current extraction snapshot.",
                "impact": "low",
                "evidence": "Signals are relatively aligned in available inputs.",
            }
        )

    return unresolved


def _build_future_steps(contradiction_pair_count: int, unresolved_areas: list[dict]) -> list[str]:
    steps: list[str] = []

    if contradiction_pair_count > 0:
        steps.append("Run targeted replication or ablation experiments for contradictory claim pairs.")
    steps.append("Expand paper collection with newer and domain-adjacent studies.")
    steps.append("Standardize evaluation metrics so paper outcomes can be compared directly.")

    if any("dataset" in str(item.get("gap", "")).lower() for item in unresolved_areas):
        steps.append("Introduce additional benchmark datasets to reduce dataset-selection bias.")
    if any("method" in str(item.get("gap", "")).lower() for item in unresolved_areas):
        steps.append("Test alternative model families to reduce single-method dependence.")

    return steps[:6]


def _build_recommended_actions(contradiction_pair_count: int, unresolved_areas: list[dict]) -> list[dict]:
    actions: list[dict] = []

    if contradiction_pair_count > 0:
        actions.append(
            {
                "priority": 1,
                "action": "Resolve high-impact contradictions first",
                "rationale": "Conflicting evidence blocks confident decision-making.",
                "next_step": "Assign each contradictory pair to a validation owner and define adjudication criteria.",
            }
        )

    actions.append(
        {
            "priority": 2,
            "action": "Close evidence coverage gaps",
            "rationale": "Sparse methods/datasets can hide failure modes.",
            "next_step": "Extract or add papers that explicitly report methodology and benchmark setup.",
        }
    )

    if any(str(item.get("impact", "")).lower() == "high" for item in unresolved_areas):
        actions.append(
            {
                "priority": 3,
                "action": "Create a contradiction monitoring loop",
                "rationale": "High-impact gaps should be re-evaluated as new papers are added.",
                "next_step": "Re-run analysis on each ingestion batch and track trend of contradiction pairs.",
            }
        )

    actions.append(
        {
            "priority": 4,
            "action": "Translate gaps into experiment backlog",
            "rationale": "Research decisions improve when unanswered questions map to concrete tests.",
            "next_step": "Convert unresolved areas into hypotheses, metrics, owners, and deadlines.",
        }
    )

    return actions[:6]


def _load_extracted_items(extracted_dir: Path, paper_ids: list[str] | None = None) -> list[dict]:
    extracted: list[dict] = []
    for path in sorted(extracted_dir.glob("*.json")):
        if paper_ids and path.stem not in set(paper_ids):
            continue
        extracted.append(json.loads(path.read_text(encoding="utf-8")))
    return extracted


def _extract_reference_lines(extracted_dir: Path, paper_ids: list[str] | None = None) -> list[str]:
    return _extract_reference_lines_from_items(_load_extracted_items(extracted_dir, paper_ids))


def _extract_reference_lines_from_items(items: list[dict]) -> list[str]:
    references: list[str] = []
    for payload in items:
        for candidate in payload.get("references", []):
            compact = " ".join(str(candidate).split())
            if compact and compact not in references:
                references.append(compact)
        if len(references) >= 200:
            return references

        section_values = payload.get("sections", {}).values()
        text = " ".join(section_values) if section_values else ""
        for line in _extract_reference_candidates(text):
            if line not in references:
                references.append(line)
            if len(references) >= 200:
                return references

    return references


def _extract_with_groq(title: str, text: str) -> dict:
    if not _groq_api_keys() or not text.strip():
        return {"mode": "heuristic", "claims": [], "methods": [], "datasets": []}

    trimmed_text = text[:12000]
    prompt = (
        "Extract research signals from paper text. Return JSON only with schema "
        "{\"claims\": [string], \"methods\": [string], \"datasets\": [string]}. "
        "Use concise phrases, max 8 items per list, no duplicates.\n"
        f"Title: {title}\n"
        f"Text:\n{trimmed_text}"
    )

    payload = _groq_chat(
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=450,
    )
    if not payload:
        return {"mode": "heuristic", "claims": [], "methods": [], "datasets": []}

    content = _extract_chat_content(payload)
    parsed = _parse_json_object(content)
    if not parsed:
        return {"mode": "heuristic", "claims": [], "methods": [], "datasets": []}

    return {
        "mode": "groq",
        "model": payload.get("model", ""),
        "claims": _sanitize_text_list(parsed.get("claims", []), max_items=25, max_chars=420),
        "methods": _sanitize_text_list(parsed.get("methods", []), max_items=15, max_chars=120),
        "datasets": _sanitize_text_list(parsed.get("datasets", []), max_items=15, max_chars=120),
    }


def _find_contradictions_with_groq(claims: list[dict]) -> list[dict]:
    if not _groq_api_keys():
        return []

    candidate_pairs: list[dict] = []
    max_checks = 40
    for index, left in enumerate(claims):
        for right in claims[index + 1 :]:
            if left.get("paper_id") == right.get("paper_id"):
                continue
            if not _claims_are_related(left.get("claim", ""), right.get("claim", "")):
                continue
            candidate_pairs.append({"paper_a": left, "paper_b": right})
            if len(candidate_pairs) >= max_checks:
                break
        if len(candidate_pairs) >= max_checks:
            break

    if not candidate_pairs:
        return []

    verdicts = _batched_contradiction_verdicts(candidate_pairs=candidate_pairs)
    if not verdicts:
        return []

    by_index: dict[int, dict] = {}
    for item in verdicts:
        raw_index = item.get("index")
        if isinstance(raw_index, int):
            by_index[raw_index] = item

    contradictions: list[dict] = []
    for pair_index, pair in enumerate(candidate_pairs):
        verdict = by_index.get(pair_index, {})
        if not bool(verdict.get("contradiction", False)):
            continue
        contradictions.append(
            {
                "paper_a": pair["paper_a"],
                "paper_b": pair["paper_b"],
                "reason": str(verdict.get("reason", "")).strip()[:300],
                "mode": "groq",
            }
        )
        if len(contradictions) >= 20:
            break

    return contradictions


def _batched_contradiction_verdicts(candidate_pairs: list[dict]) -> list[dict]:
    pair_lines = []
    for index, pair in enumerate(candidate_pairs):
        left = pair["paper_a"]
        right = pair["paper_b"]
        pair_lines.append(
            {
                "index": index,
                "paper_a_id": left.get("paper_id", ""),
                "paper_b_id": right.get("paper_id", ""),
                "claim_a": left.get("claim", ""),
                "claim_b": right.get("claim", ""),
            }
        )

    compact_pairs = json.dumps(pair_lines, ensure_ascii=True)
    prompt = (
        "For each pair of claims, decide whether the claims directly contradict each other on the same topic. "
        "Be strict and only mark true for explicit conflict. "
        "Return JSON only with schema {\"pairs\": [{\"index\": int, \"contradiction\": boolean, \"reason\": string}]}.\n"
        f"Pairs JSON:\n{compact_pairs}"
    )

    payload = _groq_chat(
        messages=[
            {
                "role": "system",
                "content": "Return valid JSON only. Keep reasons short and factual.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1200,
    )
    if not payload:
        return []

    content = _extract_chat_content(payload)
    parsed = _parse_json_object(content)
    if not parsed:
        return []

    pairs_obj = parsed.get("pairs", [])
    if not isinstance(pairs_obj, list):
        return []

    verdicts: list[dict] = []
    for item in pairs_obj:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        contradiction = bool(item.get("contradiction", False))
        reason = str(item.get("reason", "")).strip()[:300]
        if not isinstance(index, int):
            continue
        verdicts.append(
            {
                "index": index,
                "contradiction": contradiction,
                "reason": reason,
            }
        )
    return verdicts


def _groq_chat(messages: list[dict], max_tokens: int) -> dict | None:
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    keys = _groq_api_keys()
    if not keys:
        return None

    with httpx.Client(timeout=25.0) as client:
        for _ in range(len(keys)):
            api_key = _next_groq_api_key(keys)
            try:
                response = client.post(
                    GROQ_URL,
                    headers={
                        "authorization": f"Bearer {api_key}",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        "response_format": {"type": "json_object"},
                        "messages": messages,
                    },
                )
                if response.status_code == 429 or response.status_code >= 500:
                    continue
                response.raise_for_status()
                payload = response.json()
            except Exception:  # noqa: BLE001
                continue

            if isinstance(payload, dict):
                return payload

    return None


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


def _sanitize_text_list(items: list, max_items: int, max_chars: int) -> list[str]:
    clean: list[str] = []
    for item in items:
        text = " ".join(str(item).split())
        if not text or text in clean:
            continue
        clean.append(text[:max_chars])
        if len(clean) >= max_items:
            break
    return clean


def _extract_reference_candidates(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    references: list[str] = []

    patterns = [
        re.compile(r"^\[\d+\]\s+.+"),
        re.compile(r"^[A-Z][A-Za-z\-\s,.'&]+\(?(19|20)\d{2}\)?.+"),
    ]

    for line in lines:
        normalized = " ".join(line.split())
        if len(normalized) < 25 or len(normalized) > 260:
            continue
        if any(pattern.match(normalized) for pattern in patterns) and normalized not in references:
            references.append(normalized)
        if len(references) >= 80:
            break
    return references


def _recent_works(items: list[dict]) -> list[dict]:
    years = []
    for item in items:
        year_str = str(item.get("year", "")).strip()
        if year_str.isdigit():
            years.append(int(year_str))
    if not years:
        return []

    latest = max(years)
    threshold = latest - 1
    recent = []
    for item in items:
        year_str = str(item.get("year", "")).strip()
        if not year_str.isdigit():
            continue
        year = int(year_str)
        if year >= threshold:
            recent.append(
                {
                    "paper_id": item.get("paper_id", ""),
                    "title": item.get("title", ""),
                    "year": year_str,
                    "source": item.get("source", ""),
                }
            )
    return recent


def _build_in_corpus_edges(items: list[dict]) -> list[dict]:
    by_title: dict[str, str] = {}
    for item in items:
        title = str(item.get("title", "")).strip().lower()
        paper_id = str(item.get("paper_id", "")).strip()
        if title and paper_id:
            by_title[title] = paper_id

    edges: list[dict] = []
    for item in items:
        source_id = str(item.get("paper_id", "")).strip()
        if not source_id:
            continue
        for ref in item.get("references", []):
            ref_lower = str(ref).lower()
            for title, target_id in by_title.items():
                if target_id == source_id:
                    continue
                if title and title in ref_lower:
                    edge = {"source_paper_id": source_id, "target_paper_id": target_id}
                    if edge not in edges:
                        edges.append(edge)
    return edges


def _build_decision_recommendations(
    contradiction_count: int,
    gap_count: int,
    method_count: int,
    recent_count: int,
) -> list[str]:
    recommendations: list[str] = []

    if contradiction_count > 0:
        recommendations.append("Prioritize replication studies for claims flagged as contradictory.")
    if gap_count > 0:
        recommendations.append("Address top unanswered questions before locking product direction.")
    if method_count <= 2:
        recommendations.append("Expand method diversity to reduce single-approach risk.")
    if recent_count == 0:
        recommendations.append("Add newer papers to reduce stale evidence risk.")

    if not recommendations:
        recommendations.append("Current evidence appears aligned; proceed with targeted validation experiments.")

    return recommendations


def _claims_are_related(left: str, right: str) -> bool:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return False
    return len(left_tokens.intersection(right_tokens)) >= 1


def _token_set(text: str) -> set[str]:
    tokens = re.findall(r"[a-z]{5,}", text.lower())
    return {token for token in tokens if token not in STOPWORDS}


def _load_recent_debates() -> list[dict]:
    """Load recent structured debate results from disk."""
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = root_dir / "data"
    debates_dir = data_dir / "debates"

    if not debates_dir.exists():
        return []

    debates = []
    for debate_file in sorted(debates_dir.glob("*.json"), reverse=True)[:5]:  # Last 5
        try:
            debate_data = json.loads(debate_file.read_text(encoding="utf-8"))
            verdict = debate_data.get("verdict_card", {})
            debates.append({
                "file": debate_file.name,
                "paper_A": debate_data.get("paper_A", {}),
                "paper_B": debate_data.get("paper_B", {}),
                "winner": verdict.get("winner"),
                "total_score_A": verdict.get("total_score_A"),
                "total_score_B": verdict.get("total_score_B"),
                "narrative": verdict.get("narrative", ""),
            })
        except Exception:
            pass

    return debates
