from __future__ import annotations

import json
from typing import Any

import requests
import streamlit as st

st.set_page_config(page_title="Council System Checker", layout="wide")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            .main {
                background: radial-gradient(circle at 10% 10%, #f4fbff 0%, #ffffff 45%),
                            radial-gradient(circle at 90% 0%, #fff7eb 0%, rgba(255,255,255,0) 30%);
            }
            .hero {
                padding: 1rem 1.2rem;
                border: 1px solid #dbe7ef;
                border-radius: 14px;
                background: linear-gradient(120deg, #f5fbff 0%, #ffffff 50%, #fff7ef 100%);
            }
            .kpi-card {
                border: 1px solid #dde8ef;
                border-radius: 12px;
                padding: 0.7rem 0.9rem;
                background: #ffffff;
            }
            .muted {
                color: #556b7a;
                font-size: 0.92rem;
            }
            .stage-pill {
                display: inline-block;
                padding: 0.2rem 0.5rem;
                border-radius: 999px;
                border: 1px solid #c8d9e3;
                background: #f4fbff;
                margin-right: 0.35rem;
                margin-bottom: 0.35rem;
                font-size: 0.82rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _api_request(method: str, base_url: str, path: str, payload: dict | None = None) -> tuple[bool, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        response = requests.request(method=method, url=url, json=payload, timeout=180)
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            data = response.json()
        else:
            data = {"status_code": response.status_code, "text": response.text}

        if response.status_code >= 400:
            return False, {"status_code": response.status_code, "error": data}
        return True, data
    except Exception as exc:  # noqa: BLE001
        return False, {"error": str(exc)}


def _compact_status(ok: bool) -> str:
    return "OK" if ok else "Failed"


def _fetch_overview(base_url: str) -> dict[str, Any]:
    health_ok, health = _api_request("GET", base_url, "/health")
    papers_ok, papers = _api_request("GET", base_url, "/papers")
    report_ok, report = _api_request("GET", base_url, "/report")

    paper_count = papers.get("count", 0) if papers_ok and isinstance(papers, dict) else 0
    claim_count = report.get("claim_count", 0) if report_ok and isinstance(report, dict) else 0
    contradiction_count = len(report.get("contradictions", [])) if report_ok and isinstance(report, dict) else 0
    top_methods = report.get("top_methods", []) if report_ok and isinstance(report, dict) else []

    return {
        "health_ok": health_ok,
        "health": health,
        "papers_ok": papers_ok,
        "papers": papers,
        "paper_count": paper_count,
        "report_ok": report_ok,
        "report": report,
        "claim_count": claim_count,
        "contradiction_count": contradiction_count,
        "top_methods": top_methods,
    }


def _show_raw(label: str, payload: Any) -> None:
    with st.expander(f"Raw: {label}"):
        st.json(payload)


def _safe_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _stage_chip(label: str, value: str) -> None:
    st.markdown(f"<span class='stage-pill'>{label}: {value}</span>", unsafe_allow_html=True)


_inject_styles()

if "last_crawl_report" not in st.session_state:
    st.session_state["last_crawl_report"] = None
if "last_papers" not in st.session_state:
    st.session_state["last_papers"] = []
if "last_extract_status" not in st.session_state:
    st.session_state["last_extract_status"] = None
if "last_report" not in st.session_state:
    st.session_state["last_report"] = None
if "last_final_report" not in st.session_state:
    st.session_state["last_final_report"] = None
if "last_knowledge_graph" not in st.session_state:
    st.session_state["last_knowledge_graph"] = None


st.markdown(
    """
    <div class="hero">
        <h2 style="margin:0;">Council Research Dashboard</h2>
        <p class="muted" style="margin:0.4rem 0 0 0;">A visual cockpit to run and inspect Crawl -> Extract -> Analyze -> Final Report.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Connection")
    base_url = st.text_input("API Base URL", value="http://127.0.0.1:8000")
    refresh_overview = st.button("Refresh Overview", use_container_width=True)

overview = _fetch_overview(base_url) if refresh_overview or True else {}

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("API Health", _compact_status(bool(overview.get("health_ok", False))))
with k2:
    st.metric("Papers", int(overview.get("paper_count", 0)))
with k3:
    st.metric("Claims", int(overview.get("claim_count", 0)))
with k4:
    st.metric("Contradictions", int(overview.get("contradiction_count", 0)))

tabs = st.tabs(["Overview", "Run Pipeline", "Papers", "Reports", "Knowledge Graph"])

with tabs[0]:
    st.subheader("System Snapshot")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("Pipeline status")
        last_run = st.session_state.get("last_crawl_report")
        if isinstance(last_run, dict):
            _stage_chip("Crawl", "done")
            _stage_chip("Extract", "done")
            _stage_chip("Analyze", "done")
            _stage_chip("Final", "done")
        else:
            _stage_chip("Crawl", "not run")
            _stage_chip("Extract", "not run")
            _stage_chip("Analyze", "not run")
            _stage_chip("Final", "not run")

        if overview.get("report_ok"):
            top_methods = _safe_list(overview.get("top_methods"))
            if top_methods:
                st.markdown("Top methods from latest report")
                chart_data = {item.get("name", ""): item.get("count", 0) for item in top_methods[:8]}
                st.bar_chart(chart_data)
            else:
                st.info("No method data yet. Run the pipeline first.")
        else:
            st.info("No latest report found yet.")

    with c2:
        st.markdown("Recent pipeline output")
        if isinstance(st.session_state.get("last_crawl_report"), dict):
            summary = st.session_state["last_crawl_report"].get("final_report", {}).get("executive_summary", {})
            st.json(summary)
        else:
            st.caption("No run in current session.")

with tabs[1]:
    st.subheader("Run Full Pipeline")
    st.caption("Primary action for demos: one click to crawl, extract, analyze, and build final report.")

    col1, col2 = st.columns(2)
    with col1:
        question = st.text_area(
            "Question",
            value="How can graph neural networks improve traffic forecasting in smart cities?",
            height=100,
        )
    with col2:
        query = st.text_input("Query (optional)", value="")

    a1, a2, a3, a4, a5 = st.columns(5)
    with a1:
        topic_count = st.number_input("Topic Count", min_value=1, max_value=12, value=4)
    with a2:
        limit_per_source = st.number_input("Limit/Source", min_value=1, max_value=50, value=10)
    with a3:
        max_papers = st.number_input("Max Papers", min_value=1, max_value=200, value=20)
    with a4:
        concurrency = st.number_input("Concurrency", min_value=1, max_value=20, value=6)
    with a5:
        top_k = st.number_input("Top K", min_value=1, max_value=50, value=10)

    target_finding = st.text_input("Target Research Finding", value="GNN improves traffic forecasting")

    if st.button("Run Full Pipeline", type="primary", use_container_width=True):
        payload = {
            "query": query,
            "question": question,
            "topic_count": int(topic_count),
            "limit_per_source": int(limit_per_source),
            "max_papers": int(max_papers),
            "concurrency": int(concurrency),
            "target_research_finding": target_finding,
            "top_k": int(top_k),
        }
        with st.spinner("Running crawl -> extract -> analyze -> final report..."):
            ok, result = _api_request("POST", base_url, "/crawl-report", payload)

        if ok and isinstance(result, dict):
            st.success("Pipeline completed.")
            st.session_state["last_crawl_report"] = result
            crawl = result.get("crawl", {})
            extract = result.get("extract", {})
            analyze = result.get("analyze", {})

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Discovered", crawl.get("discovered", 0))
                st.metric("Saved", crawl.get("saved", 0))
            with c2:
                st.metric("Extracted", extract.get("processed_count", 0))
                st.metric("Skipped", extract.get("skipped_count", 0))
            with c3:
                st.metric("Claims", analyze.get("claim_count", 0))
                st.metric("Contradictions", analyze.get("contradiction_count", 0))

            _show_raw("crawl-report", result)
        else:
            st.error("Pipeline failed. See error details below.")
            _show_raw("crawl-report-error", result)

with tabs[2]:
    st.subheader("Papers and Extraction")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        if st.button("Load Papers", use_container_width=True):
            ok, result = _api_request("GET", base_url, "/papers")
            if ok and isinstance(result, dict):
                st.session_state["last_papers"] = _safe_list(result.get("papers", []))
            else:
                st.error("Could not load papers.")
                _show_raw("papers-error", result)

        papers = st.session_state.get("last_papers", [])
        st.metric("Papers in Corpus", len(papers))
        if papers:
            st.dataframe(papers, use_container_width=True, hide_index=True)
        else:
            st.info("No papers loaded yet.")

    with c2:
        st.markdown("Extraction controls")
        if st.button("Start Background Extraction", use_container_width=True):
            ok, result = _api_request("POST", base_url, "/extract-all/background")
            st.session_state["last_extract_status"] = result
            if ok:
                st.success("Background extraction started.")
            else:
                st.error("Could not start background extraction.")
                _show_raw("extract-start", result)

        if st.button("Refresh Extraction Status", use_container_width=True):
            ok, result = _api_request("GET", base_url, "/extract-all/status")
            if ok:
                st.session_state["last_extract_status"] = result
            else:
                st.error("Could not fetch extraction status.")
                _show_raw("extract-status", result)

        status_payload = st.session_state.get("last_extract_status")
        if isinstance(status_payload, dict):
            _stage_chip("Status", str(status_payload.get("status", "unknown")))
            st.caption(
                f"Processed: {status_payload.get('processed_count', 0)} | "
                f"Skipped: {status_payload.get('skipped_count', 0)}"
            )
            if status_payload.get("error"):
                st.error(status_payload.get("error"))

        st.markdown("Extract one paper")
        paper_id = st.text_input("Paper ID")
        if st.button("Extract Selected Paper", use_container_width=True):
            if not paper_id.strip():
                st.warning("Enter a paper ID first.")
            else:
                ok, result = _api_request("POST", base_url, f"/extract/{paper_id.strip()}")
                if ok:
                    st.success("Paper extracted.")
                    st.json(result)
                else:
                    st.error("Paper extraction failed.")
                    _show_raw("extract-paper", result)

with tabs[3]:
    st.subheader("Reports")
    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("Run Analyze", use_container_width=True):
            ok, result = _api_request("POST", base_url, "/analyze", {"paper_ids": []})
            if ok:
                st.success("Analyze complete.")
                st.json(result)
            else:
                st.error("Analyze failed.")
                _show_raw("analyze-error", result)
    with r2:
        if st.button("Load Latest Report", use_container_width=True):
            ok, result = _api_request("GET", base_url, "/report")
            if ok and isinstance(result, dict):
                st.session_state["last_report"] = result
            else:
                st.error("Could not load latest report.")
                _show_raw("report-error", result)
    with r3:
        if st.button("Generate Final Report", use_container_width=True):
            ok, result = _api_request(
                "POST",
                base_url,
                "/final-report",
                {
                    "target_research_finding": "GNN improves traffic forecasting",
                    "top_k": 10,
                    "paper_ids": [],
                },
            )
            if ok and isinstance(result, dict):
                st.session_state["last_final_report"] = result.get("report", {})
                st.success("Final report generated.")
            else:
                st.error("Final report generation failed.")
                _show_raw("final-report-error", result)

    latest = st.session_state.get("last_report")
    final = st.session_state.get("last_final_report")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Latest report")
        if isinstance(latest, dict):
            st.metric("Paper Count", latest.get("paper_count", 0))
            st.metric("Claim Count", latest.get("claim_count", 0))
            st.metric("Contradictions", len(_safe_list(latest.get("contradictions", []))))

            exec_summary = latest.get("executive_summary", {}) if isinstance(latest.get("executive_summary", {}), dict) else {}
            if exec_summary:
                st.markdown("Executive summary")
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.metric("Papers Considered", exec_summary.get("papers_considered", 0))
                with s2:
                    st.metric("Unanswered Questions", exec_summary.get("unanswered_question_count", 0))
                with s3:
                    st.metric("Recent Works", exec_summary.get("recent_works_count", 0))

            markdown_summary = latest.get("executive_summary_markdown", "")
            if markdown_summary:
                st.markdown("Executive summary (readable)")
                st.markdown(markdown_summary)

            summary_json = latest.get("executive_summary_json", {})
            if isinstance(summary_json, dict) and summary_json:
                with st.expander("Executive summary (structured JSON)"):
                    st.json(summary_json)

            top_methods = _safe_list(latest.get("top_methods", []))
            if top_methods:
                table = [{"method": item.get("name", ""), "count": item.get("count", 0)} for item in top_methods]
                st.dataframe(table, use_container_width=True, hide_index=True)
            _show_raw("latest-report", latest)
        else:
            st.caption("Load latest report to view details.")

    with c2:
        st.markdown("Final report")
        if isinstance(final, dict):
            summary = final.get("executive_summary", {})
            st.metric("Papers Considered", summary.get("papers_considered", 0))
            st.metric("Unanswered Questions", summary.get("unanswered_question_count", 0))
            st.metric("Recent Works", summary.get("recent_works_count", 0))

            recommendations = _safe_list(final.get("decision_recommendations", []))
            if recommendations:
                st.markdown("Decision recommendations")
                for item in recommendations:
                    st.write(f"- {item}")
            _show_raw("final-report", final)
        else:
            st.caption("Generate final report to view details.")

with tabs[4]:
    st.subheader("Knowledge Graph")
    st.caption("Rewritten from heatmap mode: generates graph relations and React Flow-ready layout data.")

    default_ids = []
    if isinstance(st.session_state.get("last_papers"), list):
        default_ids = [item.get("paper_id", "") for item in st.session_state["last_papers"][:6] if item.get("paper_id")]

    selected_ids_text = st.text_input(
        "Paper IDs (comma-separated)",
        value=", ".join(default_ids),
    )
    target_finding = st.text_input("Target finding (optional)", value="")

    if st.button("Build Knowledge Graph", type="primary", use_container_width=True):
        selected_ids = [item.strip() for item in selected_ids_text.split(",") if item.strip()]
        if len(selected_ids) < 2:
            st.warning("Please provide at least two paper IDs.")
        else:
            payload = {
                "paper_ids": selected_ids,
                "target_research_finding": target_finding,
                "top_k": 10,
                "save_files": True,
            }
            ok, result = _api_request("POST", base_url, "/feature/heatmap", payload)
            if ok and isinstance(result, dict):
                st.session_state["last_knowledge_graph"] = result
                st.success("Knowledge graph generated.")
            else:
                st.error("Knowledge graph generation failed.")
                _show_raw("knowledge-graph-error", result)

    graph_payload = st.session_state.get("last_knowledge_graph")
    if isinstance(graph_payload, dict):
        knowledge_graph = graph_payload.get("knowledge_graph", {})
        reactflow_graph = graph_payload.get("reactflow_graph", {})

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Graph Nodes", len(_safe_list(knowledge_graph.get("nodes", []))))
        with c2:
            st.metric("Graph Edges", len(_safe_list(knowledge_graph.get("edges", []))))
        with c3:
            st.metric("React Flow Nodes", len(_safe_list(reactflow_graph.get("nodes", []))))

        nodes = _safe_list(knowledge_graph.get("nodes", []))
        if nodes:
            st.markdown("Nodes")
            st.dataframe(nodes, use_container_width=True, hide_index=True)

        edges = _safe_list(knowledge_graph.get("edges", []))
        if edges:
            st.markdown("Edges")
            st.dataframe(edges, use_container_width=True, hide_index=True)

        with st.expander("React Flow JSON"):
            st.json(reactflow_graph)

        saved = graph_payload.get("saved_files", {})
        if isinstance(saved, dict) and saved:
            st.caption(
                "Saved: "
                + ", ".join([f"{name} -> {path}" for name, path in saved.items()])
            )
    else:
        st.caption("Build a knowledge graph to view nodes, edges, and React Flow payload.")
