# Council Backend (Hackathon MVP)

## TL;DR

- Run: `python -m pip install -r requirements.txt`
- Start API: `uvicorn app:app --reload`
- Start UI: `streamlit run streamlit_app.py`
- One-shot demo: call `POST /crawl-report`
- Output files are in `data/reports/` (`latest_report.json`, `final_report.json`)

End-to-end research pipeline for hackathon demos:

- Discover papers from OpenAlex + Semantic Scholar
- Download PDFs and extract claims, methods, and datasets (Groq-first, fallback-safe)
- Build analysis and final decision-oriented reports

## 1) Quick Start (2 minutes)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create .env from .env.example and set values:

```bash
GROQ_API_KEY=your_primary_key
GROQ_API_KEYS=key_one,key_two,key_three
GROQ_MODEL=llama-3.3-70b-versatile
CRAWL4AI_ENABLED=0
```

Notes:

- GROQ_API_KEYS is optional and enables round-robin key circulation with failover.
- GROQ_API_KEY is supported as single-key fallback.
- CRAWL4AI_ENABLED defaults to 0 and is optional.

Start API:

```bash
uvicorn app:app --reload
```

Start visual dashboard:

```bash
streamlit run streamlit_app.py
```

## 2) One-Shot Demo Call

Use this for judge/demo flow:

```bash
curl -X POST http://127.0.0.1:8000/crawl-report \
  -H "Content-Type: application/json" \
  -d '{"question":"How can graph neural networks improve traffic forecasting in smart cities?", "topic_count":4, "limit_per_source":10, "max_papers":20, "concurrency":6, "target_research_finding":"GNN improves traffic forecasting", "top_k":10}'
```

## 3) API Cheat Sheet

- GET /health: service health
- POST /crawl: discovery + download only
- GET /papers: list downloaded papers
- POST /extract/{paper_id}: extract one paper
- POST /extract-all: extract all (sync)
- POST /extract-all/background: extract all (non-blocking)
- GET /extract-all/status: background extraction status
- POST /analyze: generate latest_report.json
- GET /report: get latest report JSON
- POST /final-report: generate final_report.json
- POST /crawl-report: one-shot crawl -> extract -> analyze -> final report

## 4) Typical Flow

1. Run POST /crawl-report from Streamlit or curl.
2. Inspect papers in GET /papers.
3. Review report in GET /report.
4. Review final output from POST /final-report.

## 5) Project Structure

- app.py: API entry point
- council_api/main.py: FastAPI routes
- council_api/extraction.py: extraction + reporting logic
- research_crawler/: discovery and downloading pipeline
- streamlit_app.py: visual checker/dashboard

Generated data:

```text
data/
  pdf/
  metadata/
  extracted/
  reports/
```

## 6) Behavior Notes

- Discovery defaults to OpenAlex + Semantic Scholar.
- DuckDuckGo code exists in repo but is disabled in default search path.
- paper_id is stable: sha1(doi or normalized_title)[:16] with paper_ prefix.
