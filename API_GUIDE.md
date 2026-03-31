# Council API Guide (Dummy Examples)

## Base URL
http://127.0.0.1:8000

## Content Type
application/json

## Auth
No auth required in current version.

## Standard Error Shape
```json
{
  "detail": "Error message"
}
```

---

## 1) Health
### Request
```bash
curl http://127.0.0.1:8000/health
```

### Dummy Response
```json
{
  "status": "ok"
}
```

---

## 2) Crawl (discovery + download)
### Request
```bash
curl -X POST http://127.0.0.1:8000/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How can graph neural networks improve traffic forecasting?",
    "topic_count": 4,
    "limit_per_source": 5,
    "max_papers": 10,
    "concurrency": 4
  }'
```

### Dummy Response
```json
{
  "query": "graph neural networks traffic forecasting",
  "topics": ["graph neural network", "traffic forecasting"],
  "discovered": 15,
  "deduped": 12,
  "attempted": 10,
  "saved": 8,
  "skipped": 1,
  "failed": 1,
  "results": [
    {
      "paper_id": "paper_1ba6db38d4c86a5e",
      "status": "saved",
      "reason": "",
      "pdf_path": "data/pdf/paper_1ba6db38d4c86a5e.pdf",
      "metadata_path": "data/metadata/paper_1ba6db38d4c86a5e.json"
    }
  ]
}
```

---

## 3) List Papers
### Request
```bash
curl http://127.0.0.1:8000/papers
```

### Dummy Response
```json
{
  "count": 2,
  "papers": [
    {
      "paper_id": "paper_1ba6db38d4c86a5e",
      "title": "Graph Neural Networks for Traffic Forecasting",
      "year": "2024",
      "source": "openalex",
      "pdf_path": "data/pdf/paper_1ba6db38d4c86a5e.pdf",
      "metadata_path": "data/metadata/paper_1ba6db38d4c86a5e.json"
    }
  ]
}
```

---

## 4) Extract Single Paper
### Request
```bash
curl -X POST http://127.0.0.1:8000/extract/paper_1ba6db38d4c86a5e
```

### Dummy Response
```json
{
  "paper_id": "paper_1ba6db38d4c86a5e",
  "extracted_path": "data/extracted/paper_1ba6db38d4c86a5e.json",
  "claim_count": 12,
  "method_count": 4,
  "dataset_count": 3
}
```

---

## 5) Extract All (Sync)
### Request
```bash
curl -X POST http://127.0.0.1:8000/extract-all
```

### Dummy Response
```json
{
  "processed_count": 8,
  "skipped_count": 2,
  "processed": [
    {
      "paper_id": "paper_1ba6db38d4c86a5e",
      "claim_count": 12,
      "method_count": 4,
      "dataset_count": 3
    }
  ],
  "skipped": [
    {
      "paper_id": "paper_deadbeefdeadbe",
      "reason": "PDF file not found"
    }
  ]
}
```

---

## 6) Extract All (Background)
### Start Job Request
```bash
curl -X POST http://127.0.0.1:8000/extract-all/background
```

### Start Job Dummy Response
```json
{
  "status": "accepted",
  "message": "extract-all started in background"
}
```

### Status Request
```bash
curl http://127.0.0.1:8000/extract-all/status
```

### Status Dummy Response
```json
{
  "status": "running",
  "started_at": "2026-03-30T10:00:00+00:00",
  "finished_at": "",
  "processed_count": 3,
  "skipped_count": 1,
  "processed": [],
  "skipped": [],
  "error": ""
}
```

---

## 7) Analyze
### Request
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"paper_ids": []}'
```

### Dummy Response
```json
{
  "report_path": "data/reports/latest_report.json",
  "paper_count": 8,
  "claim_count": 90,
  "contradiction_count": 6,
  "gaps": [
    "Several contradictory claims need manual verification."
  ],
  "report": {
    "detailed_report": {
      "contradiction_analysis": {
        "total_contradicting_pairs": 3
      },
      "gaps_between_papers": [],
      "future_steps": [],
      "recommended_actions": [],
      "not_addressed": [],
      "what_to_work_on_next": []
    }
  }
}
```

---

## 8) Get Latest Report
### Request
```bash
curl http://127.0.0.1:8000/report
```

### Dummy Response
```json
{
  "paper_count": 8,
  "claim_count": 90,
  "top_methods": [
    {"name": "transformer", "count": 5}
  ],
  "top_datasets": [
    {"name": "METR-LA", "count": 4}
  ],
  "contradictions": [],
  "gaps": [],
  "executive_summary": {
    "papers_considered": 8,
    "unanswered_question_count": 2,
    "decision_topic_count": 6,
    "contradicting_paper_count": 3,
    "recent_works_count": 4,
    "individual_paper_summaries": [],
    "cross_paper_analysis": {},
    "critical_insights": {}
  },
  "executive_summary_json": {
    "individual_paper_summaries": [],
    "cross_paper_analysis": {
      "common_themes": [],
      "methodology_differences": [],
      "conflicting_conclusions": [],
      "supporting_or_validating_pairs": [],
      "evolution_of_ideas": []
    },
    "critical_insights": {
      "most_effective_approaches": [],
      "research_gaps_identified": [],
      "opportunities_for_future_work": []
    }
  },
  "executive_summary_markdown": "# Executive Summary\n...",
  "detailed_report": {
    "paper_profiles": [],
    "contradiction_analysis": {
      "total_contradiction_items": 0,
      "total_contradicting_pairs": 0,
      "pairwise_contradictions": [],
      "per_paper_contradictions": []
    },
    "gaps_between_papers": [],
    "future_steps": [],
    "recommended_actions": [],
    "not_addressed": [],
    "what_to_work_on_next": []
  }
}
```

---

## 9) Final Report
### Request
```bash
curl -X POST http://127.0.0.1:8000/final-report \
  -H "Content-Type: application/json" \
  -d '{
    "target_research_finding": "GNN improves traffic forecasting",
    "top_k": 10,
    "paper_ids": []
  }'
```

### Dummy Response
```json
{
  "report_path": "data/reports/final_report.json",
  "report": {
    "executive_summary": {
      "papers_considered": 8,
      "unanswered_question_count": 2,
      "decision_topic_count": 6,
      "contradicting_paper_count": 3,
      "recent_works_count": 4
    }
  }
}
```

---

## 10) Crawl + Report (One-Shot)
### Request
```bash
curl -X POST http://127.0.0.1:8000/crawl-report \
  -H "Content-Type: application/json" \
  -d '{
    "query": "",
    "question": "How can graph neural networks improve traffic forecasting in smart cities?",
    "topic_count": 4,
    "limit_per_source": 10,
    "max_papers": 20,
    "concurrency": 6,
    "target_research_finding": "GNN improves traffic forecasting",
    "top_k": 10
  }'
```

### Dummy Response
```json
{
  "crawl": {
    "discovered": 20,
    "saved": 12,
    "failed": 2
  },
  "extract": {
    "processed_count": 12,
    "skipped_count": 0
  },
  "analyze": {
    "report_path": "data/reports/latest_report.json",
    "paper_count": 12,
    "claim_count": 160,
    "contradiction_count": 8
  },
  "final_report": {
    "report_path": "data/reports/final_report.json",
    "executive_summary": {
      "papers_considered": 12
    }
  }
}
```

---

## 11) Feature: Citation Jump
### Request
```bash
curl -X POST http://127.0.0.1:8000/feature/citation \
  -H "Content-Type: application/json" \
  -d '{
    "paper_id": "paper_1ba6db38d4c86a5e",
    "claim_text": "We demonstrate significant improvements over prior baselines."
  }'
```

### Dummy Response
```json
{
  "paper_id": "paper_1ba6db38d4c86a5e",
  "claim_text": "We demonstrate significant improvements over prior baselines.",
  "pdf_path": "data/pdf/paper_1ba6db38d4c86a5e.pdf",
  "page_number": 6,
  "bbox": [88.1, 214.0, 468.5, 231.8]
}
```

---

## 12) Feature: Debate (SSE Stream)
### Request
```bash
curl -N -X POST http://127.0.0.1:8000/feature/debate \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "paper_id_A": "paper_1ba6db38d4c86a5e",
    "paper_id_B": "paper_2ba6db38d4c86a5e"
  }'
```

### Dummy Stream Chunks
```text
data: {"token":"A1: Paper A shows better MAE on METR-LA..."}

data: {"token":"B1: Paper B finds no consistent gain under sparse data..."}

event: done
data: [DONE]
```

---

## 13) Feature: Heatmap
Note: this endpoint now returns a knowledge graph payload for frontend graph rendering.

### Request
```bash
curl -X POST http://127.0.0.1:8000/feature/heatmap \
  -H "Content-Type: application/json" \
  -d '{
    "paper_ids": [
      "paper_1ba6db38d4c86a5e",
      "paper_2ba6db38d4c86a5e"
    ],
    "save_files": true
  }'
```

### Dummy Response
```json
{
  "paper_ids": [
    "paper_1ba6db38d4c86a5e",
    "paper_2ba6db38d4c86a5e"
  ],
  "paper_display_names": [
    {
      "paper_id": "paper_1ba6db38d4c86a5e",
      "display_name": "Graph Neural Networks for Traffic Forecasting (2024)"
    }
  ],
  "knowledge_graph": {
    "nodes": [
      {
        "id": "paper_1ba6db38d4c86a5e",
        "title": "Graph Neural Networks for Traffic Forecasting",
        "key_methodology": "graph neural network",
        "main_contribution": "Improves MAE under sparse traffic sensors.",
        "tags": ["ML"]
      }
    ],
    "edges": [
      {
        "source": "paper_1ba6db38d4c86a5e",
        "target": "paper_2ba6db38d4c86a5e",
        "relationship": "VALIDATES",
        "explanation": "No direct contradiction and shared method/problem signals.",
        "confidence": 0.68
      }
    ]
  },
  "reactflow_graph": {
    "nodes": [
      {
        "id": "paper_1ba6db38d4c86a5e",
        "data": {
          "label": "Graph Neural Networks for Traffic Forecasting",
          "description": "Method: graph neural network; Contribution: Improves MAE under sparse traffic sensors."
        },
        "position": {
          "x": 120,
          "y": 80
        }
      }
    ],
    "edges": [
      {
        "id": "e0",
        "source": "paper_1ba6db38d4c86a5e",
        "target": "paper_2ba6db38d4c86a5e",
        "label": "VALIDATES"
      }
    ]
  },
  "saved_files": {
    "knowledge_graph_json": "data/reports/knowledge_graph.json",
    "reactflow_graph_json": "data/reports/reactflow_graph.json"
  },
  "mode": "knowledge-graph"
}
```

---

## Single End-to-End Dummy Flow
1. POST /crawl
2. GET /papers
3. POST /extract-all/background
4. GET /extract-all/status (poll until completed)
5. POST /analyze
6. GET /report
7. POST /final-report
8. POST /feature/heatmap
9. POST /feature/citation
10. POST /feature/debate (stream)

Notes:
- `GET /report` is now the canonical place for executive summary markdown/json.
- `POST /feature/heatmap` is graph-only and should not be used for summary rendering.
