# Structured Debate Example Output

## Request
```bash
POST /feature/structured-debate
Content-Type: application/json

{
  "paper_id_A": "paper_2024_001",
  "paper_id_B": "paper_2024_002"
}
```

## Full Response

```json
{
  "paper_A": {
    "id": "paper_2024_001",
    "title": "Improving RAG with Adaptive Retrieval"
  },
  "paper_B": {
    "id": "paper_2024_002",
    "title": "Dense Passage Retrieval at Scale"
  },
  "axes_analysis": {
    "problem_framing": {
      "axis": "problem_framing",
      "description": "Problem Definition & Relevance",
      "paper_A": {
        "score": 8,
        "reasoning": "Clear problem statement (score: 2/3) with 3 relevance signals. Abstract length: 521 chars."
      },
      "paper_B": {
        "score": 6,
        "reasoning": "Clear problem statement (score: 1/3) with 2 relevance signals. Abstract length: 418 chars."
      },
      "winner": "A",
      "score_diff": 2
    },
    "literature_review": {
      "axis": "literature_review",
      "description": "Literature Coverage & Positioning",
      "paper_A": {
        "score": 9,
        "reasoning": "Referenced 87 sources with 5 positioning statements. Coverage score: 3/4."
      },
      "paper_B": {
        "score": 7,
        "reasoning": "Referenced 64 sources with 3 positioning statements. Coverage score: 2/4."
      },
      "winner": "A",
      "score_diff": 2
    },
    "methodology": {
      "axis": "methodology",
      "description": "Methodological Validity",
      "paper_A": {
        "score": 6,
        "reasoning": "Method diversity: 2 unique methods. Methodology detail score: 1/2. Validity signals: 3."
      },
      "paper_B": {
        "score": 8,
        "reasoning": "Method diversity: 4 unique methods. Methodology detail score: 2/2. Validity signals: 5."
      },
      "winner": "B",
      "score_diff": -2
    },
    "data_dataset": {
      "axis": "data_dataset",
      "description": "Data Quality & Credibility",
      "paper_A": {
        "score": 7,
        "reasoning": "Dataset diversity: 3 unique datasets (2 trusted). Scale signals: 2."
      },
      "paper_B": {
        "score": 8,
        "reasoning": "Dataset diversity: 5 unique datasets (3 trusted). Scale signals: 3."
      },
      "winner": "B",
      "score_diff": -1
    },
    "results_metrics": {
      "axis": "results_metrics",
      "description": "Results & Metrics Appropriateness",
      "paper_A": {
        "score": 8,
        "reasoning": "Metric rigor: 8 signals, 3/3 score. Comparison signals: 2. Evidence-backed claims: 4."
      },
      "paper_B": {
        "score": 6,
        "reasoning": "Metric rigor: 5 signals, 2/3 score. Comparison signals: 1. Evidence-backed claims: 2."
      },
      "winner": "A",
      "score_diff": 2
    },
    "reproducibility": {
      "axis": "reproducibility",
      "description": "Reproducibility & Transparency",
      "paper_A": {
        "score": 9,
        "reasoning": "Code/resource signals: 3. Implementation detail score: 2/2. Parameter transparency: 2."
      },
      "paper_B": {
        "score": 5,
        "reasoning": "Code/resource signals: 1. Implementation detail score: 1/2. Parameter transparency: 1."
      },
      "winner": "A",
      "score_diff": 4
    },
    "limitations": {
      "axis": "limitations",
      "description": "Honest Limitations Statement",
      "paper_A": {
        "score": 7,
        "reasoning": "Limitation depth: 2 signals. Conclusion quality: 2/2. Honesty signals: 1."
      },
      "paper_B": {
        "score": 5,
        "reasoning": "Limitation depth: 1 signals. Conclusion quality: 1/2. Honesty signals: 0."
      },
      "winner": "A",
      "score_diff": 2
    },
    "ethical_impact": {
      "axis": "ethical_impact",
      "description": "Ethical & Social Considerations",
      "paper_A": {
        "score": 6,
        "reasoning": "Ethical consideration signals: 2. Real-world impact discussion: 1."
      },
      "paper_B": {
        "score": 4,
        "reasoning": "Ethical consideration signals: 1. Real-world impact discussion: 0."
      },
      "winner": "A",
      "score_diff": 2
    },
    "novelty": {
      "axis": "novelty",
      "description": "Novelty vs Incremental Work",
      "paper_A": {
        "score": 7,
        "reasoning": "Novelty claims: 2. Incremental signals: 1. Method combinations: 3."
      },
      "paper_B": {
        "score": 8,
        "reasoning": "Novelty claims: 3. Incremental signals: 0. Method combinations: 4."
      },
      "winner": "B",
      "score_diff": -1
    },
    "practical_applicability": {
      "axis": "practical_applicability",
      "description": "Real-World Deployment Potential",
      "paper_A": {
        "score": 8,
        "reasoning": "Scalability signals: 2. Deployment evidence: 2. Applicability limitations: 1."
      },
      "paper_B": {
        "score": 7,
        "reasoning": "Scalability signals: 2. Deployment evidence: 1. Applicability limitations: 0."
      },
      "winner": "A",
      "score_diff": 1
    }
  },
  "verdict_card": {
    "winner": "A",
    "winner_id": "paper_2024_001",
    "winner_title": "Improving RAG with Adaptive Retrieval",
    "total_score_A": 75,
    "total_score_B": 62,
    "score_margin": 13,
    "axes_won": {
      "A": 7,
      "B": 2,
      "tie": 1
    },
    "paper_A": {
      "id": "paper_2024_001",
      "title": "Improving RAG with Adaptive Retrieval",
      "strongest_axis": "Reproducibility & Transparency"
    },
    "paper_B": {
      "id": "paper_2024_002",
      "title": "Dense Passage Retrieval at Scale",
      "strongest_axis": "Novelty vs Incremental Work"
    },
    "narrative": "📊 VERDICT: paper_2024_001 dominates across 7 of 10 axes. (Improving RAG with Adaptive Retrieval)\n🏆 Winner by score margin: 13 points\n💪 paper_2024_001's strongest axis: Reproducibility & Transparency\n📌 paper_2024_002's strongest showing: Novelty vs Incremental Work\n\n📈 Detailed breakdown by axis included in axes_analysis.\n✅ Use for literature review, grant decisions, or research direction."
  }
}
```

---

## Verdict Card Summary

### Winner
- **Paper:** Improving RAG with Adaptive Retrieval (paper_2024_001)
- **Total Score:** 75/100 vs 62/100
- **Margin:** +13 points
- **Axes Won:** 7 out of 10

### Key Findings
- **Paper A Strengths:** Problem framing, literature review, reproducibility, results clarity, honest limitations, ethical consideration, practical applicability
- **Paper B Strengths:** Methodology diversity, data quality, novelty
- **Tie:** (None) – Paper A wins decisively

### Detailed Axis Breakdown

| Axis | Paper A | Paper B | Winner | Gap |
|------|---------|---------|--------|-----|
| Problem Framing | 8 | 6 | **A** | +2 |
| Literature Review | 9 | 7 | **A** | +2 |
| Methodology | 6 | 8 | **B** | -2 |
| Data/Dataset | 7 | 8 | **B** | -1 |
| Results & Metrics | 8 | 6 | **A** | +2 |
| Reproducibility | 9 | 5 | **A** | +4 |
| Limitations | 7 | 5 | **A** | +2 |
| Ethical Impact | 6 | 4 | **A** | +2 |
| Novelty | 7 | 8 | **B** | -1 |
| Practical Applicability | 8 | 7 | **A** | +1 |
| **TOTAL** | **75** | **62** | **A** | **+13** |

---

## Key Observations

### Why Paper A Wins

1. **Reproducibility is Critical:** Paper A excels (9/10) with code availability and parameter transparency, while Paper B lags (5/10)
2. **Problem Clarity:** Paper A clearly frames the problem with multiple relevance signals
3. **Thorough Review:** Paper A references 87 sources vs 64 for Paper B, showing deeper literature engagement
4. **Honest Limitations:** Paper A acknowledges constraints more openly, building reviewer confidence

### Where Paper B Excels

1. **Methodological Rigor:** 4 unique methods vs 2 for Paper A
2. **Data Diversity:** 5 datasets (3 trusted) vs 3 datasets (2 trusted) for Paper A
3. **Novelty Claims:** Stronger positioning as new contribution vs incremental work

### Implications for Literature Review

- **Use Paper A for:** Reference implementation, reproducible baseline, robust methodology discussions
- **Use Paper B for:** Novel technique references, dataset contributions, methodology comparisons
- **Citation Guidance:** Cite A for reproducibility sections, B for novelty positioning

---

## Alternative Scenario: Expected Close Match

If the verdict had been tied (e.g., both 68 pts), the verdict card would show:

```json
"verdict_card": {
  "winner": "Tie",
  "winner_id": "Both Papers",
  "winner_title": "Equally competitive",
  "total_score_A": 68,
  "total_score_B": 68,
  "score_margin": 0,
  "axes_won": {
    "A": 4,
    "B": 4,
    "tie": 2
  },
  "narrative": "📊 VERDICT: Highly competitive papers with matched strengths.\n⚖️  paper_2024_001 and paper_2024_002 each excel in different areas.\n• paper_2024_001 leads in Problem Definition & Relevance\n• paper_2024_002 leads in Methodological Validity\n..."
}
```

---

## UI Rendering

### Verdict Card Display
```
┌──────────────────────────────────────────────────┐
│ 🏆 Improving RAG with Adaptive Retrieval wins    │
│                                                  │
│                75 pts  vs  62 pts                │
├──────────────────────────────────────────────────┤
│ 📊 VERDICT: paper_2024_001 dominates across    │
│ 7 of 10 axes.                                  │
│                                                  │
│ 🏆 Winner by score margin: 13 points            │
│ 💪 paper_2024_001's strongest axis:            │
│    Reproducibility & Transparency              │
│                                                  │
│ 📌 paper_2024_002's strongest showing:         │
│    Novelty vs Incremental Work                 │
├──────────────────────────────────────────────────┤
│ [■■■■■■■ A: 7] [■■ B: 2] [■ Tie: 1]          │
└──────────────────────────────────────────────────┘
```

### Axes Tab - Reproducibility Detail
```
┌──────────────────────────────────────────────┐
│ [Overview] [Problem] ... [Reproducibility]   │
├──────────────────────────────────────────────┤
│ Reproducibility & Transparency               │
│ Winner: A (diff: +4) ⭐ LARGEST GAP          │
│                                               │
│ Paper A (Score: 9/10)                        │
│ Code/resource signals: 3. Implementation     │
│ detail score: 2/2. Parameter transparency: 2 │
│ ✅ GitHub available, full hyperparams listed │
│                                               │
│ Paper B (Score: 5/10)                        │
│ Code/resource signals: 1. Implementation     │
│ detail score: 1/2. Parameter transparency: 1 │
│ ⚠️ Limited code release, partial details     │
└──────────────────────────────────────────────┘
```

---

## Next Steps

1. **Review Full Breakdown:** Click through each axis tab
2. **Export Verdict:** Save narrative for literature review
3. **Track in Report:** See this debate in your final report under `structured_debates`
4. **Benchmark:** Compare both papers to additional candidates

