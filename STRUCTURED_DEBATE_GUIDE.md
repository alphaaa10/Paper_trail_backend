# Structured Debate Feature Guide 🏛️

## Overview

The **Structured Debate** feature enables systematic comparison of research papers across 10 methodological axes with automated verdict generation and detailed reasoning for each comparison point.

---

## 10 Debate Axes

### 1. **Problem Framing** 🎯
- Evaluates clarity of problem statement and real-world relevance
- Scores based on: presence of relevance keywords, abstract length, problem clarity
- Attacks: Artificial construction, scope inflation, missing context

### 2. **Literature Review** 📚
- Assesses coverage of prior work and positioning against existing research
- Scores based on: reference count, positioning statements, coverage depth
- Attacks: Cherry-picking, bias, outdated references

### 3. **Methodology** 🔬
- Evaluates methodological validity and appropriateness of approach
- Scores based on: method diversity, implementation detail, validity signals
- Attacks: Unrealistic assumptions, overfitting to data, lack of rigor

### 4. **Data / Dataset** 📊
- Assesses data quality, credibility, and diversity
- Scores based on: dataset count, trusted source usage, scale signals
- Attacks: Biased data, small sample size, synthetic vs real-world gap

### 5. **Results & Metrics** 📈
- Evaluates appropriateness and rigor of metrics and baselines
- Scores based on: metric count, comparison signals, evidence-backed claims
- Attacks: Metric gaming, cherry-picked baselines, statistical insignificance

### 6. **Reproducibility** 🔄
- Assesses code/data availability and clarity of procedures
- Scores based on: resource availability signals, implementation detail, parameter transparency
- Attacks: Black-box claims, missing details, hidden parameters

### 7. **Limitations** ⚠️
- Evaluates honesty and depth of stated limitations
- Scores based on: limitation statements, conclusion quality, honesty signals
- Attacks: Omitted weaknesses, buried limitations, overreaching claims

### 8. **Ethical & Social Impact** 🌍
- Considers ethical implications and real-world impact
- Scores based on: ethical consideration signals, real-world applicability discussion
- Attacks: Ignored risks, bias, misuse potential, naive assumptions

### 9. **Novelty** ✨
- Assesses novelty vs incremental contributions
- Scores based on: novelty claims, incremental signals, method combinations
- Attacks: "Paper inflation" problem, slight modifications as major claims

### 10. **Practical Applicability** 🚀
- Evaluates real-world deployment potential and scalability
- Scores based on: scalability signals, deployment evidence, applicability limitations
- Attacks: Lab success ≠ real-world success, cost-prohibitive, unscalable

---

## API Endpoints

### POST `/feature/structured-debate`
**Compare two papers on all 10 axes**

**Request:**
```json
{
  "paper_id_A": "paper_uuid_1",
  "paper_id_B": "paper_uuid_2"
}
```

**Response:**
```json
{
  "paper_A": {
    "id": "paper_uuid_1",
    "title": "Paper A Title"
  },
  "paper_B": {
    "id": "paper_uuid_2",
    "title": "Paper B Title"
  },
  "axes_analysis": {
    "problem_framing": {
      "axis": "problem_framing",
      "description": "Problem Definition & Relevance",
      "paper_A": {
        "score": 8,
        "reasoning": "Clear problem statement (score: 2/3) with 3 relevance signals. Abstract length: 456 chars."
      },
      "paper_B": {
        "score": 6,
        "reasoning": "Clear problem statement (score: 1/3) with 1 relevance signals. Abstract length: 287 chars."
      },
      "winner": "A",
      "score_diff": 2
    },
    // ... 9 more axes ...
  },
  "verdict_card": {
    "winner": "A",
    "winner_id": "paper_uuid_1",
    "winner_title": "Paper A Title",
    "total_score_A": 72,
    "total_score_B": 58,
    "score_margin": 14,
    "axes_won": {
      "A": 7,
      "B": 2,
      "tie": 1
    },
    "paper_A": {
      "id": "paper_uuid_1",
      "title": "Paper A Title",
      "strongest_axis": "Problem Definition & Relevance"
    },
    "paper_B": {
      "id": "paper_uuid_2",
      "title": "Paper B Title",
      "strongest_axis": "Practical Applicability"
    },
    "narrative": "📊 VERDICT: paper_uuid_1 dominates across 7 of 10 axes. (Paper A Title)\n🏆 Winner by score margin: 14 points\n💪 paper_uuid_1's strongest axis: Problem Definition & Relevance\n..."
  }
}
```

### GET `/feature/debates`
**List all saved debate results**

**Response:**
```json
[
  {
    "filename": "2024-01-15T14-30-45_paper_a_vs_paper_b.json",
    "paper_A": {"id": "paper_a", "title": "Paper A"},
    "paper_B": {"id": "paper_b", "title": "Paper B"},
    "winner": "A"
  }
  // ... more debates ...
]
```

### GET `/feature/debates/{debate_id}`
**Retrieve a specific debate result**

**Response:** Full debate object (same as structured-debate endpoint)

---

## Frontend UI

### Investigation View → Structured Debate Panel

**Components:**

1. **Input Section:**
   - Paper ID A input field
   - Paper ID B input field
   - "Run Structured Analysis" button
   - "Clear" button

2. **Verdict Card** (appears after analysis):
   - Winner badge with paper ID and title
   - Score display (total points A vs B)
   - Visual bar showing axes won distribution
   - Narrative summary with key findings

3. **Axes Tabs:**
   - "Overview" tab: All 10 axes at a glance
   - Per-axis tabs: Detailed comparison for each axis
   - Each tab shows: axis description, winner badge, score diff, paper A & B reasoning

4. **Navigation:**
   - Tab buttons for quick axis switching
   - Score badges (1-10) for each paper per axis
   - Color-coded winner indicators (teal for A, slate for B, neutral for tie)

---

## Scoring System

### Per-Axis Scoring
- **Scale:** 1-10 points per paper per axis
- **Methodology:** Heuristic scoring based on text signals and metadata
- **Signals:** Keyword presence, section length, content patterns, claim counts

### Verdict Determination
- **Winner Selection:** Paper with highest cumulative score across all 10 axes
- **Score Margin:** Absolute difference between totals
- **Axes Won:** Count of individual axes where each paper scored higher

### Example Scoring Breakdown
```
Axis: Methodology
  Paper A: 8/10 
    - Method diversity: 3 unique methods (2 pts)
    - Detail score: 2/2 (detailed implementation)
    - Validity signals: 3 (algorithm, architecture, parameter mentions)
    
  Paper B: 6/10
    - Method diversity: 2 unique methods (1 pt)
    - Detail score: 1/2 (sparse implementation detail)
    - Validity signals: 2 (algorithm, architecture)

Winner: Paper A (+2 score diff)
```

---

## Storage & Persistence

### Debate Results Directory
Location: `data/debates/`

**Filename Format:** `YYYY-MM-DDTHH-MM-SS_paper_id_A_vs_paper_id_B.json`

**Example:** `2024-01-15T14-30-45_arxiv-2401-1234_vs_arxiv-2401-5678.json`

**Auto-Save:** Triggered on every `/feature/structured-debate` POST request

---

## Integration with Final Report

### Report Section: `structured_debates`
The final report (from `/report` or `/final-report` endpoints) includes:
- Last 5 debate results
- Quick verdicts (winner, score, paper IDs)
- Links to full debate narratives

**Use Cases:**
- Track competing methodologies across corpus
- Document literature review findings
- Support grant/publication decisions
- Evidence for research direction pivots

---

## Usage Examples

### Example 1: Compare Methodology Approaches
```bash
curl -X POST http://localhost:8000/feature/structured-debate \
  -H "Content-Type: application/json" \
  -d '{
    "paper_id_A": "arxiv-2401-1234",
    "paper_id_B": "arxiv-2401-5678"
  }'
```

**Result:** Detailed comparison showing which paper has more rigorous, diverse, and well-documented methodology.

### Example 2: Evaluate Reproducibility for Replication Studies
Focus on the **Reproducibility** axis to identify:
- Which paper provides code/data
- Which paper has detailed enough procedures
- Which has transparent hyperparameters

### Example 3: Select Paper for Systematic Review
- Run debate against several candidates
- Review verdict cards for each
- Select paper with highest scores on relevant axes (e.g., Literature Review + Limitations for survey)

---

## Frontend Display Examples

### Verdict Card Layout
```
┌─────────────────────────────────────┐
│ 🏆 Paper A Advances wins            │
│ 72 pts vs 58 pts                    │
├─────────────────────────────────────┤
│ 📊 VERDICT: paper_id_A dominates    │
│ across 7 of 10 axes.               │
│ 🏆 Winner by score margin: 14 pts   │
│ 💪 Strongest: Problem Framing       │
│ 📌 B's strength: Practical Appeal   │
├─────────────────────────────────────┤
│ [████ A: 7] [██ Tie: 1] [███ B: 2] │
└─────────────────────────────────────┘
```

### Axes Tab View
```
┌────────────────────────────────────────────┐
│ [Overview] [Problem] [Literature] [Meth...] │
├────────────────────────────────────────────┤
│ Problem Definition & Relevance             │
│ Winner: A (diff: +2)                       │
│                                             │
│ Paper A (Score: 8/10)                      │
│ Clear problem statement (score: 2/3)       │
│ with 3 relevance signals...                │
│                                             │
│ Paper B (Score: 6/10)                      │
│ Clear problem statement (score: 1/3)       │
│ with 1 relevance signal...                 │
└────────────────────────────────────────────┘
```

---

## Tips & Best Practices

1. **Complement with Live Debate:** Use structured debate for systematic analysis, live debate for discovery
2. **Focus on Relevant Axes:** Weight axes based on your domain (e.g., Ethics for ML safety, Reproducibility for ML infrastructure)
3. **Track Decision Rationale:** Review verdict narrative when forming literature review sections
4. **Batch Comparisons:** Compare all top papers against a "reference" to navigate corpus
5. **Iterate:** Run debates before and after adding new papers to track corpus quality

---

## Technical Notes

- **Heuristic Scoring:** All scoring is heuristic-based (keyword/pattern matching), not LLM-dependent
- **Fallback Robustness:** Works even if LLM services are unavailable
- **Performance:** Structured debate completes in <100ms (no API calls)
- **Storage:** Each debate ~8-12 KB JSON file
- **Scalability:** Can store thousands of debates; index/search optional

---

## Future Enhancements

1. **LLM-Assisted Scoring:** Optional deep LLM evaluation per axis
2. **Custom Axes:** Define domain-specific evaluation criteria
3. **Batch Debates:** Compare multiple papers in tournament format
4. **Weighted Scoring:** Assign importance weights to each axis
5. **Export Formats:** Generate LaTeX tables, PDF reports from verdicts
6. **Temporal Analysis:** Track evolution of verdict as papers cite each other

---

## Questions?

Check the main README or API_GUIDE.md for general project questions.
For debates specifically, review the payload/response examples above or explore via Postman.
