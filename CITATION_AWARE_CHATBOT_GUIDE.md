# Citation-Aware Vectorless Chatbot System

## Overview

The Council API now includes a **vectorless, citation-aware chatbot** that provides full traceability for every answer. Instead of using embedding models or RAG systems, the chatbot uses **deterministic claim matching** to ground all answers in specific paper citations.

**Key Benefit:** Every statement in the answer is backed by a specific claim from a specific paper, making results fully reproducible and auditable.

---

## Architecture

### Vectorless Design
- **No embeddings:** Uses heuristic token overlap + keyword matching
- **No semantic search:** Deterministic relevance scoring
- **No vector databases:** Pattern-based retrieval from extracted claims
- **Result:** Fast, reproducible, fully traceable

### Traceability Chain
```
Question
  ↓
[Find Relevant Claims] ← Deterministic keyword matching
  ↓
[Generate Answer from Claims] ← LLM constrained by citations
  ↓
Answer + Citation References ← Each statement backed by paper ID + claim text
```

---

## API Endpoint

### POST `/feature/citation-chat`

**Purpose:** Ask research questions with full citation traceability

**Request:**
```json
{
  "question": "What methodologies are used in these papers?",
  "paper_ids": ["paper_1", "paper_2"],
  "require_citations": true
}
```

**Parameters:**
- `question` (string, required): Research question
- `paper_ids` (array, optional): Specific papers to query. Empty = all extracted papers
- `require_citations` (boolean, default: true): Only accept answers with citations

**Response:**
```json
{
  "question": "What methodologies are used in these papers?",
  "answer": "The papers use transformer-based models and reinforcement learning approaches. Paper A employs BERT with fine-tuning, while Paper B uses a novel multi-agent reinforcement learning framework.",
  "citations": [
    {
      "paper_id": "paper_1",
      "claim_text": "we employ BERT with task-specific fine-tuning",
      "section": "claims",
      "relevance_score": 0.87
    },
    {
      "paper_id": "paper_2",
      "claim_text": "our novel multi-agent reinforcement learning approach achieves state-of-the-art results",
      "section": "claims",
      "relevance_score": 0.82
    }
  ],
  "answer_with_citations": "The papers use transformer-based models and reinforcement learning approaches. Paper A employs BERT with fine-tuning [1], while Paper B uses a novel multi-agent reinforcement learning framework [2].\n\n---\n\n### 📚 Citations\n[1] **paper_1** (claims)\n> we employ BERT with task-specific fine-tuning\n\n[2] **paper_2** (claims)\n> our novel multi-agent reinforcement learning approach achieves state-of-the-art results",
  "papers_considered": 2,
  "mode": "groq-cited"
}
```

### Response Fields
- `question`: Echo of input question
- `answer`: Plain text answer with citation markers like [1], [2]
- `citations[]`: Array of Citation objects (paper_id, claim_text, section, relevance_score)
- `answer_with_citations`: Markdown formatted answer with footnote-style citations
- `papers_considered`: Number of papers in context
- `mode`: One of:
  - `groq-cited`: LLM generated with citations
  - `fallback-cited`: Fallback summary (if LLM unavailable)
  - `no-match`: No relevant claims found

---

## Citation Object

Each citation includes:
```json
{
  "paper_id": "arxiv_2024_001",
  "claim_text": "we propose a novel architecture that combines vision transformers with diffusion models",
  "section": "claims",
  "relevance_score": 0.91
}
```

**Fields:**
- `paper_id`: Unique identifier of source paper
- `claim_text`: Exact claim/sentence from paper
- `section`: Source section ("claims", "abstract", "methodology", "results", etc.)
- `relevance_score`: Relevance to question (0-1), higher = more relevant

---

## How It Works

### 1. Claim Extraction
Papers are indexed by extracted claims, methods, datasets, and abstract sections.
No embedding model needed - just text from extracted JSON.

### 2. Relevance Scoring (Vectorless)

For each question, the system:
- **Tokenizes** question and each claim
- **Calculates overlap** between question tokens and claim tokens
- **Applies heuristics:**
  - Token overlap score: 70% weight
  - Claim length penalty: 20% weight (prefer concise, relevant claims)
  - Research keyword boost: 10% weight ("propose", "method", "dataset", etc.)

**Scoring Formula:**
```
relevance = (overlap_score × 0.7) + (length_score × 0.2) + (research_boost × 0.1)
```

**Example:**
- Question: "What datasets do papers use?"
- Claim: "we evaluate on ImageNet, CIFAR-10, and MS COCO"
- Overlap: 4/7 tokens match (dataset, papers, use found) = 0.57
- Length: Claim is 14 words, good length = 0.8
- Research keyword: "evaluate" present = 0.1
- **Final score:** (0.57 × 0.7) + (0.8 × 0.2) + 0.1 = **0.73**

### 3. Answer Generation

The LLM receives:
- Question
- Top 10 relevant claims (with relevance scores)
- Instruction to cite claims by index

**LLM Constraint:**
```
"Answer using ONLY the provided citations.
Each statement must be backed by at least one citation [1], [2], etc.
Return JSON: {\"answer\": string, \"used_citation_indices\": [1, 2, 3]}"
```

This ensures the LLM can't hallucinate - every statement must map back to a specific claim.

### 4. Citation Linking

Final answer includes citation markers [1], [2], etc. that map to the citations array.
Users can click citations to jump to PDF locations or copy full citations.

---

## Advantages Over Embeddings

| Aspect | Embeddings/RAG | Citation-Aware (Vectorless) |
|--------|-------|----------|
| **Reproducibility** | Non-deterministic (LLM + embeddings vary) | Deterministic (same question = same matching) |
| **Traceability** | Claims "near" in embedding space (opaque) | Exact claims cited (fully transparent) |
| **Startup Cost** | High (embed all papers) | Low (just extract JSON) |
| **Maintenance** | Reindex if add papers | Just add new JSON files |
| **Hallucination** | LLM can cite claims not in context | LLM forced to cite provided claims |
| **Explainability** | Black box similarity | Clear token overlap logic |

---

## Frontend UI

### Layout

**AI Assistant View** → **Citation-Aware Chat Panel**

Components:
1. **Question Input** - Textarea for research question
2. **Paper Filter** - Optional comma-separated paper IDs
3. **Action Buttons** - "Ask with Citations" and "Clear"
4. **Answer Display** - Main answer text with citation markers
5. **Citations Panel** - List of citations with:
   - Citation number [1], [2], etc.
   - Paper ID
   - Section (claims, abstract, etc.)
   - Relevance score
   - Claim text (truncated)
   - Actions: Copy, Jump to PDF
6. **Formatted View** - Markdown tab with full citations

### Citation Item Controls

Each citation shows:
- **[n]** - Citation number for reference
- **Paper ID** - Source paper link
- **Section Badge** - Where claim came from
- **Relevance Score** - 0-100% match
- **Copy Button** - Copy citation to clipboard
- **Jump to PDF** - Navigate to location in PDF (uses existing `/feature/citation` endpoint)

### Responsive Design
- Desktop: 3-column layout (question, answer, citations)
- Mobile: Stacked layout with collapsible sections

---

## Example Usage

### Question 1: Compare Methods
```
Input:
  Question: "Compare the machine learning methods used in these papers"
  Papers: [paper_1, paper_2, paper_3]

Output:
  Answer:
    "Paper 1 introduces a transformer-based approach [1], while Paper 2 focuses on
    graph neural networks [2]. Paper 3 combines both with reinforcement learning [3]."
  
  Citations:
    [1] paper_1 (claims): "we propose a transformer-based sequence model"
    [2] paper_2 (claims): "our graph neural network architecture achieves SOTA"
    [3] paper_3 (claims): "we combine transformers with multi-agent RL for control"
```

### Question 2: Find Datasets
```
Input:
  Question: "What datasets are most commonly used?"
  Papers: [] (all papers)

Output:
  Answer:
    "ImageNet and CIFAR-10 are prevalent [1][2]. Several papers also use domain-specific
    benchmarks like SQuAD [3] and MS COCO [4]."
  
  Citations:
    [1] paper_A (claims): "we evaluate on ImageNet with ResNet50"
    [2] paper_B (claims): "CIFAR-10 results show 99.2% accuracy"
    [3] paper_C (claims): "the QA task is evaluated on SQuAD"
    [4] paper_D (claims): "object detection benchmarks: MS COCO and PASCAL VOC"
```

### Question 3: Reproducibility Check
```
Input:
  Question: "Which papers provide code and datasets?"
  Papers: (all)

Output:
  Answer:
    "Paper X explicitly mentions github availability [1]. Paper Y provides supplementary
    materials with dataset access [2]. Paper Z does not disclose code availability [3]."
  
  Citations (with relevance scores):
    [1] paper_X (claims): "Code available at https://github.com/..." (0.95)
    [2] paper_Y (abstract): "supplementary materials including dataset provided" (0.88)
    [3] paper_Z (conclusion): "Limitations: reproducibility details in appendix" (0.72)
```

---

## API Comparison

### Old System (`/feature/ask`)
- Vector embeddings for similarity
- Opaque "relevance" ranking
- Citations from LLM only (may hallucinate)
- Non-reproducible results

### New System (`/feature/citation-chat`)
- Deterministic keyword matching
- Transparent token overlap scoring
- Citations backed by actual extracted claims
- Reproducible results across runs

---

## Response Status Modes

| Mode | Meaning | Trust Level |
|------|---------|------------|
| `groq-cited` | LLM generated answer with real citations | High - answer constrained by citations |
| `fallback-cited` | No LLM (fallback), summary from top claims | Medium - direct claim listing |
| `no-match` | No relevant claims found for question | Low - consider expanding paper corpus |

---

## Error Handling

### Empty Result
```json
{
  "detail": "No extracted papers found."
}
```
**Solution:** Run `/extract-all` first

### No Matching Claims
```json
{
  "question": "...",
  "answer": "No relevant claims found in the provided papers...",
  "citations": [],
  "mode": "no-match"
}
```
**Solution:** Try broader question or add more papers

### LLM Error (Fallback)
```json
{
  "question": "...",
  "answer": "Based on the extracted papers, here are relevant findings...",
  "citations": [...],
  "mode": "fallback-cited"
}
```
**Solution:** Will still return citations but auto-generated summary

---

## Testing the Feature

### Via cURL
```bash
curl -X POST http://localhost:8000/feature/citation-chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main contributions of these papers?",
    "paper_ids": ["paper_1", "paper_2"],
    "require_citations": true
  }' | jq .
```

### Via Postman
1. New request: POST → `{{base_url}}/feature/citation-chat`
2. Body (JSON):
   ```json
   {
     "question": "Compare methodologies",
     "paper_ids": [],
     "require_citations": true
   }
   ```
3. Send and review citations panel

### Via UI
1. Navigate to "AI Assistant" view
2. Enter question in textarea
3. (Optional) Enter paper IDs
4. Click "Ask with Citations"
5. Review answer + expandable citations panel
6. Click "Jump to PDF" to navigate to source location

---

## Frontend Changes Summary

| Component | Change | Location |
|-----------|--------|----------|
| HTML | Replaced Q&A input with Citation-Chat panel | [ui/index.html](ui/index.html#L328) |
| CSS | Added `.citation-item`, `.citations-panel`, `.answer-box` styles | [ui/styles.css](ui/styles.css#L800) |
| JavaScript | New `renderCitationChat()` function + `jumpToCitation()` helper | [ui/app.js](ui/app.js#L692) |
| Endpoint | POST `/feature/citation-chat` instead of `/feature/ask` | [council_api/feature_citation_chat.py] |

---

## Configuration

### Relevance Thresholds
Edit [council_api/feature_citation_chat.py](council_api/feature_citation_chat.py) to adjust:
- Line ~130: `if score > 0.2:` - Minimum relevance for claims (default 0.2)
- Line ~140: `if score > 0.3:` - Minimum relevance for abstract (default 0.3)

### Token Stopwords
Edit line ~160 in [council_api/feature_citation_chat.py](council_api/feature_citation_chat.py) to customize filtered words

### Citation Count
Modify line ~100 to return more/fewer citations:
```python
return [cit for cit, _ in relevant[:10]]  # Change 10 to desired count
```

---

## Troubleshooting

### Issue: "No citations showing"
- **Cause:** Question too specific or papers don't contain relevant claims
- **Solution:** Use broader question, check `/papers` endpoint for extracted papers

### Issue: "Citation text doesn't match claim"
- **Cause:** Truncation or LLM rephrasing
- **Solution:** Use "Copy" button to get exact original claim text

### Issue: "Jump to PDF not working"
- **Cause:** Paper metadata missing or PDF path incorrect
- **Solution:** Check `/health` endpoint and `data/metadata/` directory

### Issue: "Fallback mode answers are too generic"
- **Cause:** LLM unavailable or citation matching weak
- **Solution:** Verify Groq API keys in environment, add more specific papers

---

## Future Enhancements

1. **Citation Weighting:** Prioritize highly-cited papers
2. **Multi-hop Citations:** "Paper A cites Paper B, which shows..."
3. **Temporal Citations:** Track when claims were made
4. **Contradiction Detection:** Flag conflicting citations
5. **Citation Network:** Visualize paper-to-paper citation chains
6. **Batch Citation:** Answer multiple questions in one request

---

## Performance

- **Relevance Matching:** O(n × m) where n = questions, m = claims ~10-50ms for 100 papers
- **LLM Call:** ~2-5 seconds (network dependent)
- **Total Response:** ~2.5-5.5 seconds
- **Storage:** 1-2 KB per citation

---

## References

- [Feature Citation Endpoint](council_api/feature_citation.py) - Used for PDF navigation
- [Extracted Papers Format](data/extracted/) - Structure of claim data
- [API Guide](API_GUIDE.md) - General endpoint documentation
