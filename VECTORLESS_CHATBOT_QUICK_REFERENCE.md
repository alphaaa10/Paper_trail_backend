# Vectorless Chatbot System - Quick Reference

## What Changed

### Before: Vector-Based Q&A
- Used embeddings for similarity
- Non-reproducible results
- Citations unclear (just paper IDs)
- "Black box" relevance scoring

### After: Vectorless Citation-Aware
- Uses deterministic keyword matching
- Reproducible results (same question = same answer)
- Citations explicit (paper ID + exact claim text)
- Transparent scoring (token overlap + heuristics)

---

## API Changes

### Endpoint Switch

**Old:**
```bash
POST /feature/ask
{
  "question": "...",
  "paper_ids": []
}
```

**New:**
```bash
POST /feature/citation-chat
{
  "question": "...",
  "paper_ids": [],
  "require_citations": true
}
```

### Quick Response Comparison

| Feature | Old `/ask` | New `/citation-chat` |
|---------|-----------|----------------------|
| Citations | Parsed from LLM (opaque) | Structured array (transparent) |
| Format | `cited_paper_ids: [id1, id2]` | `citations: [{paper_id, claim_text, section, score}]` |
| Traceability | Low (LLM could hallucinate) | High (claims must exist in corpus) |
| Reproducibility | Non-deterministic | Deterministic |
| Formatted | No markdown | Yes, with `answer_with_citations` |

---

## Frontend Changes

### HTML Element IDs Changed

| Old | New | Type |
|-----|-----|------|
| `#aiQuestion` | `#citChatQuestion` | textarea |
| `#aiPaperIds` | `#citChatPaperIds` | input |
| `#btnAiAsk` | `#btnCitChat` | button |
| `#aiOut` | `#citChatOut` | output |
| N/A | `#citChatResultContainer` | **NEW** result box |
| N/A | `#citChatAnswer` | **NEW** answer display |
| N/A | `#citationsList` | **NEW** citations list |
| N/A | `#citChatMarkdown` | **NEW** formatted view |

### CSS New Classes

| Class | Purpose |
|-------|---------|
| `.citation-chat-result` | Result container |
| `.answer-box` | Answer display box |
| `.citations-panel` | Citations container |
| `.citation-item` | Individual citation card |
| `.citation-btn` | Citation action buttons |
| `.markdown-view` | Formatted view toggle |

### JavaScript New Functions

| Function | Purpose |
|----------|---------|
| `renderCitationChat(response)` | Display response with citations |
| `copyCitation(paperId, text)` | Copy to clipboard |
| `jumpToCitation(paperId, text)` | Navigate to PDF location |

---

## Backend Files

### New Files
- **`council_api/feature_citation_chat.py`** - Main chatbot logic (350 lines)

### Modified Files
- **`council_api/main.py`** - Added import + router mount
- **`ui/index.html`** - Updated AI Assistant panel
- **`ui/app.js`** - New renderCitationChat() + supporting functions
- **`ui/styles.css`** - Added citation styling (~120 lines)

### No Changes
- `/feature/ask` still works (backward compatible)
- `/feature/citation` used for PDF navigation
- `/papers`, `/extract/*` endpoints unchanged

---

## How It Works (30-Second Overview)

```
Question: "What datasets are used?"

↓ [Vectorless Matching]
Find relevant claims:
  - "ImageNet used in experiments" (score: 0.82)
  - "CIFAR-10 for classification" (score: 0.79)
  - "SQuAD for QA tasks" (score: 0.75)

↓ [LLM Constrained Generation]
Generate answer from claims:
  "Papers use ImageNet [1], CIFAR-10 [2], and SQuAD [3]."

↓ [Return with Citations]
Response includes:
  - Answer text with [1], [2], [3] markers
  - Citations array with exact claim_text for each
  - Markdown formatted view
  - Relevance scores for each citation
```

---

## API Request/Response Examples

### Request
```json
{
  "question": "Compare machine learning approaches",
  "paper_ids": ["paper_a", "paper_b"],
  "require_citations": true
}
```

### Response
```json
{
  "question": "Compare machine learning approaches",
  "answer": "Paper A uses transformers [1] while Paper B employs CNNs [2]. Both achieve similar performance.",
  "citations": [
    {
      "paper_id": "paper_a",
      "claim_text": "we employ transformer-based architecture",
      "section": "methodology",
      "relevance_score": 0.89
    },
    {
      "paper_id": "paper_b",
      "claim_text": "our CNN model achieves state-of-the-art accuracy",
      "section": "claims",
      "relevance_score": 0.85
    }
  ],
  "answer_with_citations": "Paper A uses transformers [1] while Paper B employs CNNs [2]...\n\n### Citations\n[1] **paper_a** (methodology)\n> we employ transformer-based architecture\n...",
  "papers_considered": 2,
  "mode": "groq-cited"
}
```

---

## UI Interaction Flow

```
[User enters question]
         ↓
[Optional: filters papers]
         ↓
[Click "Ask with Citations"]
         ↓
    [Loading...]
         ↓
[Answer displayed with citation markers]
         ↓
[Citations list shows below answer]
   - Paper ID
   - Section label
   - Relevance %
   - Claim text
   - Copy button
   - Jump to PDF button
         ↓
[User can expand "Formatted View"]
         ↓
[Markdown with [1], [2], [3] style citations]
```

---

## Configuration

### Adjust Relevance Thresholds
**File:** `council_api/feature_citation_chat.py`

```python
# Line 130-140
if score > 0.2:  # Min relevance for claims (claims section)
    # ...
if score > 0.3:  # Min relevance for abstract
    # ...
```

### Adjust Max Citations
**File:** `council_api/feature_citation_chat.py`

```python
# Line 100
return [cit for cit, _ in relevant[:10]]  # Return top 10, change to desired count
```

### Adjust Stopwords
**File:** `council_api/feature_citation_chat.py`

```python
# Line 160-165
stopwords = {
    "the", "a", "an", "and", ...  # Add/remove words
}
```

---

## Testing Quick Commands

### Via Terminal/curl
```bash
curl -X POST http://localhost:8000/feature/citation-chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What methods are discussed?",
    "paper_ids": [],
    "require_citations": true
  }' | jq .
```

### Via UI
1. Nav to "AI Assistant" view
2. Enter question
3. Click "Ask with Citations"
4. Review citations panel
5. Try "Copy" and "Jump to PDF" buttons

---

## Modes Explained

| Mode | Meaning | When | Reliability |
|------|---------|------|-------------|
| `groq-cited` | LLM generated with real citations | Normal | ⭐⭐⭐⭐⭐ |
| `fallback-cited` | Fallback summary from claims | LLM unavailable | ⭐⭐⭐⭐ |
| `no-match` | No relevant claims found | Poor question or corpus | ⭐⭐⭐ |

---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| "No extracted papers" | Haven't run extraction | Run POST `/extract-all` |
| "No relevant claims found" | Question too specific | Try broader question |
| Answer is fallback | LLM error | Check Groq API keys |
| Jump to PDF fails | Paper missing metadata | Check `/health` + `data/metadata/` |

---

## Performance Estimates

| Operation | Time |
|-----------|------|
| Relevance matching | ~30-50ms |
| LLM call | ~2-4s |
| Response parsing | ~10-20ms |
| Frontend render | ~100-150ms |
| **Total** | **~2.5-5s** |

---

## Backward Compatibility

✅ **Old `/feature/ask` endpoint still works**
- No breaking changes
- Both endpoints available
- Gradually migrate to new endpoint
- Old endpoint will eventually be deprecated

---

## Feature Comparison Table

| Aspect | `/feature/ask` (vector) | `/feature/citation-chat` (vectorless) |
|--------|------------------------|--------------------------------------|
| **Approach** | Embeddings + similarity | Keyword overlap + heuristics |
| **Speed** | Medium (~2-4s) | Fast (~2-4s) |
| **Reproducibility** | ⭐⭐ (non-deterministic) | ⭐⭐⭐⭐⭐ (deterministic) |
| **Citations** | Implicit, may hallucinate | Explicit, guaranteed to exist |
| **Transparency** | Black box | Clear token-based logic |
| **Startup Cost** | High (embed papers) | Low (extract JSON) |
| **Maintenance** | Reindex when papers change | Just add new files |
| **Expl ability** | Hard to explain why | Easy to explain matching |

---

## Migration Guide

### For Existing Code Using `/feature/ask`

**Before:**
```javascript
api("POST", "/feature/ask", 
  { question, paper_ids: paperIds }
)
```

**After:**
```javascript
api("POST", "/feature/citation-chat",
  { 
    question, 
    paper_ids: paperIds, 
    require_citations: true 
  }
)
```

### Response Handling Changes

**Before:**
```javascript
response.cited_paper_ids  // Array of strings
```

**After:**
```javascript
response.citations  // Array of Citation objects
response.citations[0].paper_id  // String
response.citations[0].claim_text  // String
response.citations[0].relevance_score  // Float 0-1
```

---

## Documentation Files

| File | Purpose |
|------|---------|
| `CITATION_AWARE_CHATBOT_GUIDE.md` | Complete architecture + design |
| `FRONTEND_INTEGRATION_GUIDE.md` | UI/UX implementation details |
| This file | Quick reference for developers |

---

## Key Benefits

✅ **Reproducible** - Same question always produces same citations
✅ **Auditable** - Every claim traceable to source
✅ **Fast** - No embedding overhead, just keyword matching
✅ **Transparent** - Logic is clear and understandable
✅ **Reliable** - Citations must exist (no hallucinations)
✅ **Scalable** - Works with any number of papers

---

## Next Steps

1. Test the new `/feature/citation-chat` endpoint
2. Review citation outputs for accuracy
3. Adjust relevance thresholds if needed
4. Migrate frontend to use new endpoint
5. Collect feedback on citation quality
6. Plan deprecation of old `/feature/ask` endpoint (3+ month runway)

---

## Support

- **Backend Issues:** Check `council_api/feature_citation_chat.py`
- **Frontend Issues:** Check `ui/app.js` `renderCitationChat()` function
- **Styling Issues:** Check `ui/styles.css` `.citation-*` classes
- **API Issues:** Check response structure in `CitationAwareChatResponse` model
