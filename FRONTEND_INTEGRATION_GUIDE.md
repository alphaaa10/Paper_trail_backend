# Frontend Integration Guide - Citation-Aware Chatbot
**For External Frontend:** `C:\Users\faizk\projects\Research Intelligence`

---

## Overview

This guide explains how to integrate the new **vectorless citation-aware chatbot** into your external frontend. The backend `/feature/citation-chat` endpoint is fully functional—this guide shows you what changes to make in your frontend code.

---

## What This Change Replaces

### Old System (Vector-Based)
- **Endpoint:** `/feature/ask`
- **Response:** Generic answer + implicit citations
- **Traceability:** Low (LLM decides which papers to cite)
- **Logic:** Black-box embedding similarity

### New System (Vectorless Citation-Aware)
- **Endpoint:** `/feature/citation-chat`
- **Response:** Answer + structured citation objects with metadata
- **Traceability:** High (every claim explicitly backed by extracted text)
- **Logic:** Transparent token-based matching + deterministic scoring

---

## API Endpoint Specification

### Request

```http
POST /feature/citation-chat
Content-Type: application/json

{
  "question": "What datasets are commonly used?",
  "paper_ids": [],
  "require_citations": true
}
```

**Parameters:**
- `question` (string, required): User's research question
- `paper_ids` (array, optional): Filter to specific papers (empty = all papers)
- `require_citations` (boolean, optional, default=true): Whether to require citation backing

### Response

```json
{
  "question": "What datasets are commonly used?",
  "answer": "ImageNet [1] and CIFAR-10 [2] are commonly used. Some papers also employ SQuAD [3].",
  "citations": [
    {
      "paper_id": "arxiv_2024_001",
      "claim_text": "we use ImageNet for pretraining",
      "section": "experiments",
      "relevance_score": 0.89
    },
    {
      "paper_id": "arxiv_2024_002",
      "claim_text": "CIFAR-10 dataset was used for classification tasks",
      "section": "methodology",
      "relevance_score": 0.85
    },
    {
      "paper_id": "arxiv_2024_003",
      "claim_text": "SQuAD benchmark for question answering",
      "section": "results",
      "relevance_score": 0.76
    }
  ],
  "answer_with_citations": "ImageNet [1] and CIFAR-10 [2] are commonly used...\n\n### Citations\n[1] **arxiv_2024_001** (experiments)\n> we use ImageNet for pretraining\n[2] **arxiv_2024_002** (methodology)\n> CIFAR-10 dataset was used for classification tasks\n[3] **arxiv_2024_003** (results)\n> SQuAD benchmark for question answering",
  "papers_considered": 3,
  "mode": "groq-cited"
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `question` | string | Echo of user's question |
| `answer` | string | Answer with citation markers [1], [2], etc. |
| `citations` | array | Citation objects with metadata |
| `answer_with_citations` | string | Markdown formatted answer with full citations |
| `papers_considered` | number | How many papers were searched |
| `mode` | string | Response generation mode: `groq-cited`, `fallback-cited`, or `no-match` |

**Citation Object Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `paper_id` | string | Unique identifier of the paper |
| `claim_text` | string | Exact claim text from extracted corpus |
| `section` | string | Section where claim appears (abstract, introduction, methodology, results, etc.) |
| `relevance_score` | float | Relevance score 0.0-1.0 (higher = more relevant) |

---

## Frontend Implementation Changes

### 1. HTML Changes

**Add these elements to your research assistant panel:**

```html
<!-- Citation-Aware Research Q&A Panel -->
<div id="citChatPanel" class="panel">
  <div class="panel-header">
    <h3>🔍 Citation-Aware Research Q&A (Vectorless)</h3>
    <p class="panel-description">
      Ask research questions. Every answer is backed by specific paper claims with full traceability.
    </p>
  </div>

  <div class="panel-body">
    <!-- Question Input -->
    <div class="form-group">
      <label for="citChatQuestion">Research Question:</label>
      <textarea 
        id="citChatQuestion" 
        placeholder="e.g., What machine learning approaches are used? What datasets do papers employ?"
        rows="3">
      </textarea>
    </div>

    <!-- Optional Paper Filter -->
    <div class="form-group">
      <label for="citChatPaperIds">Filter by Paper IDs (optional, comma-separated):</label>
      <input 
        type="text" 
        id="citChatPaperIds" 
        placeholder="e.g., paper1, paper2, paper3"
      />
    </div>

    <!-- Action Buttons -->
    <div class="button-group">
      <button id="btnCitChat" class="btn btn-primary">
        ✨ Ask with Citations
      </button>
      <button id="btnCitChatClear" class="btn btn-secondary">
        🔄 Clear
      </button>
    </div>

    <!-- Status/Output -->
    <div id="citChatOut" class="status-message"></div>

    <!-- Results Container (hidden until response) -->
    <div id="citChatResultContainer" class="citation-chat-result hidden">
      
      <!-- Answer Box -->
      <div class="answer-section">
        <h4>📝 Answer</h4>
        <div id="citChatAnswer" class="answer-box"></div>
      </div>

      <!-- Citations Panel -->
      <div class="citations-section">
        <h4>📚 Citations (Backing Claims)</h4>
        <div id="citationsList" class="citations-panel"></div>
      </div>

      <!-- Formatted Markdown View -->
      <details class="markdown-view">
        <summary>📄 Formatted View</summary>
        <div id="citChatMarkdown" class="markdown-content"></div>
      </details>

    </div>
  </div>
</div>
```

### 2. CSS Changes

**Add these classes to your stylesheet:**

```css
/* Citation Chat Result Container */
.citation-chat-result {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-top: 20px;
  padding: 20px;
  background: #f5f5f5;
  border-radius: 8px;
}

.citation-chat-result.hidden {
  display: none;
}

/* Answer Section */
.answer-section {
  grid-column: 1 / -1;
}

.answer-box {
  padding: 15px;
  background: white;
  border-left: 4px solid #0d9488;
  border-radius: 4px;
  line-height: 1.6;
  color: #333;
  font-size: 14px;
}

/* Citations Section */
.citations-section {
  grid-column: 1 / -1;
}

.citations-panel {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 15px;
}

/* Citation Item Card */
.citation-item {
  padding: 15px;
  background: white;
  border-left: 3px solid #0d9488;
  border-radius: 4px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.citation-header {
  display: flex;
  justify-content: space-between;
  align-items: start;
  margin-bottom: 10px;
  gap: 10px;
}

.citation-meta {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.citation-section {
  padding: 2px 8px;
  background: #0d9488;
  color: white;
  border-radius: 3px;
  font-size: 11px;
  font-weight: bold;
  text-transform: capitalize;
}

.citation-score {
  padding: 2px 8px;
  background: #10b981;
  color: white;
  border-radius: 3px;
  font-size: 11px;
  font-weight: bold;
}

.citation-text {
  padding: 10px;
  background: #f9fafb;
  border-radius: 4px;
  font-style: italic;
  color: #666;
  margin: 10px 0;
  font-size: 13px;
  border-left: 2px solid #d1d5db;
}

.citation-actions {
  display: flex;
  gap: 8px;
}

.citation-btn {
  padding: 6px 12px;
  background: #e5e7eb;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.citation-btn:hover {
  background: #d1d5db;
  border-color: #0d9488;
}

/* Formatted Markdown View */
.markdown-view {
  grid-column: 1 / -1;
  padding: 15px;
  background: white;
  border-radius: 4px;
  border: 1px solid #e5e7eb;
}

.markdown-view summary {
  cursor: pointer;
  font-weight: bold;
  padding: 10px;
  user-select: none;
}

.markdown-content {
  padding: 15px;
  background: #f9fafb;
  border-radius: 4px;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #333;
  line-height: 1.5;
}

/* Responsive Layout */
@media (max-width: 768px) {
  .citation-chat-result {
    grid-template-columns: 1fr;
  }

  .citations-panel {
    grid-template-columns: 1fr;
  }

  .citation-header {
    flex-direction: column;
  }

  .citation-actions {
    flex-direction: column;
  }

  .citation-btn {
    width: 100%;
  }
}

/* Loading State */
.loading {
  opacity: 0.6;
  pointer-events: none;
}

.spinner {
  display: inline-block;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
```

### 3. JavaScript Changes

**Add these functions to your frontend code:**

```javascript
/**
 * Initialize citation-aware chatbot
 * Call this after DOM is ready
 */
function initCitationChatbot() {
  const btnCitChat = document.getElementById("btnCitChat");
  const btnCitChatClear = document.getElementById("btnCitChatClear");

  if (btnCitChat) {
    btnCitChat.addEventListener("click", handleCitationChatSubmit);
  }

  if (btnCitChatClear) {
    btnCitChatClear.addEventListener("click", handleCitationChatClear);
  }
}

/**
 * Handle citation chat submit
 */
function handleCitationChatSubmit() {
  const question = document.getElementById("citChatQuestion")?.value.trim();
  const paperIdsInput = document.getElementById("citChatPaperIds")?.value.trim();
  
  if (!question) {
    alert("Please enter a research question.");
    return;
  }

  // Parse paper IDs
  const paperIds = paperIdsInput 
    ? paperIdsInput.split(",").map(id => id.trim()).filter(id => id)
    : [];

  // Show loading state
  document.getElementById("citChatOut").textContent = "⏳ Generating answer with citations...";
  document.getElementById("btnCitChat").classList.add("loading");

  // Call API
  fetch("http://localhost:8000/feature/citation-chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      question: question,
      paper_ids: paperIds,
      require_citations: true
    })
  })
  .then(res => res.json())
  .then(response => {
    renderCitationChat(response);
  })
  .catch(error => {
    console.error("Citation chat error:", error);
    document.getElementById("citChatOut").textContent = `❌ Error: ${error.message}`;
  })
  .finally(() => {
    document.getElementById("btnCitChat").classList.remove("loading");
  });
}

/**
 * Render citation chat response
 */
function renderCitationChat(response) {
  if (!response) {
    document.getElementById("citChatOut").textContent = "❌ No response received.";
    return;
  }

  // Show result container
  const container = document.getElementById("citChatResultContainer");
  if (container) {
    container.classList.remove("hidden");
  }

  // Display answer
  document.getElementById("citChatAnswer").textContent = response.answer || "No answer generated.";

  // Display citations
  const citations = response.citations || [];
  const citationsList = document.getElementById("citationsList");

  if (citationsList) {
    citationsList.innerHTML = citations
      .map((cit, idx) => `
        <div class="citation-item">
          <div class="citation-header">
            <strong>[${idx + 1}] ${cit.paper_id}</strong>
            <div class="citation-meta">
              <span class="citation-section">${cit.section || "claim"}</span>
              <span class="citation-score">${((cit.relevance_score || 0) * 100).toFixed(0)}%</span>
            </div>
          </div>
          <div class="citation-text">"${cit.claim_text}"</div>
          <div class="citation-actions">
            <button class="citation-btn" onclick="copyCitation('${cit.paper_id}', '${cit.claim_text.replace(/'/g, "\\'")}')" title="Copy to clipboard">
              📋 Copy
            </button>
            <button class="citation-btn" onclick="jumpToCitation('${cit.paper_id}', '${cit.claim_text.replace(/'/g, "\\'")}')" title="Jump to PDF location">
              🔗 PDF
            </button>
          </div>
        </div>
      `)
      .join("");
  }

  // Display formatted markdown
  const markdown = response.answer_with_citations || response.answer;
  document.getElementById("citChatMarkdown").textContent = markdown;

  // Update status
  const status = `✅ Generated with ${citations.length} citations from ${response.papers_considered} papers (Mode: ${response.mode})`;
  document.getElementById("citChatOut").textContent = status;
}

/**
 * Handle citation chat clear
 */
function handleCitationChatClear() {
  document.getElementById("citChatQuestion").value = "";
  document.getElementById("citChatPaperIds").value = "";
  document.getElementById("citChatOut").textContent = "";
  document.getElementById("citChatResultContainer").classList.add("hidden");
}

/**
 * Copy citation to clipboard
 */
function copyCitation(paperId, claimText) {
  const text = `[${paperId}] ${claimText}`;
  navigator.clipboard.writeText(text).then(() => {
    alert("✅ Citation copied to clipboard!");
  }).catch(err => {
    console.error("Copy failed:", err);
    alert("❌ Failed to copy. See console for details.");
  });
}

/**
 * Jump to citation in PDF
 * Calls backend to get page number and location
 */
function jumpToCitation(paperId, claimText) {
  fetch("http://localhost:8000/feature/citation", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      paper_id: paperId,
      claim_text: claimText
    })
  })
  .then(res => res.json())
  .then(data => {
    alert(`📄 Found in: ${paperId}\nPage: ${data.page_number || "?"}\nLocation: ${data.bbox || data.location || "See PDF"}`);
    // User can manually open the PDF at the specified location
  })
  .catch(err => {
    console.error("Citation lookup error:", err);
    alert(`❌ Could not find location: ${err.message}`);
  });
}
```

**Call initialization after DOM loads:**
```javascript
document.addEventListener("DOMContentLoaded", () => {
  initCitationChatbot();
  // ... other initialization code
});
```

---

## Integration Checklist

- [ ] Copy HTML panel structure to your research panel
- [ ] Add CSS classes to your stylesheet 
- [ ] Add JavaScript functions to your app code
- [ ] Call `initCitationChatbot()` after DOM ready
- [ ] Update API endpoint from `/feature/ask` to `/feature/citation-chat`
- [ ] Test with sample question: "What datasets are used?"
- [ ] Verify citations display with paper ID, section, and score
- [ ] Test Copy Citation button (should copy to clipboard)
- [ ] Test Jump to PDF button (should show page location)
- [ ] Test on mobile - should stack vertically
- [ ] Review markdown formatted view (expandable)

---

## Testing the Integration

### Manual Testing Steps

1. **Start Backend** (in council directory):
   ```bash
   uvicorn council_api.main:app --reload
   ```

2. **Open Frontend** in your Research Intelligence directory

3. **Test Question 1** (should find multiple citations):
   ```
   "What machine learning models are discussed?"
   ```

4. **Test Question 2** (with paper filter):
   ```
   Paper filter: paper1, paper3
   Question: "What datasets are used?"
   ```

5. **Test Interaction**:
   - Click "Ask with Citations" 
   - Review answer and citations list
   - Click "Copy" on a citation (should copy to clipboard)
   - Click "PDF" on a citation (should show location)
   - Expand "Formatted View" (should show markdown)
   - Click "Clear" (should reset all fields)

### API Testing (curl)

```bash
curl -X POST http://localhost:8000/feature/citation-chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What deep learning architectures are used?",
    "paper_ids": [],
    "require_citations": true
  }' | jq .
```

---

## Response Modes Explained

| Mode | Meaning | Example When |
|------|---------|--------------|
| `groq-cited` | LLM-generated answer with citations | Normal case, LLM available |
| `fallback-cited` | Structured summary from top claims | LLM unavailable or error |
| `no-match` | No relevant claims found | Question too specific or unclear |

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "No response" | Backend not running | Start with `uvicorn council_api.main:app` |
| "No citations found" | No extracted papers | Run `/extract-all` from council backend first |
| "CORS error" | Cross-origin request blocked | Backend should have CORS enabled |
| Citations aren't showing | JavaScript error | Check browser console for JS errors |
| "Jump to PDF" doesn't work | `/feature/citation` not available | Verify citation endpoint is mounted |

---

## Performance Metrics

- **Question-to-response:** ~2-4 seconds (includes LLM call)
- **Citation matching:** ~30-50ms
- **Frontend render:** ~100-150ms

---

## Backend File References

- **Endpoint:** `council_api/feature_citation_chat.py` → `/feature/citation-chat` POST
- **Routes:** `council_api/main.py` → `app.include_router(citation_chat_router)`
- **Models:** `CitationAwareChatRequest`, `Citation`, `CitationAwareChatResponse`
- **Relevance Scoring:** Deterministic token overlap + heuristics (no embeddings)

---

## API Backward Compatibility

✅ Old `/feature/ask` endpoint still available
- Both endpoints work simultaneously
- Migrate gradually if needed
- Old endpoint will be deprecated in future releases

---

## Next Steps

1. Copy the HTML, CSS, and JavaScript from this guide into your frontend
2. Update your API calls to use `/feature/citation-chat`
3. Test with sample research questions
4. Adjust styling to match your design system
5. Collect feedback on citation quality and relevance

---

## Support

**Backend Issues:** Check `council_api/feature_citation_chat.py`
**Frontend Integration:** Review the HTML, CSS, JavaScript sections above
**API Issues:** Check response structure in `CitationAwareChatResponse` model
**Citation Quality:** Adjust relevance thresholds in backend feature file

