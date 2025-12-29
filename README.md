# Agentic Resume Matcher

An intelligent, multi-agent system for automated resume screening. This project moves beyond simple keyword matching by using LLMs to "read" resumes, reason about candidate fit, and score them against a Job Description with human-like nuance.

---

## 📊 Performance Evaluation

To validate the system, I tested it against a **Synthetic Evaluation Dataset** comprising 15 resumes ranging from "Perfect Matches" to "Completely Irrelevant" profiles.

### 1. Retrieval Performance
I measured the ability of the Hybrid Search (FAISS + BM25) to surface relevant candidates from the pool.

| K | Precision@K | Recall@K | Insight |
| :--- | :--- | :--- | :--- |
| **3** | **1.000** | **0.429** | The top 3 candidates Ire all Excellent Matches (Bob, Alice, Carol). |
| **5** | **1.000** | **0.714** | The top 5 included strong "Partial Matches" (Eve, Frank), proving the system prioritizes quality. |
| **10** | **0.600** | **0.857** | By K=10, nearly all relevant talent was captured. |
| **15** | **0.467** | **1.000** | **100% Recall achieved.** No viable candidate was missed in the retrieval phase. |

### 2. Matching Accuracy
I measured the accuracy of the **Matching Agent** (Gemini 2.5 Pro) by comparing its generated score (0-100) against the human-verified score range.

* **Overall Accuracy:** **100.0%**
* **Detailed Results:**

| Candidate | System Score | Target Range | Match |
| :--- | :--- | :--- | :--- |
| **Alice Chen** | **1.000** | 0.90 - 1.00 | ✅ Perfect Match |
| **Carol Davis** | **0.980** | 0.90 - 1.00 | ✅ Perfect Match |
| **Bob Smith** | **0.965** | 0.90 - 1.00 | ✅ Perfect Match |
| **Eve Miller** | **0.575** | 0.50 - 0.70 | ✅ Correctly identified as partial fit (Junior/ML vs Senior/AI). |
| **David Wilson** | **0.485** | 0.40 - 0.60 | ✅ Correctly identified backend gaps. |
| **Frank White** | **0.480** | 0.40 - 0.60 | ✅ Correctly identified lack of deep AI experience. |
| **Jack Black** | **0.295** | 0.20 - 0.40 | ✅ Correctly identified as DevOps (Role Mismatch). |
| **Kelly King** | **0.130** | 0.00 - 0.20 | ✅ Correctly rejected (Java Monolith vs Python AI). |

---

## Architecture

### 1) Extraction Agent
- **Model:** `gemini-2.5-flash`
- **Input:** Raw resume files (PDF/DOCX/TXT)
- **Goal:** Structural normalization
- **Tools:** `ResumeReaderTool` (wraps `pdfplumber` / `docx2txt`)
- **Output:**
  - `ResumeData` (JSON): structured fields (Name, Skills, Experience, Education, etc.)
  - `normalized_text`: clean string optimized for embedding

### 2) Hybrid Indexing
- **Embeddings:** `gemini-embedding-001` (Google GenAI)
- **Storage:**
  - **Vector:** FAISS (semantic search)
  - **Lexical:** BM25 (keyword search)
- **Strategy:** **Reciprocal Rank Fusion (RRF)** to combine vector + keyword results

### 3) Retrieval
1. Generate query embedding from job description  
2. Parallel search (FAISS + BM25)  
3. Fuse rankings (RRF)  
4. Return Top‑K candidates  

### 4) Matching Agent
- **Model:** `gemini-2.5-pro`
- **Input:** Job Description + Candidate Metadata + Normalized Resume Text
- **Goal:** Reasoning and final decision
- **Tools:** `ScoringTool` (deterministic Iighted calculation)
- **Output:**
  - **Score (0–100)**: quantitative fit
  - **Reasoning**: qualitative explanation (why the candidate fits / fails)

---

## Strands Primitives Used
- **Agent:** `strands.Agent`
- **Tool:** `strands.tool` decorator (for `ScoringTool` and file handlers)
- **Model:** `strands.models.GeminiModel` (Gemini 2.5 family)

---

## Scoring Logic

The `ScoringTool` implements a strict Iighted formula to prevent hallucinated math:

```python
Final Score = (0.30 * Experience_Score) + 
              (0.40 * Skills_Score) + 
              (0.10 * Education_Score) + 
              (0.20 * Projects_Score)
```

---

## 1. Model Selection & Justification

| Provider | Model family | Variant | Max input tokens (context) | Price (USD) — input / output per 1M tokens | Role in pipeline | Notes / decision |
|---|---|---:|---:|---:|---|---|
| **Google (Gemini API)** | **Gemini 2.5** | **Flash** (`gemini-2.5-flash`) | **1,048,576** | **$0.30 / $2.50** *(text/image/video)* | **Extraction** | **Selected.** Low cost + 1M context fits long resumes/portfolios with feIr chunking edge-cases. |
| **Google (Gemini API)** | **Gemini 2.5** | **Pro** (`gemini-2.5-pro`) | **1,048,576** | **≤200K:** **$1.25 / $10.00**; **>200K:** **$2.50 / $15.00** | **Matching** | **Selected.** Strong reasoning; keep **math deterministic via tool** to avoid scoring drift. |
| **Google (Gemini API)** | **Gemini 3** | **Flash (Preview)** (`gemini-3-flash-preview`) | **1,048,576** | **$0.50 / $3.00** *(text/image/video)* | Optional / future | **Deferred.** Preview model; consider once Strands + Gemini 3 stability is confirmed. |
| **Google (Gemini API)** | **Gemini 3** | **Pro (Preview)** (`gemini-3-pro-preview`) | **1,048,576** | **≤200K:** **$2.00 / $12.00**; **>200K:** **$4.00 / $18.00** | Optional / future | **Deferred.** PoIrful but **preview** + higher cost than Gemini 2.5 Pro. |
| **OpenAI** | **GPT-5** | **Mini** (`gpt-5-mini`) | **400,000** | **$0.25 / $2.00** | N/A | **Rejected (for this design).** Good pricing, but context is still < Gemini 1M. |
| **OpenAI** | **GPT-5.2** | **Standard** (`gpt-5.2`) | **400,000** | **$1.75 / $14.00** | N/A | **Rejected.** Costly vs Gemini 2.5 Pro + smaller context. |
| **OpenAI** | **GPT-5.2** | **Pro** (`gpt-5.2-pro`) | **400,000** | **$21.00 / $168.00** | N/A | **Rejected.** Prohibitive for batch resume screening. |
| **OpenAI** | **GPT-4o** | **Standard** (`gpt-4o`) | **128,000** | **$2.50 / $10.00** | N/A | **Rejected.** Smaller context and higher input cost vs Gemini 2.5 Pro. |
| **Anthropic** | **Claude** | **Sonnet 4.5** (`claude-sonnet-4-5`) | **200,000** *(1M beta available for some orgs)* | **≤200K:** **$3 / $15**; **>200K:** **$6 / $22.50** | N/A | **Rejected.** Strong model, but base pricing is higher than Gemini Flash and long-context is premium-priced. |

---

## 🚀 Usage

### Requirements
* Python 3.10+
* Google API Key

### Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Create a .env file or Configure settings in `config/settings.py`.
4. Run Ib-App:
    ```bash
    streamlit run app.py
    ```

### Running Evaluation
* **Full Pipeline Test (Retrieval + Matching)**:
    ```bash
    python evaluation/run_evaluation.py
    ```
* **Ib Application**:
    ```bash
    streamlit run app.py

    ```

### Use [Web-App Directly](https://resumematching-qnvbswfupsdmsjnxxmxa7p.streamlit.app/)
