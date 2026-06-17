# 🔬 ArXiv Multi-Agent Research Assistant

A production-ready AI research assistant that uses **multi-agent orchestration** (LangGraph) to automatically search scientific papers on arXiv, ingest them into a RAG pipeline, and synthesize comprehensive research answers — all powered by **Groq's ultra-fast LLaMA 3.1**.

> Built as a portfolio project demonstrating LLM system design, agentic workflows, and RAG pipelines.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌──────────────┐
│   Planner   │────▶│   Researcher     │────▶│   Reader    │────▶│  Synthesizer │
│             │     │                  │     │   (RAG)     │     │              │
│ Breaks query│     │ arXiv search     │     │ FAISS query │     │ Final answer │
│ into tasks  │     │ Web search       │     │ Sentence    │     │ with sources │
│             │     │ Auto PDF ingest  │     │ Transformers│     │              │
└─────────────┘     └──────────────────┘     └─────────────┘     └──────────────┘
                                                    │
                                           ┌────────┴────────┐
                                           │   FAISS Index   │
                                           │ (vector store)  │
                                           └─────────────────┘
```

**Tech stack:**

| Component | Technology |
|-----------|-----------|
| Agent orchestration | LangGraph StateGraph |
| LLM | LLaMA 3.1 70B via Groq API |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector store | FAISS (local, no infra needed) |
| Paper search | arXiv API + DuckDuckGo |
| API | FastAPI + SSE streaming |
| UI | Streamlit |
| Evaluation | RAGAS (faithfulness, relevancy, recall) |

---

## 🚀 Quick start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/arxiv-research-assistant
cd arxiv-research-assistant
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get a free key at: https://console.groq.com
```

### 3. Start the API server

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Start the Streamlit UI (new terminal)

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` — ask a research question!

---

## 📡 API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/status` | RAG index stats |
| `POST` | `/research` | Run full pipeline (blocking) |
| `GET` | `/research/stream?query=...` | SSE streaming with live agent updates |
| `POST` | `/ingest` | Upload a PDF to the knowledge base |

### Example API call

```python
import httpx

result = httpx.post(
    "http://localhost:8000/research",
    json={"query": "What are the latest advances in RAG?"},
    timeout=120,
).json()

print(result["final_answer"])
```

---

## 🤖 How agents work

### 1. Planner
Decomposes the user's question into 2–4 focused research sub-tasks using LLaMA 3.1.

### 2. Researcher
- Builds a precise arXiv search query from sub-tasks
- Fetches top 5 relevant papers, downloads PDFs
- Runs a parallel DuckDuckGo web search
- Auto-ingests downloaded PDFs into the FAISS index

### 3. Reader (RAG)
- Embeds the original query with Sentence Transformers
- Retrieves top-5 most relevant chunks from the FAISS index
- Returns passage-level context with source + page citations

### 4. Synthesizer
Combines all gathered information into a structured markdown answer with:
- Overview summary
- Key findings with citations
- Methodologies discussed
- Open questions
- Recommended further reading

---

## 📊 Evaluation (RAGAS)

Run the built-in evaluation suite:

```bash
python eval/evaluate.py
```

Scores three RAGAS metrics across sample research questions:
- **Answer Relevancy** — does the answer address the question?
- **Faithfulness** — is every claim grounded in retrieved context?
- **Context Recall** — did retrieval surface the right information?

Results saved to `eval/results.json`.

---

## 📁 Project structure

```
arxiv-research-assistant/
├── agents/
│   └── graph.py          # LangGraph multi-agent orchestration
├── rag/
│   └── pipeline.py       # FAISS + Sentence Transformers RAG pipeline
├── tools/
│   └── search_tools.py   # arXiv API + DuckDuckGo search tools
├── api/
│   └── main.py           # FastAPI backend with SSE streaming
├── ui/
│   └── app.py            # Streamlit frontend
├── eval/
│   └── evaluate.py       # RAGAS evaluation
├── data/
│   ├── docs/             # Downloaded/uploaded PDFs
│   └── faiss_index.*     # Persisted vector index
├── requirements.txt
└── .env.example
```

---

## 💡 Key design decisions

**Why LangGraph over plain LangChain?**
LangGraph allows stateful, graph-based agent orchestration with explicit edges between agents. This enables conditional routing, error recovery, and clear separation of concerns — closer to how production AI systems are built.

**Why FAISS over a cloud vector DB?**
FAISS runs entirely locally with no infrastructure cost, making the project reproducible and demo-friendly. Swapping to Pinecone or Weaviate requires only changing the RAG module.

**Why Groq?**
Groq's LPU hardware delivers sub-second token generation for LLaMA 3.1 70B — fast enough for streaming UI demos, and completely free on the dev tier.

**Chunking strategy:**
512-word chunks with 64-word overlap balances context preservation with retrieval precision. Each chunk retains source + page metadata for citation.

---

## 📹 Demo

[Link to Loom demo video — add yours here]

[Live deployment on HuggingFace Spaces — add yours here]

---

## 🛠️ Extending this project

- **Add memory**: Use LangGraph's persistence to maintain research history across sessions
- **Conditional routing**: Route to different agents based on query type (e.g., math questions → Wolfram tool)
- **Multi-modal**: Add image extraction from PDFs using `pdfplumber` + GPT-4V for figure analysis
- **Auth + multi-user**: Add FastAPI JWT auth and per-user FAISS indexes

---

## 📄 License

MIT

---

*Built by Rohit Patil | [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) | [GitHub](https://github.com/YOUR_USERNAME)*
