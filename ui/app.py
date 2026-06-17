"""
ui/app.py

Streamlit frontend for the ArXiv Multi-Agent Research Assistant.
Shows live agent status, final answer with markdown rendering,
arXiv paper cards, and a PDF upload panel.

Run:  streamlit run ui/app.py
      (make sure FastAPI is running on port 8000 first)
"""

import time
import httpx
import json
import streamlit as st

API_URL = "http://localhost:8000"

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="ArXiv Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------

st.markdown("""
<style>
.agent-card {
    padding: 10px 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 14px;
    border: 1px solid #e0e0e0;
}
.agent-active  { background: #e8f4fd; border-color: #1976d2; color: #1976d2; }
.agent-done    { background: #e8f5e9; border-color: #388e3c; color: #388e3c; }
.agent-waiting { background: #f5f5f5; border-color: #bdbdbd; color: #9e9e9e; }
.paper-card {
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    margin-bottom: 10px;
    background: #fafafa;
}
.metric-box {
    background: #f0f4ff;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Sidebar — RAG status & PDF upload
# ------------------------------------------------------------------

with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("Multi-agent • LangGraph • Groq • RAG")

    st.divider()

    # RAG status
    st.subheader("📚 Knowledge Base")
    try:
        status = httpx.get(f"{API_URL}/status", timeout=3).json()
        col1, col2 = st.columns(2)
        col1.metric("Docs indexed", status["rag_docs_ingested"])
        col2.metric("Chunks", status["rag_chunks_indexed"])
        if status["index_ready"]:
            st.success("RAG index ready")
        else:
            st.info("No docs ingested yet — upload PDFs below or ask a question to auto-fetch from arXiv")
    except Exception:
        st.warning("API not reachable — start FastAPI first:\nuvicorn api.main:app --reload")

    st.divider()

    # PDF upload
    st.subheader("📄 Upload Paper PDF")
    uploaded = st.file_uploader("Drop a PDF here", type=["pdf"])
    if uploaded:
        with st.spinner("Ingesting..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=60,
                )
                data = resp.json()
                st.success(f"✅ {data['chunks_added']} chunks added")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.divider()
    st.caption("Built with LangGraph · LangChain · Groq · FAISS · Sentence Transformers")


# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------

st.title("ArXiv Multi-Agent Research Assistant")
st.markdown("*Ask a research question — agents will search arXiv, scan papers, and synthesize an answer.*")

# Example questions
examples = [
    "What are the latest techniques in retrieval-augmented generation?",
    "How do vision transformers compare to CNNs for image classification?",
    "Explain recent advances in reinforcement learning from human feedback (RLHF)",
    "What is the current state of large language model alignment?",
]

with st.expander("💡 Try an example question"):
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["query_input"] = ex

query = st.text_area(
    "Your research question",
    value=st.session_state.get("query_input", ""),
    height=80,
    placeholder="e.g. What are the latest techniques in retrieval-augmented generation?",
    key="query_input",
)

run_btn = st.button("🚀 Research", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# Research execution
# ------------------------------------------------------------------

if run_btn and query.strip():
    st.divider()

    # Agent status panel
    agents = [
        ("planner", "🗺️", "Planner", "Breaking query into sub-tasks"),
        ("researcher", "🔍", "Researcher", "Searching arXiv + web"),
        ("reader", "📖", "Reader", "Querying RAG index"),
        ("synthesizer", "✍️", "Synthesizer", "Writing final answer"),
    ]

    col_agents, col_answer = st.columns([1, 2])

    with col_agents:
        st.subheader("Agent pipeline")
        agent_placeholders = {}
        for key, icon, name, desc in agents:
            ph = st.empty()
            ph.markdown(
                f'<div class="agent-card agent-waiting">{icon} <b>{name}</b><br>'
                f'<span style="font-size:12px">{desc}</span></div>',
                unsafe_allow_html=True,
            )
            agent_placeholders[key] = ph

    with col_answer:
        st.subheader("Research answer")
        answer_placeholder = st.empty()
        answer_placeholder.markdown("*Waiting for agents...*")

    # Call API
    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{API_URL}/research", json={"query": query})
            result = resp.json()

        # Animate agent cards
        for i, (key, icon, name, desc) in enumerate(agents):
            time.sleep(0.3)
            agent_placeholders[key].markdown(
                f'<div class="agent-card agent-done">{icon} <b>{name}</b> ✓<br>'
                f'<span style="font-size:12px">{desc}</span></div>',
                unsafe_allow_html=True,
            )

        # Display sub-tasks
        with col_agents:
            st.subheader("Sub-tasks planned")
            for i, task in enumerate(result.get("sub_tasks", []), 1):
                st.markdown(f"**{i}.** {task}")

            if result.get("rag_context_used"):
                st.success("📚 RAG context contributed to the answer")
            else:
                st.info("ℹ️ Answer based on live search (no matching PDFs in index)")

        # Stream final answer with typewriter effect
        answer = result.get("final_answer", "No answer generated.")
        displayed = ""
        words = answer.split(" ")
        for i in range(0, len(words), 6):
            displayed += " ".join(words[i:i + 6]) + " "
            answer_placeholder.markdown(displayed + "▌")
            time.sleep(0.04)
        answer_placeholder.markdown(answer)

        # arXiv papers section
        st.divider()
        st.subheader("📄 Papers found on arXiv")
        arxiv_text = result.get("arxiv_results", "")
        if arxiv_text and "No arXiv" not in arxiv_text:
            st.markdown(arxiv_text)
        else:
            st.info("No arXiv papers retrieved for this query.")

    except Exception as e:
        st.error(f"Research failed: {e}\n\nMake sure the FastAPI server is running:\n`uvicorn api.main:app --reload`")

elif run_btn:
    st.warning("Please enter a research question.")
