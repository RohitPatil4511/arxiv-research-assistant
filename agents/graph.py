```python
"""
ui/app.py

Streamlit frontend for the ArXiv Multi-Agent Research Assistant.
Runs LangGraph directly without FastAPI.
"""

import time
import streamlit as st
from agents.graph import run_research

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

.agent-active {
    background: #e8f4fd;
    border-color: #1976d2;
    color: #1976d2;
}

.agent-done {
    background: #e8f5e9;
    border-color: #388e3c;
    color: #388e3c;
}

.agent-waiting {
    background: #f5f5f5;
    border-color: #bdbdbd;
    color: #9e9e9e;
}

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
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("Multi-agent • LangGraph • Groq • RAG")

    st.divider()

    st.subheader("📚 Knowledge Base")
    st.success("✅ Research engine ready")
    st.info("Running directly inside Streamlit")

    st.divider()

    st.subheader("📄 Upload Paper PDF")

    st.file_uploader(
        "PDF upload coming soon",
        type=["pdf"],
        disabled=True
    )

    st.divider()

    st.caption(
        "Built with LangGraph · LangChain · Groq · FAISS · Sentence Transformers"
    )

# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------

st.title("ArXiv Multi-Agent Research Assistant")

st.markdown(
    "*Ask a research question — agents will search arXiv, scan papers, and synthesize an answer.*"
)

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

run_btn = st.button(
    "🚀 Research",
    type="primary",
    use_container_width=True
)

# ------------------------------------------------------------------
# Research execution
# ------------------------------------------------------------------

if run_btn and query.strip():

    st.divider()

    agents = [
        ("planner", "🗺️", "Planner", "Breaking query into sub-tasks"),
        ("researcher", "🔍", "Researcher", "Searching arXiv"),
        ("reader", "📖", "Reader", "Querying RAG index"),
        ("synthesizer", "✍️", "Synthesizer", "Writing final answer"),
    ]

    col_agents, col_answer = st.columns([1, 2])

    with col_agents:
        st.subheader("Agent Pipeline")

        agent_placeholders = {}

        for key, icon, name, desc in agents:
            ph = st.empty()

            ph.markdown(
                f"""
                <div class="agent-card agent-waiting">
                {icon} <b>{name}</b><br>
                <span style="font-size:12px">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            agent_placeholders[key] = ph

    with col_answer:
        st.subheader("Research Answer")
        answer_placeholder = st.empty()
        answer_placeholder.markdown("*Waiting for agents...*")

    try:

        # Run LangGraph directly
        state = run_research(query)

        result = {
            "sub_tasks": state.get("sub_tasks", []),
            "final_answer": state.get("final_answer", ""),
            "arxiv_results": state.get("arxiv_results", ""),
            "rag_context_used": bool(state.get("rag_context")),
        }

        # Animate agent cards
        for key, icon, name, desc in agents:

            time.sleep(0.3)

            agent_placeholders[key].markdown(
                f"""
                <div class="agent-card agent-done">
                {icon} <b>{name}</b> ✓<br>
                <span style="font-size:12px">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_agents:

            st.subheader("Sub-tasks Planned")

            for i, task in enumerate(result["sub_tasks"], start=1):
                st.markdown(f"**{i}.** {task}")

            if result["rag_context_used"]:
                st.success("📚 RAG context contributed to the answer")
            else:
                st.info("ℹ️ Answer generated from live research")

        answer = result["final_answer"]

        displayed = ""

        words = answer.split()

        for i in range(0, len(words), 6):
            displayed += " ".join(words[i:i + 6]) + " "
            answer_placeholder.markdown(displayed + "▌")
            time.sleep(0.04)

        answer_placeholder.markdown(answer)

        st.divider()

        st.subheader("📄 Papers Found on arXiv")

        arxiv_text = result["arxiv_results"]

        if arxiv_text:
            st.markdown(arxiv_text)
        else:
            st.info("No arXiv papers retrieved.")

    except Exception as e:
        st.error(f"Research failed: {e}")

elif run_btn:
    st.warning("Please enter a research question.")
```
