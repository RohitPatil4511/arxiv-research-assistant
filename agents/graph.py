"""
agents/graph.py

Multi-agent research pipeline using LangGraph.

Agents:
  1. Planner      — breaks the user query into sub-tasks
  2. Researcher   — searches arXiv + web for relevant papers/info
  3. Reader       — queries the RAG pipeline for ingested PDF context
  4. Synthesizer  — writes the final structured answer

State flows:  Planner → Researcher → Reader → Synthesizer → END
"""

import os
from typing import TypedDict, List, Optional, Annotated
import operator

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from tools.search_tools import arxiv_search, format_arxiv_results, web_search, format_web_results
from rag.pipeline import RAGPipeline

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

# Shared LLM instance (Groq is fast enough for all agents)
llm = ChatGroq(model=GROQ_MODEL, temperature=0.2, streaming=True)

# Shared RAG pipeline
rag = RAGPipeline()


# ------------------------------------------------------------------
# State schema
# ------------------------------------------------------------------

class ResearchState(TypedDict):
    query: str                        # original user question
    sub_tasks: List[str]              # Planner output
    arxiv_results: str                # Researcher output
    web_results: str                  # Researcher output
    rag_context: str                  # Reader output
    final_answer: str                 # Synthesizer output
    current_agent: str                # for UI streaming
    messages: Annotated[List, operator.add]   # full message log


# ------------------------------------------------------------------
# Agent 1 — Planner
# ------------------------------------------------------------------

def planner_agent(state: ResearchState) -> dict:
    """Decompose the user query into focused research sub-tasks."""
    print("[Planner] Running...")

    prompt = f"""You are a research planning assistant.
Break the following research question into 2-4 specific, focused sub-tasks
that together will produce a comprehensive answer.

Research question: {state['query']}

Respond as a numbered list only. Each sub-task should be one clear sentence.
Example format:
1. Find recent papers on X
2. Look up definitions and background for Y
3. Identify key algorithms used in Z"""

    response = llm.invoke([HumanMessage(content=prompt)])
    sub_tasks_text = response.content

    # Parse numbered list
    sub_tasks = []
    for line in sub_tasks_text.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            task = line.split(".", 1)[-1].strip()
            if task:
                sub_tasks.append(task)

    return {
        "sub_tasks": sub_tasks,
        "current_agent": "planner",
        "messages": [{"role": "planner", "content": sub_tasks_text}],
    }


# ------------------------------------------------------------------
# Agent 2 — Researcher
# ------------------------------------------------------------------

def researcher_agent(state: ResearchState) -> dict:
    """Search arXiv and web based on the planner's sub-tasks."""
    print("[Researcher] Running...")

    # Build a focused search query from sub-tasks
    tasks_str = "\n".join(state["sub_tasks"])
    query_prompt = f"""Given these research sub-tasks:
{tasks_str}

Write ONE concise arXiv search query (max 10 words) that covers the main topic."""

    query_response = llm.invoke([HumanMessage(content=query_prompt)])
    search_query = query_response.content.strip().strip('"')

    # Search arXiv (download top 2 PDFs for RAG ingestion)
    arxiv_papers = arxiv_search(
        query=search_query,
        max_results=5,
        download_pdfs=True,
    )
    arxiv_str = format_arxiv_results(arxiv_papers)

    # Auto-ingest any newly downloaded PDFs into RAG
    newly_ingested = rag.ingest_directory("data/docs")
    if newly_ingested:
        print(f"[Researcher] Auto-ingested {newly_ingested} new chunks into RAG")

    # Web search for additional context
    web_results = web_search(query=search_query, max_results=4)
    web_str = format_web_results(web_results)

    return {
        "arxiv_results": arxiv_str,
        "web_results": web_str,
        "current_agent": "researcher",
        "messages": [{"role": "researcher", "content": f"arXiv:\n{arxiv_str}\n\nWeb:\n{web_str}"}],
    }


# ------------------------------------------------------------------
# Agent 3 — Reader (RAG)
# ------------------------------------------------------------------

def reader_agent(state: ResearchState) -> dict:
    """Query the FAISS RAG index for context from ingested PDFs."""
    print("[Reader] Running...")

    context = rag.get_context_string(state["query"], top_k=5)

    return {
        "rag_context": context,
        "current_agent": "reader",
        "messages": [{"role": "reader", "content": context}],
    }


# ------------------------------------------------------------------
# Agent 4 — Synthesizer
# ------------------------------------------------------------------

def synthesizer_agent(state: ResearchState) -> dict:
    """Combine all gathered information into a structured final answer."""
    print("[Synthesizer] Running...")

    sub_tasks_str = "\n".join(f"- {t}" for t in state["sub_tasks"])

    system = """You are an expert research synthesizer.
Your job is to produce a clear, well-structured research summary.
Always cite sources when using specific information.
Use markdown formatting with headers, bullet points, and bold text."""

    user = f"""Research question: {state['query']}

Research sub-tasks addressed:
{sub_tasks_str}

== arXiv Papers Found ==
{state['arxiv_results']}

== Web Search Results ==
{state['web_results']}

== Context from Ingested PDFs (RAG) ==
{state['rag_context']}

---

Write a comprehensive, well-structured answer to the research question.
Include:
1. A brief overview (2-3 sentences)
2. Key findings with citations
3. Important methodologies or techniques mentioned
4. Open questions or future directions
5. Recommended papers to read further"""

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])

    return {
        "final_answer": response.content,
        "current_agent": "synthesizer",
        "messages": [{"role": "synthesizer", "content": response.content}],
    }


# ------------------------------------------------------------------
# Build the LangGraph
# ------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("reader", reader_agent)
    graph.add_node("synthesizer", synthesizer_agent)

    # Linear pipeline: planner → researcher → reader → synthesizer → END
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "reader")
    graph.add_edge("reader", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


# Compiled graph (import this in API and UI)
research_graph = build_graph()


# ------------------------------------------------------------------
# Convenience runner
# ------------------------------------------------------------------

def run_research(query: str) -> ResearchState:
    """Run the full multi-agent pipeline and return the final state."""
    initial_state = ResearchState(
        query=query,
        sub_tasks=[],
        arxiv_results="",
        web_results="",
        rag_context="",
        final_answer="",
        current_agent="",
        messages=[],
    )
    final_state = research_graph.invoke(initial_state)
    return final_state
