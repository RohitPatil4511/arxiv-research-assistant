"""
api/main.py

FastAPI backend with:
  - POST /research         → run full multi-agent pipeline (blocking)
  - GET  /research/stream  → SSE streaming with live agent status updates
  - POST /ingest           → manually upload and ingest a PDF
  - GET  /status           → RAG index stats
"""

import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agents.graph import research_graph, ResearchState
from rag.pipeline import RAGPipeline

app = FastAPI(
    title="ArXiv Research Assistant API",
    description="Multi-agent AI research assistant for scientific papers",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    query: str
    sub_tasks: list[str]
    final_answer: str
    arxiv_results: str
    rag_context_used: bool


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "ArXiv Research Assistant API", "status": "running"}


@app.get("/status")
def get_status():
    """Return RAG index statistics."""
    return {
        "rag_docs_ingested": rag.doc_count,
        "rag_chunks_indexed": rag.chunk_count,
        "index_ready": rag.index is not None,
    }


@app.post("/research", response_model=ResearchResponse)
def run_research(request: ResearchRequest):
    """Run the full multi-agent pipeline synchronously."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    initial_state = ResearchState(
        query=request.query,
        sub_tasks=[],
        arxiv_results="",
        web_results="",
        rag_context="",
        final_answer="",
        current_agent="",
        messages=[],
    )

    final_state = research_graph.invoke(initial_state)

    return ResearchResponse(
        query=final_state["query"],
        sub_tasks=final_state["sub_tasks"],
        final_answer=final_state["final_answer"],
        arxiv_results=final_state["arxiv_results"],
        rag_context_used=bool(final_state["rag_context"] and "No relevant" not in final_state["rag_context"]),
    )


@app.get("/research/stream")
async def stream_research(query: str):
    """
    SSE endpoint — streams agent status updates in real time.
    Each event has: {agent, status, data}
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def event_generator() -> AsyncGenerator[dict, None]:
        agents = ["planner", "researcher", "reader", "synthesizer"]
        agent_labels = {
            "planner": "Planning research tasks...",
            "researcher": "Searching arXiv and web...",
            "reader": "Reading ingested papers (RAG)...",
            "synthesizer": "Synthesizing final answer...",
        }

        # Run graph in thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        state_holder = {}

        def run_graph():
            initial = ResearchState(
                query=query, sub_tasks=[], arxiv_results="",
                web_results="", rag_context="", final_answer="",
                current_agent="", messages=[],
            )
            state_holder["result"] = research_graph.invoke(initial)

        # Stream agent status as graph runs
        task = loop.run_in_executor(None, run_graph)

        for agent in agents:
            yield {
                "event": "agent_start",
                "data": json.dumps({"agent": agent, "label": agent_labels[agent]}),
            }
            # Small delay to let graph progress
            await asyncio.sleep(0.5)

        # Wait for graph to finish
        await task
        result = state_holder.get("result", {})

        # Stream the final answer in chunks for a typewriter effect
        answer = result.get("final_answer", "No answer generated.")
        words = answer.split(" ")
        chunk_size = 8
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size]) + " "
            yield {
                "event": "answer_chunk",
                "data": json.dumps({"chunk": chunk}),
            }
            await asyncio.sleep(0.05)

        # Final event with full metadata
        yield {
            "event": "done",
            "data": json.dumps({
                "sub_tasks": result.get("sub_tasks", []),
                "rag_used": bool(result.get("rag_context")),
            }),
        }

    return EventSourceResponse(event_generator())


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload and ingest a PDF into the RAG vector store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    save_path = f"data/docs/{file.filename}"
    Path("data/docs").mkdir(parents=True, exist_ok=True)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    chunks_added = rag.ingest_pdf(save_path, source_name=file.filename)

    return {
        "message": f"Successfully ingested {file.filename}",
        "chunks_added": chunks_added,
        "total_chunks": rag.chunk_count,
    }
