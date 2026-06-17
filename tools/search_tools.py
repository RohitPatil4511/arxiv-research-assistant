"""
tools/search_tools.py
Two tools available to agents:
  1. arxiv_search  — search arXiv and optionally download PDFs
  2. web_search    — DuckDuckGo web search for general context
"""

import os
import time
from pathlib import Path
from typing import List, Optional

import arxiv
from duckduckgo_search import DDGS


# ------------------------------------------------------------------
# arXiv tool
# ------------------------------------------------------------------

def arxiv_search(
    query: str,
    max_results: int = 5,
    download_pdfs: bool = False,
    download_dir: str = "data/docs",
) -> List[dict]:
    """
    Search arXiv for papers matching the query.

    Args:
        query:         Natural language research query.
        max_results:   How many papers to return (max 10).
        download_pdfs: If True, download PDFs to download_dir.
        download_dir:  Where to save PDFs.

    Returns:
        List of paper dicts: {title, authors, abstract, url, pdf_path}
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=min(max_results, 10),
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    for paper in client.results(search):
        entry = {
            "title": paper.title,
            "authors": [a.name for a in paper.authors[:3]],
            "abstract": paper.summary[:600] + "..." if len(paper.summary) > 600 else paper.summary,
            "url": paper.entry_id,
            "published": str(paper.published.date()),
            "pdf_path": None,
        }

        if download_pdfs:
            Path(download_dir).mkdir(parents=True, exist_ok=True)
            safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in paper.title)
            pdf_path = f"{download_dir}/{safe_name[:60]}.pdf"
            if not Path(pdf_path).exists():
                try:
                    paper.download_pdf(filename=pdf_path)
                    time.sleep(1)   # be polite to arXiv
                except Exception as e:
                    print(f"[arXiv] PDF download failed: {e}")
                    pdf_path = None
            entry["pdf_path"] = pdf_path

        results.append(entry)

    return results


def format_arxiv_results(results: List[dict]) -> str:
    """Format arXiv results as a readable string for the LLM."""
    if not results:
        return "No arXiv papers found for this query."

    lines = []
    for i, p in enumerate(results, 1):
        authors = ", ".join(p["authors"])
        lines.append(
            f"{i}. **{p['title']}**\n"
            f"   Authors: {authors} ({p['published']})\n"
            f"   URL: {p['url']}\n"
            f"   Abstract: {p['abstract']}"
        )
    return "\n\n".join(lines)


# ------------------------------------------------------------------
# Web search tool
# ------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Search the web using DuckDuckGo (no API key required).

    Returns:
        List of {title, url, snippet}
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        print(f"[WebSearch] Error: {e}")
        return []


def format_web_results(results: List[dict]) -> str:
    """Format web search results as a string for the LLM."""
    if not results:
        return "No web results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r['title']}\n"
            f"   URL: {r['url']}\n"
            f"   {r['snippet']}"
        )
    return "\n\n".join(lines)
