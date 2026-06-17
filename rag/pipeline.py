"""
rag/pipeline.py
Handles PDF ingestion, chunking, embedding with Sentence Transformers,
and FAISS vector store for retrieval.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 5))
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")


class RAGPipeline:
    """
    End-to-end RAG pipeline:
      1. Ingest PDFs → extract text
      2. Chunk text with overlap
      3. Embed with Sentence Transformers
      4. Store/load FAISS index
      5. Retrieve top-k chunks for a query
    """

    def __init__(self):
        print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunks: List[str] = []
        self.metadata: List[dict] = []  # {source, page, chunk_id}

        # Load existing index if available
        if Path(f"{INDEX_PATH}.faiss").exists():
            self._load_index()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_pdf(self, pdf_path: str, source_name: str = None) -> int:
        """Extract text from a PDF, chunk it, embed it, add to FAISS."""
        source_name = source_name or Path(pdf_path).stem
        print(f"[RAG] Ingesting: {source_name}")

        reader = PdfReader(pdf_path)
        new_chunks = []
        new_meta = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            page_chunks = self._chunk_text(text)
            for i, chunk in enumerate(page_chunks):
                if len(chunk.strip()) < 50:   # skip near-empty chunks
                    continue
                new_chunks.append(chunk)
                new_meta.append({
                    "source": source_name,
                    "page": page_num + 1,
                    "chunk_id": len(self.chunks) + len(new_chunks) - 1,
                })

        if not new_chunks:
            print(f"[RAG] Warning: no usable text found in {source_name}")
            return 0

        embeddings = self.embedder.encode(new_chunks, show_progress_bar=True)
        self._add_to_index(embeddings, new_chunks, new_meta)
        self._save_index()
        print(f"[RAG] Added {len(new_chunks)} chunks from {source_name}")
        return len(new_chunks)

    def ingest_directory(self, dir_path: str = "data/docs") -> int:
        """Ingest all PDFs in a directory."""
        total = 0
        for pdf in Path(dir_path).glob("*.pdf"):
            total += self.ingest_pdf(str(pdf))
        return total

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[dict]:
        """
        Retrieve top-k most relevant chunks for a query.
        Returns list of {text, source, page, score}.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = self.embedder.encode([query])
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append({
                "text": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "page": self.metadata[idx]["page"],
                "score": float(1 / (1 + dist)),   # convert L2 distance to similarity
            })
        return results

    def get_context_string(self, query: str, top_k: int = TOP_K) -> str:
        """Return retrieved chunks formatted as a context block for the LLM."""
        results = self.retrieve(query, top_k)
        if not results:
            return "No relevant context found in the knowledge base."

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[Source {i}: {r['source']}, page {r['page']} | relevance: {r['score']:.2f}]\n{r['text']}"
            )
        return "\n\n---\n\n".join(lines)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + CHUNK_SIZE
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def _add_to_index(self, embeddings: np.ndarray, chunks: List[str], meta: List[dict]):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)
        self.metadata.extend(meta)

    def _save_index(self):
        Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{INDEX_PATH}.faiss")
        with open(f"{INDEX_PATH}.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)
        print(f"[RAG] Index saved ({self.index.ntotal} vectors)")

    def _load_index(self):
        self.index = faiss.read_index(f"{INDEX_PATH}.faiss")
        with open(f"{INDEX_PATH}.pkl", "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.metadata = data["metadata"]
        print(f"[RAG] Loaded index with {self.index.ntotal} vectors")

    @property
    def doc_count(self) -> int:
        return len(set(m["source"] for m in self.metadata))

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
