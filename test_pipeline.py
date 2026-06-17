"""
test_pipeline.py
Quick smoke test to verify everything is wired correctly.
Run BEFORE starting the full API/UI.
Usage:  python test_pipeline.py
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def check_env():
    print("1. Checking environment...")
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key == "your_groq_api_key_here":
        print("   ❌ GROQ_API_KEY not set. Copy .env.example → .env and add your key.")
        print("      Get a free key at https://console.groq.com")
        return False
    print(f"   ✅ GROQ_API_KEY found ({key[:8]}...)")
    return True

def check_imports():
    print("2. Checking imports...")
    try:
        import groq, langchain_groq, langgraph, faiss
        import sentence_transformers, arxiv, duckduckgo_search
        import fastapi, streamlit
        print("   ✅ All packages installed")
        return True
    except ImportError as e:
        print(f"   ❌ Missing package: {e}")
        print("      Run: pip install -r requirements.txt")
        return False

def check_rag():
    print("3. Testing RAG pipeline...")
    try:
        from rag.pipeline import RAGPipeline
        rag = RAGPipeline()
        result = rag.get_context_string("test query")
        print(f"   ✅ RAG pipeline initialized ({rag.chunk_count} chunks in index)")
        return True
    except Exception as e:
        print(f"   ❌ RAG pipeline error: {e}")
        return False

def check_groq():
    print("4. Testing Groq LLM connection...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        llm = ChatGroq(model=model, max_tokens=50)
        resp = llm.invoke([HumanMessage(content="Say 'hello' only.")])
        print(f"   ✅ Groq connected. Response: {resp.content[:50]}")
        return True
    except Exception as e:
        print(f"   ❌ Groq error: {e}")
        return False

def check_arxiv():
    print("5. Testing arXiv search...")
    try:
        from tools.search_tools import arxiv_search
        results = arxiv_search("transformer neural network", max_results=1)
        if results:
            print(f"   ✅ arXiv search works. Found: {results[0]['title'][:50]}...")
        else:
            print("   ⚠️  arXiv returned 0 results (may be a network issue)")
        return True
    except Exception as e:
        print(f"   ❌ arXiv error: {e}")
        return False

if __name__ == "__main__":
    print("\n🔬 ArXiv Research Assistant — Setup Check\n" + "="*45)
    results = [
        check_env(),
        check_imports(),
        check_rag(),
        check_groq(),
        check_arxiv(),
    ]
    print("\n" + "="*45)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"✅ All {total} checks passed! You're ready to run:")
        print("   Terminal 1: uvicorn api.main:app --reload")
        print("   Terminal 2: streamlit run ui/app.py")
    else:
        print(f"⚠️  {passed}/{total} checks passed. Fix the issues above before running.")