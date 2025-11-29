#!/usr/bin/env python3
"""
Verify that the RAG system setup is correct.
"""
import sys
import os


def check_imports():
    """Check if all required packages are installed."""
    print("Checking imports...")
    try:
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        print("  [OK] llama-index-core")

        from llama_index.embeddings.openai import OpenAIEmbedding
        print("  [OK] llama-index-embeddings-openai")

        from llama_index.vector_stores.chroma import ChromaVectorStore
        print("  [OK] llama-index-vector-stores-chroma")

        from llama_index.postprocessor.cohere_rerank import CohereRerank
        print("  [OK] llama-index-postprocessor-cohere-rerank")

        import chromadb
        print("  [OK] chromadb")

        from mcp.server.fastmcp import FastMCP
        print("  [OK] fastmcp")

        return True
    except ImportError as e:
        print(f"  [ERROR] Import error: {e}")
        return False


def check_rag_package():
    """Check if rag package is importable."""
    print("\nChecking rag package...")
    try:
        from rag.config import settings, logger
        print("  [OK] rag.config")

        from rag.models import SearchMode, ProjectMetadata
        print("  [OK] rag.models")

        from rag.storage import ChromaManager, IndexManager
        print("  [OK] rag.storage")

        from rag.retrieval import SearchEngine
        print("  [OK] rag.retrieval")

        from rag.ingestion import DocumentLoader, DocumentProcessor
        print("  [OK] rag.ingestion")

        from rag.project import ProjectManager
        print("  [OK] rag.project")

        from rag.tools import register_all_tools
        print("  [OK] rag.tools")

        return True
    except ImportError as e:
        print(f"  [ERROR] Import error: {e}")
        return False


def check_openai_key():
    """Check if OpenAI API key is set."""
    print("\nChecking OpenAI API key...")
    if os.getenv("OPENAI_API_KEY"):
        key = os.getenv("OPENAI_API_KEY")
        if key.startswith("sk-"):
            print("  [OK] OPENAI_API_KEY is set")
            return True
        else:
            print("  [WARNING] OPENAI_API_KEY doesn't look valid (should start with 'sk-')")
            return True  # Still allow to continue
    else:
        print("  [ERROR] OPENAI_API_KEY not set")
        print("    Set it: export OPENAI_API_KEY='your-key'")
        return False


def check_optional_keys():
    """Check optional API keys."""
    print("\nChecking optional API keys...")
    
    cohere_key = os.getenv("COHERE_API_KEY")
    if cohere_key:
        print("  [OK] COHERE_API_KEY is set (reranking enabled)")
    else:
        print("  [INFO] COHERE_API_KEY not set (reranking disabled)")
    
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    if firecrawl_key:
        print("  [OK] FIRECRAWL_API_KEY is set (web crawling enabled)")
    else:
        print("  [INFO] FIRECRAWL_API_KEY not set (web crawling disabled)")
    
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        print("  [OK] GITHUB_TOKEN is set (GitHub indexing enabled)")
    else:
        print("  [INFO] GITHUB_TOKEN not set (GitHub indexing disabled)")
    
    return True


def check_data_folder():
    """Check if data folder exists and has files."""
    print("\nChecking data folder...")
    if os.path.exists("data"):
        files = os.listdir("data")
        if files:
            print(f"  [OK] Data folder exists with {len(files)} file(s)")
            for f in files[:5]:
                print(f"    - {f}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")
            return True
        else:
            print("  [WARNING] Data folder is empty")
            print("    Add documents: cp /path/to/docs/* data/")
            return True
    else:
        print("  [INFO] Data folder not found (will be created when needed)")
        return True


def check_mcp_server():
    """Check if MCP server file exists."""
    print("\nChecking MCP server...")
    if os.path.exists("mcp_server.py"):
        print("  [OK] mcp_server.py exists")
        return True
    else:
        print("  [ERROR] mcp_server.py not found")
        return False


def check_storage():
    """Check storage directory."""
    print("\nChecking storage...")
    if os.path.exists("storage"):
        print("  [OK] Storage directory exists")
        if os.path.exists("storage/chroma_db"):
            print("  [OK] ChromaDB initialized")
        if os.path.exists("storage/docstore.json"):
            print("  [OK] Docstore present")
        return True
    else:
        print("  [INFO] Storage directory not found (will be created on first index)")
        return True


def main():
    print("=" * 50)
    print("RAG System Setup Verification")
    print("=" * 50)

    results = []
    results.append(check_imports())
    results.append(check_rag_package())
    results.append(check_openai_key())
    results.append(check_optional_keys())
    results.append(check_data_folder())
    results.append(check_mcp_server())
    results.append(check_storage())

    print("\n" + "=" * 50)
    if all(results):
        print("[OK] All checks passed!")
        print("\nNext steps:")
        print("1. Add your documents to data/ folder")
        print("2. Run: python build_index.py")
        print("3. Configure MCP client with mcp_server.py")
        print("\nQuick test:")
        print("  python -c \"from mcp_server import query_rag; print(query_rag('test'))\"")
    else:
        print("[ERROR] Some checks failed. Please fix the issues above.")
        sys.exit(1)
    print("=" * 50)


if __name__ == "__main__":
    main()
