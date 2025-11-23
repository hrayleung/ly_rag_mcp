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
        print("  ✓ llama-index-core")

        from llama_index.embeddings.openai import OpenAIEmbedding
        print("  ✓ llama-index-embeddings-openai")

        from llama_index.vector_stores.chroma import ChromaVectorStore
        print("  ✓ llama-index-vector-stores-chroma")

        from llama_index.postprocessor.cohere_rerank import CohereRerank
        print("  ✓ llama-index-postprocessor-cohere-rerank")

        import chromadb
        print("  ✓ chromadb")

        from mcp.server.fastmcp import FastMCP
        print("  ✓ fastmcp")

        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def check_openai_key():
    """Check if OpenAI API key is set."""
    print("\nChecking OpenAI API key...")
    if os.getenv("OPENAI_API_KEY"):
        key = os.getenv("OPENAI_API_KEY")
        if key.startswith("sk-"):
            print("  ✓ OPENAI_API_KEY is set")
            return True
        else:
            print("  ✗ OPENAI_API_KEY doesn't look valid (should start with 'sk-')")
            return False
    else:
        print("  ✗ OPENAI_API_KEY not set")
        print("    Set it: export OPENAI_API_KEY='your-key'")
        return False

def check_data_folder():
    """Check if data folder exists and has files."""
    print("\nChecking data folder...")
    if os.path.exists("data"):
        files = os.listdir("data")
        if files:
            print(f"  ✓ Data folder exists with {len(files)} file(s)")
            for f in files:
                print(f"    - {f}")
            return True
        else:
            print("  ⚠ Data folder is empty")
            print("    Add documents: cp /path/to/docs/* data/")
            return True
    else:
        print("  ✗ Data folder not found")
        return False

def check_mcp_server():
    """Check if MCP server file exists."""
    print("\nChecking MCP server...")
    if os.path.exists("mcp_server.py"):
        print("  ✓ mcp_server.py exists")
        return True
    else:
        print("  ✗ mcp_server.py not found")
        return False

def main():
    print("=" * 50)
    print("RAG System Setup Verification")
    print("=" * 50)

    results = []
    results.append(check_imports())
    results.append(check_openai_key())
    results.append(check_data_folder())
    results.append(check_mcp_server())

    print("\n" + "=" * 50)
    if all(results):
        print("✓ All checks passed!")
        print("\nNext steps:")
        print("1. Add your documents to data/ folder")
        print("2. Run: python build_index.py")
        print("3. Configure MCP client with config from mcp_config_example.json")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    print("=" * 50)

if __name__ == "__main__":
    main()
