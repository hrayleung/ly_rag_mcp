#!/usr/bin/env python3
"""
Test script for codebase indexing functionality.
Run this to test indexing without MCP client.
"""
import os
import sys
from pathlib import Path

# Ensure the script can import from current directory
sys.path.insert(0, str(Path(__file__).parent))

def test_local_codebase_indexing():
    """Test indexing the current llamaindex project."""
    print("=" * 60)
    print("Testing Local Codebase Indexing")
    print("=" * 60)

    # Import after setting up path
    from mcp_server import index_local_codebase, get_index_stats, query_rag

    # Get current project directory
    project_dir = Path(__file__).parent.absolute()
    print(f"\n[INDEXING] Directory: {project_dir}")

    # Test 1: Index with auto-detection (no language filter)
    print("\n[TEST 1] Auto-detect all code files...")
    result = index_local_codebase(
        directory_path=str(project_dir),
        exclude_patterns=["storage", "__pycache__", "*.pyc", "test_*"]
    )

    print("\nResult:")
    if "success" in result and result["success"]:
        print(f"  [SUCCESS] Complete!")
        print(f"  [STATS] Files indexed: {result['files_indexed']}")
        print(f"  [STATS] Chunks created: {result['chunks_created']}")
        print(f"  [STATS] Language breakdown:")
        for lang, count in result.get('language_breakdown', {}).items():
            print(f"      - {lang}: {count} files")
    else:
        print(f"  [ERROR] {result.get('error', 'Unknown error')}")
        return

    # Test 2: Check index stats
    print("\n[TEST 2] Checking index statistics...")
    stats = get_index_stats()
    print(f"  [STATS] Total documents in index: {stats.get('document_count', 0)}")
    print(f"  [STATS] Storage location: {stats.get('storage_location', 'N/A')}")

    # Test 3: Query the indexed code
    print("\n[TEST 3] Querying indexed code...")
    print("  Query: 'How does the RAG indexing work?'")
    response = query_rag(
        question="How does the RAG indexing work?",
        similarity_top_k=3,
        use_rerank=False  # Set to False to avoid Cohere API call in test
    )

    print("\n  Response preview:")
    lines = response.split('\n')[:15]
    for line in lines:
        print(f"    {line}")
    if len(response.split('\n')) > 15:
        print("    ... (truncated)")

    print("\n" + "=" * 60)
    print("[SUCCESS] All tests completed!")
    print("=" * 60)


def test_language_filter():
    """Test indexing with specific language filter."""
    print("\n" + "=" * 60)
    print("Testing Language-Specific Indexing")
    print("=" * 60)

    from mcp_server import index_local_codebase

    project_dir = Path(__file__).parent.absolute()

    print("\n[INDEXING] Only Python files...")
    result = index_local_codebase(
        directory_path=str(project_dir),
        language_filter=["python"],  # Only Python files
        exclude_patterns=["storage", "test_*"]
    )

    print("\nResult:")
    if "success" in result and result["success"]:
        print(f"  [SUCCESS] Complete!")
        print(f"  [STATS] Files indexed: {result['files_indexed']}")
        print(f"  [STATS] Language breakdown: {result.get('language_breakdown', {})}")
        print(f"  [STATS] Filters applied: {result.get('filters_applied', {})}")
    else:
        print(f"  [ERROR] {result.get('error', 'Unknown error')}")


def main():
    """Run all tests."""
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        print("   Please run: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    print("\n[STARTING] Codebase Indexing Tests\n")

    try:
        # Run tests
        test_local_codebase_indexing()

        # Optional: Test language filtering
        response = input("\n\nRun language filter test? (y/n): ")
        if response.lower() == 'y':
            test_language_filter()

        print("\n[SUCCESS] All tests completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
