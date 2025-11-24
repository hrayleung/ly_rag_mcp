#!/usr/bin/env python3
"""
Comprehensive debugging and profiling tool for the RAG system.
Run this to diagnose issues, profile performance, and validate configurations.
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def check_environment():
    """Check environment variables and API keys."""
    print_section("Environment Variables")

    checks = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY")
    }

    for key, value in checks.items():
        if value:
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"✓ {key}: {masked}")
        else:
            status = "⚠" if key == "COHERE_API_KEY" else "✗"
            print(f"{status} {key}: Not set")
            if key == "OPENAI_API_KEY":
                print(f"  ERROR: {key} is required!")
                return False
            else:
                print(f"  WARNING: {key} is optional (reranking will be disabled)")

    return True


def check_storage():
    """Check storage directory and files."""
    print_section("Storage Status")

    storage_path = Path("./storage")

    if not storage_path.exists():
        print("✗ Storage directory not found")
        print("  Run: python build_index.py /path/to/documents")
        return False

    print(f"✓ Storage directory exists: {storage_path.absolute()}")

    # Check critical files
    files_to_check = [
        "docstore.json",
        "index_store.json",
        "indexed_files.json",
        "graph_store.json"
    ]

    print("\nStorage files:")
    for file in files_to_check:
        file_path = storage_path / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✓ {file}: {size:,} bytes")
        else:
            print(f"  ⚠ {file}: Missing")

    # Check ChromaDB
    chroma_path = storage_path / "chroma_db"
    if chroma_path.exists():
        chroma_size = sum(f.stat().st_size for f in chroma_path.rglob('*') if f.is_file())
        print(f"  ✓ chroma_db/: {chroma_size:,} bytes")
    else:
        print(f"  ✗ chroma_db/: Missing")
        return False

    return True


def load_index_metadata():
    """Load and analyze index metadata."""
    print_section("Index Metadata Analysis")

    try:
        # Load tracking data
        tracking_path = Path("./storage/indexed_files.json")
        if tracking_path.exists():
            with open(tracking_path) as f:
                tracking_data = json.load(f)

            print(f"Tracked files: {len(tracking_data)}")

            # Analyze file types
            extensions = {}
            total_size = 0
            for file_path, metadata in tracking_data.items():
                ext = Path(file_path).suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
                total_size += metadata.get('size', 0)

            print(f"Total indexed size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
            print("\nFile types distribution:")
            for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
                print(f"  {ext or '(no extension)'}: {count} files")
        else:
            print("⚠ No tracking data found (indexed_files.json missing)")

        # Load docstore
        docstore_path = Path("./storage/docstore.json")
        if docstore_path.exists():
            with open(docstore_path) as f:
                docstore = json.load(f)

            doc_count = len(docstore.get('docstore/data', {}))
            print(f"\nDocstore entries: {doc_count}")

    except Exception as e:
        print(f"✗ Error loading metadata: {e}")
        return False

    return True


def profile_retrieval(query: str = "test query", top_k_values: List[int] = None):
    """Profile retrieval performance with different parameters."""
    print_section("Retrieval Performance Profiling")

    if top_k_values is None:
        top_k_values = [1, 3, 6, 10, 15, 20]

    try:
        # Import here to avoid issues if dependencies not installed
        from mcp_server import get_index, get_reranker

        print(f"Query: '{query}'")
        print(f"Testing top_k values: {top_k_values}\n")

        index = get_index()
        reranker = get_reranker()

        print(f"Reranker available: {'Yes' if reranker else 'No (COHERE_API_KEY not set)'}\n")

        results = []

        for top_k in top_k_values:
            # Test without reranking
            start = time.time()
            retriever = index.as_retriever(similarity_top_k=top_k)
            nodes_no_rerank = retriever.retrieve(query)
            time_no_rerank = time.time() - start

            result = {
                "top_k": top_k,
                "time_no_rerank": time_no_rerank,
                "results_no_rerank": len(nodes_no_rerank)
            }

            # Test with reranking if available
            if reranker:
                start = time.time()
                initial_top_k = max(top_k * 2, 10)
                retriever_rerank = index.as_retriever(similarity_top_k=initial_top_k)
                nodes_rerank = retriever_rerank.retrieve(query)
                nodes_rerank = reranker.postprocess_nodes(nodes_rerank, query_str=query, top_n=top_k)
                time_rerank = time.time() - start

                result["time_rerank"] = time_rerank
                result["results_rerank"] = len(nodes_rerank)
                result["candidates_retrieved"] = initial_top_k

            results.append(result)

        # Print results table
        print(f"{'top_k':<8} {'No Rerank':<20} {'With Rerank':<30}")
        print(f"{'':8} {'Time (s)':<12} {'Results':<8} {'Time (s)':<12} {'Results':<8} {'Candidates':<10}")
        print("-" * 70)

        for r in results:
            line = f"{r['top_k']:<8} {r['time_no_rerank']:<12.4f} {r['results_no_rerank']:<8}"
            if 'time_rerank' in r:
                line += f" {r['time_rerank']:<12.4f} {r['results_rerank']:<8} {r['candidates_retrieved']:<10}"
            else:
                line += " " + "N/A".center(30)
            print(line)

        # Analysis
        print("\nAnalysis:")
        if reranker:
            avg_overhead = sum(r.get('time_rerank', 0) - r['time_no_rerank'] for r in results) / len(results)
            print(f"  Average reranking overhead: {avg_overhead:.4f}s")
            print(f"  Reranking adds: {avg_overhead/results[0]['time_no_rerank']*100:.1f}% latency on average")
        else:
            print("  Reranking not available (set COHERE_API_KEY to test)")

        # Check if results are as expected
        unexpected = [r for r in results if r['results_no_rerank'] != r['top_k']]
        if unexpected:
            print(f"\n  ⚠ WARNING: {len(unexpected)} queries returned fewer results than requested")
            print("    This indicates the index has fewer documents than some top_k values")

        return results

    except ImportError as e:
        print(f"✗ Cannot import dependencies: {e}")
        print("  Make sure you're in the correct conda environment")
        return None
    except Exception as e:
        print(f"✗ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_edge_cases():
    """Test edge cases and potential issues."""
    print_section("Edge Case Testing")

    try:
        from mcp_server import query_rag, iterative_search, get_index_stats

        tests = []

        # Test 1: Empty query
        print("Test 1: Empty query")
        try:
            result = query_rag("", similarity_top_k=3)
            tests.append(("Empty query", "PASS" if isinstance(result, str) else "FAIL"))
            print(f"  ✓ Handled gracefully: {result[:100]}...")
        except Exception as e:
            tests.append(("Empty query", f"FAIL: {e}"))
            print(f"  ✗ Error: {e}")

        # Test 2: Very large top_k
        print("\nTest 2: Very large top_k (100)")
        try:
            result = query_rag("test", similarity_top_k=100)
            tests.append(("Large top_k", "PASS" if isinstance(result, str) else "FAIL"))
            print(f"  ✓ Handled: {result[:100]}...")
        except Exception as e:
            tests.append(("Large top_k", f"FAIL: {e}"))
            print(f"  ✗ Error: {e}")

        # Test 3: top_k = 0
        print("\nTest 3: Invalid top_k (0)")
        try:
            result = query_rag("test", similarity_top_k=0)
            tests.append(("Zero top_k", "PASS" if isinstance(result, str) else "FAIL"))
            print(f"  ✓ Handled: {result[:100]}...")
        except Exception as e:
            tests.append(("Zero top_k", f"FAIL: {e}"))
            print(f"  ✗ Error: {e}")

        # Test 4: Special characters
        print("\nTest 4: Special characters in query")
        try:
            result = query_rag("test 中文 émojis 🔍", similarity_top_k=3)
            tests.append(("Special chars", "PASS" if isinstance(result, str) else "FAIL"))
            print(f"  ✓ Handled: {result[:100]}...")
        except Exception as e:
            tests.append(("Special chars", f"FAIL: {e}"))
            print(f"  ✗ Error: {e}")

        # Test 5: Iterative search
        print("\nTest 5: Iterative search")
        try:
            result = iterative_search("test", initial_top_k=3)
            tests.append(("Iterative search", "PASS" if isinstance(result, dict) else "FAIL"))
            print(f"  ✓ Returned: {list(result.keys())}")
        except Exception as e:
            tests.append(("Iterative search", f"FAIL: {e}"))
            print(f"  ✗ Error: {e}")

        # Test 6: Stats retrieval
        print("\nTest 6: Index stats")
        try:
            stats = get_index_stats()
            tests.append(("Index stats", "PASS" if 'document_count' in stats else "FAIL"))
            print(f"  ✓ Stats: {stats}")
        except Exception as e:
            tests.append(("Index stats", f"FAIL: {e}"))
            print(f"  ✗ Error: {e}")

        # Summary
        print("\nTest Summary:")
        passed = sum(1 for _, status in tests if status == "PASS")
        print(f"  Passed: {passed}/{len(tests)}")

        for test_name, status in tests:
            symbol = "✓" if status == "PASS" else "✗"
            print(f"  {symbol} {test_name}: {status}")

        return passed == len(tests)

    except ImportError as e:
        print(f"✗ Cannot import dependencies: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_index_quality():
    """Analyze the quality and distribution of indexed documents."""
    print_section("Index Quality Analysis")

    try:
        import chromadb
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.core import Settings

        # Configure
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

        # Load ChromaDB
        chroma_client = chromadb.PersistentClient(path="./storage/chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("rag_collection")

        # Get collection stats
        count = chroma_collection.count()
        print(f"Total documents in ChromaDB: {count}")

        if count == 0:
            print("⚠ Index is empty!")
            return False

        # Sample some documents
        if count > 0:
            sample_size = min(10, count)
            results = chroma_collection.get(limit=sample_size, include=['metadatas', 'documents'])

            print(f"\nSampling {sample_size} documents:")

            # Analyze document lengths
            lengths = [len(doc) for doc in results['documents']]
            avg_length = sum(lengths) / len(lengths)
            min_length = min(lengths)
            max_length = max(lengths)

            print(f"  Document lengths:")
            print(f"    Average: {avg_length:.0f} chars")
            print(f"    Min: {min_length} chars")
            print(f"    Max: {max_length} chars")

            # Analyze metadata
            if results['metadatas']:
                metadata_keys = set()
                for metadata in results['metadatas']:
                    if metadata:
                        metadata_keys.update(metadata.keys())

                print(f"\n  Metadata fields found: {', '.join(sorted(metadata_keys))}")

            # Check for quality issues
            print(f"\n  Quality checks:")

            too_short = sum(1 for l in lengths if l < 50)
            if too_short > 0:
                print(f"    ⚠ {too_short} documents < 50 chars (may be too short)")
            else:
                print(f"    ✓ No suspiciously short documents")

            too_long = sum(1 for l in lengths if l > 50000)
            if too_long > 0:
                print(f"    ⚠ {too_long} documents > 50K chars (may need chunking)")
            else:
                print(f"    ✓ No suspiciously long documents")

        return True

    except Exception as e:
        print(f"✗ Error analyzing index: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Debug and profile the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debug_rag.py                    # Run all checks
  python debug_rag.py --quick            # Quick checks only (skip profiling)
  python debug_rag.py --profile          # Profile only
  python debug_rag.py --query "test"     # Profile with custom query
        """
    )

    parser.add_argument('--quick', action='store_true', help='Skip performance profiling')
    parser.add_argument('--profile', action='store_true', help='Run profiling only')
    parser.add_argument('--query', default='test query', help='Query for profiling')

    args = parser.parse_args()

    print("=" * 70)
    print(" RAG System Debug & Profiling Tool")
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_passed = True

    if not args.profile:
        # Run checks
        all_passed &= check_environment()
        all_passed &= check_storage()
        all_passed &= load_index_metadata()

        if not all_passed:
            print("\n" + "="*70)
            print(" ✗ Critical checks failed! Fix issues before proceeding.")
            print("="*70)
            sys.exit(1)

        # Run quality analysis
        analyze_index_quality()

        # Run edge case tests
        test_edge_cases()

    if not args.quick:
        # Run performance profiling
        profile_retrieval(query=args.query)

    print("\n" + "="*70)
    if all_passed:
        print(" ✓ All checks passed!")
    else:
        print(" ⚠ Some issues detected - review output above")
    print("="*70)


if __name__ == "__main__":
    main()
