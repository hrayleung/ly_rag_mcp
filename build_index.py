#!/usr/bin/env python3
"""
Build a local RAG database using LlamaIndex and OpenAI embeddings.
Supports incremental updates to minimize API token usage.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
    Document,
)
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore


TRACKING_FILE = "./storage/indexed_files.json"


def load_tracking_data():
    """Load tracking data of previously indexed files."""
    if not Path(TRACKING_FILE).exists():
        return {}

    try:
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}


def save_tracking_data(tracking_data):
    """Save tracking data of indexed files."""
    Path(TRACKING_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(TRACKING_FILE, 'w') as f:
        json.dump(tracking_data, f, indent=2)


def get_file_metadata(file_path):
    """Get file metadata for change detection."""
    stat = file_path.stat()
    return {
        'path': str(file_path),
        'mtime': stat.st_mtime,
        'size': stat.st_size
    }


def is_file_changed(file_path, tracking_data):
    """Check if file is new or modified."""
    file_key = str(file_path)

    if file_key not in tracking_data:
        return True  # New file

    current_meta = get_file_metadata(file_path)
    stored_meta = tracking_data[file_key]

    # Check if modified
    if current_meta['mtime'] != stored_meta['mtime'] or \
       current_meta['size'] != stored_meta['size']:
        return True

    return False


def build_index(data_dir: str = "./data", rebuild: bool = False):
    """
    Build or update the RAG index from documents in the specified directory.

    Args:
        data_dir: Path to directory containing documents to index
        rebuild: If True, rebuild entire index from scratch
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables.\n"
            "Please set it: export OPENAI_API_KEY='your-api-key'"
        )

    # Configure LlamaIndex settings
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    # Validate data directory
    data_path = Path(data_dir).resolve()

    if not data_path.exists():
        print(f"Error: Directory not found: {data_path}")
        sys.exit(1)

    if not data_path.is_dir():
        print(f"Error: Path is not a directory: {data_path}")
        sys.exit(1)

    if not any(data_path.iterdir()):
        print(f"Warning: Directory is empty: {data_path}")
        sys.exit(1)

    # Define supported file extensions
    SUPPORTED_EXTS = {
        '.txt', '.pdf', '.docx', '.md', '.epub',
        '.ppt', '.pptx', '.pptm',
        '.csv', '.json', '.xml',
        '.ipynb', '.html',
        '.jpg', '.jpeg', '.png',
        '.hwp', '.mbox'
    }

    # Scan directory for all files
    all_files = []
    for item in data_path.rglob('*'):
        if item.is_file() and not item.name.startswith('.'):
            all_files.append(item)

    # Categorize files
    supported_files = []
    unsupported_files = []

    for file_path in all_files:
        ext = file_path.suffix.lower()
        if ext in SUPPORTED_EXTS:
            supported_files.append(file_path)
        else:
            unsupported_files.append(file_path)

    # Report unsupported files
    if unsupported_files:
        print(f"\n⚠ Skipping {len(unsupported_files)} unsupported file(s):")
        for f in unsupported_files[:10]:
            print(f"  - {f.name} ({f.suffix})")
        if len(unsupported_files) > 10:
            print(f"  ... and {len(unsupported_files) - 10} more")

    if not supported_files:
        print(f"\nError: No supported files found in {data_path}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTS))}")
        sys.exit(1)

    # Incremental update logic
    storage_path = Path("./storage")
    tracking_data = {} if rebuild else load_tracking_data()

    # Determine which files need processing
    files_to_process = []
    unchanged_files = []

    if rebuild:
        files_to_process = supported_files
        print(f"\n🔄 Full rebuild mode: processing all {len(supported_files)} file(s)...")
    else:
        for file_path in supported_files:
            if is_file_changed(file_path, tracking_data):
                files_to_process.append(file_path)
            else:
                unchanged_files.append(file_path)

        if unchanged_files:
            print(f"\n✓ {len(unchanged_files)} file(s) unchanged, skipping")

        if files_to_process:
            print(f"📝 {len(files_to_process)} new or modified file(s) to process:")
            for f in files_to_process[:10]:
                print(f"  - {f.name}")
            if len(files_to_process) > 10:
                print(f"  ... and {len(files_to_process) - 10} more")
        else:
            print("\n✓ No new or modified files. Index is up to date!")
            return

    # Load only files that need processing
    print(f"\nLoading {len(files_to_process)} document(s)...")

    documents = SimpleDirectoryReader(
        input_files=[str(f) for f in files_to_process],
        errors='ignore'
    ).load_data()

    print(f"Loaded {len(documents)} documents")

    # Clean document text
    print("Cleaning document encoding...")
    cleaned_documents = []

    for doc in documents:
        try:
            content = doc.get_content()

            try:
                cleaned = content.encode('utf-8', errors='surrogateescape').decode('utf-8', errors='ignore')
            except:
                cleaned = content.encode('utf-8', errors='replace').decode('utf-8')

            cleaned = ''.join(char for char in cleaned if char.isprintable() or char in '\n\t ')

            if len(cleaned.strip()) > 10:
                new_doc = Document(
                    text=cleaned,
                    metadata=doc.metadata,
                    excluded_llm_metadata_keys=doc.excluded_llm_metadata_keys,
                    excluded_embed_metadata_keys=doc.excluded_embed_metadata_keys,
                )
                cleaned_documents.append(new_doc)

        except Exception as e:
            print(f"Warning: Skipping problematic document: {e}")
            continue

    skipped = len(documents) - len(cleaned_documents)
    documents = cleaned_documents
    print(f"After cleaning: {len(documents)} valid documents ({skipped} skipped)")

    if not documents:
        print("No valid documents to index!")
        return

    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    storage_path.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path="./storage/chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("rag_collection")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Load existing index or create new one
    if not rebuild and (storage_path / "docstore.json").exists():
        print("Loading existing index for incremental update...")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir="./storage"
        )
        index = load_index_from_storage(storage_context)

        # Insert new documents
        print(f"Adding {len(documents)} new document(s) to existing index...")
        for doc in documents:
            index.insert(doc)
    else:
        # Create new index
        print("Building new index...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )

    # Persist index
    print("Persisting index...")
    index.storage_context.persist(persist_dir="./storage")

    # Update tracking data
    for file_path in files_to_process:
        tracking_data[str(file_path)] = get_file_metadata(file_path)

    save_tracking_data(tracking_data)

    # Get final stats
    final_count = chroma_collection.count()

    print("\n✓ Index updated successfully!")
    print(f"  - Source directory: {data_path}")
    print(f"  - Documents processed: {len(documents)}")
    print(f"  - Total documents in index: {final_count}")
    print(f"  - Storage location: {storage_path.resolve()}")
    print(f"\nYou can now query the index using the MCP server.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build or update RAG index from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_index.py                              # Incremental update of ./data
  python build_index.py /path/to/documents           # Incremental update of custom directory
  python build_index.py --rebuild                    # Force full rebuild of ./data
  python build_index.py /path/to/docs --rebuild      # Force full rebuild of custom directory
        """
    )

    parser.add_argument(
        'data_dir',
        nargs='?',
        default='./data',
        help='Directory containing documents to index (default: ./data)'
    )

    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force full rebuild instead of incremental update'
    )

    args = parser.parse_args()

    try:
        build_index(data_dir=args.data_dir, rebuild=args.rebuild)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
