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
from llama_index.core.node_parser import SentenceSplitter
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

    # Configure text splitter to handle large documents
    # chunk_size=1024 tokens (~4000 chars), overlap=200 chars for context continuity
    Settings.text_splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )

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
        # Documents
        '.txt', '.pdf', '.docx', '.md', '.epub',
        '.ppt', '.pptx', '.pptm',
        '.csv', '.json', '.xml',
        '.ipynb', '.html',
        '.jpg', '.jpeg', '.png',
        '.hwp', '.mbox',
        # Code files
        '.py', '.js', '.ts', '.jsx', '.tsx',
        '.java', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.go', '.rs', '.rb', '.php',
        '.swift', '.kt', '.scala', '.r',
        '.sh', '.bash', '.zsh',
        '.sql', '.yaml', '.yml', '.toml',
        '.dockerfile', '.makefile',
        '.vue', '.svelte', '.astro'
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
        print(f"\nWARNING: Skipping {len(unsupported_files)} unsupported file(s):")
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
        print(f"\nFull rebuild mode: processing all {len(supported_files)} file(s)...")
    else:
        for file_path in supported_files:
            if is_file_changed(file_path, tracking_data):
                files_to_process.append(file_path)
            else:
                unchanged_files.append(file_path)

        if unchanged_files:
            print(f"\n[OK] {len(unchanged_files)} file(s) unchanged, skipping")

        if files_to_process:
            print(f"[PROCESSING] {len(files_to_process)} new or modified file(s) to process:")
            for f in files_to_process[:10]:
                print(f"  - {f.name}")
            if len(files_to_process) > 10:
                print(f"  ... and {len(files_to_process) - 10} more")
        else:
            print("\n[OK] No new or modified files. Index is up to date!")
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
                # --- Context Injection Logic ---
                # Prepend folder context and filename to text
                file_path_str = doc.metadata.get('file_path')
                if file_path_str:
                    path_obj = Path(file_path_str)
                    file_name = path_obj.name
                    
                    # Try to calculate relative path from the indexed root (data_path)
                    try:
                        # data_path is available in local scope
                        rel_path = path_obj.relative_to(data_path)
                        context_parts = list(rel_path.parent.parts)
                    except ValueError:
                        # Fallback: Use last 2 parent folders
                        parents = list(path_obj.parent.parts)
                        context_parts = [p for p in parents if p and p not in ('/', '\\')]
                        if len(parents) >= 2:
                            context_parts = parents[-2:]
                        
                    # Filter noise
                    context_parts = [p for p in context_parts if p and p != "."]
                    context_str = " / ".join(context_parts)
                    
                    prefix = ""
                    if context_str:
                        prefix += f"Context: {context_str}\n"
                    prefix += f"Filename: {file_name}\n\n"
                    
                    if not cleaned.startswith("Context: "):
                        cleaned = prefix + cleaned
                        
                    # Add to metadata so it gets picked up by clean_metadata loop below
                    doc.metadata['filename'] = file_name
                    if context_str:
                        doc.metadata['folder_context'] = context_str
                # -------------------------------

                # Clean metadata - ChromaDB only accepts str, int, float, None
                clean_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, type(None))):
                        clean_metadata[key] = value
                    elif isinstance(value, bool):
                        clean_metadata[key] = str(value)
                    elif isinstance(value, (list, dict)):
                        # Skip complex types that ChromaDB doesn't support
                        continue
                    else:
                        # Convert other types to string
                        clean_metadata[key] = str(value)

                new_doc = Document(
                    text=cleaned,
                    metadata=clean_metadata,
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

    # Check for large documents that will be chunked
    large_docs = [doc for doc in documents if len(doc.text) > 10000]
    if large_docs:
        total_chars = sum(len(doc.text) for doc in large_docs)
        print(f"\n[INFO] {len(large_docs)} large document(s) detected ({total_chars:,} total chars)")
        print(f"   Will be automatically chunked into smaller pieces (1024 tokens/chunk)")
        print(f"   This prevents 'max tokens per request' errors")

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

        # Insert new documents in batch (much faster than one-by-one)
        print(f"Adding {len(documents)} new document(s) to existing index...")

        # Use the configured text splitter to chunk documents properly
        text_splitter = Settings.text_splitter
        nodes = []
        for doc in documents:
            # Split document into chunks if needed
            doc_nodes = text_splitter.get_nodes_from_documents([doc])
            nodes.extend(doc_nodes)

        print(f"Created {len(nodes)} node(s) from {len(documents)} document(s) (with chunking)")
        index.insert_nodes(nodes, show_progress=True)
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

    print("\n[SUCCESS] Index updated successfully!")
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
        print(f"\n[ERROR] {e}")
        sys.exit(1)
