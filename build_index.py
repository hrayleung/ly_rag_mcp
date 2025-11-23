#!/usr/bin/env python3
"""
Build a local RAG database using LlamaIndex and OpenAI embeddings.
Supports command-line directory specification.
"""
import os
import sys
import argparse
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore


def build_index(data_dir: str = "./data"):
    """
    Build the RAG index from documents in the specified directory.

    Args:
        data_dir: Path to directory containing documents to index
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables.\n"
            "Please set it: export OPENAI_API_KEY='your-api-key'"
        )

    # Configure LlamaIndex settings (only embedding model, no LLM needed)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    # Validate data directory
    data_path = Path(data_dir).resolve()

    print(f"Loading documents from {data_path}...")

    if not data_path.exists():
        print(f"Error: Directory not found: {data_path}")
        sys.exit(1)

    if not data_path.is_dir():
        print(f"Error: Path is not a directory: {data_path}")
        sys.exit(1)

    if not any(data_path.iterdir()):
        print(f"Warning: Directory is empty: {data_path}")
        print("Please add documents before building the index.")
        sys.exit(1)

    # Define supported file extensions
    SUPPORTED_EXTS = {
        '.txt', '.pdf', '.docx', '.md', '.epub',  # Documents
        '.ppt', '.pptx', '.pptm',  # Presentations
        '.csv', '.json', '.xml',  # Data
        '.ipynb', '.html',  # Code & Notebooks
        '.jpg', '.jpeg', '.png',  # Images
        '.hwp', '.mbox'  # Other
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
        for f in unsupported_files[:10]:  # Show max 10
            print(f"  - {f.name} ({f.suffix})")
        if len(unsupported_files) > 10:
            print(f"  ... and {len(unsupported_files) - 10} more")

    if not supported_files:
        print(f"\nError: No supported files found in {data_path}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTS))}")
        sys.exit(1)

    print(f"\nProcessing {len(supported_files)} supported file(s)...")

    # Load documents recursively (will only load supported formats)
    documents = SimpleDirectoryReader(
        input_dir=str(data_path),
        recursive=True,
        required_exts=list(SUPPORTED_EXTS),  # Only load supported formats
        errors='ignore'  # Ignore encoding errors
    ).load_data()

    print(f"Loaded {len(documents)} documents")

    # Clean document text to handle encoding issues (critical for OpenAI API)
    print("Cleaning document encoding...")
    cleaned_documents = []

    for doc in documents:
        try:
            content = doc.get_content()

            # Aggressive cleaning: remove surrogates and non-UTF8 characters
            try:
                # Remove surrogate pairs that cause UTF-8 encoding errors
                cleaned = content.encode('utf-8', errors='surrogateescape').decode('utf-8', errors='ignore')
            except:
                # Fallback: replace all problematic characters
                cleaned = content.encode('utf-8', errors='replace').decode('utf-8')

            # Remove zero-width and control characters
            cleaned = ''.join(char for char in cleaned if char.isprintable() or char in '\n\t ')

            # Only keep documents with meaningful content
            if len(cleaned.strip()) > 10:
                # Create new Document object with cleaned text
                from llama_index.core import Document
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

    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    storage_path = Path("./storage")
    storage_path.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path="./storage/chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("rag_collection")

    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index
    print("Building index with OpenAI embeddings...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    # Persist index
    print("Persisting index to ./storage...")
    index.storage_context.persist(persist_dir="./storage")

    print("\n✓ Index built successfully!")
    print(f"  - Source directory: {data_path}")
    print(f"  - Documents indexed: {len(documents)}")
    print(f"  - Storage location: {storage_path.resolve()}")
    print(f"\nYou can now query the index using the MCP server.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build RAG index from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_index.py                              # Use default ./data directory
  python build_index.py /path/to/documents           # Specify directory directly
  python build_index.py ~/Documents/research         # Use absolute path
        """
    )

    parser.add_argument(
        'data_dir',
        nargs='?',
        default='./data',
        help='Directory containing documents to index (default: ./data)'
    )

    args = parser.parse_args()

    try:
        build_index(data_dir=args.data_dir)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
