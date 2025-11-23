#!/usr/bin/env python3
"""
MCP Server for querying the local RAG database.
Enhanced with 2025 best practices:
- MCP resources pattern for document listing
- Dynamic document ingestion
- Contextual retrieval support
"""
import os
from pathlib import Path
from datetime import datetime
from typing import List
from mcp.server.fastmcp import FastMCP
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Initialize FastMCP server
mcp = FastMCP("LlamaIndex RAG")

# Global variables to cache
_index = None
_chroma_client = None
_chroma_collection = None
_reranker = None

# File upload constraints (2025 best practices)
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = [
    # Documents
    '.txt', '.pdf', '.docx', '.md', '.epub',
    # Presentations
    '.ppt', '.pptx', '.pptm',
    # Data
    '.csv', '.json', '.xml',
    # Code & Notebooks
    '.ipynb', '.html',
    # Images (can extract text via OCR if needed)
    '.jpg', '.jpeg', '.png',
    # Other
    '.hwp', '.mbox'
]

# Get absolute paths (important for MCP server execution)
SCRIPT_DIR = Path(__file__).parent.absolute()
STORAGE_PATH = SCRIPT_DIR / "storage"
CHROMA_PATH = STORAGE_PATH / "chroma_db"


def get_chroma_client():
    """Get or create ChromaDB client (cached)."""
    global _chroma_client, _chroma_collection

    if _chroma_client is not None:
        return _chroma_client, _chroma_collection

    # Ensure directories exist
    STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    # Configure ChromaDB with settings for MCP environment
    chroma_settings = chromadb.Settings(
        allow_reset=True,
        anonymized_telemetry=False
    )

    _chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=chroma_settings
    )
    _chroma_collection = _chroma_client.get_or_create_collection("rag_collection")

    return _chroma_client, _chroma_collection


def get_reranker():
    """Get or create Cohere Reranker (cached)."""
    global _reranker

    if _reranker is not None:
        return _reranker

    # Check for Cohere API key
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        return None  # Reranker is optional

    # Create CohereRerank with latest model
    _reranker = CohereRerank(
        api_key=cohere_api_key,
        model="rerank-v3.5",  # Latest Cohere rerank model
        top_n=3  # Return top 3 after reranking
    )

    return _reranker


def get_index():
    """Load the RAG index from storage (cached). Creates new index if doesn't exist."""
    global _index

    if _index is not None:
        return _index

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables.\n"
            "Please set it in your MCP server configuration."
        )

    # Configure LlamaIndex settings
    # Only need embedding model - MCP client's LLM will generate answers
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    # Get cached ChromaDB client and collection
    chroma_client, chroma_collection = get_chroma_client()

    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Load or create index
    docstore_path = STORAGE_PATH / "docstore.json"
    if docstore_path.exists():
        # Load existing index
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(STORAGE_PATH)
        )
        _index = load_index_from_storage(storage_context)
    else:
        # Create new empty index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _index = VectorStoreIndex([], storage_context=storage_context)
        _index.storage_context.persist(persist_dir=str(STORAGE_PATH))

    return _index


@mcp.tool()
def query_rag(question: str, similarity_top_k: int = 3, use_rerank: bool = True) -> str:
    """
    Retrieve relevant documents from the RAG database with optional reranking.
    Returns raw text chunks for the MCP client's LLM to process.
    This saves costs by using your client's LLM instead of OpenAI's LLM.

    Args:
        question: The question/query to search for
        similarity_top_k: Number of final documents to return (default: 3)
        use_rerank: Whether to use Cohere reranking for better accuracy (default: True)

    Returns:
        Formatted string with retrieved document chunks and metadata
    """
    try:
        index = get_index()
        reranker = get_reranker() if use_rerank else None

        # If using reranker, retrieve more candidates (10) for reranking
        # Otherwise, retrieve the requested number
        initial_top_k = 10 if reranker else similarity_top_k
        retriever = index.as_retriever(similarity_top_k=initial_top_k)
        nodes = retriever.retrieve(question)

        if not nodes:
            return "No relevant documents found in the knowledge base."

        # Apply reranking if available
        if reranker:
            nodes = reranker.postprocess_nodes(nodes, query_str=question)
            # Take top similarity_top_k after reranking
            nodes = nodes[:similarity_top_k]
            rerank_used = " (reranked with Cohere Rerank v3.5)"
        else:
            rerank_used = ""

        # Format the retrieved chunks for the client's LLM
        result = f"Found {len(nodes)} relevant document(s){rerank_used}:\n\n"

        for i, node in enumerate(nodes, 1):
            result += f"--- Document {i} ---\n"
            result += f"{node.node.get_content()}\n"

            # Add metadata if available
            if node.node.metadata:
                result += f"\nMetadata: {node.node.metadata}\n"

            # Add similarity score if available
            if node.score:
                result += f"Relevance Score: {node.score:.3f}\n"

            result += "\n"

        return result

    except Exception as e:
        return f"Error retrieving documents: {str(e)}"


@mcp.tool()
def query_rag_with_sources(question: str, similarity_top_k: int = 3, use_rerank: bool = True) -> dict:
    """
    Query the RAG database and return source documents with full metadata.
    Similar to query_rag but returns structured data instead of formatted text.

    Args:
        question: The question to ask the RAG system
        similarity_top_k: Number of most similar documents to retrieve (default: 3)
        use_rerank: Whether to use Cohere reranking for better accuracy (default: True)

    Returns:
        Dictionary with 'sources' and metadata
    """
    try:
        index = get_index()
        reranker = get_reranker() if use_rerank else None

        # If using reranker, retrieve more candidates for reranking
        initial_top_k = 10 if reranker else similarity_top_k
        retriever = index.as_retriever(similarity_top_k=initial_top_k)
        nodes = retriever.retrieve(question)

        if not nodes:
            return {
                "sources": [],
                "message": "No relevant documents found"
            }

        # Apply reranking if available
        if reranker:
            nodes = reranker.postprocess_nodes(nodes, query_str=question)
            nodes = nodes[:similarity_top_k]

        sources = []
        for node in nodes:
            sources.append({
                "text": node.node.get_content(),
                "text_preview": node.node.get_content()[:200] + "...",
                "score": float(node.score) if node.score else None,
                "metadata": node.node.metadata,
            })

        return {
            "sources": sources,
            "total_found": len(sources),
            "reranked": reranker is not None,
            "rerank_model": "rerank-v3.5" if reranker else None
        }
    except Exception as e:
        return {"error": f"Error querying RAG: {str(e)}"}


@mcp.tool()
def get_index_stats() -> dict:
    """
    Get statistics about the RAG index.

    Returns:
        Dictionary with index statistics
    """
    try:
        if not STORAGE_PATH.exists():
            return {"error": "Index not found. Run build_index.py first."}

        # Use cached ChromaDB client
        chroma_client, chroma_collection = get_chroma_client()
        count = chroma_collection.count()

        return {
            "status": "ready",
            "document_count": count,
            "storage_location": str(STORAGE_PATH),
            "embedding_model": "text-embedding-3-large",
            "llm_model": "Using MCP client's LLM (not configured here)",
        }
    except Exception as e:
        return {"error": f"Error getting stats: {str(e)}"}


@mcp.tool()
def add_document_from_text(text: str, metadata: dict = None) -> dict:
    """
    Add a document from raw text to the RAG index.
    Useful for ingesting content from chat, web scraping, or API responses.

    Args:
        text: The text content to add
        metadata: Optional metadata dictionary (e.g., {'source': 'chat', 'topic': 'AI'})

    Returns:
        Dictionary with success status and document ID
    """
    try:
        if not text or len(text.strip()) == 0:
            return {"error": "Text cannot be empty"}

        # Create document
        doc_metadata = metadata or {}
        doc_metadata['added_via'] = 'mcp_tool'
        doc_metadata['added_at'] = datetime.now().isoformat()

        document = Document(text=text, metadata=doc_metadata)

        # Add to index
        index = get_index()
        node = document.as_node()
        index.insert_nodes([node])

        # Persist
        index.storage_context.persist(persist_dir=str(STORAGE_PATH))

        return {
            "success": True,
            "message": "Document added successfully",
            "node_id": node.node_id,
            "text_length": len(text)
        }
    except Exception as e:
        return {"error": f"Error adding document: {str(e)}"}


@mcp.tool()
def add_documents_from_directory(directory_path: str) -> dict:
    """
    Ingest all documents from a specified directory into the RAG index.
    Supports multiple file formats: txt, pdf, docx, md, csv, json.

    Args:
        directory_path: Absolute path to the directory containing documents

    Returns:
        Dictionary with ingestion results
    """
    try:
        dir_path = Path(directory_path)

        # Validate directory
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory_path}"}

        if not dir_path.is_dir():
            return {"error": f"Path is not a directory: {directory_path}"}

        # Load documents
        documents = SimpleDirectoryReader(
            input_dir=str(dir_path),
            recursive=True
        ).load_data()

        if not documents:
            return {"error": "No documents found in directory"}

        # Add metadata
        for doc in documents:
            doc.metadata['ingested_via'] = 'mcp_tool'
            doc.metadata['ingested_at'] = datetime.now().isoformat()
            doc.metadata['source_directory'] = str(dir_path)

        # Add to index
        index = get_index()
        nodes = [doc.as_node() for doc in documents]
        index.insert_nodes(nodes)

        # Persist
        index.storage_context.persist(persist_dir=str(STORAGE_PATH))

        return {
            "success": True,
            "message": f"Successfully ingested {len(documents)} documents",
            "document_count": len(documents),
            "directory": str(dir_path)
        }
    except Exception as e:
        return {"error": f"Error ingesting documents: {str(e)}"}


@mcp.tool()
def list_indexed_documents() -> dict:
    """
    List all documents currently in the RAG index with their metadata.

    Returns:
        Dictionary with list of documents and their metadata
    """
    try:
        # Use get_index_stats to get count without creating new ChromaDB client
        stats = get_index_stats()

        if "error" in stats:
            return {"error": stats["error"]}

        count = stats.get("document_count", 0)

        if count == 0:
            return {
                "success": True,
                "document_count": 0,
                "documents": [],
                "message": "No documents in the index"
            }

        # Query for a sample of documents using the retriever
        index = get_index()
        retriever = index.as_retriever(similarity_top_k=10)

        # Do a broad query to get sample documents
        from llama_index.core.schema import QueryBundle
        query = QueryBundle(query_str="文档 内容 代码")  # Broad search terms
        nodes = retriever.retrieve(query)

        docs = []
        for i, node in enumerate(nodes, 1):
            content = node.node.get_content()
            docs.append({
                "index": i,
                "text_preview": content[:150] + "..." if len(content) > 150 else content,
                "metadata": node.node.metadata,
                "text_length": len(content),
                "relevance_score": float(node.score) if node.score else None
            })

        return {
            "success": True,
            "document_count": count,
            "documents_shown": len(docs),
            "documents": docs,
            "message": f"Showing {len(docs)} sample documents out of {count} total"
        }
    except Exception as e:
        return {"error": f"Error listing documents: {str(e)}"}


@mcp.tool()
def clear_index() -> dict:
    """
    Clear all documents from the RAG index.
    WARNING: This is a destructive operation!

    Returns:
        Dictionary with operation status
    """
    try:
        global _index

        # Delete storage
        if STORAGE_PATH.exists():
            import shutil
            shutil.rmtree(STORAGE_PATH)

        # Reset cache
        _index = None

        # Reinitialize empty index
        get_index()

        return {
            "success": True,
            "message": "Index cleared successfully"
        }
    except Exception as e:
        return {"error": f"Error clearing index: {str(e)}"}


# MCP Resources: Expose indexed documents as resources (2025 best practice)
@mcp.resource("rag://documents")
def get_documents_resource() -> str:
    """
    MCP Resource: List all indexed documents
    Access pattern: rag://documents
    """
    try:
        result = list_indexed_documents()
        if "error" in result:
            return f"Error: {result['error']}"

        output = f"# Indexed Documents ({result['document_count']} total)\n\n"
        for doc in result["documents"]:
            output += f"## Document: {doc['node_id'][:8]}...\n"
            output += f"**Preview:** {doc['text_preview']}\n"
            output += f"**Length:** {doc['text_length']} chars\n"
            if doc['metadata']:
                output += f"**Metadata:** {doc['metadata']}\n"
            output += "\n---\n\n"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.resource("rag://stats")
def get_stats_resource() -> str:
    """
    MCP Resource: Get RAG index statistics
    Access pattern: rag://stats
    """
    try:
        stats = get_index_stats()
        if "error" in stats:
            return f"Error: {stats['error']}"

        output = "# RAG Index Statistics\n\n"
        for key, value in stats.items():
            output += f"**{key}:** {value}\n"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
