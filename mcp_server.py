#!/usr/bin/env python3
"""
MCP Server for querying the local RAG database.
Enhanced with 2025 best practices:
- MCP resources pattern for document listing
- Dynamic document ingestion
- Contextual retrieval support
- Comprehensive validation and error handling
- Performance optimization with caching
"""
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from functools import wraps
import time
import nest_asyncio
from mcp.server.fastmcp import FastMCP

# Allow nested event loops (needed for GitHub reader)
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=os.getenv("RAG_LOG_LEVEL", "WARNING"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_mcp_server")
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
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

# Cache performance metrics
_cache_stats = {
    "index_loads": 0,
    "index_cache_hits": 0,
    "reranker_loads": 0,
    "reranker_cache_hits": 0,
    "chroma_loads": 0,
    "chroma_cache_hits": 0
}

# File upload constraints
MAX_FILE_SIZE_MB = 300
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
]

# Get absolute paths (important for MCP server execution)
SCRIPT_DIR = Path(__file__).parent.absolute()
STORAGE_PATH = SCRIPT_DIR / "storage"
CHROMA_PATH = STORAGE_PATH / "chroma_db"

# Configuration constants
MIN_TOP_K = 1
MAX_TOP_K = 50  # Increased from implicit 20 for flexibility
DEFAULT_TOP_K = 6
RERANK_CANDIDATE_MULTIPLIER = 2
MIN_RERANK_CANDIDATES = 10


def validate_top_k(top_k: int) -> int:
    """Validate and clamp top_k parameter."""
    if not isinstance(top_k, int):
        logger.warning(f"top_k must be int, got {type(top_k).__name__}, converting...")
        top_k = int(top_k)

    if top_k < MIN_TOP_K:
        logger.warning(f"top_k={top_k} too small, using minimum {MIN_TOP_K}")
        return MIN_TOP_K
    elif top_k > MAX_TOP_K:
        logger.warning(f"top_k={top_k} too large, using maximum {MAX_TOP_K}")
        return MAX_TOP_K

    return top_k


def validate_query(query: str) -> str:
    """Validate and clean query string."""
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty or whitespace only")

    if len(query) > 10000:
        logger.warning(f"Query is very long ({len(query)} chars), truncating to 10000")
        query = query[:10000]

    return query


def log_performance(func):
    """Decorator to log function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__

        logger.info(f"Starting {function_name}")

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Completed {function_name} in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed {function_name} after {elapsed:.3f}s: {e}")
            raise

    return wrapper


def get_chroma_client():
    """Get or create ChromaDB client (cached)."""
    global _chroma_client, _chroma_collection, _cache_stats

    if _chroma_client is not None:
        _cache_stats["chroma_cache_hits"] += 1
        logger.debug("ChromaDB client: cache hit")
        return _chroma_client, _chroma_collection

    _cache_stats["chroma_loads"] += 1
    logger.debug("ChromaDB client: loading from disk")

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
    global _reranker, _cache_stats

    if _reranker is not None:
        _cache_stats["reranker_cache_hits"] += 1
        logger.debug("Reranker: cache hit")
        return _reranker

    _cache_stats["reranker_loads"] += 1

    # Check for Cohere API key
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        return None  # Reranker is optional

    # Create CohereRerank with latest model
    # Set top_n high enough to handle any query, we'll slice results manually
    _reranker = CohereRerank(
        api_key=cohere_api_key,
        model="rerank-v3.5",  # Latest Cohere rerank model
        top_n=MAX_TOP_K  # Set to max, we'll slice results manually for dynamic control
    )

    return _reranker


def get_index():
    """Load the RAG index from storage (cached). Creates new index if doesn't exist."""
    global _index, _cache_stats

    if _index is not None:
        _cache_stats["index_cache_hits"] += 1
        logger.debug("Index: cache hit")
        return _index

    _cache_stats["index_loads"] += 1
    logger.debug("Index: loading from storage")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables.\n"
            "Please set it in your MCP server configuration."
        )

    # Configure LlamaIndex settings
    # Only need embedding model - MCP client's LLM will generate answers
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    # Configure text splitter (must match build_index.py settings)
    Settings.text_splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )

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
@log_performance
def query_rag(question: str, similarity_top_k: int = DEFAULT_TOP_K, use_rerank: bool = True) -> str:
    """
    Retrieve relevant documents from the RAG database with optional reranking.
    Returns raw text chunks for the MCP client's LLM to process.
    This saves costs by using your client's LLM instead of OpenAI's LLM.

    Args:
        question: The question/query to search for
        similarity_top_k: Number of final documents to return (default: 6, range: 1-50)
        use_rerank: Whether to use Cohere reranking for better accuracy (default: True)

    Returns:
        Formatted string with retrieved document chunks and metadata
    """
    try:
        # Validate inputs
        question = validate_query(question)
        similarity_top_k = validate_top_k(similarity_top_k)

        logger.debug(f"Query: '{question[:100]}...', top_k={similarity_top_k}, rerank={use_rerank}")

        index = get_index()
        reranker = get_reranker() if use_rerank else None

        # If using reranker, retrieve more candidates for reranking
        # Use 2x similarity_top_k or MIN_RERANK_CANDIDATES, whichever is larger
        if reranker:
            initial_top_k = max(similarity_top_k * RERANK_CANDIDATE_MULTIPLIER, MIN_RERANK_CANDIDATES)
            logger.debug(f"Using reranker: retrieving {initial_top_k} candidates for top {similarity_top_k}")
        else:
            initial_top_k = similarity_top_k
            logger.debug(f"No reranker: retrieving {initial_top_k} documents directly")

        retriever = index.as_retriever(similarity_top_k=initial_top_k)
        nodes = retriever.retrieve(question)

        logger.debug(f"Retrieved {len(nodes)} initial candidates")

        if not nodes:
            logger.info("No relevant documents found")
            return "No relevant documents found in the knowledge base."

        # Apply reranking if available
        if reranker:
            rerank_start = time.time()
            # Rerank all candidates
            nodes = reranker.postprocess_nodes(nodes, query_str=question)
            # Manually slice to get desired top_k
            nodes = nodes[:similarity_top_k]
            rerank_time = time.time() - rerank_start
            logger.debug(f"Reranking took {rerank_time:.3f}s, returned {len(nodes)} nodes")
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

        logger.info(f"Successfully retrieved and formatted {len(nodes)} documents")
        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return f"Invalid input: {str(e)}"
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}", exc_info=True)
        return f"Error retrieving documents: {str(e)}"


@mcp.tool()
@log_performance
def query_rag_with_sources(question: str, similarity_top_k: int = DEFAULT_TOP_K, use_rerank: bool = True) -> dict:
    """
    Query the RAG database and return source documents with full metadata.
    Similar to query_rag but returns structured data instead of formatted text.

    Args:
        question: The question to ask the RAG system
        similarity_top_k: Number of most similar documents to retrieve (default: 6, range: 1-50)
        use_rerank: Whether to use Cohere reranking for better accuracy (default: True)

    Returns:
        Dictionary with 'sources' and metadata
    """
    try:
        # Validate inputs
        question = validate_query(question)
        similarity_top_k = validate_top_k(similarity_top_k)

        logger.debug(f"Query with sources: '{question[:100]}...', top_k={similarity_top_k}")

        index = get_index()
        reranker = get_reranker() if use_rerank else None

        # If using reranker, retrieve more candidates for reranking
        initial_top_k = max(similarity_top_k * RERANK_CANDIDATE_MULTIPLIER, MIN_RERANK_CANDIDATES) if reranker else similarity_top_k
        retriever = index.as_retriever(similarity_top_k=initial_top_k)
        nodes = retriever.retrieve(question)

        if not nodes:
            logger.info("No relevant documents found")
            return {
                "sources": [],
                "message": "No relevant documents found"
            }

        # Apply reranking if available
        if reranker:
            nodes = reranker.postprocess_nodes(nodes, query_str=question)
            # Manually slice to get desired top_k
            nodes = nodes[:similarity_top_k]
            logger.debug(f"After reranking: {len(nodes)} nodes")

        sources = []
        for node in nodes:
            content = node.node.get_content()
            sources.append({
                "text": content,
                "text_preview": content[:200] + "..." if len(content) > 200 else content,
                "score": float(node.score) if node.score else None,
                "metadata": node.node.metadata,
            })

        logger.info(f"Successfully retrieved {len(sources)} sources")
        return {
            "sources": sources,
            "total_found": len(sources),
            "reranked": reranker is not None,
            "rerank_model": "rerank-v3.5" if reranker else None
        }
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {"error": f"Invalid input: {str(e)}"}
    except Exception as e:
        logger.error(f"Error querying RAG: {e}", exc_info=True)
        return {"error": f"Error querying RAG: {str(e)}"}


@mcp.tool()
def get_cache_stats() -> dict:
    """
    Get cache performance metrics to understand caching efficiency.

    Returns:
        Dictionary with cache hit/miss statistics
    """
    global _cache_stats

    total_index = _cache_stats["index_loads"] + _cache_stats["index_cache_hits"]
    total_reranker = _cache_stats["reranker_loads"] + _cache_stats["reranker_cache_hits"]
    total_chroma = _cache_stats["chroma_loads"] + _cache_stats["chroma_cache_hits"]

    index_hit_rate = (_cache_stats["index_cache_hits"] / total_index * 100) if total_index > 0 else 0
    reranker_hit_rate = (_cache_stats["reranker_cache_hits"] / total_reranker * 100) if total_reranker > 0 else 0
    chroma_hit_rate = (_cache_stats["chroma_cache_hits"] / total_chroma * 100) if total_chroma > 0 else 0

    return {
        "index": {
            "cache_hits": _cache_stats["index_cache_hits"],
            "loads": _cache_stats["index_loads"],
            "hit_rate": f"{index_hit_rate:.1f}%"
        },
        "reranker": {
            "cache_hits": _cache_stats["reranker_cache_hits"],
            "loads": _cache_stats["reranker_loads"],
            "hit_rate": f"{reranker_hit_rate:.1f}%"
        },
        "chroma": {
            "cache_hits": _cache_stats["chroma_cache_hits"],
            "loads": _cache_stats["chroma_loads"],
            "hit_rate": f"{chroma_hit_rate:.1f}%"
        },
        "summary": f"Cache is working well" if min(index_hit_rate, chroma_hit_rate) > 50 else "Cache warming up"
    }


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

        # Add to index with chunking
        index = get_index()
        text_splitter = Settings.text_splitter
        nodes = text_splitter.get_nodes_from_documents([document])
        index.insert_nodes(nodes)

        # Persist
        index.storage_context.persist(persist_dir=str(STORAGE_PATH))

        return {
            "success": True,
            "message": f"Document added successfully ({len(nodes)} chunk(s) created)",
            "chunks_created": len(nodes),
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

        # Add to index with chunking
        index = get_index()
        text_splitter = Settings.text_splitter
        nodes = []
        for doc in documents:
            doc_nodes = text_splitter.get_nodes_from_documents([doc])
            nodes.extend(doc_nodes)
        index.insert_nodes(nodes)

        # Persist
        index.storage_context.persist(persist_dir=str(STORAGE_PATH))

        return {
            "success": True,
            "message": f"Successfully ingested {len(documents)} documents ({len(nodes)} chunks)",
            "document_count": len(documents),
            "chunks_created": len(nodes),
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
                "node_id": node.node.node_id,
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
@log_performance
def iterative_search(question: str, initial_top_k: int = 3, detailed_top_k: int = 10) -> dict:
    """
    Perform a two-phase iterative search: first get a small set of highly relevant results,
    then optionally retrieve more for deeper exploration.

    This tool encourages multi-round search patterns by:
    1. Returning initial focused results (default 3)
    2. Providing metadata to help decide if more context is needed
    3. Suggesting if a follow-up search with more results would be beneficial

    The MCP client's LLM can then decide whether to:
    - Use the initial results if they're sufficient
    - Request more results by calling query_rag() with higher similarity_top_k
    - Refine the question and search again
    - Search for specific aspects mentioned in the initial results

    Args:
        question: The question/query to search for
        initial_top_k: Number of initial results to return (default: 3, recommended: 2-5)
        detailed_top_k: Number of results available for detailed exploration (default: 10)

    Returns:
        Dictionary with initial results and suggestions for refinement
    """
    try:
        # Validate inputs
        question = validate_query(question)
        initial_top_k = validate_top_k(initial_top_k)
        detailed_top_k = validate_top_k(detailed_top_k)

        logger.debug(f"Iterative search: '{question[:100]}...', initial={initial_top_k}, detailed={detailed_top_k}")

        index = get_index()
        reranker = get_reranker()

        # Get initial focused results
        candidate_count = max(initial_top_k * RERANK_CANDIDATE_MULTIPLIER, MIN_RERANK_CANDIDATES)
        initial_retriever = index.as_retriever(similarity_top_k=candidate_count)
        initial_nodes = initial_retriever.retrieve(question)

        logger.debug(f"Retrieved {len(initial_nodes)} candidates for iterative search")

        if not initial_nodes:
            return {
                "initial_results": [],
                "message": "No relevant documents found",
                "suggestion": "Try rephrasing the question or using different keywords"
            }

        # Rerank and get top initial results
        if reranker:
            initial_nodes = reranker.postprocess_nodes(initial_nodes, query_str=question)
            # Manually slice to get desired top_k
            initial_nodes = initial_nodes[:initial_top_k]
        else:
            initial_nodes = initial_nodes[:initial_top_k]

        # Format initial results
        initial_results = []
        topics_found = set()

        for i, node in enumerate(initial_nodes, 1):
            content = node.node.get_content()
            result = {
                "rank": i,
                "text": content,
                "preview": content[:300] + "..." if len(content) > 300 else content,
                "score": float(node.score) if node.score else None,
                "metadata": node.node.metadata,
                "length": len(content)
            }
            initial_results.append(result)

            # Extract potential topics for refinement suggestions
            if node.node.metadata and 'file_name' in node.node.metadata:
                topics_found.add(node.node.metadata['file_name'])

        # Analyze score distribution to suggest if more results needed
        if initial_results:
            scores = [r['score'] for r in initial_results if r['score'] is not None]
            avg_score = sum(scores) / len(scores) if scores else 0

            # Generate refinement suggestions
            suggestions = []

            if avg_score < 0.7 and len(scores) > 0:
                suggestions.append(
                    f"Initial results have moderate relevance (avg score: {avg_score:.2f}). "
                    "Consider refining the question or trying different keywords."
                )

            if len(initial_results) < initial_top_k:
                suggestions.append(
                    f"Only {len(initial_results)} documents found. The knowledge base may have limited information on this topic."
                )
            else:
                suggestions.append(
                    f"Found {len(initial_results)} initial results. "
                    f"You can retrieve up to {detailed_top_k} more results using query_rag(similarity_top_k={detailed_top_k}) "
                    "if you need broader context or want to explore related topics."
                )

            if topics_found:
                suggestions.append(
                    f"Documents found in: {', '.join(list(topics_found)[:3])}. "
                    "You can search for specific aspects within these sources."
                )

        return {
            "initial_results": initial_results,
            "total_initial": len(initial_results),
            "reranked": reranker is not None,
            "suggestions": suggestions,
            "next_steps": [
                f"If satisfied: Use these {len(initial_results)} results to answer the question",
                f"If need more context: Call query_rag(question='{question}', similarity_top_k={detailed_top_k})",
                "If need clarification: Refine the question and search again",
                "If found specific topics: Search for those specific aspects"
            ]
        }

    except ValueError as e:
        logger.error(f"Validation error in iterative search: {e}")
        return {"error": f"Invalid input: {str(e)}"}
    except Exception as e:
        logger.error(f"Error in iterative search: {e}", exc_info=True)
        return {"error": f"Error in iterative search: {str(e)}"}


@mcp.tool()
def index_github_repository(
    owner: str,
    repo: str,
    branch: str = "main",
    filter_dirs: Optional[List[str]] = None,
    filter_extensions: Optional[List[str]] = None
) -> dict:
    """
    Index a GitHub repository for semantic code search.
    Clones and indexes code from a public or private GitHub repository.

    Automatically uses GITHUB_TOKEN from environment variables.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        branch: Branch to index (default: "main")
        filter_dirs: Optional list of directories to include (e.g., ["src", "lib"])
        filter_extensions: Optional list of file extensions to include (e.g., [".py", ".js"])

    Returns:
        Dictionary with indexing results

    Example:
        index_github_repository(
            owner="OpenBMB",
            repo="UltraRAG",
            branch="main"
        )
    """
    try:
        from llama_index.readers.github import GithubRepositoryReader, GithubClient

        # Read GitHub token from environment variable
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            return {
                "error": "GitHub token not found",
                "suggestion": "Set GITHUB_TOKEN environment variable in your .mcp.json config file"
            }

        logger.info(f"Indexing GitHub repository: {owner}/{repo} (branch: {branch})")

        # Initialize GitHub client
        github_client = GithubClient(github_token=token, verbose=True)

        # Configure reader
        reader = GithubRepositoryReader(
            github_client=github_client,
            owner=owner,
            repo=repo,
            filter_directories=(filter_dirs, GithubRepositoryReader.FilterType.INCLUDE) if filter_dirs else None,
            filter_file_extensions=(filter_extensions, GithubRepositoryReader.FilterType.INCLUDE) if filter_extensions else None,
            verbose=True,
        )

        # Load documents from the repository
        logger.info("Loading documents from repository...")
        documents = reader.load_data(branch=branch)

        if not documents:
            return {
                "error": f"No documents found in {owner}/{repo}",
                "suggestion": "Check filter_dirs and filter_extensions, or verify the repository has content"
            }

        logger.info(f"Loaded {len(documents)} documents from repository")

        # Add metadata
        for doc in documents:
            doc.metadata['source'] = 'github'
            doc.metadata['repository'] = f"{owner}/{repo}"
            doc.metadata['branch'] = branch
            doc.metadata['indexed_at'] = datetime.now().isoformat()
            doc.metadata['indexed_via'] = 'github_indexer'

        # Add to index with chunking
        index = get_index()
        text_splitter = Settings.text_splitter
        nodes = []
        for doc in documents:
            doc_nodes = text_splitter.get_nodes_from_documents([doc])
            nodes.extend(doc_nodes)

        logger.info(f"Created {len(nodes)} chunks from {len(documents)} documents")
        index.insert_nodes(nodes, show_progress=True)

        # Persist
        index.storage_context.persist(persist_dir=str(STORAGE_PATH))

        return {
            "success": True,
            "message": f"Successfully indexed {owner}/{repo}",
            "repository": f"{owner}/{repo}",
            "branch": branch,
            "documents_indexed": len(documents),
            "chunks_created": len(nodes),
            "filters_applied": {
                "directories": filter_dirs or "all",
                "extensions": filter_extensions or "all"
            }
        }

    except ImportError:
        return {
            "error": "GitHub reader not installed",
            "fix": "Run: pip install llama-index-readers-github"
        }
    except Exception as e:
        logger.error(f"Error indexing GitHub repository: {e}", exc_info=True)
        return {"error": f"Error indexing repository: {str(e)}"}


@mcp.tool()
def index_local_codebase(
    directory_path: str,
    language_filter: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    include_hidden: bool = False
) -> dict:
    """
    Index a local codebase directory with code-aware processing.
    Optimized for source code with language-specific filtering and common excludes.

    Args:
        directory_path: Absolute path to the codebase directory
        language_filter: Optional list of languages/extensions (e.g., ["python", "javascript", ".ts"])
        exclude_patterns: Optional list of glob patterns to exclude (e.g., ["node_modules", "*.test.js"])
        include_hidden: Whether to include hidden files/directories (default: False)

    Returns:
        Dictionary with indexing results

    Example:
        index_local_codebase(
            directory_path="/path/to/project",
            language_filter=["python", "typescript"],
            exclude_patterns=["node_modules", "__pycache__", "*.pyc"]
        )
    """
    try:
        dir_path = Path(directory_path).resolve()

        # Validate directory
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory_path}"}

        if not dir_path.is_dir():
            return {"error": f"Path is not a directory: {directory_path}"}

        logger.info(f"Indexing local codebase: {dir_path}")

        # Language to extension mapping
        LANGUAGE_EXTENSIONS = {
            "python": [".py"],
            "javascript": [".js", ".jsx", ".mjs", ".cjs"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "cpp": [".cpp", ".hpp", ".cc", ".cxx", ".c", ".h"],
            "csharp": [".cs"],
            "go": [".go"],
            "rust": [".rs"],
            "ruby": [".rb"],
            "php": [".php"],
            "swift": [".swift"],
            "kotlin": [".kt"],
            "scala": [".scala"],
            "shell": [".sh", ".bash", ".zsh"],
            "sql": [".sql"],
            "html": [".html", ".htm"],
            "css": [".css", ".scss", ".sass", ".less"],
            "yaml": [".yaml", ".yml"],
            "json": [".json"],
            "xml": [".xml"],
            "markdown": [".md", ".markdown"],
            "vue": [".vue"],
            "svelte": [".svelte"],
            "astro": [".astro"]
        }

        # Common directories to exclude by default
        DEFAULT_EXCLUDES = [
            "node_modules", "__pycache__", ".git", ".svn", ".hg",
            "venv", "env", ".venv", ".env",
            "build", "dist", "target", "out",
            ".idea", ".vscode", ".vs",
            "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll",
            ".DS_Store", "Thumbs.db"
        ]

        # Combine default and user excludes
        exclude_patterns = exclude_patterns or []
        all_excludes = set(DEFAULT_EXCLUDES + exclude_patterns)

        # Determine which extensions to include
        allowed_exts = set()
        if language_filter:
            for lang in language_filter:
                # Check if it's a language name
                if lang.lower() in LANGUAGE_EXTENSIONS:
                    allowed_exts.update(LANGUAGE_EXTENSIONS[lang.lower()])
                # Or a direct extension
                elif lang.startswith('.'):
                    allowed_exts.add(lang.lower())
                else:
                    allowed_exts.add(f".{lang.lower()}")
        else:
            # Include all code extensions from ALLOWED_EXTENSIONS
            allowed_exts = set([ext for ext in ALLOWED_EXTENSIONS if ext not in ['.pdf', '.docx', '.ppt', '.pptx', '.pptm', '.jpg', '.jpeg', '.png', '.hwp', '.mbox', '.epub']])

        logger.info(f"Scanning for extensions: {sorted(allowed_exts)}")
        logger.info(f"Excluding patterns: {sorted(all_excludes)}")

        # Collect files
        files_to_index = []
        skipped_files = []

        for item in dir_path.rglob('*'):
            # Skip hidden files/dirs if requested
            if not include_hidden and any(part.startswith('.') for part in item.parts):
                continue

            # Skip non-files
            if not item.is_file():
                continue

            # Check exclusion patterns
            relative_path = item.relative_to(dir_path)
            excluded = False
            for pattern in all_excludes:
                if pattern in str(relative_path):
                    excluded = True
                    break
                # Check glob patterns
                if '*' in pattern and relative_path.match(pattern):
                    excluded = True
                    break

            if excluded:
                skipped_files.append(item)
                continue

            # Check extension
            if item.suffix.lower() in allowed_exts:
                files_to_index.append(item)
            else:
                skipped_files.append(item)

        if not files_to_index:
            return {
                "error": "No files found matching criteria",
                "scanned_directory": str(dir_path),
                "filters": {
                    "languages": language_filter or "all code files",
                    "extensions": sorted(allowed_exts),
                    "excluded": sorted(all_excludes)
                },
                "suggestion": "Adjust language_filter or exclude_patterns"
            }

        logger.info(f"Found {len(files_to_index)} files to index, skipped {len(skipped_files)} files")

        # Load documents
        documents = SimpleDirectoryReader(
            input_files=[str(f) for f in files_to_index],
            errors='ignore'
        ).load_data()

        if not documents:
            return {"error": "Failed to load documents from codebase"}

        # Add code-specific metadata
        for doc in documents:
            doc.metadata['source'] = 'local_codebase'
            doc.metadata['codebase_root'] = str(dir_path)
            doc.metadata['indexed_at'] = datetime.now().isoformat()
            doc.metadata['indexed_via'] = 'codebase_indexer'
            doc.metadata['content_type'] = 'code'

            # Add language info if available
            if 'file_path' in doc.metadata:
                file_ext = Path(doc.metadata['file_path']).suffix.lower()
                for lang, exts in LANGUAGE_EXTENSIONS.items():
                    if file_ext in exts:
                        doc.metadata['language'] = lang
                        break

        # Add to index with chunking
        index = get_index()
        text_splitter = Settings.text_splitter
        nodes = []
        for doc in documents:
            doc_nodes = text_splitter.get_nodes_from_documents([doc])
            nodes.extend(doc_nodes)

        logger.info(f"Created {len(nodes)} chunks from {len(documents)} code files")
        index.insert_nodes(nodes, show_progress=True)

        # Persist
        index.storage_context.persist(persist_dir=str(STORAGE_PATH))

        # Calculate statistics
        language_stats = {}
        for doc in documents:
            lang = doc.metadata.get('language', 'unknown')
            language_stats[lang] = language_stats.get(lang, 0) + 1

        return {
            "success": True,
            "message": f"Successfully indexed codebase at {dir_path}",
            "codebase_root": str(dir_path),
            "files_indexed": len(documents),
            "files_skipped": len(skipped_files),
            "chunks_created": len(nodes),
            "language_breakdown": language_stats,
            "filters_applied": {
                "languages": language_filter or "all",
                "extensions": sorted(allowed_exts),
                "excluded_patterns": sorted(all_excludes)
            }
        }

    except Exception as e:
        logger.error(f"Error indexing local codebase: {e}", exc_info=True)
        return {"error": f"Error indexing codebase: {str(e)}"}


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
