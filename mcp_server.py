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
import json
import re
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
from llama_index.core.node_parser import SentenceSplitter, CodeSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Initialize FastMCP server
mcp = FastMCP("LlamaIndex RAG")

# Get absolute paths (important for MCP server execution)
SCRIPT_DIR = Path(__file__).parent.absolute()
STORAGE_PATH = SCRIPT_DIR / "storage"
CHROMA_PATH = STORAGE_PATH / "chroma_db"

# Global variables to cache
_index = None
_chroma_client = None
_chroma_collection = None
_reranker = None
_bm25_retriever = None
_doc_count_at_bm25_build = 0

# Project Management
DEFAULT_PROJECT = "rag_collection"
_current_project = DEFAULT_PROJECT

# Cache performance metrics
_cache_stats = {
    "index_loads": 0,
    "index_cache_hits": 0,
    "reranker_loads": 0,
    "reranker_cache_hits": 0,
    "chroma_loads": 0,
    "chroma_cache_hits": 0,
    "bm25_builds": 0,
    "bm25_cache_hits": 0
}

# File upload constraints
MAX_FILE_SIZE_MB = 100

# Global Constants
SUPPORTED_CODE_EXTS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
    '.sh', '.bash', '.zsh', '.sql', '.yaml', '.yml', '.toml', '.dockerfile',
    '.makefile', '.vue', '.svelte', '.astro', '.html', '.css'
}

SUPPORTED_DOC_EXTS = {
    '.txt', '.pdf', '.docx', '.doc', '.md', '.epub',
    '.ppt', '.pptx', '.pptm', '.xls', '.xlsx', '.csv', '.json', '.xml',
    '.ipynb', '.hwp', '.mbox', '.rtf'
}

ALLOWED_EXTENSIONS = list(SUPPORTED_CODE_EXTS | SUPPORTED_DOC_EXTS | {'.jpg', '.jpeg', '.png'})
ALLOWED_EXTENSIONS_SET = {ext.lower() for ext in ALLOWED_EXTENSIONS}
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

DEFAULT_EXCLUDES = [
    "node_modules", "__pycache__", ".git", ".svn", ".hg",
    "venv", "env", ".venv", ".env",
    "build", "dist", "target", "out",
    ".idea", ".vscode", ".vs",
    "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll",
    ".DS_Store", "Thumbs.db"
]

CODE_INDICATORS = {
    'files': {'package.json', 'requirements.txt', 'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle', 'Makefile', 'CMakeLists.txt'},
    'dirs': {'.git', '.vscode', '.idea', 'src', 'lib', 'include', 'node_modules', 'venv'}
}

# ---------------------------------------------------------------------------
# File validation helpers
# ---------------------------------------------------------------------------
def _should_ingest_file(path: Path):
    """Return (bool, reason) indicating whether the file is eligible for ingestion."""
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS_SET:
        return False, "unsupported_extension"
    try:
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return False, "file_too_large"
    except OSError as exc:
        logger.warning(f"Unable to stat {path}: {exc}. Skipping.")
        return False, "stat_error"
    return True, ""

# ... (get_index and other helpers)

# Configuration constants
MIN_TOP_K = 1
MAX_TOP_K = 50  # Increased from implicit 20 for flexibility
DEFAULT_TOP_K = 6
RERANK_CANDIDATE_MULTIPLIER = 2
MIN_RERANK_CANDIDATES = 10

# Search strategy heuristics
HYBRID_QUERY_TOKENS = re.compile(r"[A-Z_]{2,}|::|->|\.|\d{3,}")
CODE_SNIPPET_RE = re.compile(r"[{}();=<>*/+-]")
LOW_SCORE_THRESHOLD = 0.2
RERANK_DELTA_THRESHOLD = 0.05
RERANK_MIN_RESULTS = 3
HYDE_TRIGGER_MIN_RESULTS = 1
HYDE_TRIGGER_SCORE = 0.1


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


def get_chroma_client(collection_name: str = None):
    """
    Get ChromaDB client for a specific project.
    ENFORCES STRICT ISOLATION: Each project gets its own 'chroma_db' folder.
    """
    global _chroma_client, _chroma_collection, _cache_stats, _current_project

    target_project = collection_name or _current_project
    
    # Path: ./storage/{project_name}/chroma_db
    project_storage = STORAGE_PATH / target_project
    chroma_path = project_storage / "chroma_db"

    # Check if we can reuse the cached client
    # We reuse ONLY if the target project matches the current loaded one
    if _chroma_client is not None:
        # Chroma client doesn't expose its path easily, so we track it via _current_project
        # If the request is for the same project we currently have loaded:
        if target_project == _current_project:
            _cache_stats["chroma_cache_hits"] += 1
            return _chroma_client, _chroma_collection

    _cache_stats["chroma_loads"] += 1
    logger.info(f"Switching ChromaDB to project: {target_project} at {chroma_path}")

    # Ensure directories exist
    chroma_path.mkdir(parents=True, exist_ok=True)

    # Configure ChromaDB
    chroma_settings = chromadb.Settings(
        allow_reset=True,
        anonymized_telemetry=False
    )

    # Initialize NEW client for this specific project path
    _chroma_client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=chroma_settings
    )
    
    # In this isolated mode, the collection name can just be "default" or the project name.
    # Let's use "default" since the folder separates them, but keeping project name is safer for debug.
    _chroma_collection = _chroma_client.get_or_create_collection(target_project)
    
    # Update current project tracker if we just switched the global client
    if collection_name is None:
        _current_project = target_project

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


def get_index(collection_name: str = None):
    """Load the RAG index from storage (cached)."""
    global _index, _cache_stats, _current_project

    target_project = collection_name or _current_project

    # Smart Cache Check:
    # If we have an index loaded AND it belongs to the requested project
    if _index is not None and _current_project == target_project:
        _cache_stats["index_cache_hits"] += 1
        logger.debug(f"Index: cache hit for {target_project}")
        return _index

    _cache_stats["index_loads"] += 1
    logger.info(f"Index: loading for {target_project}")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables.\n"
            "Please set it in your MCP server configuration."
        )

    # Configure LlamaIndex settings
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

    # Get isolated ChromaDB client
    chroma_client, chroma_collection = get_chroma_client(target_project)

    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Define Persistence Paths
    # Format: ./storage/{project_name}/
    project_storage_path = STORAGE_PATH / target_project
    project_storage_path.mkdir(parents=True, exist_ok=True)
    
    docstore_path = project_storage_path / "docstore.json"
    
    if docstore_path.exists():
        # Load existing index
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(project_storage_path)
        )
        _index = load_index_from_storage(storage_context)
    else:
        # Create new empty index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _index = VectorStoreIndex([], storage_context=storage_context)
        _index.storage_context.persist(persist_dir=str(project_storage_path))

    # Update global state
    _current_project = target_project
    return _index


# --- Project Management Tools ---

def _discover_projects_on_disk() -> List[str]:
    """Return sorted list of available project names in storage."""
    if not STORAGE_PATH.exists():
        return []

    projects = []
    for item in STORAGE_PATH.iterdir():
        if not item.is_dir():
            continue
        if item.name.startswith('.'):
            continue
        if item.name == "chroma_db":
            continue
        projects.append(item.name)
    return sorted(projects)


def _infer_project_from_text(text: str) -> Optional[str]:
    """
    Attempt to infer the best matching project for the given text.
    Returns project name if a confident match is found, otherwise None.
    """
    projects = _discover_projects_on_disk()
    if not projects:
        return None

    text_lower = text.lower()
    best_project = None
    best_score = 0

    for project in projects:
        project_lower = project.lower()
        variants = {
            project_lower,
            project_lower.replace('_', ' '),
            project_lower.replace('-', ' ')
        }

        for variant in variants:
            normalized = variant.strip()
            if not normalized or len(normalized) < 3:
                continue

            score = len(normalized)
            if ' ' in normalized:
                if normalized in text_lower and score > best_score:
                    best_project = project
                    best_score = score
            else:
                pattern = r'\b{}\b'.format(re.escape(normalized))
                if re.search(pattern, text_lower) and score > best_score:
                    best_project = project
                    best_score = score

    return best_project


def _require_project_selection(
    action: str,
    project: Optional[str],
    suggested: Optional[str] = None
) -> tuple:
    """
    Ensure an ingestion action has an explicit project target.

    Returns (error_dict, resolved_project_name).
    """
    available_projects = _discover_projects_on_disk()

    if not project:
        message = (
            f"{action} requires an explicit project. "
            "Ask the user whether to create a new project or choose an existing one, "
            "then call this tool again with the `project` parameter."
        )
        return (
            {
                "error": "project_confirmation_required",
                "message": message,
                "current_project": _current_project,
                "available_projects": available_projects,
                "suggested_project": suggested,
                "next_steps": [
                    "If new workspace: create_project('<name>')",
                    "If existing workspace: switch_project('<name>')",
                    f"Retry: {action}(..., project='<name>')"
                ]
            },
            None
        )

    project_name = project.strip()
    if not project_name:
        return (
            {
                "error": "invalid_project_name",
                "message": "Project name cannot be empty. Use alphanumeric characters, '-' or '_'."
            },
            None
        )

    if any(not (c.isalnum() or c in ('_', '-')) for c in project_name):
        return (
            {
                "error": "invalid_project_name",
                "message": "Project names may only contain letters, numbers, '-' and '_'.",
                "received": project_name
            },
            None
        )

    if project_name not in available_projects:
        return (
            {
                "error": "project_not_found",
                "message": (
                    f"Project '{project_name}' not found. "
                    "Create it first with create_project(...) before indexing."
                ),
                "available_projects": available_projects
            },
            None
        )

    if project_name != _current_project:
        get_index(project_name)

    return None, project_name


def _load_project_manifest(project: str) -> dict:
    """Load or initialize the change-tracking manifest for a project."""
    manifest_path = STORAGE_PATH / project / "ingest_manifest.json"
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                data = json.load(manifest_file)
                if isinstance(data, dict):
                    data.setdefault("roots", {})
                    return data
        except Exception as exc:
            logger.warning(f"Failed to load manifest for {project}: {exc}. Reinitializing.")

    return {"roots": {}, "updated_at": datetime.now().isoformat()}


def _save_project_manifest(project: str, manifest: dict) -> None:
    """Persist the project manifest to disk."""
    manifest_path = STORAGE_PATH / project / "ingest_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now().isoformat()
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)


def _analyze_query_signals(question: str) -> dict:
    """Extract lightweight heuristics from the query text."""
    tokens = question.strip().split()
    has_hybrid_tokens = bool(HYBRID_QUERY_TOKENS.search(question))
    has_code_chars = bool(CODE_SNIPPET_RE.search(question))
    avg_token_len = sum(len(t) for t in tokens) / len(tokens) if tokens else 0
    uppercase_ratio = sum(1 for t in tokens if t.isupper() and len(t) > 1) / len(tokens) if tokens else 0
    digit_ratio = sum(1 for t in tokens if any(ch.isdigit() for ch in t)) / len(tokens) if tokens else 0
    return {
        "tokens": tokens,
        "has_hybrid_tokens": has_hybrid_tokens,
        "has_code_chars": has_code_chars,
        "avg_token_len": avg_token_len,
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
    }


def _select_search_mode(question: str, requested_mode: str) -> str:
    """
    Choose an appropriate search mode.

    If the caller explicitly requests keyword/hybrid we respect that.
    Otherwise we use heuristics to flip between semantic and hybrid.
    """
    requested_mode = (requested_mode or "semantic").lower()
    if requested_mode in ("keyword", "hybrid"):
        return requested_mode

    signals = _analyze_query_signals(question)

    if signals["has_hybrid_tokens"] or signals["has_code_chars"]:
        logger.debug("Query signals indicate technical content; using hybrid search.")
        return "hybrid"

    if signals["uppercase_ratio"] > 0.3 or signals["digit_ratio"] > 0.4:
        logger.debug("Query heavy in uppercase/digits; switching to hybrid search.")
        return "hybrid"

    return "semantic"


def _should_apply_reranker(nodes, requested: bool) -> bool:
    """Decide whether running the reranker is worth it."""
    if not requested or not nodes:
        return False

    if len(nodes) < RERANK_MIN_RESULTS:
        return False

    scores = [node.score for node in nodes if node.score is not None]
    if len(scores) < 2:
        return True

    top_score = scores[0]
    second_score = scores[1]
    if top_score is None or second_score is None:
        return True

    if abs(top_score - second_score) < RERANK_DELTA_THRESHOLD:
        return True

    logger.debug("Skipping reranker (clear winner among candidates).")
    return False


def _should_trigger_hyde(nodes) -> bool:
    """Determine whether to fall back to HyDE augmentation."""
    if not nodes:
        return True

    scores = [node.score for node in nodes if node.score is not None]
    if not scores:
        return True

    if len(scores) <= HYDE_TRIGGER_MIN_RESULTS and scores[0] < HYDE_TRIGGER_SCORE:
        return True

    if max(scores) < LOW_SCORE_THRESHOLD:
        return True

    return False


def _retrieve_nodes_for_index(
    index,
    question: str,
    search_query: str,
    similarity_top_k: int,
    search_mode: str,
    use_rerank: bool,
) -> tuple[List, bool]:
    """Run retrieval against a specific index."""
    # Determine candidate count (fetch more if reranking)
    initial_top_k = similarity_top_k
    if use_rerank:
        initial_top_k = max(
            similarity_top_k * RERANK_CANDIDATE_MULTIPLIER,
            MIN_RERANK_CANDIDATES
        )

    # Select Retriever
    if search_mode == "hybrid":
        vector_retriever = index.as_retriever(similarity_top_k=initial_top_k)
        bm25_retriever = get_bm25_retriever(index)

        if bm25_retriever:
            bm25_retriever.similarity_top_k = initial_top_k
            retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=initial_top_k,
                num_queries=1,  # Simple RRF
                mode="reciprocal_rerank",
                use_async=False,
                verbose=True
            )
        else:
            logger.warning("BM25 retriever creation failed, falling back to semantic")
            retriever = vector_retriever

    elif search_mode == "keyword":
        bm25_retriever = get_bm25_retriever(index)
        if bm25_retriever:
            bm25_retriever.similarity_top_k = initial_top_k
            retriever = bm25_retriever
        else:
            logger.warning("BM25 retriever creation failed, falling back to semantic")
            retriever = index.as_retriever(similarity_top_k=initial_top_k)

    else:  # semantic (default)
        retriever = index.as_retriever(similarity_top_k=initial_top_k)

    # Retrieve
    nodes = retriever.retrieve(search_query)

    reranker = get_reranker()
    apply_rerank = _should_apply_reranker(nodes, use_rerank) and reranker is not None
    rerank_used = False

    if apply_rerank:
        nodes = reranker.postprocess_nodes(nodes, query_str=question)
        nodes = nodes[:similarity_top_k]
        rerank_used = True
    else:
        nodes = nodes[:similarity_top_k]

    return nodes, rerank_used

@mcp.tool()
def create_project(name: str) -> dict:
    """Create a new isolated project (workspace)."""
    try:
        safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-'))
        if not safe_name:
            return {"error": "Invalid project name"}
            
        # Check if directory exists
        project_path = STORAGE_PATH / safe_name
        if project_path.exists():
             return {"error": f"Project '{safe_name}' already exists at {project_path}"}
        
        # Bootstrap the project by initializing its storage
        # This creates ./storage/{name}/chroma_db
        logger.info(f"Creating project workspace: {safe_name}")
        get_chroma_client(safe_name)
        
        return {"success": True, "message": f"Created project '{safe_name}'. Use switch_project('{safe_name}') to activate."}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def list_projects() -> dict:
    """List all available projects (directories in storage)."""
    try:
        projects = _discover_projects_on_disk()
        return {
            "projects": sorted(projects),
            "current_project": _current_project,
            "count": len(projects)
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def switch_project(name: str) -> dict:
    """Switch the active workspace."""
    global _current_project, _index, _bm25_retriever, _chroma_client, _chroma_collection
    try:
        project_path = STORAGE_PATH / name
        if not project_path.exists() or not project_path.is_dir():
            return {"error": f"Project '{name}' not found. Use list_projects() to see available ones."}
            
        logger.info(f"Switching to project: {name}")
        _current_project = name
        
        # Reset caches to force reload
        _index = None
        _bm25_retriever = None
        # Force reload of chroma client for this project
        _chroma_client = None 
        _chroma_collection = None
        
        # Pre-load the new index
        get_index(name)
        
        return {"success": True, "message": f"Switched to project '{name}'"}
    except Exception as e:
        return {"error": str(e)}


# Language mapping for CodeSplitter
CODE_LANG_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cs': 'c_sharp',
    '.go': 'go',
    '.rs': 'rust',
    '.rb': 'ruby',
    '.html': 'html',
    '.css': 'css',
    '.php': 'php',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.json': 'json'
}

def get_splitter_for_file(file_path: str):
    """Return optimal splitter based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext in CODE_LANG_MAP:
        try:
            return CodeSplitter(
                language=CODE_LANG_MAP[ext],
                chunk_lines=40,
                chunk_lines_overlap=15,
                max_chars=1500,
            )
        except Exception as e:
            logger.warning(f"Failed to init CodeSplitter for {ext}: {e}. Fallback to text.")
            
    return SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )


def get_bm25_retriever(index):
    """Get or create BM25 retriever (cached)."""
    global _bm25_retriever, _doc_count_at_bm25_build, _cache_stats, _chroma_collection

    # Try to get nodes from docstore
    nodes = list(index.docstore.docs.values())
    current_doc_count = len(nodes)

    # Fallback: If docstore is empty, try retrieving from ChromaDB directly
    if current_doc_count == 0:
        logger.warning("Docstore is empty. Attempting to load nodes from ChromaDB for BM25...")
        try:
            # Ensure we have the collection
            if _chroma_collection is None:
                get_chroma_client()
            
            # Fetch all documents from Chroma
            result = _chroma_collection.get()
            
            if result and result['documents']:
                from llama_index.core.schema import TextNode
                
                chroma_docs = result['documents']
                chroma_metadatas = result['metadatas']
                chroma_ids = result['ids']
                
                nodes = []
                for i, text in enumerate(chroma_docs):
                    if text: # Skip empty text
                        node = TextNode(
                            text=text,
                            id_=chroma_ids[i],
                            metadata=chroma_metadatas[i] if chroma_metadatas else {}
                        )
                        nodes.append(node)
                
                current_doc_count = len(nodes)
                logger.info(f"Recovered {current_doc_count} nodes from ChromaDB")
        except Exception as e:
            logger.error(f"Failed to load from ChromaDB: {e}")

    if current_doc_count == 0:
        logger.warning("No documents available for BM25. Hybrid/Keyword search will fail.")
        return None

    if _bm25_retriever is not None and _doc_count_at_bm25_build == current_doc_count:
        _cache_stats["bm25_cache_hits"] += 1
        logger.debug("BM25 Retriever: cache hit")
        return _bm25_retriever

    _cache_stats["bm25_builds"] += 1
    logger.info(f"Building BM25 Retriever with {current_doc_count} nodes...")

    start_time = time.time()
    
    _bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=DEFAULT_TOP_K,
        stemmer=None,
        language="english"
    )
    _doc_count_at_bm25_build = current_doc_count

    logger.info(f"Built BM25 Retriever in {time.time() - start_time:.3f}s")
    return _bm25_retriever


def _generate_hyde_query(question: str) -> str:
    """Generate a hypothetical answer for the question using OpenAI."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.warning("OPENAI_API_KEY missing, skipping HyDE")
        return question

    try:
        # Use a lightweight model for HyDE if possible, but standard is fine
        llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_key)
        prompt = (
            "Please write a brief passage to answer the question.\n"
            "Question: {question}\n"
            "Passage:"
        )
        response = llm.complete(prompt.format(question=question))
        logger.debug(f"HyDE generated: {str(response)[:100]}...")
        return str(response)
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        return question


def _retrieve_nodes(
    question: str,
    similarity_top_k: int,
    search_mode: str,
    use_rerank: bool,
    use_hyde: bool
):
    """Common retrieval logic for RAG tools."""
    # Validation
    question = validate_query(question)
    similarity_top_k = validate_top_k(similarity_top_k)

    requested_mode = (search_mode or "semantic").lower()
    effective_mode = _select_search_mode(question, requested_mode)
    if effective_mode != requested_mode:
        logger.info(f"Adjusted search mode from '{requested_mode}' to '{effective_mode}' for query: {question[:80]}")

    inferred_project = _infer_project_from_text(question)

    candidate_projects: List[str] = []
    if inferred_project:
        candidate_projects.append(inferred_project)
    if _current_project not in candidate_projects:
        candidate_projects.append(_current_project)

    for project in _discover_projects_on_disk():
        if project not in candidate_projects:
            candidate_projects.append(project)

    original_project = _current_project
    best_project = None
    best_nodes = []
    best_score = float("-inf")
    fallback_nodes = []
    fallback_project = None
    fallback_query = question
    fallback_rerank_used = False
    best_query = question
    best_rerank_used = False

    for project in candidate_projects:
        index = get_index(project)
        hyde_attempted = bool(use_hyde)
        search_query = _generate_hyde_query(question) if hyde_attempted else question
        project_nodes = []
        project_query_used = search_query
        project_rerank_used = False

        for attempt in range(2):
            nodes, rerank_used = _retrieve_nodes_for_index(
                index=index,
                question=question,
                search_query=search_query,
                similarity_top_k=similarity_top_k,
                search_mode=effective_mode,
                use_rerank=use_rerank,
            )

            if nodes:
                project_nodes = nodes
                project_rerank_used = rerank_used
                if hyde_attempted or not _should_trigger_hyde(nodes):
                    break

            if hyde_attempted:
                break

            logger.info(f"HyDE fallback triggered for project '{project}' on query: {question[:80]}")
            hyde_attempted = True
            search_query = _generate_hyde_query(question)
            project_query_used = search_query
        else:
            project_nodes = nodes
            project_rerank_used = rerank_used

        fallback_nodes = project_nodes
        fallback_project = project
        fallback_query = project_query_used
        fallback_rerank_used = project_rerank_used

        if project_nodes:
            top_score = project_nodes[0].score if project_nodes[0].score is not None else 0.0
        else:
            top_score = float("-inf")

        if inferred_project and project == inferred_project and project_nodes:
            best_project = project
            best_nodes = project_nodes
            best_score = top_score
            best_query = project_query_used
            best_rerank_used = project_rerank_used
            break

        if project_nodes and top_score > best_score:
            best_project = project
            best_nodes = project_nodes
            best_score = top_score
            best_query = project_query_used
            best_rerank_used = project_rerank_used

    if best_project:
        if best_project != _current_project:
            get_index(best_project)
        if best_project != original_project:
            logger.info(f"Auto-selected project '{best_project}' for query: {question[:80]}")
        return best_nodes, best_query, effective_mode, best_rerank_used

    # No strong project match found; revert to original project context.
    if original_project != _current_project:
        get_index(original_project)

    return fallback_nodes, fallback_query, effective_mode, fallback_rerank_used


@mcp.tool()
@log_performance
def query_rag(
    question: str, 
    similarity_top_k: int = DEFAULT_TOP_K, 
    search_mode: str = "semantic",
    use_rerank: bool = True,
    use_hyde: bool = False
) -> str:
    """
    Retrieve documents from RAG.
    
    Args:
        question: Query string.
        similarity_top_k: Documents to return (default: 6).
        search_mode: 'semantic' (default, concepts), 'hybrid' (technical/ids), 'keyword' (exact).
        use_rerank: Enable Cohere reranking (default: True).
        use_hyde: Enable HyDE for ambiguous queries (default: False).
    """
    try:
        nodes, used_query, used_mode, rerank_used = _retrieve_nodes(
            question,
            similarity_top_k,
            search_mode,
            use_rerank,
            use_hyde
        )

        logger.debug(f"Retrieved {len(nodes)} nodes")

        if not nodes:
            logger.info("No relevant documents found")
            return "No relevant documents found in the knowledge base."

        # Format the retrieved chunks
        rerank_label = " + Cohere Rerank" if rerank_used else ""
        hyde_label = " + HyDE" if used_query != question else ""
        mode_label = f" ({used_mode}{hyde_label}{rerank_label})"
        
        result = f"Found {len(nodes)} relevant document(s){mode_label}:\n\n"

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

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return f"Invalid input: {str(e)}"
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}", exc_info=True)
        return f"Error retrieving documents: {str(e)}"


@mcp.tool()
@log_performance
def query_rag_with_sources(
    question: str, 
    similarity_top_k: int = DEFAULT_TOP_K, 
    search_mode: str = "semantic",
    use_rerank: bool = True,
    use_hyde: bool = False
) -> dict:
    """
    Retrieve documents with metadata (structured).
    
    Args:
        question: Query string.
        similarity_top_k: Documents to return.
        search_mode: 'semantic', 'hybrid', 'keyword'.
        use_rerank: Enable reranking (default: True).
        use_hyde: Enable HyDE (default: False).
    """
    try:
        nodes, used_query, used_mode, rerank_used = _retrieve_nodes(
            question,
            similarity_top_k,
            search_mode,
            use_rerank,
            use_hyde
        )

        if not nodes:
            logger.info("No relevant documents found")
            return {
                "sources": [],
                "message": "No relevant documents found"
            }

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
            "search_mode": used_mode,
            "reranked": rerank_used,
            "used_hyde": used_query != question,
            "generated_query": used_query if use_hyde else None
        }
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {"error": f"Invalid input: {str(e)}"}
    except Exception as e:
        logger.error(f"Error querying RAG: {e}", exc_info=True)
        return {"error": f"Error querying RAG: {str(e)}"}


@mcp.tool()
def get_cache_stats() -> dict:
    """Get cache performance metrics."""
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
    """Get RAG index statistics."""
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
    Add raw text to index.
    
    Args:
        text: Content to add.
        metadata: Optional dict.
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
def add_documents_from_directory(directory_path: str, project: Optional[str] = None) -> dict:
    """
    Index all docs in directory (recursive).
    
    Args:
        directory_path: Absolute path.
        project: Target project/workspace to store ingested data.
    """
    try:
        project_error, resolved_project = _require_project_selection(
            "add_documents_from_directory",
            project,
            suggested=Path(directory_path).name if directory_path else None
        )
        if project_error:
            return project_error

        dir_path = Path(directory_path)

        # Validate directory
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory_path}"}

        if not dir_path.is_dir():
            return {"error": f"Path is not a directory: {directory_path}"}

        files_to_ingest = []
        skipped_unsupported = []
        skipped_oversize = []
        skipped_other = []

        for path in dir_path.rglob('*'):
            if not path.is_file():
                continue

            allowed, reason = _should_ingest_file(path)
            if not allowed:
                target_list = {
                    "unsupported_extension": skipped_unsupported,
                    "file_too_large": skipped_oversize
                }.get(reason, skipped_other)
                target_list.append(str(path))
                continue

            files_to_ingest.append(path)

        if not files_to_ingest:
            message = "No documents found in directory"
            if skipped_unsupported or skipped_oversize or skipped_other:
                return {
                    "error": message,
                    "skipped_unsupported": len(skipped_unsupported),
                    "skipped_oversize": len(skipped_oversize),
                    "skipped_other": len(skipped_other),
                    "max_file_size_mb": MAX_FILE_SIZE_MB,
                    "allowed_extensions": sorted(ALLOWED_EXTENSIONS_SET),
                }
            return {"error": message}

        # Load documents
        documents = SimpleDirectoryReader(
            input_files=[str(p) for p in files_to_ingest],
            errors='ignore'
        ).load_data()

        if not documents:
            return {"error": "No documents could be loaded from directory"}

        # Add metadata
        for doc in documents:
            doc.metadata['ingested_via'] = 'mcp_tool'
            doc.metadata['ingested_at'] = datetime.now().isoformat()
            doc.metadata['source_directory'] = str(dir_path)

        # Add to index with SMART chunking
        index = get_index(resolved_project)
        nodes = []
        
        for doc in documents:
            # Determine splitter per document based on file path
            file_path = doc.metadata.get('file_path', '')
            if not file_path and 'source_directory' in doc.metadata: 
                # fallback if file_path missing but we have dir context (less accurate but safer)
                pass 
            
            splitter = get_splitter_for_file(file_path)
            doc_nodes = splitter.get_nodes_from_documents([doc])
            nodes.extend(doc_nodes)
            
        index.insert_nodes(nodes)

        # Persist
        project_storage_path = STORAGE_PATH / _current_project
        index.storage_context.persist(persist_dir=str(project_storage_path))

        return {
            "success": True,
            "message": f"Successfully ingested {len(documents)} documents ({len(nodes)} chunks)",
            "document_count": len(documents),
            "chunks_created": len(nodes),
            "directory": str(dir_path),
            "skipped_unsupported": len(skipped_unsupported),
            "skipped_oversize": len(skipped_oversize),
            "skipped_other": len(skipped_other),
            "max_file_size_mb": MAX_FILE_SIZE_MB,
        }
    except Exception as e:
        return {"error": f"Error ingesting documents: {str(e)}"}


@mcp.tool()
def crawl_website(url: str, max_depth: int = 1, max_pages: int = 10) -> dict:
    """
    Crawl website to RAG index (requires FIRECRAWL_API_KEY).
    
    Args:
        url: Starting URL.
        max_depth: 0=single page, 1=direct links, 2+=deep.
        max_pages: Limit pages (e.g., 1 for single, 50+ for docs).
    """
    try:
        from firecrawl import FirecrawlApp

        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            return {
                "error": "FIRECRAWL_API_KEY not found",
                "suggestion": "Set FIRECRAWL_API_KEY environment variable"
            }

        if not url.startswith("http"):
            return {"error": "Invalid URL. Must start with http:// or https://"}

        logger.info(f"Crawling {url} (depth={max_depth}, pages={max_pages})")

        app = FirecrawlApp(api_key=api_key)

        logger.info("Initiating crawl...")
        
        # Firecrawl v2 SDK 'crawl' is synchronous and takes kwargs
        try:
            crawl_result = app.crawl(
                url,
                limit=max_pages,
                max_discovery_depth=max_depth,
                scrape_options={'formats': ['markdown']}
            )
        except Exception as e:
            return {"error": f"Firecrawl API error: {str(e)}"}
        
        # Extract pages from result
        # Result is typically a CrawlJob object with a .data attribute (list of items)
        # or a dict depending on exact version/response
        pages = []
        if hasattr(crawl_result, 'data'):
            pages = crawl_result.data
        elif isinstance(crawl_result, dict) and 'data' in crawl_result:
            pages = crawl_result['data']
        else:
            # Maybe it's the list directly?
            # Or maybe we need to fetch status if it didn't wait? 
            # help(crawl) said "wait for it to complete"
            logger.warning(f"Unexpected crawl result type: {type(crawl_result)}")
            if isinstance(crawl_result, list):
                pages = crawl_result
            else:
                return {"error": f"Unexpected response structure: {str(crawl_result)[:100]}"}

        if not pages:
            return {"error": "No pages found"}

        # Convert to Documents
        documents = []
        for page in pages:
            # Handle object or dict access
            if isinstance(page, dict):
                text = page.get('markdown') or page.get('text')
                metadata = page.get('metadata', {})
                source = page.get('source', url)
            else:
                # Assume object
                text = getattr(page, 'markdown', None) or getattr(page, 'text', None)
                metadata = getattr(page, 'metadata', {})
                source = getattr(page, 'source', url)

            if not text:
                continue
                
            # Safely convert metadata to dict
            metadata_dict = {}
            if isinstance(metadata, dict):
                metadata_dict = metadata
            elif hasattr(metadata, 'model_dump'): # Pydantic v2
                metadata_dict = metadata.model_dump()
            elif hasattr(metadata, 'dict'): # Pydantic v1
                metadata_dict = metadata.dict()
            elif hasattr(metadata, '__dict__'):
                metadata_dict = metadata.__dict__
                
            # Ensure metadata is flat/clean for Chroma
            clean_metadata = {}
            if metadata_dict:
                for k, v in metadata_dict.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_metadata[k] = v
                    else:
                        clean_metadata[k] = str(v)
            
            clean_metadata['source'] = source
            clean_metadata['crawled_at'] = datetime.now().isoformat()
            clean_metadata['crawled_via'] = 'firecrawl'
            
            doc = Document(text=text, metadata=clean_metadata)
            documents.append(doc)

        if not documents:
            return {"error": "No valid content extracted from pages"}

        # Index
        index = get_index()
        text_splitter = Settings.text_splitter
        nodes = []
        for doc in documents:
            nodes.extend(text_splitter.get_nodes_from_documents([doc]))
            
        index.insert_nodes(nodes)
        index.storage_context.persist(persist_dir=str(STORAGE_PATH))

        return {
            "success": True,
            "message": f"Successfully crawled and indexed {len(documents)} pages from {url}",
            "pages_crawled": len(documents),
            "chunks_created": len(nodes)
        }

    except ImportError:
        return {"error": "firecrawl-py not installed"}
    except Exception as e:
        logger.error(f"Error crawling website: {e}", exc_info=True)
        return {"error": f"Error crawling website: {str(e)}"}


@mcp.tool()
def list_indexed_documents() -> dict:
    """List sample of indexed documents with metadata."""
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
def iterative_search(
    question: str, 
    initial_top_k: int = 3, 
    detailed_top_k: int = 10,
    search_mode: str = "semantic",
    use_rerank: bool = True,
    use_hyde: bool = False
) -> dict:
    """
    Two-phase search: quick initial results + suggestions.
    
    Args:
        question: Query string.
        initial_top_k: Initial results (default: 3).
        detailed_top_k: Potential results for follow-up (default: 10).
        search_mode: 'semantic', 'hybrid', 'keyword'.
        use_rerank: Enable reranking.
        use_hyde: Enable HyDE.
    """
    try:
        # Validate inputs
        question = validate_query(question)
        initial_top_k = validate_top_k(initial_top_k)
        detailed_top_k = validate_top_k(detailed_top_k)

        logger.debug(f"Iterative search: '{question[:100]}...', initial={initial_top_k}")

        # Use shared retrieval logic
        # Note: _retrieve_nodes handles retrieval + reranking + cutting to top_k
        initial_nodes, used_query, initial_mode, initial_rerank = _retrieve_nodes(
            question,
            initial_top_k,
            search_mode,
            use_rerank,
            use_hyde
        )

        if not initial_nodes:
            return {
                "initial_results": [],
                "message": "No relevant documents found",
                "suggestion": "Try rephrasing the question or using different keywords"
            }

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

            if node.node.metadata and 'file_name' in node.node.metadata:
                topics_found.add(node.node.metadata['file_name'])

        # Analyze score distribution
        scores = [r['score'] for r in initial_results if r['score'] is not None]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Generate suggestions
        suggestions = []
        
        # Add mode-specific suggestions
        if initial_mode == "semantic" and avg_score < 0.7:
            suggestions.append("Results have low relevance. Consider search_mode='hybrid' if searching for specific terms.")
        
        if avg_score < 0.7 and len(scores) > 0:
            suggestions.append(f"Relevance is moderate (avg: {avg_score:.2f}). Consider refining the query.")

        suggestions.append(
            f"Found {len(initial_results)} initial results. "
            f"You can retrieve more using query_rag(similarity_top_k={detailed_top_k}, search_mode='{initial_mode}')"
        )

        return {
            "initial_results": initial_results,
            "total_initial": len(initial_results),
            "search_mode": initial_mode,
            "reranked": initial_rerank,
            "used_hyde": used_query != question,
            "suggestions": suggestions,
            "next_steps": [
                f"If satisfied: Answer the question",
                f"If need more context: query_rag(question='{question}', similarity_top_k={detailed_top_k}, search_mode='{initial_mode}')",
                "If searching for exact term: Retry with search_mode='keyword'"
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
    Index GitHub repo (requires GITHUB_TOKEN).
    
    Args:
        owner: User/Org.
        repo: Repo name.
        branch: Branch (default: main).
        filter_dirs: Directories to include.
        filter_extensions: Extensions to include.
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
def index_hybrid_folder(
    path: str,
    exclude_patterns: Optional[List[str]] = None,
    project: Optional[str] = None
) -> dict:
    """
    Index a folder containing mixed content (Code + Documents).
    Smartly filters out noise (node_modules, .git) while including both
    source code (.py, .ts) and documentation (.md, .pdf).
    
    Args:
        path: Directory path.
        exclude_patterns: Additional glob patterns to ignore.
        project: Target project/workspace to store ingested data.
    """
    try:
        dir_path = Path(path).resolve()
        
        if not dir_path.exists():
            return {"error": f"Path not found: {path}"}
        if not dir_path.is_dir():
            return {"error": f"Path is not a directory: {path}"}

        project_error, resolved_project = _require_project_selection(
            "index_hybrid_folder",
            project,
            suggested=dir_path.name
        )
        if project_error:
            return project_error

        logger.info(f"Hybrid indexing: {dir_path} -> project {resolved_project}")

        # Prepare Excludes
        user_excludes = exclude_patterns or []
        all_excludes = set(DEFAULT_EXCLUDES + user_excludes)
        
        # Allowed Extensions (Code + Docs)
        valid_exts = SUPPORTED_CODE_EXTS | SUPPORTED_DOC_EXTS

        files_to_index = []
        skipped_count = 0

        for item in dir_path.rglob('*'):
            # Skip hidden files/dirs
            if any(part.startswith('.') for part in item.parts):
                continue
            
            if not item.is_file():
                continue

            # Check exclusion patterns
            relative_path = item.relative_to(dir_path)
            excluded = False
            for pattern in all_excludes:
                if pattern in str(relative_path):
                    excluded = True
                    break
                if '*' in pattern and relative_path.match(pattern):
                    excluded = True
                    break
            
            if excluded:
                skipped_count += 1
                continue

            # Check extension
            if item.suffix.lower() in valid_exts:
                files_to_index.append(item)
            else:
                skipped_count += 1

        if not files_to_index:
            return {"error": "No valid files found to index"}

        logger.info(f"Found {len(files_to_index)} files (skipped {skipped_count})")

        # Load documents
        documents = SimpleDirectoryReader(
            input_files=[str(f) for f in files_to_index],
            errors='ignore'
        ).load_data()

        # Add metadata
        for doc in documents:
            doc.metadata['source'] = 'hybrid_folder'
            doc.metadata['codebase_root'] = str(dir_path)
            doc.metadata['ingested_at'] = datetime.now().isoformat()
            
            # Tag content type
            ext = Path(doc.metadata.get('file_path', '')).suffix.lower()
            if ext in SUPPORTED_CODE_EXTS:
                doc.metadata['content_type'] = 'code'
            elif ext in SUPPORTED_DOC_EXTS:
                doc.metadata['content_type'] = 'document'

        # SMART Chunking
        index = get_index(resolved_project)
        nodes = []
        for doc in documents:
            file_path = doc.metadata.get('file_path', '')
            splitter = get_splitter_for_file(file_path)
            doc_nodes = splitter.get_nodes_from_documents([doc])
            nodes.extend(doc_nodes)

        logger.info(f"Created {len(nodes)} chunks from {len(documents)} files")
        index.insert_nodes(nodes, show_progress=True)
        
        index.storage_context.persist(persist_dir=str(STORAGE_PATH / _current_project))

        return {
            "success": True,
            "message": f"Indexed {len(documents)} mixed files ({len(nodes)} chunks)",
            "files_indexed": len(documents),
            "chunks_created": len(nodes),
            "root": str(dir_path)
        }

    except Exception as e:
        logger.error(f"Error in hybrid indexing: {e}", exc_info=True)
        return {"error": f"Hybrid indexing failed: {str(e)}"}


@mcp.tool()
def index_modified_files(
    path: str,
    project: Optional[str] = None
) -> dict:
    """
    Index only new or modified files under a directory for a given project.
    Tracks file mtimes in a per-project manifest to avoid reprocessing unchanged files.
    
    Args:
        path: Directory path to monitor.
        project: Target project/workspace to store ingested data.
    """
    try:
        dir_path = Path(path).resolve()

        if not dir_path.exists():
            return {"error": f"Path not found: {path}"}
        if not dir_path.is_dir():
            return {"error": f"Path is not a directory: {path}"}

        project_error, resolved_project = _require_project_selection(
            "index_modified_files",
            project,
            suggested=dir_path.name
        )
        if project_error:
            return project_error

        index = get_index(resolved_project)
        existing_doc_count = len(index.docstore.docs)

        manifest = _load_project_manifest(resolved_project)
        roots = manifest.setdefault("roots", {})
        root_key = str(dir_path)
        root_entry = roots.get(root_key, {})

        previous_files_raw = root_entry.get("files", {})
        previous_files: dict[str, int] = {}
        for rel_path, info in previous_files_raw.items():
            if isinstance(info, dict):
                prev_mtime = info.get("mtime_ns")
            else:
                prev_mtime = info
            try:
                previous_files[rel_path] = int(prev_mtime)
            except (TypeError, ValueError):
                previous_files[rel_path] = 0

        changed_paths: List[Path] = []
        change_map: dict[str, dict[str, str]] = {}
        current_files: dict[str, int] = {}
        new_files: List[str] = []
        modified_files: List[str] = []
        skipped_unsupported: List[str] = []
        skipped_oversize: List[str] = []
        skipped_other: List[str] = []

        for item in dir_path.rglob('*'):
            if not item.is_file():
                continue

            allowed, reason = _should_ingest_file(item)
            if not allowed:
                target_list = {
                    "unsupported_extension": skipped_unsupported,
                    "file_too_large": skipped_oversize
                }.get(reason, skipped_other)
                target_list.append(str(item))
                continue

            rel_path = str(item.relative_to(dir_path))
            try:
                mtime_ns = item.stat().st_mtime_ns
            except OSError as exc:
                logger.warning(f"Failed to stat {item}: {exc}. Skipping.")
                skipped_other.append(str(item))
                continue

            current_files[rel_path] = mtime_ns
            prev_mtime = previous_files.get(rel_path)

            if prev_mtime is None:
                changed_paths.append(item)
                new_files.append(rel_path)
                change_map[str(item.resolve())] = {
                    "change": "new",
                    "relative_path": rel_path
                }
            elif mtime_ns > prev_mtime:
                changed_paths.append(item)
                modified_files.append(rel_path)
                change_map[str(item.resolve())] = {
                    "change": "modified",
                    "relative_path": rel_path
                }

        removed_files = sorted(set(previous_files.keys()) - set(current_files.keys()))

        baseline_run = len(previous_files) == 0

        if not changed_paths:
            # Persist manifest updates (removals) even if nothing was indexed
            roots[root_key] = {
                "files": {
                    rel: {"mtime_ns": current_files[rel]}
                    for rel in sorted(current_files.keys())
                },
                "last_scan": datetime.now().isoformat()
            }
            _save_project_manifest(resolved_project, manifest)

            message = "No new or modified files detected."
            if removed_files:
                message += f" {len(removed_files)} previously indexed file(s) were removed."
            elif baseline_run and existing_doc_count > 0:
                message = (
                    "Baseline recorded for incremental indexing. "
                    "No files were re-indexed because the project already contained data."
                )
            return {
                "success": True,
                "message": message,
                "project": resolved_project,
                "directory": str(dir_path),
                "removed_files": removed_files,
                "skipped_unsupported": len(skipped_unsupported),
                "skipped_oversize": len(skipped_oversize),
                "skipped_other": len(skipped_other),
            }

        if baseline_run and existing_doc_count > 0:
            roots[root_key] = {
                "files": {
                    rel: {"mtime_ns": current_files[rel]}
                    for rel in sorted(current_files.keys())
                },
                "last_scan": datetime.now().isoformat()
            }
            _save_project_manifest(resolved_project, manifest)
            return {
                "success": True,
                "message": (
                    "Baseline recorded for incremental indexing. "
                    "No files were re-indexed because the project already contained data."
                ),
                "project": resolved_project,
                "directory": str(dir_path),
                "tracked_files": sorted(current_files.keys()),
                "removed_files": removed_files,
                "skipped_unsupported": len(skipped_unsupported),
                "skipped_oversize": len(skipped_oversize),
                "skipped_other": len(skipped_other),
            }

        # Load documents only for changed files
        documents = SimpleDirectoryReader(
            input_files=[str(p) for p in changed_paths],
            errors='ignore'
        ).load_data()

        if not documents:
            return {"error": "Failed to load changed files for indexing"}

        nodes = []
        chunk_count = 0

        for doc in documents:
            file_path_str = doc.metadata.get('file_path')
            change_meta = None
            if file_path_str:
                change_meta = change_map.get(str(Path(file_path_str).resolve()))

            rel_path = change_meta['relative_path'] if change_meta else None
            change_label = change_meta['change'] if change_meta else "modified"

            doc.metadata['sync_root'] = str(dir_path)
            doc.metadata['project'] = resolved_project
            if rel_path:
                doc.metadata['relative_path'] = rel_path
            doc.metadata['change_type'] = change_label
            doc.metadata['ingested_via'] = 'index_modified_files'
            doc.metadata['ingested_at'] = datetime.now().isoformat()

            splitter = get_splitter_for_file(file_path_str or "")
            doc_nodes = splitter.get_nodes_from_documents([doc])
            nodes.extend(doc_nodes)
            chunk_count += len(doc_nodes)

        index.insert_nodes(nodes)
        project_storage_path = STORAGE_PATH / _current_project
        index.storage_context.persist(persist_dir=str(project_storage_path))

        # Update manifest with latest mtimes (and drop removed files)
        roots[root_key] = {
            "files": {
                rel: {"mtime_ns": current_files[rel]}
                for rel in sorted(current_files.keys())
            },
            "last_scan": datetime.now().isoformat()
        }
        _save_project_manifest(resolved_project, manifest)

        return {
            "success": True,
            "message": f"Indexed {len(changed_paths)} changed file(s) ({chunk_count} chunks)",
            "project": resolved_project,
            "directory": str(dir_path),
            "files_indexed": len(changed_paths),
            "new_files": new_files,
            "modified_files": modified_files,
            "removed_files": removed_files,
            "chunks_created": chunk_count,
            "skipped_unsupported": len(skipped_unsupported),
            "skipped_oversize": len(skipped_oversize),
            "skipped_other": len(skipped_other),
        }

    except Exception as e:
        logger.error(f"Error indexing modified files: {e}", exc_info=True)
        return {"error": f"Error indexing modified files: {str(e)}"}


@mcp.tool()
def inspect_directory(path: str) -> dict:
    """
    Analyze folder to recommend 'index_local_codebase' or 'add_documents_from_directory'.
    
    CRITICAL INSTRUCTION FOR LLM:
    1.  Use this tool FIRST before indexing anything.
    2.  Check the output for the suggested project name (derived from the folder).
    3.  If the current project (from list_projects) is 'rag_collection' (default),
        you MUST suggest creating/switching to the new project name BEFORE indexing.
        Example: "I see this is the 'frontend' repo. Shall I create a 'frontend' project for it?"
    
    Args:
        path: Absolute path.
    """
    try:
        dir_path = Path(path).resolve()
        
        if not dir_path.exists():
            return {"error": f"Path not found: {path}"}
        if not dir_path.is_dir():
            return {"error": f"Path is not a directory: {path}"}

        # Stats
        stats = {
            "total_files": 0,
            "code_files": 0,
            "doc_files": 0,
            "extensions": {},
            "markers_found": []
        }
        
        # Quick scan (limit to 1000 files to be fast)
        file_limit = 1000
        scanned = 0
        
        for root, dirs, files in os.walk(dir_path):
            # Check markers
            root_path = Path(root)
            if root_path == dir_path:
                for d in dirs:
                    if d in CODE_INDICATORS['dirs']:
                        stats['markers_found'].append(d + "/")
                for f in files:
                    if f in CODE_INDICATORS['files']:
                        stats['markers_found'].append(f)

            for f in files:
                if scanned >= file_limit:
                    break
                    
                ext = Path(f).suffix.lower()
                stats["total_files"] += 1
                stats["extensions"][ext] = stats["extensions"].get(ext, 0) + 1
                
                if ext in SUPPORTED_CODE_EXTS:
                    stats["code_files"] += 1
                elif ext in SUPPORTED_DOC_EXTS:
                    stats["doc_files"] += 1
                
                scanned += 1
            
            if scanned >= file_limit:
                break

        # Decision Logic
        recommendation = "ask_user"
        reason = "Ambiguous content"
        
        is_codebase_structure = bool(stats['markers_found'])
        code_ratio = stats['code_files'] / stats['total_files'] if stats['total_files'] > 0 else 0
        doc_ratio = stats['doc_files'] / stats['total_files'] if stats['total_files'] > 0 else 0

        if stats['total_files'] == 0:
            recommendation = "none"
            reason = "Directory is empty"
        elif is_codebase_structure or code_ratio > 0.5:
            recommendation = "index_local_codebase"
            reason = f"High code ratio ({code_ratio:.1%}) and/or project markers found ({', '.join(stats['markers_found'][:3])})"
        elif doc_ratio > 0.5:
            recommendation = "add_documents_from_directory"
            reason = f"High document ratio ({doc_ratio:.1%}) found"
        else:
            reason = "Mixed content: contains both code and documents"
            recommendation = "index_hybrid_folder"

        return {
            "path": str(dir_path),
            "suggested_project_name": dir_path.name,
            "current_active_project": _current_project,
            "stats": {
                "total_files": stats['total_files'] if scanned < file_limit else f"{file_limit}+",
                "code_files": stats['code_files'],
                "doc_files": stats['doc_files'],
                "top_extensions": sorted(stats['extensions'].items(), key=lambda x: x[1], reverse=True)[:5]
            },
            "recommendation": recommendation,
            "reason": reason
        }

    except Exception as e:
        logger.error(f"Error inspecting directory: {e}", exc_info=True)
        return {"error": f"Error inspecting directory: {str(e)}"}


@mcp.tool()
def index_local_codebase(
    directory_path: str,
    language_filter: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    include_hidden: bool = False,
    project: Optional[str] = None
) -> dict:
    """
    Index local code with smart filtering.
    
    Args:
        directory_path: Codebase root.
        language_filter: Languages (e.g., ['python', '.ts']).
        exclude_patterns: Glob patterns to ignore.
        include_hidden: Include dotfiles (default: False).
        project: Target project/workspace to store ingested data.
    """
    try:
        dir_path = Path(directory_path).resolve()

        # Validate directory
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory_path}"}

        if not dir_path.is_dir():
            return {"error": f"Path is not a directory: {directory_path}"}

        project_error, resolved_project = _require_project_selection(
            "index_local_codebase",
            project,
            suggested=dir_path.name
        )
        if project_error:
            return project_error

        logger.info(f"Indexing local codebase: {dir_path} -> project {resolved_project}")

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

        # SMART Chunking
        index = get_index(resolved_project)
        nodes = []
        for doc in documents:
            file_path = doc.metadata.get('file_path', '')
            splitter = get_splitter_for_file(file_path)
            doc_nodes = splitter.get_nodes_from_documents([doc])
            nodes.extend(doc_nodes)

        logger.info(f"Created {len(nodes)} chunks from {len(documents)} code files")
        index.insert_nodes(nodes, show_progress=True)

        # Persist
        project_storage_path = STORAGE_PATH / _current_project
        index.storage_context.persist(persist_dir=str(project_storage_path))

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
    """Clear all documents (DESTRUCTIVE)."""
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
