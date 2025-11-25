# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a LlamaIndex-based RAG (Retrieval-Augmented Generation) MCP server that provides semantic search over local documents and codebases. The system uses OpenAI embeddings for vector search, Cohere for reranking, and ChromaDB for vector storage. The MCP server only retrieves and ranks documents - the MCP client's LLM generates the final answers.

**Key Features**:
- Semantic search over documents (PDF, DOCX, Markdown, etc.)
- **Codebase indexing**: Index and search GitHub repositories and local code projects
- Two-stage retrieval: Vector search + Cohere reranking for high accuracy
- Code-aware processing: Supports 20+ programming languages with language detection
- Smart filtering: Automatic exclusion of build artifacts, dependencies, and IDE files
- Incremental updates: Only re-index changed files to save API costs

## Development Commands

### Environment Setup
```bash
# Activate conda environment
conda activate deep-learning

# Install dependencies
pip install llama-index llama-index-embeddings-openai llama-index-vector-stores-chroma
pip install llama-index-postprocessor-cohere-rerank chromadb fastmcp nest-asyncio

# For codebase indexing (optional)
pip install llama-index-readers-github llama-index-readers-file
```

### Debugging and Profiling
```bash
# Run comprehensive debug and profiling tool
python debug_rag.py                    # Full diagnostics
python debug_rag.py --quick            # Skip profiling
python debug_rag.py --profile          # Profile only
python debug_rag.py --query "test"     # Custom query for profiling

# Enable debug logging in MCP server
export RAG_LOG_LEVEL=DEBUG  # Add to MCP config env section
```

### Building the Index
```bash
# Incremental update (only processes new/modified files)
python build_index.py /path/to/documents

# Force full rebuild
python build_index.py /path/to/documents --rebuild

# Default directory (./data)
python build_index.py
```

### Running and Testing
```bash
# Verify setup
python verify_setup.py

# Run MCP server directly (for testing)
python mcp_server.py

# Check index stats programmatically
python -c "from mcp_server import get_index_stats; print(get_index_stats())"

# Test codebase indexing (direct testing without MCP client)
python test_codebase_indexing.py
```

### Environment Variables
Required:
- `OPENAI_API_KEY`: For text-embedding-3-large model

Optional:
- `COHERE_API_KEY`: Enables reranking with rerank-v3.5 model

## Architecture

### Two-Stage Retrieval Pipeline
1. **Vector Search**: Retrieves 10 candidates using OpenAI embeddings (text-embedding-3-large, 3072 dimensions)
2. **Semantic Reranking**: Cohere rerank-v3.5 reorders by semantic relevance, returns top 6
3. **Client Processing**: MCP client's LLM generates the answer

### Key Components

**build_index.py** - Indexing system with incremental update support
- Tracks file metadata (mtime, size) in `storage/indexed_files.json`
- Only processes new/modified files to save 95%+ on API tokens
- Cleans document encoding to handle non-UTF-8 characters
- Filters ChromaDB metadata to supported types (str, int, float, None)
- Uses batch insertion for better performance

**mcp_server.py** - FastMCP server implementation
- Caches index, ChromaDB client, and reranker instances globally
- Uses absolute paths (SCRIPT_DIR) for reliable MCP execution
- Lazy initialization: creates empty index if none exists
- MCP Resources pattern: `rag://documents` and `rag://stats`

**Storage Structure**:
```
storage/
├── chroma_db/          # ChromaDB vector database (persistent)
├── indexed_files.json  # File tracking for incremental updates
├── docstore.json       # LlamaIndex document store
├── index_store.json    # LlamaIndex index metadata
├── graph_store.json    # LlamaIndex graph store
└── *.json             # Other index metadata
```

### Embedding Model Configuration
Both `build_index.py` and `mcp_server.py` must use the same embedding model:
```python
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")  # 3072 dims
```

Changing the embedding model requires a full rebuild with `--rebuild` flag.

### Reranking Configuration
Located in `mcp_server.py:98-102`:
```python
_reranker = CohereRerank(
    api_key=cohere_api_key,
    model="rerank-v3.5",
    top_n=6  # Number of documents returned after reranking
)
```

## MCP Integration

### Client Configuration
Add to MCP client config (e.g., `~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "llamaindex-rag": {
      "command": "/path/to/conda/envs/deep-learning/bin/python",
      "args": ["/absolute/path/to/mcp_server.py"],
      "cwd": "/absolute/path/to/llamaindex",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "COHERE_API_KEY": "..."
      }
    }
  }
}
```

### Available MCP Tools

**Query Tools**:
- `query_rag(question, similarity_top_k=6, use_rerank=True)` - Returns formatted text
  - `similarity_top_k` can be set to 1-20 based on needs (LLM decides dynamically)
  - Retrieves 2x candidates for reranking to ensure quality
- `query_rag_with_sources(question, similarity_top_k=6, use_rerank=True)` - Returns structured dict
  - Same flexibility as query_rag
- `iterative_search(question, initial_top_k=3, detailed_top_k=10)` - Two-phase search
  - Returns small focused set first (default 3)
  - Provides suggestions for follow-up searches
  - Encourages multi-round refinement instead of single-turn search

**Management Tools**:
- `get_index_stats()` - Index statistics
- `list_indexed_documents()` - Sample of indexed documents
- `add_document_from_text(text, metadata)` - Dynamic text ingestion
- `add_documents_from_directory(path)` - Bulk directory import
- `clear_index()` - Destructive: clears all documents

**Codebase Indexing Tools**:
- `index_github_repository(github_token, owner, repo, branch, filter_dirs, filter_extensions)` - Index a GitHub repository
  - Supports public and private repositories with token authentication
  - Optional filtering by directories and file extensions
  - Automatically adds repository metadata to indexed code
- `index_local_codebase(directory_path, language_filter, exclude_patterns, include_hidden)` - Index local codebase with code-aware processing
  - Language-specific filtering (e.g., ["python", "typescript", "go"])
  - Smart exclusion of common build/dependency directories (node_modules, __pycache__, etc.)
  - Automatic language detection and metadata tagging
  - Statistics breakdown by programming language

### MCP Resources
- `rag://documents` - Browse indexed documents
- `rag://stats` - View index statistics

## Important Implementation Details

### Incremental Update System
File change detection in `build_index.py:57-72`:
- Checks modification time (mtime) and file size
- Tracks metadata in `storage/indexed_files.json`
- Only processes changed/new files on subsequent runs
- Use `--rebuild` flag to force full reindex

When to use `--rebuild`:
- Changed embedding model
- Corrupted index
- Need to remove deleted files from index

### Metadata Filtering
ChromaDB only accepts `str`, `int`, `float`, `None` types (see `build_index.py:205-217`). Complex types like lists, dicts, and bools are converted or skipped during ingestion.

### Document Cleaning
Text cleaning pipeline in `build_index.py:194-233`:
- Handles encoding issues with surrogateescape/ignore
- Removes non-printable characters (except \n, \t, space)
- Skips documents shorter than 10 characters
- Reports skipped documents

### Supported File Formats
Defined in both scripts (must match):
- Documents: `.txt`, `.pdf`, `.docx`, `.md`, `.epub`
- Presentations: `.ppt`, `.pptx`, `.pptm`
- Data: `.csv`, `.json`, `.xml`
- Code/Notebooks: `.ipynb`, `.html`
- Images: `.jpg`, `.jpeg`, `.png` (OCR if configured)
- Other: `.hwp`, `.mbox`
- **Code files**: `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.java`, `.cpp`, `.c`, `.h`, `.hpp`, `.cs`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`, `.r`, `.sh`, `.bash`, `.zsh`, `.sql`, `.yaml`, `.yml`, `.toml`, `.dockerfile`, `.makefile`, `.vue`, `.svelte`, `.astro`

File size limit: 300MB per file

## Common Workflows

### Adding New Documents
```bash
# Just re-run - automatically detects new/modified files
python build_index.py /path/to/documents
```

### Dynamic Retrieval Configuration

**Flexible Result Count**:
The system now allows the MCP client's LLM to decide how many results to retrieve:
- `similarity_top_k` parameter accepts 1-20 results per query
- No hardcoded limits - the LLM can request 3 results for focused answers or 15+ for comprehensive analysis
- Reranker automatically retrieves 2x candidates to ensure quality (e.g., 20 candidates for top 10 results)

**Retrieval Settings** (`mcp_server.py`):
- Lines 150, 209: `similarity_top_k=6` as default (LLM can override)
- Line 170, 227: `initial_top_k = max(similarity_top_k * 2, 10)` - dynamic candidate pool
- Lines 99-102: Reranker initialized without hardcoded top_n (controlled per query)

### Multi-Round Search Patterns

**Recommended Search Strategies**:

1. **Iterative Refinement** (Preferred):
   - Start with `iterative_search(question, initial_top_k=3)` for focused results
   - Review initial results and relevance scores
   - Follow up with `query_rag(question, similarity_top_k=10)` if more context needed
   - Or refine the question based on what was found

2. **Adaptive Result Count**:
   - Specific questions: Use `similarity_top_k=3-5` for focused answers
   - Broad research: Use `similarity_top_k=10-15` for comprehensive coverage
   - Exploratory: Use `similarity_top_k=20` to find edge cases

3. **Progressive Deepening**:
   - Query 1: Broad question with `similarity_top_k=5` to understand scope
   - Query 2: Refined question targeting specific aspects found in Query 1
   - Query 3: Deep dive into specific documents or topics

4. **Score-Based Adaptation**:
   - If relevance scores < 0.7: Rephrase question or try different keywords
   - If relevance scores > 0.9: Results are highly relevant, fewer needed
   - If scores widely distributed: Request more results to capture variety

**Why Multi-Round is Better**:
- Single large query (20 results) may include noise and reduce LLM focus
- Multiple focused queries (3-5 results each) provide better signal-to-noise ratio
- Iterative approach allows course correction based on what's actually found
- Saves tokens by only retrieving what's needed

### Codebase Indexing

The system supports indexing source code from GitHub repositories and local codebases with code-aware processing.

#### Indexing GitHub Repositories

```python
# Basic usage - index entire repository
index_github_repository(
    github_token="ghp_xxxxxxxxxxxx",
    owner="facebook",
    repo="react",
    branch="main"
)

# Filter by directories - only index specific folders
index_github_repository(
    github_token="ghp_xxxxxxxxxxxx",
    owner="vercel",
    repo="next.js",
    branch="canary",
    filter_dirs=["packages/next", "packages/react"]
)

# Filter by file extensions - only Python files
index_github_repository(
    github_token="ghp_xxxxxxxxxxxx",
    owner="python",
    repo="cpython",
    filter_extensions=[".py", ".c", ".h"]
)
```

**Getting a GitHub Token**:
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic) with `repo` scope for private repos, or no scopes for public repos
3. Use the token in the `github_token` parameter

#### Indexing Local Codebases

**Language Auto-Detection**: The system automatically detects programming languages from file extensions. The `language_filter` parameter is optional - omit it to index all code files.

```python
# Basic usage - index all code files (AUTO-DETECTS ALL LANGUAGES)
index_local_codebase(
    directory_path="/Users/username/projects/my-app"
)
# Returns language breakdown: {"python": 45, "javascript": 23, "typescript": 12, ...}

# Optional: Filter by specific programming languages
index_local_codebase(
    directory_path="/Users/username/projects/my-app",
    language_filter=["python", "javascript", "typescript"]
)

# Custom exclusion patterns
index_local_codebase(
    directory_path="/Users/username/projects/my-app",
    language_filter=["python"],
    exclude_patterns=["tests", "*.test.py", "migrations"]
)

# Include hidden files (disabled by default)
index_local_codebase(
    directory_path="/Users/username/projects/my-app",
    include_hidden=True
)
```

**Supported Languages**:
- `python`, `javascript`, `typescript`, `java`, `cpp` (C/C++)
- `csharp`, `go`, `rust`, `ruby`, `php`
- `swift`, `kotlin`, `scala`, `shell`
- `sql`, `html`, `css`, `yaml`, `json`, `xml`, `markdown`
- `vue`, `svelte`, `astro`

**Default Exclusions**:
The local codebase indexer automatically excludes:
- Dependencies: `node_modules`, `venv`, `env`, `.venv`
- Build outputs: `build`, `dist`, `target`, `out`
- Version control: `.git`, `.svn`, `.hg`
- IDE files: `.idea`, `.vscode`, `.vs`
- Compiled files: `*.pyc`, `*.pyo`, `*.so`, `*.dll`
- Cache: `__pycache__`, `.DS_Store`

#### Querying Indexed Code

After indexing, use regular RAG tools to search your codebase:

```python
# Find authentication implementation
query_rag("How does user authentication work?", similarity_top_k=5)

# Find error handling patterns
query_rag("Show me error handling patterns in the API layer", similarity_top_k=8)

# Understand a specific feature
query_rag("How is file upload implemented?", similarity_top_k=3)

# Get sources with metadata (includes file paths, language info)
query_rag_with_sources("Find database query optimization code", similarity_top_k=5)
```

**Code Search Tips**:
1. **Be specific about what you're looking for**: Instead of "authentication", try "JWT token validation" or "OAuth2 implementation"
2. **Use technical terms**: "database connection pooling", "rate limiting middleware", "Redux state management"
3. **Mention file types or patterns**: "React hooks", "API endpoints", "database migrations"
4. **Use iterative search**: Start with `iterative_search()` for broad questions, then refine based on results

#### Example Workflow: Indexing a New Project

```bash
# 1. Index your local project
index_local_codebase(
    directory_path="/Users/username/projects/myapp",
    language_filter=["python", "javascript"],
    exclude_patterns=["tests", "docs"]
)

# Output:
# {
#   "success": true,
#   "files_indexed": 127,
#   "chunks_created": 483,
#   "language_breakdown": {
#     "python": 89,
#     "javascript": 38
#   }
# }

# 2. Verify indexing
get_index_stats()

# 3. Search your code
query_rag("How does the payment processing work?")

# 4. Get detailed sources
query_rag_with_sources("Find all API endpoint definitions")
```

### Debugging Index Issues
```bash
# Quick checks
ls storage/                              # Verify storage exists
cat storage/indexed_files.json | jq     # View tracking data
python debug_rag.py --quick              # Run diagnostic tool

# Detailed profiling
python debug_rag.py                      # Full diagnostics with profiling
python debug_rag.py --query "your query" # Profile specific query

# Check MCP server functionality
python -c "from mcp_server import get_index_stats; print(get_index_stats())"
python -c "from mcp_server import get_cache_stats; print(get_cache_stats())"
```

## Debugging and Optimization

### Debug Tool (debug_rag.py)

The `debug_rag.py` tool provides comprehensive diagnostics:

**Features**:
- Environment validation (API keys, paths)
- Storage integrity checks
- Index metadata analysis
- Quality checks (document lengths, metadata)
- Performance profiling (with/without reranking)
- Edge case testing
- Cache efficiency analysis

**Usage Examples**:
```bash
# Full diagnostic - use this first when troubleshooting
python debug_rag.py

# Quick checks only (skip slow profiling)
python debug_rag.py --quick

# Profile retrieval performance
python debug_rag.py --profile --query "machine learning"

# The tool will report:
# - Missing API keys
# - Storage issues
# - Index quality problems
# - Performance bottlenecks
# - Cache efficiency
```

### Logging and Monitoring

**Enable Debug Logging**:
Add to MCP client config `env` section:
```json
"env": {
  "OPENAI_API_KEY": "...",
  "COHERE_API_KEY": "...",
  "RAG_LOG_LEVEL": "DEBUG"  // or INFO, WARNING, ERROR
}
```

**Log Levels**:
- `DEBUG`: Detailed query/retrieval information
- `INFO`: High-level operations (recommended for troubleshooting)
- `WARNING`: Default - only warnings and errors
- `ERROR`: Only errors

**What Gets Logged**:
- Query validation and parameters
- Cache hits/misses
- Retrieval candidates and reranking
- Performance timings
- Errors with full stack traces

### Validation and Error Handling

**Input Validation** (`mcp_server.py:78-108`):
- `validate_top_k()`: Clamps to 1-50, converts types
- `validate_query()`: Checks for empty/whitespace, truncates at 10K chars

**Error Types**:
- `ValueError`: Invalid inputs (empty query, bad parameters)
- Generic `Exception`: System errors (missing index, API failures)

All errors are logged with stack traces when `RAG_LOG_LEVEL=DEBUG`.

### Performance Optimization

**Caching Strategy**:
Three levels of caching with performance metrics:
1. **Index Cache**: LlamaIndex object (expensive to load)
2. **ChromaDB Cache**: Vector store client
3. **Reranker Cache**: Cohere rerank model

**Monitor Cache Performance**:
```python
# Via MCP tool
get_cache_stats()

# Returns:
# {
#   "index": {"cache_hits": 10, "loads": 1, "hit_rate": "90.9%"},
#   "reranker": {"cache_hits": 8, "loads": 1, "hit_rate": "88.9%"},
#   "chroma": {"cache_hits": 12, "loads": 1, "hit_rate": "92.3%"}
# }
```

**Cache Hit Rates**:
- First query: 0% (cold start - everything loads)
- Subsequent queries: >90% (all cached)
- If hit rate < 50%: Something is wrong (cache not persisting)

**Performance Tuning**:
- `RERANK_CANDIDATE_MULTIPLIER = 2`: Retrieve 2x candidates for reranking
- `MIN_RERANK_CANDIDATES = 10`: Minimum candidates even for small top_k
- `MAX_TOP_K = 50`: Maximum results allowed per query

### Common Issues and Solutions

**Issue**: "No relevant documents found"
- **Check**: Run `get_index_stats()` - is document_count > 0?
- **Fix**: Run `python build_index.py /path/to/documents`

**Issue**: Slow queries
- **Check**: Run `debug_rag.py --profile` - is reranking taking > 2s?
- **Fix**: Set `use_rerank=False` or reduce `similarity_top_k`
- **Monitor**: Use `get_cache_stats()` - cache should be >90% after warmup

**Issue**: Poor relevance
- **Check**: Relevance scores in results - are they < 0.7?
- **Fix**: Use `iterative_search()` and refine query based on suggestions
- **Alternative**: Increase `similarity_top_k` to get more candidates

**Issue**: "OPENAI_API_KEY not found"
- **Check**: MCP config `env` section has API key
- **Fix**: Add key to MCP client config, restart client

**Issue**: Cache not persisting
- **Check**: `get_cache_stats()` shows low hit rates across queries
- **Root cause**: Server restarting between queries
- **Fix**: Check MCP client logs for server crashes/restarts

### Edge Cases Handled

The system handles these edge cases (tested by `debug_rag.py`):
1. Empty queries → Validation error
2. Very long queries (>10K chars) → Truncated with warning
3. `top_k=0` or negative → Clamped to MIN_TOP_K (1)
4. `top_k>50` → Clamped to MAX_TOP_K (50)
5. Special characters/Unicode → Properly encoded
6. Missing index → Creates empty index
7. No results found → Graceful message
8. Reranker unavailable → Falls back to vector search only

### Design Philosophy
The server is retrieval-only - it does not generate answers. This design:
- Allows MCP clients to use any LLM for generation
- Reduces operational costs (no OpenAI LLM calls)
- Provides better context visibility to users
- Separates concerns: retrieval vs. generation
