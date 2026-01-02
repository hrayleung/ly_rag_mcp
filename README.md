# LlamaIndex RAG MCP Server

Local RAG system with OpenAI/Gemini embeddings, hybrid search (semantic + BM25), Cohere reranking, HyDE query augmentation, and MCP server integration.

## Features

- **Incremental Updates**: Only processes new/modified files based on mtime/size tracking
- **Hybrid Search**: Semantic (vector) + keyword (BM25) retrieval with RRF fusion
- **Adaptive Mode Selection**: Automatically detects code-like queries and switches to hybrid search
- **HyDE Query Augmentation**: Hypothetical Document Embeddings for better conceptual matching
- **Two-Stage Retrieval**: Flexible top_k (1-50) + optional Cohere v3.5 reranking
- **Smart Project Routing**: Multi-project isolation with automatic query routing to best project
- **Multiple Embedding Providers**: OpenAI or Google Gemini
- **Models**: OpenAI `text-embedding-3-large` or Gemini `text-embedding-004` + Cohere `rerank-v3.5`
- **MCP Integration**: Works with standard MCP clients (Claude Desktop, Chatwise, Cherry Studio, etc.)
- **68+ File Formats**: Code (49 ext), Docs (16 ext), Images (3 ext) via LlamaIndex readers
- **Local Storage**: Per-project ChromaDB vector databases under `storage/{project}/`
- **Optional Vue.js UI**: Real-time MCP monitoring at `/` when running api_server.py
- **Thread-Safe**: All managers use singleton pattern with locks; cross-platform file locking for metadata

## Installation

```bash
git clone https://github.com/hrayleung/ly_rag_mcp.git
cd ly_rag_mcp

conda create -n deep-learning python=3.10
conda activate deep-learning

# Core dependencies
pip install llama-index llama-index-embeddings-openai llama-index-vector-stores-chroma
pip install llama-index-postprocessor-cohere-rerank chromadb fastmcp

# Gemini support (optional)
pip install google-genai

# New features (Hybrid Search & Web Crawling)
pip install rank-bm25 llama-index-retrievers-bm25 firecrawl-py
```

## Quick Start

### 1. Set API Keys

**Option A: OpenAI Embeddings (default)**
```bash
export EMBEDDING_PROVIDER=openai
export EMBEDDING_MODEL=text-embedding-3-large
export OPENAI_API_KEY='your-openai-api-key'
export COHERE_API_KEY='your-cohere-api-key'     # Optional: For Reranking
export FIRECRAWL_API_KEY='your-firecrawl-key'   # Optional: For Web Crawling
```

**Option B: Gemini Embeddings**
```bash
export EMBEDDING_PROVIDER=gemini
export EMBEDDING_MODEL=text-embedding-004
export GEMINI_API_KEY='your-gemini-api-key'
export COHERE_API_KEY='your-cohere-api-key'     # Optional: For Reranking
export FIRECRAWL_API_KEY='your-firecrawl-key'   # Optional: For Web Crawling
```

### 2. Index Documents

```bash
# Incremental update (default - only processes new/modified files)
python build_index.py /path/to/your/documents

# Force full rebuild
python build_index.py /path/to/your/documents --rebuild
```

### 3. Configure MCP Client

Add to your MCP client configuration:

**Option A: OpenAI Embeddings**
```json
{
  "mcpServers": {
    "llamaindex-rag": {
      "command": "/path/to/conda/envs/deep-learning/bin/python",
      "args": ["/path/to/ly_rag_mcp/mcp_server.py"],
      "cwd": "/path/to/ly_rag_mcp",
      "env": {
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "OPENAI_API_KEY": "your-openai-key",
        "COHERE_API_KEY": "your-cohere-key",
        "FIRECRAWL_API_KEY": "your-firecrawl-key"
      }
    }
  }
}
```

**Option B: Gemini Embeddings**
```json
{
  "mcpServers": {
    "llamaindex-rag": {
      "command": "/path/to/conda/envs/deep-learning/bin/python",
      "args": ["/path/to/ly_rag_mcp/mcp_server.py"],
      "cwd": "/path/to/ly_rag_mcp",
      "env": {
        "EMBEDDING_PROVIDER": "gemini",
        "EMBEDDING_MODEL": "text-embedding-004",
        "GEMINI_API_KEY": "your-gemini-key",
        "COHERE_API_KEY": "your-cohere-key",
        "FIRECRAWL_API_KEY": "your-firecrawl-key"
      }
    }
  }
}
```

**Config locations:**
- **Claude Code**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Chatwise/Cherry Studio**: Application settings

### 4. Query

Restart your MCP client and ask questions:

```
What are these documents about?
Use query_rag to search for "parallel computing"
Show me the index statistics
```

## Supported File Formats

| Category | Formats |
|----------|---------|
| Code (49 extensions) | `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.java`, `.cpp`, `.c`, `.h`, `.hpp`, `.cs`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`, `.r`, `.sh`, `.bash`, `.zsh`, `.sql`, `.yaml`, `.yml`, `.toml`, `.dockerfile`, `.makefile`, `.vue`, `.svelte`, `.astro`, `.html`, `.css` |
| Documents (16 extensions) | `.txt`, `.pdf`, `.docx`, `.doc`, `.md`, `.epub`, `.ppt`, `.pptx`, `.pptm`, `.xls`, `.xlsx`, `.csv`, `.json`, `.xml`, `.ipynb`, `.hwp`, `.mbox`, `.rtf` |
| Images (3 extensions) | `.jpg`, `.jpeg`, `.png` |

File size limit: 100MB per file

**Excluded by default:** `node_modules`, `__pycache__`, `.git`, venv, build outputs, and IDE files.

## MCP Tools

### Query Tools
- `query_rag(question, top_k=6, search_mode='semantic', use_rerank=True, use_hyde=False, return_metadata=False, project=None)`
  - `search_mode`: 'semantic' (vector), 'hybrid' (BM25+Vector with RRF), or 'keyword' (BM25-only). Auto-detects code patterns for hybrid.
  - `use_hyde`: Enables Hypothetical Document Embeddings - generates synthetic answer for better retrieval on conceptual queries
  - `top_k`: Number of results (1-50)
  - `return_metadata`: Returns structured JSON with sources, scores, metadata when True; otherwise returns formatted text
  - `project`: Explicit project name, or auto-route via smart routing

### Ingestion Tools
- `index_documents(path, project=None)` - Index documents from directory (supports all formats)
- `add_text(text, metadata=None, project=None)` - Add raw text to index
- `inspect_directory(path)` - Analyze folder content before indexing (shows file types, counts)
- `crawl_website(url, max_pages=10, project=None)` - Crawl websites via Firecrawl

**Project Confirmation:** Ingestion tools require a `project` argument. If omitted, returns `{"action_required": "select_project", ...}` with suggestions.

### Admin Tools
- `manage_project(action, project=None, keywords=None, description=None)`
  - Actions: `list`, `create`, `switch`, `update`, `analyze`, `choose`
- `get_stats(stat_type='index')` - System statistics
- `list_documents(project=None, limit=100)` - List indexed documents
- `clear_index(project=None, confirm=False)` - Destructive: clear project index

### HTTP API Server (Optional)
Run `python api_server.py` for telemetry endpoints and Vue UI:
- `GET /api/mcp/tools` - Tool metadata (name, type, description)
- `GET /api/mcp/requests` - Recent request samples (status, route, latency, tool, user)
- `GET /api/mcp/logs` - Server log ring buffer
- `GET /api/mcp/stats` - System metrics (uptime, RPM, errors, index stats)
- `GET /api/mcp/health` - Health check with uptime status
- `GET /` - Vue UI frontend (build with `npm run build` in `frontend/`)

See `README_UI.md` for detailed frontend documentation.

### Debugging
- `python debug_rag.py` - Comprehensive diagnostics (environment, storage, index quality, performance)
- `python verify_setup.py` - Quick setup verification
- Set `RAG_LOG_LEVEL=DEBUG` for detailed logs

## Architecture

### Retrieval Pipeline
```
Query → Search Mode Detection (Adaptive)
    ↓
(Optional) HyDE Query Augmentation (if weak results)
    ↓
Parallel: Vector Search + BM25 Keyword Search
    ↓
RRF (Reciprocal Rank Fusion) Merge
    ↓
(Optional) Cohere Reranking
    ↓
Results (top_k)
```

### Ingestion Pipeline
```
Documents → DocumentLoader (20+ formats)
    ↓
DocumentProcessor (UTF-8 sanitize + context injection)
    ↓
DocumentChunker (AST for code, sentence for docs)
    ↓
IndexManager → LlamaIndex Storage
    ↓
Persist: ChromaDB + JSON manifests
```

### Multi-Project Isolation
```
storage/
└── {project_name}/
    ├── chroma_db/          # ChromaDB vector collection
    ├── project_metadata.json    # Project config, keywords
    ├── ingest_manifest.json     # File change tracking
    ├── indexed_files.json       # mtime/size tracking
    ├── docstore.json       # LlamaIndex document store
    ├── index_store.json    # LlamaIndex index metadata
    └── graph_store.json    # LlamaIndex graph store
```

## Search Strategies

### Adaptive Mode Selection
Automatically switches to hybrid search when query contains:
- 3+ digit tokens (e.g., "HTTP_200")
- Uppercase patterns: `[A-Z_]{2,}`
- Code characters: `{}();=<>*/+-`
- Path-like tokens: `/`, `\`
- Dots in tokens with 3+ trailing chars (e.g., `module.function`)
- >30% uppercase or >40% digits

### HyDE (Hypothetical Document Embeddings)
- **Trigger**: Results ≤ 1 OR max score ≤ 0.1 OR all scores < 0.2
- **Process**: Generates synthetic answer using OpenAI GPT-3.5, embeds it, searches with that embedding
- **Timeout**: 30 seconds
- **Retries**: 2 with exponential backoff (0.5s → 1s)

### Optional Vue.js UI
- **Aesthetic**: Minimal dark industrial style using Inter and JetBrains Mono fonts.
- **Mechanism**: Auto-polls `api_server.py` every 3 seconds using `Promise.all` for parallel data fetching.
- **Features**: Live request monitoring, latency tracking, terminal-style log viewer.
- **Run**: `python api_server.py` and visit `http://localhost:8000`.

### Reranking (Cohere v3.5)
- **Retrieval**: 2x `top_k` candidates (minimum 10)
- **Skips**: When score delta > 0.05 or too few results (< 3)
- **Benefit**: 15-40% accuracy improvement on domain-specific queries

## Project Management

### Isolated Workspaces
1. `manage_project(action="create", project="backend")`
2. `manage_project(action="switch", project="backend")`
3. Index backend files using `index_documents(path, project="backend")`
4. Switch to "frontend" to keep contexts clean
5. **Auto-routing**: Queries mentioning a project name automatically route to that workspace

### Project Metadata & Smart Routing
- Call `list_projects()` to see each workspace's details (display name, keywords, default paths, last indexed timestamp)
- Before indexing a new repo, run `inspect_directory(<path>)` for file analysis, then create/update the project
- Use `manage_project(action="update", project="frontend", keywords=["nextjs","api"], description="Customer portal")` to add hints
- For routing decisions: `manage_project(action="choose", question="<user request>")` shows scored candidates
- Ingestion tools automatically update `default_paths` and maintain `last_indexed_at` timestamps

### Keyword Learning
Successful queries automatically update project keywords via `MetadataManager.learn_from_query()`. Over time, routing improves as the system learns query-project associations.

## Reranking Strategy

### Multi-Round Search (Recommended)

Instead of requesting many results in a single query, use iterative refinement:

**Example 1: Start Focused, Expand if Needed**
```
1. Use iterative_search("Python async patterns", initial_top_k=3)
2. Review the 3 most relevant results
3. If more context needed: query_rag("Python async patterns", similarity_top_k=10)
4. Or refine: query_rag("asyncio event loop internals", similarity_top_k=5)
```

**Example 2: Progressive Deepening**
```
1. query_rag("machine learning", similarity_top_k=5) - understand scope
2. query_rag("neural network backpropagation", similarity_top_k=5) - focus on specific topic
3. query_rag("gradient descent optimization", similarity_top_k=3) - deep dive
```

**Benefits**:
- Better accuracy: Focus on most relevant results
- Token efficiency: Only retrieve what you need
- Course correction: Refine based on actual findings
- Less noise: Avoid diluting context with marginally relevant documents

### Two-Stage Retrieval
1. **Vector Search**: Retrieve 2x candidates (e.g., 20 candidates for top 10, minimum 10)
2. **Reranking**: Cohere v3.5 reorders by semantic relevance

Dynamic adjustment based on `top_k`:
- `top_k=3` → retrieves 10 → reranks → returns 3
- `top_k=10` → retrieves 20 → reranks → returns 10
- `top_k=15` → retrieves 30 → reranks → returns 15

**Enable**: Add `COHERE_API_KEY` to environment
**Disable**: Set `use_rerank=False` in queries

## Testing

The project uses pytest with a target coverage of ≥70%.

### Running Tests
```bash
# Full suite with coverage
pytest tests/ --cov=rag --cov-report=term -v

# Single file
pytest tests/test_query_tools.py --cov-report=term -v

# Single test
pytest tests/test_query_tools.py::test_specific_function --cov-report=term -v
```

### Test Files
| Test File | Module Tested | Notes |
|-----------|---------------|-------|
| `test_query_tools.py` | `rag.tools.query` | Query tool integration tests |
| `test_query_tools_small.py` | `rag.tools.query` | Fast unit tests (mocked) |
| `test_ingest_tools.py` | `rag.tools.ingest` | Ingestion integration tests |
| `test_ingest_tools_small.py` | `rag.tools.ingest` | Fast unit tests (mocked) |
| `test_admin_tools_small.py` | `rag.tools.admin` | Admin tool unit tests |
| `test_index_manager.py` | `rag.storage.index` | Index manager tests |
| `test_index_manager_small.py` | `rag.storage.index` | Fast unit tests (mocked) |
| `test_metadata.py` | `rag.project.metadata` | Metadata tests |
| `test_metadata_small.py` | `rag.project.metadata` | Fast unit tests (mocked) |
| `test_hyde.py` | `rag.retrieval.hyde` | HyDE tests |
| `test_hyde_timeout.py` | `rag.retrieval.hyde` | HyDE timeout tests |
| `test_bm25_cache_invalidation.py` | `rag.retrieval.bm25` | BM25 cache tests |
| `test_search_validation.py` | `rag.retrieval.search` | Search validation tests |
| `test_reranker_decision.py` | `rag.retrieval.reranker` | Reranker decision logic |
| `test_project_manager_choose_project.py` | `rag.project.manager` | Project routing tests |
| `test_project_discovery_validation.py` | `rag.project.manager` | Project discovery tests |
| `test_index_validation.py` | `rag.storage.index` | Index validation tests |
| `test_api_server.py` | `api_server` | API endpoint tests |

**Total:** 18 test files

### Test Patterns
- **Integration tests** (`test_*.py`): Full module testing with mocks/patches
- **Fast unit tests** (`test_*_small.py`): Lightweight mocked tests for CI speed
- **Mocking strategies**: `DummyMCP`/`FakeMCP` classes, `SimpleNamespace` mocks, `unittest.mock.patch`
- **Thread safety tests**: Multi-threaded tests for concurrent operations
- **Backward compatibility tests**: Loading legacy JSON formats, missing fields

## Configuration

### Embedding Models

Set via environment variables in your MCP config:

**OpenAI Models:**
```json
"env": {
  "EMBEDDING_PROVIDER": "openai",
  "EMBEDDING_MODEL": "text-embedding-3-large"  // or "text-embedding-3-small"
}
```

**Gemini Models:**
```json
"env": {
  "EMBEDDING_PROVIDER": "gemini",
  "EMBEDDING_MODEL": "text-embedding-004"  // or "embedding-001"
}
```

**Note:** Indexes created with one embedding model are not compatible with another. If you switch models, rebuild your index with `--rebuild`.

### Reranking Models

Set `COHERE_API_KEY` environment variable. Default model: `rerank-v3.5` (used in `rag/retrieval/reranker.py`).

### Configuration Parameters (RAGSettings)

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| **Chunking** | `chunk_size` | `1024` | Text chunk size |
| | `chunk_overlap` | `200` | Overlap between chunks |
| | `code_chunk_lines` | `40` | Lines per code chunk |
| | `code_chunk_overlap` | `15` | Overlap lines for code |
| | `code_max_chars` | `1500` | Max chars per code chunk |
| **Retrieval** | `min_top_k` | `1` | Minimum results |
| | `max_top_k` | `50` | Maximum results |
| | `default_top_k` | `6` | Default results |
| | `rerank_candidate_multiplier` | `2` | Multiplier for rerank candidates |
| | `min_rerank_candidates` | `10` | Minimum candidates for rerank |
| **Thresholds** | `low_score_threshold` | `0.2` | Low relevance threshold |
| | `rerank_delta_threshold` | `0.05` | Minimum score delta |
| | `rerank_min_results` | `3` | Minimum results to rerank |
| | `hyde_trigger_min_results` | `1` | Trigger HyDE if fewer results |
| | `hyde_trigger_score` | `0.1` | Trigger HyDE if max score below this |
| | `hyde_timeout` | `30.0` | HyDE query generation timeout (sec) |
| | `hyde_max_retries` | `2` | Max HyDE retry attempts |
| | `hyde_initial_backoff` | `0.5` | Initial backoff for HyDE retries |
| **API Server** | `request_buffer_size` | `200` | Recent requests buffer |
| | `log_buffer_size` | `400` | Log buffer size |
| **Locking** | `lock_retry_attempts` | `3` | File lock retry attempts |
| | `lock_retry_delay` | `0.1` | Delay between retries (sec) |
| **File constraints** | `max_file_size_mb` | `100` | Max file size in MB |
| | `max_query_length` | `10000` | Max query character length |
| **Project defaults** | `default_project` | `"rag_collection"` | Default project name |
| | `storage_path` | `"./storage"` | Root storage directory |

### File Extensions Supported

**Code (49 extensions):** `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.java`, `.cpp`, `.c`, `.h`, `.hpp`, `.cs`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`, `.r`, `.sh`, `.bash`, `.zsh`, `.sql`, `.yaml`, `.yml`, `.toml`, `.dockerfile`, `.makefile`, `.vue`, `.svelte`, `.astro`, `.html`, `.css`

**Documents (16 extensions):** `.txt`, `.pdf`, `.docx`, `.doc`, `.md`, `.epub`, `.ppt`, `.pptx`, `.pptm`, `.xls`, `.xlsx`, `.csv`, `.json`, `.xml`, `.ipynb`, `.hwp`, `.mbox`, `.rtf`

**Images (3 extensions):** `.jpg`, `.jpeg`, `.png`

**Default Excludes:** `node_modules`, `__pycache__`, `.git`, `.svn`, `.hg`, `venv`, `env`, `.venv`, `.env`, `build`, `dist`, `target`, `out`, `.idea`, `.vscode`, `.vs`, `*.pyc`, `*.pyo`, `*.so`, `*.dylib`, `*.dll`, `.DS_Store`, `Thumbs.db`

## Project Structure

```
ly_rag_mcp/
├── mcp_server.py           # FastMCP server entry point (MCP stdio)
├── api_server.py           # Optional HTTP API + Vue UI
├── build_index.py          # CLI index builder (incremental)
├── verify_setup.py         # Setup verification
├── debug_rag.py            # Debug & profiling tool
├── rag/                    # Core package (modular architecture)
│   ├── __init__.py         # Lazy exports
│   ├── config.py           # RAGSettings, logging, constants
│   ├── models.py           # Data models (SearchMode, ProjectMetadata, etc.)
│   ├── embeddings.py       # Embedding factory (OpenAI/Gemini)
│   ├── storage/            # Storage layer
│   │   ├── chroma.py       # ChromaDB client manager (singleton)
│   │   └── index.py        # LlamaIndex storage manager (singleton)
│   ├── retrieval/          # Retrieval layer
│   │   ├── search.py       # Unified search engine (hybrid/semantic/keyword)
│   │   ├── reranker.py     # Cohere reranking manager
│   │   ├── bm25.py         # BM25 keyword search manager
│   │   └── hyde.py         # HyDE query augmentation
│   ├── ingestion/          # Ingestion layer
│   │   ├── loader.py       # Multi-format document loading
│   │   ├── processor.py    # Text cleaning, context injection
│   │   └── chunker.py      # Smart chunking (AST for code, sentence for docs)
│   ├── project/            # Multi-project isolation
│   │   ├── manager.py      # Project lifecycle, smart routing/selection
│   │   └── metadata.py     # Project metadata storage (atomic writes)
│   └── tools/              # MCP tool definitions
│       ├── __init__.py     # Tool registration facade
│       ├── query.py        # Search & retrieval tools
│       ├── ingest.py       # Document ingestion tools
│       └── admin.py        # Admin & management tools
├── tests/                  # Test suite (pytest)
├── frontend/               # Optional Vue 3 + Vite UI
│   ├── src/
│   │   ├── components/     # Vue components
│   │   ├── composables/    # API & polling composables
│   │   └── views/          # Page views
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── .env.example            # Environment template
│
└── storage/                # Generated indexes (git-ignored)
    └── {project}/          # Per-project isolation
        ├── chroma_db/      # ChromaDB vector database
        ├── project_metadata.json  # Project config
        ├── ingest_manifest.json   # Ingestion tracking
        ├── indexed_files.json     # File change tracking
        ├── docstore.json       # LlamaIndex document store
        ├── index_store.json    # LlamaIndex index metadata
        └── graph_store.json    # LlamaIndex graph store
```

### Architecture Benefits

- **Modular**: Each module has a single responsibility (~200-300 lines per file)
- **Testable**: Clean interfaces with 18 test files, target ≥70% coverage
- **Thread-safe**: All managers use singleton pattern with `threading.Lock()` or `RLock()`
- **Atomic writes**: Metadata uses temp-file + fsync + cross-platform file locks
- **Extensible**: Easy to add new retrievers, chunkers, or tools

### Manager Singletons (lazy initialization, thread-safe)

| Manager | Module | Key Methods |
|---------|--------|-------------|
| `get_index_manager()` | `rag.storage.index` | `get_index()`, `insert_nodes()`, `persist()`, `reset()`, `switch_project()`, `validate_index()` |
| `get_project_manager()` | `rag.project.manager` | `discover_projects()`, `create_project()`, `switch_project()`, `list_projects()`, `choose_project()`, `set_project_metadata()` |
| `get_chroma_manager()` | `rag.storage.chroma` | ChromaDB connection and collection management |
| `get_reranker_manager()` | `rag.retrieval.reranker` | Cohere reranking with caching |
| `get_bm25_manager()` | `rag.retrieval.bm25` | BM25 keyword search with cache invalidation |
| `get_metadata_manager()` | `rag.project.metadata` | Project metadata with atomic writes |
| `get_search_engine()` | `rag.retrieval.search` | Unified search with adaptive mode selection |

### Data Models (`rag/models.py`)

**Enums:**
- `SearchMode`: `SEMANTIC`, `HYBRID`, `KEYWORD`
- `ChangeType`: `NEW`, `MODIFIED`, `REMOVED`
- `ContentType`: `CODE`, `DOCUMENT`, `MIXED`

**Dataclasses:**
- `FileMetadata`: `path`, `mtime_ns`, `size`
- `ProjectMetadata`: `name`, `display_name`, `description`, `keywords[]`, `default_paths[]`, `last_indexed`, `created_at`, `updated_at`, `last_chat_time`, `chat_turn_count`
- `RetrievalResult`: `text`, `score`, `metadata`, `node_id`, `preview`
- `SearchResult`: `results[]`, `query`, `search_mode`, `reranked`, `used_hyde`, `generated_query`, `project`, `total`
- `IngestResult`: `success`, `message`, `documents_processed`, `chunks_created`, `skipped_unsupported`, `skipped_oversize`, `skipped_other`, `error`
- `CacheStats`: Performance metrics for index, reranker, chroma, bm25

## Troubleshooting

### Quick Diagnostics
```bash
# Run the comprehensive debug tool first!
python debug_rag.py

# This will check:
# - Environment variables (API keys)
# - Storage integrity
# - Index quality
# - Performance metrics
# - Edge cases
```

### MCP Server Not Connecting
1. Verify Python path in MCP config
2. Check API keys in `env` section
3. Ensure `cwd` points to project directory
4. Restart MCP client
5. Check logs with `RAG_LOG_LEVEL=DEBUG` in MCP config

### No Documents Retrieved
```bash
# Check if index has documents
python -c "from mcp_server import get_index_stats; print(get_index_stats())"

# If document_count is 0:
python build_index.py /path/to/your/documents
```

### Slow Performance
```bash
# Profile retrieval performance
python debug_rag.py --profile

# Check cache efficiency
python -c "from mcp_server import get_cache_stats; print(get_cache_stats())"

# If cache hit rate < 90% after multiple queries:
# - Check MCP client logs for server restarts
# - Server should stay running between queries
```

### Poor Relevance
- Try `iterative_search()` instead of `query_rag()` for better refinement
- Check relevance scores in results (should be > 0.7 for good matches)
- Use multi-round search: start with 3 results, refine based on findings
- Increase `similarity_top_k` to get more candidates

### Adding New Documents

```bash
# Just re-run the same command - it will only process new/modified files
python build_index.py /path/to/your/documents

# Force full rebuild if needed
python build_index.py /path/to/your/documents --rebuild
```

**Incremental update detects:**
- New files added to the directory
- Modified files (based on modification time and file size)
- Skips unchanged files automatically

**When to use `--rebuild`:**
- Changed embedding model
- Corrupted index
- Want to remove deleted files from index

### Debug Logging

Add to your MCP config:
```json
"env": {
  "OPENAI_API_KEY": "...",
  "COHERE_API_KEY": "...",
  "RAG_LOG_LEVEL": "DEBUG"
}
```

This will log:
- Query validation and parameters
- Cache hits/misses
- Retrieval and reranking details
- Performance timings
- Full error stack traces

## Technical Details

### Models

| Component | Model |
|-----------|-------|
| Embedding | `text-embedding-3-large` |
| Reranking | `rerank-v3.5` |
| Vector Store | ChromaDB |
| LLM | MCP Client's model |

### Design Philosophy

The server only retrieves and ranks documents. The MCP client's LLM generates answers, providing:
- Flexibility to use any LLM
- Lower operational costs
- Better context visibility

## License

MIT License

## Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [ChromaDB](https://www.trychroma.com/)
- [OpenAI](https://openai.com/)
- [Cohere](https://cohere.com/)
