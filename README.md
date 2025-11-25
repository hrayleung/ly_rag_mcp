# LlamaIndex RAG MCP Server

Local RAG system with OpenAI embeddings, Cohere reranking, and MCP server integration.

## Features

- **Incremental Updates**: Only processes new/modified files, saves 95%+ API tokens
- **Two-Stage Retrieval**: Vector search + semantic reranking
- **Models**: OpenAI `text-embedding-3-large` + Cohere `rerank-v3.5`
- **MCP Integration**: Works with Claude Code, Chatwise, Cherry Studio
- **20+ File Formats**: PDF, DOCX, PPTX, MD, IPYNB, CSV, JSON, and more
- **Local Storage**: ChromaDB vector database
- **Flexible Indexing**: Command-line interface for any directory

## Installation

```bash
git clone https://github.com/hrayleung/ly_rag_mcp.git
cd ly_rag_mcp

conda create -n rag-env python=3.10
conda activate rag-env

# Core dependencies
pip install llama-index llama-index-embeddings-openai llama-index-vector-stores-chroma
pip install llama-index-postprocessor-cohere-rerank chromadb fastmcp

# New features (Hybrid Search & Web Crawling)
pip install rank-bm25 llama-index-retrievers-bm25 firecrawl-py
```

## Quick Start

### 1. Set API Keys

```bash
export OPENAI_API_KEY='your-openai-api-key'
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

```json
{
  "mcpServers": {
    "llamaindex-rag": {
      "command": "/path/to/conda/envs/rag-env/bin/python",
      "args": ["/path/to/ly_rag_mcp/mcp_server.py"],
      "cwd": "/path/to/ly_rag_mcp",
      "env": {
        "OPENAI_API_KEY": "your-openai-key",
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
| Documents | `.txt`, `.pdf`, `.docx`, `.md`, `.epub` |
| Presentations | `.ppt`, `.pptx`, `.pptm` |
| Data | `.csv`, `.json`, `.xml` |
| Code/Notebooks | `.ipynb`, `.html` |
| Media | `.jpg`, `.jpeg`, `.png`, `.mp3`, `.mp4` |
| Other | `.hwp`, `.mbox` |

File size limit: 300MB per file

## MCP Tools

### Query
- `query_rag(question, similarity_top_k=6, search_mode='semantic', use_rerank=True, use_hyde=False)`
  - **NEW**: `search_mode`: Choose 'semantic' (default), 'hybrid' (BM25+Vector), or 'keyword' (BM25)
  - **NEW**: `use_hyde`: Enable Hypothetical Document Embeddings for better conceptual matching
  - **NEW**: `similarity_top_k`: Fully flexible (1-50)
- `query_rag_with_sources(...)` - Same options as above, returns metadata
- `iterative_search(question, initial_top_k=3, detailed_top_k=10)` - Two-phase search

### Ingestion
- `inspect_directory(path)` - Analyze folder content
- `index_hybrid_folder(path, project)` - **NEW** Index mixed content (Code + Docs)
- `index_modified_files(path, project)` - Sync only new/changed files since last scan
- `crawl_website(url, max_depth=1, max_pages=10)` - Crawl websites
- `add_document_from_text(text, metadata)` - Add text dynamically
- `add_documents_from_directory(path, project)` - Bulk import
- `index_github_repository(...)` - Index GitHub repos
- `index_local_codebase(..., project)` - Index local code
- Reminder: Folder ingestion tools (including `index_modified_files`) require a `project` argument. If omitted, the server returns `project_confirmation_required` so you can confirm whether to create or reuse a workspace before indexing. The first run of `index_modified_files` seeds a baseline when the project already has data, so subsequent runs only add new or changed files.

### Advanced Features
- **Multi-Project**: Isolate indexes.
  - `create_project(name)`, `switch_project(name)`, `list_projects()`
- **Smart Project Routing**: Queries scan project workspaces and pick the best match (explicit mentions win).
- **Smart Splitting**: Automatically uses AST (tree-sitter) for code and sentence splitting for docs.

### Management
- `inspect_directory(path)` - Analyze folder content
- `get_index_stats()` - Index statistics
- `get_cache_stats()` - Cache performance metrics
- `list_indexed_documents()` - List all documents
- `clear_index()` - Clear index

### Debugging
- `python debug_rag.py` - Diagnostics tool
- Set `RAG_LOG_LEVEL=DEBUG` for detailed logging

### Resources
- `rag://documents` - Browse documents
- `rag://stats` - View statistics

## Architecture

```
Query → HyDE? → Hybrid Search?
                     ↓
Code → AST Splitter ─┐           Top Candidates
Docs → Text Splitter ┴→ Vector       ↓
                          Store → Reranker → LLM
```

## Search Strategies

### Hybrid Search (Technical Terms)
`search_mode='hybrid'` combines vector + keyword search. Best for error codes and acronyms.

### Workspaces
1. `create_project("backend")`
2. `switch_project("backend")`
3. Index backend files.
4. Switch to "frontend" to keep contexts clean.
5. Or just ask about "frontend" directly—queries mentioning a project name auto-route, and the server falls back to the most relevant project when nothing is explicit.

## Reranking


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

## Reranking

Two-stage retrieval process:

1. **Vector Search**: Retrieve 2x candidates (e.g., 20 candidates for top 10 results, minimum 10)
2. **Reranking**: Cohere v3.5 reorders by semantic relevance, returns requested top N

The system dynamically adjusts candidate retrieval based on your `similarity_top_k` setting:
- Request 3 results → retrieves 10 candidates → reranks → returns top 3
- Request 10 results → retrieves 20 candidates → reranks → returns top 10
- Request 15 results → retrieves 30 candidates → reranks → returns top 15

**Enable**: Add `COHERE_API_KEY` to MCP config
**Disable**: Set `use_rerank=False` in queries

## Configuration

### Embedding Models

Edit `build_index.py` and `mcp_server.py`:

```python
# Current
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")  # 3072 dims

# Alternative
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")  # 1536 dims
```

### Reranking Models

Edit `mcp_server.py`:

```python
model="rerank-v3.5"      # Current
model="rerank-3-nimble"  # Faster alternative
```

## Project Structure

```
ly_rag_mcp/
├── build_index.py          # Index builder with incremental update support
├── mcp_server.py           # FastMCP server
├── verify_setup.py         # Setup verification
├── .env.example            # Environment template
├── .gitignore              # Git ignore
├── README.md               # This file
├── data/                   # Documents (excluded from git)
└── storage/                # Generated index (excluded from git)
    ├── chroma_db/          # Vector database
    ├── indexed_files.json  # Tracking file for incremental updates
    └── *.json              # Index metadata
```

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
