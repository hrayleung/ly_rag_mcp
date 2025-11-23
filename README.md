# LlamaIndex RAG MCP Server

A production-ready Retrieval-Augmented Generation (RAG) system built with LlamaIndex, featuring advanced two-stage retrieval with OpenAI embeddings and Cohere reranking, exposed via Model Context Protocol (MCP) for seamless integration with AI assistants.

## Features

- 🚀 **Two-Stage Retrieval**: Initial vector search + semantic reranking for 15-40% accuracy improvement
- 🎯 **Advanced Models**: OpenAI `text-embedding-3-large` + Cohere `rerank-v3.5`
- 🔌 **MCP Integration**: Query from Claude Code, Chatwise, Cherry Studio, or any MCP client
- 📁 **20+ File Formats**: PDF, DOCX, PPTX, MD, IPYNB, CSV, JSON, and more
- 💾 **Local Storage**: All embeddings stored locally in ChromaDB
- 💰 **Cost Optimized**: Uses MCP client's LLM for generation (no OpenAI LLM costs)
- 🌍 **Multilingual**: Supports 100+ languages including Chinese, English, and more
- ⚡ **Flexible Indexing**: Index any directory via simple command-line interface

## Architecture

```
Documents → text-embedding-3-large → ChromaDB Vector Store
                                           ↓
Query → Embedding → Vector Search → Top 10 candidates
                                           ↓
                              Cohere Rerank v3.5 (optional)
                                           ↓
                              Top 3 most relevant → MCP Client
                                                          ↓
                                                   Client LLM → Answer
```

**Key Design**: The system retrieves and ranks documents, while your MCP client's LLM generates the final answer, saving costs and providing flexibility.

## Quick Start

### Prerequisites

- Python 3.10+ (conda environment recommended)
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Cohere API key (optional, [get one here](https://dashboard.cohere.com/api-keys))

### Installation

```bash
# Clone the repository
git clone https://github.com/hrayleung/ly_ray_mcp.git
cd ly_ray_mcp

# Create conda environment
conda create -n rag-env python=3.10
conda activate rag-env

# Install dependencies
pip install llama-index llama-index-embeddings-openai llama-index-vector-stores-chroma
pip install llama-index-postprocessor-cohere-rerank chromadb fastmcp
```

### Configuration

1. **Set API Keys**

```bash
export OPENAI_API_KEY='your-openai-api-key'
export COHERE_API_KEY='your-cohere-api-key'  # Optional but recommended
```

2. **Index Your Documents**

```bash
# Index default ./data directory
python build_index.py

# Or specify custom directory
python build_index.py /path/to/your/documents
python build_index.py ~/Documents/research
```

3. **Configure MCP Client**

Add to your MCP client configuration file:

```json
{
  "mcpServers": {
    "llamaindex-rag": {
      "command": "/path/to/conda/envs/rag-env/bin/python",
      "args": ["/path/to/ly_ray_mcp/mcp_server.py"],
      "cwd": "/path/to/ly_ray_mcp",
      "env": {
        "OPENAI_API_KEY": "your-openai-key",
        "COHERE_API_KEY": "your-cohere-key"
      }
    }
  }
}
```

**Config locations:**
- **Claude Code**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
- **Chatwise/Cherry Studio**: Check application settings

4. **Restart MCP Client & Query**

```
What are these documents about?

Use query_rag to search for information about [topic]

Show me the index statistics
```

## Supported File Formats

| Category | Formats |
|----------|---------|
| **Documents** | `.txt`, `.pdf`, `.docx`, `.md`, `.epub` |
| **Presentations** | `.ppt`, `.pptx`, `.pptm` |
| **Data** | `.csv`, `.json`, `.xml` |
| **Code & Notebooks** | `.ipynb`, `.html` |
| **Media** | `.jpg`, `.jpeg`, `.png`, `.mp3`, `.mp4` |
| **Other** | `.hwp`, `.mbox` |

**File size limit**: 10MB per file (configurable)

## Available MCP Tools

### Query Tools

- **`query_rag(question, similarity_top_k=3, use_rerank=True)`**
  - Retrieve relevant documents with optional reranking
  - Returns formatted text chunks for the MCP client's LLM to process

- **`query_rag_with_sources(question, similarity_top_k=3, use_rerank=True)`**
  - Returns structured data with full source attribution and metadata

### Management Tools

- **`get_index_stats()`** - View document count and system status
- **`list_indexed_documents()`** - List all indexed documents with previews
- **`add_document_from_text(text, metadata)`** - Add documents dynamically
- **`add_documents_from_directory(path)`** - Bulk import from directory
- **`clear_index()`** - Reset the entire index

### MCP Resources

- **`rag://documents`** - Browse indexed documents
- **`rag://stats`** - View system statistics

## Two-Stage Retrieval

The system implements a sophisticated two-stage retrieval pipeline:

### Stage 1: Vector Search
- Converts query to embedding via `text-embedding-3-large`
- Performs semantic similarity search in ChromaDB
- Retrieves top 10 candidate documents

### Stage 2: Reranking (Optional)
- Applies Cohere `rerank-v3.5` for deep semantic understanding
- Reorders candidates by true relevance
- Returns top 3 most relevant documents

**Performance**: 15-40% accuracy improvement with reranking enabled.

## Cost Analysis

### Storage Costs

| Documents | Storage Size | Growth Rate |
|-----------|-------------|-------------|
| 500 | ~70 MB | Linear |
| 1,000 | ~140 MB | ~140 KB/doc |
| 5,000 | ~685 MB | Predictable |
| 10,000 | ~1.4 GB | No surprise growth |

**Storage Breakdown:**
- `chroma.sqlite3`: ~75% (vector embeddings)
- Vector index: ~25%
- Metadata JSON: < 1 MB

**Space Efficiency**: Index uses only 2-3% of original document size.

### API Costs (per 1,000 queries)

| Component | Cost | Notes |
|-----------|------|-------|
| OpenAI Embedding | $0.05 | Required for vector search |
| Cohere Rerank | $0.50 | Optional, highly recommended |
| LLM Generation | $0.00 | Uses MCP client's LLM |
| **Total** | **$0.05-$0.55** | Extremely cost-effective |

## Usage Examples

### Basic Workflow

```bash
# 1. Add documents to a directory
mkdir my-docs
cp *.pdf *.pptx *.md my-docs/

# 2. Index the directory
python build_index.py my-docs

# 3. Query via MCP client
"Summarize the key concepts in my documents"
"Find information about [specific topic]"
```

### Advanced Queries

**In your MCP client:**

```
# Basic search
Use query_rag to find information about machine learning

# Search with source attribution
Use query_rag_with_sources to get details about neural networks

# Check system status
How many documents are in my RAG index?

# List all documents
Show me all indexed documents with their metadata

# Disable reranking for faster (but less accurate) results
Use query_rag with use_rerank=False to search for "algorithms"
```

### Re-indexing

```bash
# Add new documents to your directory
cp new-paper.pdf my-docs/

# Rebuild index (overwrites existing)
python build_index.py my-docs

# Index a different directory entirely
python build_index.py ~/Documents/new-project
```

## Configuration Options

### Embedding Models

Edit `build_index.py` and `mcp_server.py`:

```python
# Current (best quality)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")  # 3072 dims

# Alternative (50% smaller, slightly lower quality)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")  # 1536 dims
```

### Reranking Models

Edit `mcp_server.py`:

```python
# Current (best quality)
model="rerank-v3.5"

# Alternative (faster)
model="rerank-3-nimble"
```

### Retrieval Parameters

```python
# Retrieve more documents
query_rag(question, similarity_top_k=5, use_rerank=True)

# Disable reranking
query_rag(question, similarity_top_k=3, use_rerank=False)
```

## Project Structure

```
ly_ray_mcp/
├── build_index.py          # CLI tool for indexing documents
├── mcp_server.py           # FastMCP server with retrieval tools
├── verify_setup.py         # Setup verification script
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── mcp_config_example.json # MCP client configuration template
├── RERANK_GUIDE.md         # Detailed reranking documentation
├── README.md               # This file
├── data/                   # Default document directory (excluded from git)
└── storage/                # Generated index (excluded from git)
    ├── chroma_db/          # ChromaDB vector store
    └── *.json              # Index metadata
```

## Troubleshooting

### MCP Server Not Connecting

1. **Verify Python path** in MCP config points to your conda environment
2. **Check API keys** are set in `env` section of MCP config
3. **Ensure `cwd`** points to the project directory
4. **Restart MCP client** after configuration changes

### No Documents Retrieved

```bash
# Check index status
python -c "from mcp_server import get_index_stats; print(get_index_stats())"

# Expected output: {"status": "ready", "document_count": N, ...}
```

### Encoding Errors

The system automatically handles encoding issues by:
- Removing surrogate characters
- Cleaning non-UTF8 characters
- Skipping problematic documents (logs warnings)

### Unsupported File Formats

Files with unsupported extensions are automatically skipped during indexing with console notification.

## Performance Optimization

### For Speed
- Use `text-embedding-3-small` (50% faster indexing)
- Disable reranking: `use_rerank=False`
- Reduce `similarity_top_k` to 1-2

### For Accuracy
- Use `text-embedding-3-large` (current default)
- Enable reranking: `use_rerank=True` (default)
- Increase `similarity_top_k` to 5-7

### For Cost Savings
- Use `text-embedding-3-small` ($0.02 vs $0.13 per 1M tokens)
- Disable reranking (saves $0.50 per 1K queries)
- MCP client's LLM already saves costs vs. using OpenAI's LLM

## Security & Privacy

- ✅ **API keys** stored only in environment variables and MCP config (never in code)
- ✅ **Local storage** - All embeddings and documents stay on your machine
- ✅ **No telemetry** - ChromaDB telemetry disabled by default
- ✅ **Data exclusion** - User documents (in `data/`) excluded from git by default

## Technical Details

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Embedding | `text-embedding-3-large` | Convert text to 3072-dim vectors |
| Reranking | `rerank-v3.5` | Semantic relevance scoring |
| Vector Store | ChromaDB | Local vector database |
| LLM | MCP Client's Model | Answer generation (not in this server) |

### Why No LLM in Server?

This design saves costs and provides flexibility:
- **Cost**: No OpenAI LLM charges (save ~$0.50 per 1K queries)
- **Flexibility**: Use any LLM via your MCP client (Claude, GPT-4, local models)
- **Quality**: Often better than using gpt-4o-mini
- **Transparency**: MCP client sees retrieved documents for better context

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## License

MIT License

## Acknowledgments

Built with:
- [LlamaIndex](https://www.llamaindex.ai/) - Data framework for LLM applications
- [FastMCP](https://github.com/jlowin/fastmcp) - Fast MCP server framework
- [ChromaDB](https://www.trychroma.com/) - Open-source embedding database
- [OpenAI](https://openai.com/) - Embedding models
- [Cohere](https://cohere.com/) - Reranking models

## Support

For issues, questions, or suggestions, please [open an issue](https://github.com/hrayleung/ly_ray_mcp/issues).

---

**Built with LlamaIndex 🦙 | Powered by OpenAI & Cohere**
