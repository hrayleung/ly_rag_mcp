# LlamaIndex RAG MCP Server

Local RAG system with OpenAI embeddings, Cohere reranking, and MCP server integration.

## Features

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

pip install llama-index llama-index-embeddings-openai llama-index-vector-stores-chroma
pip install llama-index-postprocessor-cohere-rerank chromadb fastmcp
```

## Quick Start

### 1. Set API Keys

```bash
export OPENAI_API_KEY='your-openai-api-key'
export COHERE_API_KEY='your-cohere-api-key'  # Optional
```

### 2. Index Documents

```bash
# Index default ./data directory
python build_index.py

# Index custom directory
python build_index.py /path/to/your/documents
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
        "COHERE_API_KEY": "your-cohere-key"
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

File size limit: 10MB per file

## MCP Tools

### Query
- `query_rag(question, similarity_top_k=3, use_rerank=True)` - Retrieve documents
- `query_rag_with_sources(question, similarity_top_k=3, use_rerank=True)` - With metadata

### Management
- `get_index_stats()` - Index statistics
- `list_indexed_documents()` - List all documents
- `add_document_from_text(text, metadata)` - Add text dynamically
- `add_documents_from_directory(path)` - Bulk import
- `clear_index()` - Clear index

### Resources
- `rag://documents` - Browse documents
- `rag://stats` - View statistics

## Architecture

```
Documents → text-embedding-3-large → ChromaDB
                                         ↓
Query → Embedding → Vector Search → 10 candidates
                                         ↓
                            Cohere Rerank v3.5
                                         ↓
                            Top 3 documents → MCP Client
                                                    ↓
                                             Client's LLM
```

## Reranking

Two-stage retrieval process:

1. **Vector Search**: Retrieve 10 candidates via embedding similarity
2. **Reranking**: Cohere v3.5 reorders by semantic relevance, returns top 3

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
├── build_index.py          # Index builder with CLI
├── mcp_server.py           # FastMCP server
├── verify_setup.py         # Setup verification
├── .env.example            # Environment template
├── .gitignore              # Git ignore
├── mcp_config_example.json # MCP config template
├── RERANK_GUIDE.md         # Reranking documentation
├── README.md               # This file
├── data/                   # Documents (excluded from git)
└── storage/                # Generated index (excluded from git)
```

## Troubleshooting

### MCP Server Not Connecting
1. Verify Python path in MCP config
2. Check API keys in `env` section
3. Ensure `cwd` points to project directory
4. Restart MCP client

### No Documents Retrieved
```bash
python -c "from mcp_server import get_index_stats; print(get_index_stats())"
```

### Update Index
```bash
python build_index.py /path/to/updated/documents
```

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
