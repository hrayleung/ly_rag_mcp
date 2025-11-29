# Repository Guidelines

## Project Structure & Module Organization
This repo is script-driven: `build_index.py` ingests folders with incremental tracking (`storage/indexed_files.json`), `mcp_server.py` exposes FastMCP tools, while `debug_rag.py` and `verify_setup.py` are diagnostics. Place raw docs under `data/` (git-ignored) and expect generated vectors under `storage/`. `.env.example` documents the required keys; `mcp.json` mirrors the MCP client entry.

## Build, Test, and Development Commands
- `conda create -n rag-env python=3.10 && conda activate rag-env` (baseline interpreter).
- `pip install llama-index llama-index-embeddings-openai llama-index-vector-stores-chroma llama-index-postprocessor-cohere-rerank chromadb fastmcp rank-bm25 llama-index-retrievers-bm25 firecrawl-py` – install dependencies listed in the README.
- `python build_index.py ./data` for incremental updates; add `--rebuild` before changing embedding settings.
- `python mcp_server.py` (or reference it in `mcp.json`) to register the tools with Claude/Cherry Studio.
- `python verify_setup.py` ensures env vars and storage paths are reachable.
- `python debug_rag.py` exercises retrieval/rerank paths; treat it as the integration test.

## Project Metadata & Routing
Start ingestion by running `list_projects()` and `inspect_directory(<path>)`, then `create_project`/`switch_project` before indexing. Lock in intent via `set_project_metadata(project="frontend", keywords=["nextjs","api"], description="Customer portal docs")` so auto-selection has real tags. If unsure which corpus to search, call `choose_project(question="user request")` and surface the recommendation before `query_rag`.

## Coding Style & Naming Conventions
Follow PEP8, 4-space indents, and snake_case helpers. Public functions carry type hints and concise docstrings; module-level constants (e.g., `SUPPORTED_EXTS`) stay upper snake case near the top. Prefer `pathlib.Path` over `os.path`, gate behavior behind environment checks, and log with the shared `logger` instead of bare prints (except for intentional CLI status).

## Testing Guidelines
There is no pytest suite—use the CLI scripts. Run `python verify_setup.py` after dependency or env changes, then `python build_index.py ./data --rebuild` against a small fixture directory to validate ingestion. Finish with `python debug_rag.py --profile` or a representative MCP query (`query_rag("parallel computing patterns")`) to confirm retrieval quality.

## Commit & Pull Request Guidelines
History follows conventional commits (`feat(rag): inject folder context`). Keep the subject <72 chars, describe user-facing effects in the body, and mention any new env vars or CLI flags. PRs should include: scope summary, test/command output snippets, links to related issues, and screenshots/GIFs only when UI-facing (otherwise, sample query transcripts suffice).

## Security & Configuration Tips
Never commit `.env`, `data/`, or `storage/`; they contain API keys and embeddings. Document any new configuration knobs in `.env.example` and `README.md`, and remind reviewers to export `OPENAI_API_KEY`, `COHERE_API_KEY`, and optional `FIRECRAWL_API_KEY`. When logging, scrub secrets and prefer the provided `RAG_LOG_LEVEL=DEBUG` toggle for deep dives.
