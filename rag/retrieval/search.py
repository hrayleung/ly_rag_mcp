"""
Unified search engine combining different retrieval strategies.
"""

import re
import time
import threading
from typing import List, Optional, Tuple

from llama_index.core.retrievers import QueryFusionRetriever

from rag.config import settings, logger
from rag.models import SearchMode, SearchResult, RetrievalResult
from rag.storage.index import get_index_manager
from rag.storage.chroma import get_chroma_manager
from rag.retrieval.reranker import get_reranker_manager
from rag.retrieval.bm25 import get_bm25_manager
from rag.retrieval.hyde import generate_hyde_query, should_trigger_hyde


# Regex patterns for query analysis
HYBRID_QUERY_TOKENS = re.compile(r"[A-Z_]{2,}|::|->|\d{3,}|\w+\.\w+")
CODE_SNIPPET_RE = re.compile(r"[{}();=<>*/+-]")


class SearchEngine:
    """
    Unified search engine supporting multiple retrieval strategies.
    """

    def __init__(self):
        self._index_manager = None
        self._reranker_manager = None
        self._bm25_manager = None

    def _ensure_managers(self):
        """Lazily initialize managers to avoid import-time failures."""
        if self._index_manager is None:
            self._index_manager = get_index_manager()
        if self._reranker_manager is None:
            self._reranker_manager = get_reranker_manager()
        if self._bm25_manager is None:
            self._bm25_manager = get_bm25_manager()

    def search(
        self,
        question: str,
        similarity_top_k: int = None,
        search_mode: str = "semantic",
        use_rerank: bool = True,
        use_hyde: bool = False,
        project: Optional[str] = None
    ) -> SearchResult:
        """
        Execute a search query.

        Args:
            question: Query string
            similarity_top_k: Number of results to return
            search_mode: 'semantic', 'hybrid', or 'keyword'
            use_rerank: Enable Cohere reranking
            use_hyde: Enable HyDE for ambiguous queries
            project: Target project

        Returns:
            SearchResult containing retrieved documents
        """
        # Ensure managers are initialized
        self._ensure_managers()

        # Validate inputs
        question = self._validate_query(question)
        top_k = self._validate_top_k(similarity_top_k)

        # Determine effective search mode
        try:
            requested_mode = SearchMode(search_mode.lower())
        except ValueError:
            logger.warning(f"Invalid search mode '{search_mode}', defaulting to 'semantic'")
            requested_mode = SearchMode.SEMANTIC

        effective_mode = self._select_search_mode(question, requested_mode)

        # Get index
        index_start = time.perf_counter()
        index = self._index_manager.get_index(project)
        index_elapsed = time.perf_counter() - index_start
        current_project = self._index_manager.current_project
        logger.info(
            "search:get_index timing",
            extra={"project": current_project, "elapsed_ms": round(index_elapsed * 1000, 2)},
        )

        # Execute retrieval
        retrieve_start = time.perf_counter()
        nodes, search_query, rerank_used = self._retrieve(
            index=index,
            question=question,
            top_k=top_k,
            mode=effective_mode,
            use_rerank=use_rerank,
            use_hyde=use_hyde,
            project=current_project
        )
        retrieve_elapsed = time.perf_counter() - retrieve_start
        logger.info(
            "search:retrieve timing",
            extra={
                "project": current_project,
                "elapsed_ms": round(retrieve_elapsed * 1000, 2),
                "mode": effective_mode.value,
                "top_k": top_k,
                "rerank": rerank_used,
                "hyde": use_hyde,
            },
        )

        # Convert to results
        results = [
            RetrievalResult(
                text=node.node.get_content(),
                score=float(node.score) if node.score else None,
                metadata=node.node.metadata,
                node_id=node.node.node_id
            )
            for node in nodes
        ]

        return SearchResult(
            results=results,
            query=question,
            search_mode=effective_mode,
            reranked=rerank_used,
            used_hyde=(search_query != question),
            generated_query=search_query if use_hyde else None,
            project=current_project
        )

    def _validate_query(self, query: str) -> str:
        """Validate and clean query string."""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty or whitespace only")

        if len(query) > settings.max_query_length:
            logger.warning(f"Query too long ({len(query)}), truncating")
            query = query[:settings.max_query_length]

        if not query:
            raise ValueError("Query cannot be empty after truncation")

        return query

    def _validate_top_k(self, top_k: Optional[int]) -> int:
        """Validate and clamp top_k parameter."""
        if top_k is None:
            return settings.default_top_k

        if not isinstance(top_k, int):
            top_k = int(top_k)

        if top_k < settings.min_top_k:
            logger.warning(f"top_k={top_k} too small, using {settings.min_top_k}")
            return settings.min_top_k

        if top_k > settings.max_top_k:
            logger.warning(f"top_k={top_k} too large, using {settings.max_top_k}")
            return settings.max_top_k

        return top_k

    def _select_search_mode(
        self,
        question: str,
        requested: SearchMode
    ) -> SearchMode:
        """
        Select appropriate search mode based on query analysis.
        """
        # Respect ALL explicit mode requests including SEMANTIC (Bug M4)
        if requested in (SearchMode.KEYWORD, SearchMode.HYBRID, SearchMode.SEMANTIC):
            return requested

        # Analyze query for technical content
        has_hybrid_tokens = bool(HYBRID_QUERY_TOKENS.search(question))
        has_code_chars = bool(CODE_SNIPPET_RE.search(question))

        tokens = question.split()
        num_tokens = len(tokens)  # Cache len() call
        if num_tokens <= 1:
            return SearchMode.KEYWORD

        if any("/" in t or "\\" in t or ("." in t and len(t) > 3) for t in tokens):
            logger.debug("Path-like token detected, using hybrid search")
            return SearchMode.HYBRID

        # Cache repeated len(tokens) calls in ratio calculations
        uppercase_ratio = (
            sum(1 for t in tokens if t.isupper() and len(t) > 1) / num_tokens
            if tokens else 0
        )
        digit_ratio = (
            sum(1 for t in tokens if any(c.isdigit() for c in t)) / num_tokens
            if tokens else 0
        )

        # Switch to hybrid for technical queries
        if has_hybrid_tokens or has_code_chars:
            logger.debug("Technical content detected, using hybrid search")
            return SearchMode.HYBRID

        if uppercase_ratio > 0.3 or digit_ratio > 0.4:
            logger.debug("High uppercase/digit ratio, using hybrid search")
            return SearchMode.HYBRID

        return SearchMode.SEMANTIC

    def _retrieve(
        self,
        index,
        question: str,
        top_k: int,
        mode: SearchMode,
        use_rerank: bool,
        use_hyde: bool,
        project: str
    ) -> Tuple[List, str, bool]:
        """
        Execute retrieval operation.

        Returns:
            Tuple of (nodes, search_query, rerank_used)
        """
        # Calculate candidate count for reranking
        candidate_top_k = top_k
        if use_rerank:
            candidate_top_k = max(
                top_k * settings.rerank_candidate_multiplier,
                settings.min_rerank_candidates
            )

        # Clamp to available documents
        available = self._index_manager.get_node_count(project)
        if available > 0:
            candidate_top_k = min(candidate_top_k, available)

        # Get retriever based on mode
        retriever = self._get_retriever(index, mode, candidate_top_k, project)

        # Apply HyDE if requested
        search_query = question
        hyde_attempted = False

        if use_hyde:
            search_query = generate_hyde_query(question)
            hyde_attempted = True

        # Initial retrieval
        nodes = retriever.retrieve(search_query)

        # Auto-trigger HyDE if results are poor
        if not hyde_attempted and should_trigger_hyde(nodes):
            logger.info("HyDE fallback triggered")
            search_query = generate_hyde_query(question)
            nodes = retriever.retrieve(search_query)

        # Apply reranking with error handling
        rerank_used = False
        if self._reranker_manager.should_apply_rerank(nodes, use_rerank):
            try:
                nodes = self._reranker_manager.rerank(nodes, question, top_k)
                rerank_used = True
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")
                rerank_used = False
                # Slice to top_k on failure (reranker.rerank() already slices on success)
                nodes = nodes[:min(top_k, len(nodes))]
        else:
            nodes = nodes[:min(top_k, len(nodes))]

        return nodes, search_query, rerank_used

    def _get_retriever(
        self,
        index,
        mode: SearchMode,
        top_k: int,
        project: str
    ):
        """Get appropriate retriever based on search mode."""
        if mode == SearchMode.HYBRID:
            return self._get_hybrid_retriever(index, top_k, project)
        elif mode == SearchMode.KEYWORD:
            return self._get_keyword_retriever(index, top_k, project)
        else:
            return index.as_retriever(similarity_top_k=top_k)

    def _get_hybrid_retriever(self, index, top_k: int, project: str):
        """Get hybrid retriever combining vector and BM25."""
        self._ensure_managers()
        vector_retriever = index.as_retriever(similarity_top_k=top_k)

        chroma_manager = get_chroma_manager()
        _, collection = chroma_manager.get_client(project)

        # Validate collection before using for BM25 fallback
        if collection is not None:
            bm25_retriever = self._bm25_manager.get_retriever(index, collection, project)
        else:
            logger.warning("ChromaDB collection unavailable, BM25 disabled")
            bm25_retriever = None

        if bm25_retriever:
            bm25_retriever.similarity_top_k = top_k
            return QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=top_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
                verbose=True
            )

        logger.warning("BM25 unavailable, falling back to vector-only")
        return vector_retriever

    def _get_keyword_retriever(self, index, top_k: int, project: str):
        """Get BM25-only retriever."""
        self._ensure_managers()
        chroma_manager = get_chroma_manager()
        _, collection = chroma_manager.get_client(project)

        # Validate collection before using for BM25 fallback
        if collection is not None:
            bm25_retriever = self._bm25_manager.get_retriever(index, collection, project)
        else:
            logger.warning("ChromaDB collection unavailable, BM25 disabled")
            bm25_retriever = None

        if bm25_retriever:
            bm25_retriever.similarity_top_k = top_k
            return bm25_retriever

        logger.warning("BM25 unavailable, falling back to vector search")
        return index.as_retriever(similarity_top_k=top_k)


# Global singleton
_search_engine: Optional[SearchEngine] = None
_search_engine_lock = threading.Lock()


def get_search_engine() -> SearchEngine:
    """Get the global SearchEngine instance with thread-safe initialization."""
    global _search_engine
    if _search_engine is None:
        with _search_engine_lock:
            if _search_engine is None:
                _search_engine = SearchEngine()
    return _search_engine
